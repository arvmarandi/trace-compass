#!/bin/bash
set -e

# Full pipeline: mini-swe-agent (trace generation) → KGCompass (patch generation)
#
# Usage:
#   ./pipeline.sh <instance_id|all> [subset]
#
# Arguments:
#   instance_id   SWE-bench instance ID, e.g. django__django-10914
#                 Pass "all" to run every instance in the configured subset
#   subset        "verified" (default) or "lite" — overrides SUBSET in config.env
#
# Environment variables (can also be set in config.env):
#   MODEL         LLM in litellm format, e.g. deepseek/deepseek-v4-flash
#   SUBSET        SWE-bench subset: "verified" (default) or "lite"
#   PARALLEL      Number of instances to run concurrently in batch mode (default: 1)
#   SKIP_TRACES   Set to 1 to skip trace generation when trajectory already exists

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source shared config
if [ -f "$SCRIPT_DIR/config.env" ]; then
    set -a
    source "$SCRIPT_DIR/config.env"
    set +a
fi

INSTANCE_ID=${1:-}
# Positional arg overrides config.env SUBSET
if [ -n "${2:-}" ]; then
    SUBSET="$2"
fi
SUBSET=${SUBSET:-"verified"}
MODEL=${MODEL:-"deepseek/deepseek-chat"}
PARALLEL=${PARALLEL:-1}

if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: $0 <instance_id|all> [subset]"
    echo "Example: $0 django__django-10914"
    echo "         $0 all verified"
    exit 1
fi

MINI_DIR="$SCRIPT_DIR/mini-swe-agent"
KGCOMPASS_DIR="$SCRIPT_DIR/kgCompass"
TRACES_DIR="$MINI_DIR/outputs/stack-traces"

case "$SUBSET" in
    "verified") DATASET="princeton-nlp/SWE-Bench_Verified" ;;
    "lite")     DATASET="princeton-nlp/SWE-Bench_Lite" ;;
    *)          DATASET="$SUBSET" ;;
esac

# ----------------------------------------------------------------
# run_one: run the full pipeline for a single instance
# ----------------------------------------------------------------
run_one() {
    local instance_id="$1"
    local traj_file="$TRACES_DIR/$instance_id/$instance_id.traj.json"

    echo ""
    echo "================================================="
    echo "Pipeline: $instance_id  (subset: $SUBSET, model: $MODEL)"
    echo "================================================="

    # Step 1: Trace generation
    echo -e "\n--- Step 1: Trace Generation (mini-swe-agent) ---"
    if [ "${SKIP_TRACES:-0}" = "1" ] && [ -f "$traj_file" ]; then
        echo "✅ Trajectory already exists, skipping."
    else
        cd "$MINI_DIR"
        python -m minisweagent.run.benchmarks.swebench \
            --subset "$SUBSET" \
            --split test \
            --instance-ids "$instance_id" \
            --model "$MODEL" \
            --output "$TRACES_DIR" \
            --workers 1
        echo "✅ Trajectory saved to $traj_file"
        cd "$SCRIPT_DIR"
    fi

    # Step 2: KGCompass repair
    echo -e "\n--- Step 2: KGCompass Repair ---"
    cd "$KGCOMPASS_DIR"
    TRACES_DIR="$TRACES_DIR" SUBSET="$SUBSET" ./run_repair.sh "$instance_id"
    cd "$SCRIPT_DIR"

    echo "✅ Done: $instance_id"
}

export -f run_one
export SCRIPT_DIR MINI_DIR KGCOMPASS_DIR TRACES_DIR SUBSET MODEL DATASET SKIP_TRACES

# ----------------------------------------------------------------
# Batch mode
# ----------------------------------------------------------------
if [ "$INSTANCE_ID" = "all" ]; then
    echo "Fetching instance IDs for subset '$SUBSET' ($DATASET)..."
    INSTANCE_IDS=$(python3 -c "
from datasets import load_dataset
ds = load_dataset('$DATASET', split='test')
for row in ds:
    print(row['instance_id'])
")
    TOTAL=$(echo "$INSTANCE_IDS" | wc -l | tr -d ' ')
    echo "Running pipeline for $TOTAL instances (parallelism: $PARALLEL)"

    if [ "$PARALLEL" -gt 1 ]; then
        echo "$INSTANCE_IDS" | xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}
    else
        while IFS= read -r iid; do
            run_one "$iid" || echo "⚠️  Failed: $iid (continuing)"
        done <<< "$INSTANCE_IDS"
    fi

    echo -e "\n================================================="
    echo "Batch complete for subset: $SUBSET"
    echo "================================================="
else
    run_one "$INSTANCE_ID"
fi
