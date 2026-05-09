#!/bin/bash
set -e

# Full pipeline: mini-swe-agent (trace generation) → KGCompass (patch generation)
#
# Usage:
#   ./pipeline.sh <instance_id> [subset]
#
# Arguments:
#   instance_id   SWE-bench instance ID, e.g. django__django-12345
#   subset        SWE-bench subset: "verified" (default) or "lite"
#
# Environment variables:
#   MODEL         Model for mini-swe-agent (default: claude-sonnet-4-5)
#   SKIP_TRACES   Set to 1 to skip trace generation if traj already exists

INSTANCE_ID=$1
SUBSET=${2:-"verified"}
MODEL=${MODEL:-"claude-sonnet-4-5"}

if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: $0 <instance_id> [subset]"
    echo "Example: $0 django__django-12345 verified"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINI_DIR="$SCRIPT_DIR/mini-swe-agent"
KGCOMPASS_DIR="$SCRIPT_DIR/kgCompass"
TRACES_DIR="$MINI_DIR/outputs/stack-traces"
TRAJ_FILE="$TRACES_DIR/$INSTANCE_ID/$INSTANCE_ID.traj.json"

echo "================================================="
echo "Pipeline: $INSTANCE_ID  (subset: $SUBSET)"
echo "================================================="

# --- Step 1: Generate stack trace via mini-swe-agent ---
echo -e "\n--- Step 1: Trace Generation (mini-swe-agent) ---"

if [ "${SKIP_TRACES:-0}" = "1" ] && [ -f "$TRAJ_FILE" ]; then
    echo "✅ Trajectory already exists, skipping trace generation."
else
    cd "$MINI_DIR"
    python -m minisweagent.run.benchmarks.swebench \
        --subset "$SUBSET" \
        --split test \
        --instance-ids "$INSTANCE_ID" \
        --model "$MODEL" \
        --output "$TRACES_DIR" \
        --workers 1
    echo "✅ Trajectory saved to $TRAJ_FILE"
    cd "$SCRIPT_DIR"
fi

# --- Step 2: KGCompass repair ---
echo -e "\n--- Step 2: KGCompass Repair ---"
cd "$KGCOMPASS_DIR"
TRACES_DIR="$TRACES_DIR" ./run_repair.sh "$INSTANCE_ID"
cd "$SCRIPT_DIR"

echo -e "\n================================================="
echo "Pipeline complete for: $INSTANCE_ID"
echo "================================================="
