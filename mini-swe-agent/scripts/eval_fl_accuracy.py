#!/usr/bin/env python3
"""
Compare function-level fault localization accuracy:
  - KGCompass: Jaccard of predicted functions vs ground truth
  - Stack traces: Jaccard of functions with call_depth <= k (k=0..10) vs ground truth

Metric per arxiv 2503.21710 eq. (2-4):
    a_i^func = |E*_i ∩ Ê_i| / |E*_i ∪ Ê_i|
    FuncAcc  = (1/|S|) Σ a_i^func

Usage:
    python scripts/eval_fl_accuracy.py \\
        --tests-dir kgCompass/tmp/tests \\
        --traces-dir mini-swe-agent/outputs/stack-traces
"""

import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Ground truth from golden patch
# ---------------------------------------------------------------------------

def gt_functions(patch: str) -> set[str]:
    """Function/class names from @@ hunk headers only (not +def/-def body lines)."""
    names = set()
    for m in re.finditer(r'^@@[^@]+@@\s*(.*)', patch, re.MULTILINE):
        ctx = m.group(1).strip()
        name_m = re.match(r'(?:async\s+)?(?:def|class)\s+(\w+)', ctx)
        if name_m:
            names.add(name_m.group(1))
    return names


# ---------------------------------------------------------------------------
# Predicted sets
# ---------------------------------------------------------------------------

def _short_name(qualified: str) -> str:
    return qualified.split('.')[-1] if qualified else qualified


def load_kg_funcs(instance_id: str, tests_dir: Path, top_k: int | None = None) -> set[str]:
    """Short function/class names from kg_locations only, sorted by similarity score.

    Uses only kg_locations (not llm_locations or final_locations) so that we
    evaluate the KG-mined candidates in isolation, before any LLM augmentation.
    Entities are sorted by descending similarity and capped at top_k if given.
    """
    path = tests_dir / f"{instance_id}_gemini" / "kg_locations" / f"{instance_id}.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    entities = data.get("related_entities", {})
    all_entities = entities.get("methods", []) + entities.get("classes", [])
    all_entities.sort(key=lambda e: e.get("similarity", 0.0), reverse=True)
    if top_k is not None:
        all_entities = all_entities[:top_k]
    return {_short_name(e["name"]) for e in all_entities if e.get("name")}


def load_trace_funcs_by_depth(instance_id: str, traces_dir: Path, max_depth: int) -> set[str]:
    """Function names from all_calls where call_depth <= max_depth."""
    traj_path = traces_dir / instance_id / f"{instance_id}.traj.json"
    if not traj_path.exists():
        return set()

    info = json.loads(traj_path.read_text()).get("info", {})
    settrace = info.get("settrace_traces", {})
    if not settrace:
        return set()

    funcs: set[str] = set()
    for entry in settrace.values():
        calls = entry.get("all_calls", entry) if isinstance(entry, dict) else entry
        for frame in calls:
            depth = frame.get("call_depth")
            if depth is not None and depth <= max_depth and frame.get("func"):
                funcs.add(frame["func"])
    return funcs


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------

def jaccard(pred: set, truth: set) -> float:
    if not pred and not truth:
        return 1.0
    union = pred | truth
    return len(pred & truth) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-dir",  default="kgCompass/tmp/tests",
                        help="Directory with <id>_gemini/ KGCompass output subdirs")
    parser.add_argument("--traces-dir", default=None,
                        help="Directory with per-instance stack trace .traj.json files")
    parser.add_argument("--max-depth",  type=int, default=10,
                        help="Maximum call depth to evaluate (default: 10)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Limit KG candidates to top-k by similarity (default: all). "
                             "Paper uses 15.")
    args, _ = parser.parse_known_args()

    tests_dir  = Path(args.tests_dir)
    traces_dir = Path(args.traces_dir) if args.traces_dir else None
    max_depth  = args.max_depth
    top_k      = args.top_k

    print("Loading SWE-bench dataset...")
    from datasets import load_dataset
    instances = {inst["instance_id"]: inst
                 for inst in load_dataset("princeton-nlp/SWE-bench_Lite", split="test")}
    print(f"Loaded {len(instances)} instances.\n")

    # Collect per-instance results
    rows = []
    skipped = 0

    for run_dir in sorted(tests_dir.glob("*_gemini")):
        instance_id = run_dir.name.replace("_gemini", "")
        if instance_id not in instances:
            continue

        patch = instances[instance_id].get("patch", "")
        if not patch:
            continue

        truth = gt_functions(patch)
        if not truth:
            continue

        kg_funcs = load_kg_funcs(instance_id, tests_dir, top_k=top_k)
        if not kg_funcs:
            skipped += 1
            continue

        row = {
            "instance_id": instance_id,
            "repo":        instances[instance_id].get("repo", ""),
            "truth":       sorted(truth),
            "kg_func_acc": jaccard(kg_funcs, truth),
        }

        if traces_dir is not None:
            row["trace_func_acc"] = {}
            for k in range(max_depth + 1):
                tr_funcs = load_trace_funcs_by_depth(instance_id, traces_dir, k)
                row["trace_func_acc"][k] = jaccard(tr_funcs, truth)

        rows.append(row)

    n = len(rows)
    if n == 0:
        print("No results.")
        return

    # ---- Summary table ----
    kg_mean = sum(r["kg_func_acc"] for r in rows) / n

    print(f"Instances evaluated: {n}  (skipped {skipped} with no KG prediction)\n")
    print(f"{'Source':<30}  {'FuncAcc':>8}")
    print("-" * 42)
    kg_label = f"KGCompass (top-{top_k})" if top_k else "KGCompass (all)"
    print(f"  {kg_label:<28}  {kg_mean:>8.1%}")

    if traces_dir is not None:
        trace_rows = [r for r in rows if "trace_func_acc" in r]
        m = len(trace_rows)
        for k in range(max_depth + 1):
            mean_k = sum(r["trace_func_acc"][k] for r in trace_rows) / m
            print(f"  {'Stack trace (depth ≤ ' + str(k) + ')':<28}  {mean_k:>8.1%}")

    # ---- Per-repo breakdown ----
    by_repo: dict[str, list] = {}
    for r in rows:
        by_repo.setdefault(r["repo"], []).append(r)

    if traces_dir is not None:
        depths_to_show = [0, 1, 3, 5, 10]
        depths_to_show = [k for k in depths_to_show if k <= max_depth]
        header = f"  {'Repo':<43}  {'n':>4}  {'KG':>7}"
        for k in depths_to_show:
            header += f"  {'TR≤' + str(k):>7}"
        print(f"\n{header}")
        print("-" * (len(header) + 5))
        for repo, recs in sorted(by_repo.items()):
            kg_r = sum(r["kg_func_acc"] for r in recs) / len(recs)
            line = f"  {repo:<43}  {len(recs):>4}  {kg_r:>7.1%}"
            for k in depths_to_show:
                tr_r = sum(r["trace_func_acc"][k] for r in recs if "trace_func_acc" in r)
                tr_n = sum(1 for r in recs if "trace_func_acc" in r)
                line += f"  {tr_r/tr_n if tr_n else 0:>7.1%}"
            print(line)
    else:
        print(f"\n{'Repo':<45}  {'n':>4}  {'FuncAcc':>8}")
        print("-" * 62)
        for repo, recs in sorted(by_repo.items()):
            fa = sum(r["kg_func_acc"] for r in recs) / len(recs)
            print(f"  {repo:<43}  {len(recs):>4}  {fa:>8.1%}")

    # ---- Save ----
    out_path = tests_dir / "fl_accuracy.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
