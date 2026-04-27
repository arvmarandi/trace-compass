#!/usr/bin/env python3
"""
Rank-based evaluation of KG fault localization candidates.

For each instance, ranks KG candidates by:
  - Baseline:   original KG similarity score
  - Augmented:  similarity + TRACE_WEIGHT * trace_score(f)
                where trace_score(f) = 1/(call_depth+1)     if f in all_calls
                                       1/(depth_from_inner+1) if f in exception_frames
                                       (max of both taken)

Metrics reported (with and without augmentation):
  - Recall@k  (k = 1, 3, 5, 10, 15, 20): fraction of instances where a buggy
               function appears in the top-k candidates
  - MRR       (Mean Reciprocal Rank): mean of 1/rank of first buggy function hit

Ground truth: functions from @@ hunk headers of the golden patch (same as
eval_settrace_recall.py — existing modified functions only, not new defs).

Usage:
    python scripts/eval_kg_ranking.py \\
        --tests-dir   kgCompass/tmp/tests \\
        --traces-dir  mini-swe-agent/outputs/stack-traces \\
        --trace-weight 0.3
"""

import argparse
import json
import re
from pathlib import Path


RECALL_K = [1, 3, 5, 10, 15, 20]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def gt_functions(patch: str) -> set[str]:
    names = set()
    for m in re.finditer(r'^@@[^@]+@@\s*(.*)', patch, re.MULTILINE):
        ctx = m.group(1).strip()
        nm = re.match(r'(?:async\s+)?(?:def|class)\s+(\w+)', ctx)
        if nm:
            names.add(nm.group(1))
    return names


# ---------------------------------------------------------------------------
# KG candidates
# ---------------------------------------------------------------------------

def _short(name: str) -> str:
    return name.split('.')[-1] if name else ''


def load_kg_candidates(instance_id: str, tests_dir: Path) -> list[dict]:
    """Return list of candidate dicts from kg_locations, sorted by similarity desc."""
    path = tests_dir / f"{instance_id}_gemini" / "kg_locations" / f"{instance_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    entities = data.get("related_entities", {})
    candidates = entities.get("methods", []) + entities.get("classes", [])
    candidates.sort(key=lambda e: e.get("similarity", 0.0), reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Trace scores
# ---------------------------------------------------------------------------

def load_trace_scores(instance_id: str, traces_dir: Path) -> dict:
    """Return {(file_path, func_name): score} from settrace_traces.

    score = max(
        1/(depth_from_innermost+1)  if in exception_frames,
        1/(call_depth+1)            if in all_calls
    )
    """
    traj_path = traces_dir / instance_id / f"{instance_id}.traj.json"
    if not traj_path.exists():
        return {}

    info = json.loads(traj_path.read_text()).get("info", {})
    settrace = info.get("settrace_traces", {})
    if not settrace:
        return {}

    scores: dict[tuple, float] = {}
    for entry in settrace.values():
        if isinstance(entry, dict):
            exc_frames = entry.get("exception_frames", [])
            all_calls  = entry.get("all_calls", [])
        else:
            exc_frames = entry
            all_calls  = entry

        for depth_from_inner, frame in enumerate(reversed(exc_frames)):
            key = (frame.get("file", ""), frame.get("func", ""))
            if key[0] and key[1]:
                scores[key] = max(scores.get(key, 0.0), 1.0 / (depth_from_inner + 1))

        for frame in all_calls:
            key = (frame.get("file", ""), frame.get("func", ""))
            if key[0] and key[1]:
                d = frame.get("call_depth") or 0
                scores[key] = max(scores.get(key, 0.0), 1.0 / (d + 1))

    return scores


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def first_hit_rank(candidates: list[dict], truth: set[str], score_key: str) -> int | None:
    """Rank (1-indexed) of the first candidate whose short name is in truth, or None."""
    sorted_cands = sorted(candidates, key=lambda e: e.get(score_key, 0.0), reverse=True)
    for rank, entity in enumerate(sorted_cands, 1):
        if _short(entity.get("name", "")) in truth:
            return rank
    return None


def augment_candidates(candidates: list[dict], trace_scores: dict, weight: float) -> list[dict]:
    """Add 'augmented_score' = similarity + weight * trace_score to each candidate."""
    for e in candidates:
        key = (e.get("file_path", ""), _short(e.get("name", "")))
        trace = trace_scores.get(key, 0.0)
        e["augmented_score"] = e.get("similarity", 0.0) + weight * trace
    return candidates


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def recall_at_k(ranks: list[int | None], k: int) -> float:
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    rr = [1.0 / r for r in ranks if r is not None]
    return sum(rr) / len(ranks) if ranks else 0.0


def print_metrics(label: str, ranks: list[int | None]):
    n = len(ranks)
    hits_any = sum(1 for r in ranks if r is not None)
    print(f"\n  [{label}]  n={n}  coverage={hits_any/n:.1%}  MRR={mrr(ranks):.4f}")
    for k in RECALL_K:
        print(f"    Recall@{k:<3} {recall_at_k(ranks, k):.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-dir",    default="kgCompass/tmp/tests")
    parser.add_argument("--traces-dir",   default=None)
    parser.add_argument("--trace-weight", type=float, default=0.3,
                        help="γ in: augmented_score = similarity + γ * trace_score")
    args, _ = parser.parse_known_args()

    tests_dir  = Path(args.tests_dir)
    traces_dir = Path(args.traces_dir) if args.traces_dir else None
    weight     = args.trace_weight

    print("Loading SWE-bench dataset...")
    from datasets import load_dataset
    instances = {inst["instance_id"]: inst
                 for inst in load_dataset("princeton-nlp/SWE-bench_Lite", split="test")}
    print(f"Loaded {len(instances)} instances.\n")

    baseline_ranks:  list[int | None] = []
    augmented_ranks: list[int | None] = []
    skipped = {"no_kg": 0, "no_patch": 0, "no_truth": 0}
    results = []

    for run_dir in sorted(tests_dir.glob("*_gemini")):
        instance_id = run_dir.name.replace("_gemini", "")
        if instance_id not in instances:
            continue

        patch = instances[instance_id].get("patch", "")
        if not patch:
            skipped["no_patch"] += 1
            continue

        truth = gt_functions(patch)
        if not truth:
            skipped["no_truth"] += 1
            continue

        candidates = load_kg_candidates(instance_id, tests_dir)
        if not candidates:
            skipped["no_kg"] += 1
            continue

        base_rank = first_hit_rank(candidates, truth, "similarity")
        baseline_ranks.append(base_rank)

        aug_rank = None
        if traces_dir is not None:
            trace_scores = load_trace_scores(instance_id, traces_dir)
            augment_candidates(candidates, trace_scores, weight)
            aug_rank = first_hit_rank(candidates, truth, "augmented_score")
            augmented_ranks.append(aug_rank)

        results.append({
            "instance_id":   instance_id,
            "repo":          instances[instance_id].get("repo", ""),
            "truth_funcs":   sorted(truth),
            "n_candidates":  len(candidates),
            "baseline_rank": base_rank,
            "augmented_rank": aug_rank,
        })

    # ---- Overall summary ----
    n = len(results)
    print(f"{'='*55}")
    print(f"Instances evaluated : {n}")
    for k, v in skipped.items():
        if v: print(f"  Skipped ({k}): {v}")
    print(f"{'='*55}")

    print_metrics("Baseline  (KG similarity)", baseline_ranks)
    if traces_dir is not None:
        print_metrics(f"Augmented (+ trace, γ={weight})", augmented_ranks)

        # Delta summary
        improved = sum(
            1 for r in results
            if r["baseline_rank"] is not None and r["augmented_rank"] is not None
            and r["augmented_rank"] < r["baseline_rank"]
        )
        worsened = sum(
            1 for r in results
            if r["baseline_rank"] is not None and r["augmented_rank"] is not None
            and r["augmented_rank"] > r["baseline_rank"]
        )
        unchanged = n - improved - worsened
        print(f"\n  Rank change vs baseline:")
        print(f"    Improved : {improved}  ({improved/n:.1%})")
        print(f"    Unchanged: {unchanged}  ({unchanged/n:.1%})")
        print(f"    Worsened : {worsened}  ({worsened/n:.1%})")

    # ---- Per-repo breakdown ----
    by_repo: dict[str, list] = {}
    for r in results:
        by_repo.setdefault(r["repo"], []).append(r)

    header = f"\n  {'Repo':<43}  {'n':>4}  {'MRR(base)':>10}"
    if traces_dir:
        header += f"  {'MRR(aug)':>10}  {'R@5(base)':>10}  {'R@5(aug)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 3))

    for repo, recs in sorted(by_repo.items()):
        b_ranks = [r["baseline_rank"] for r in recs]
        line = f"  {repo:<43}  {len(recs):>4}  {mrr(b_ranks):>10.4f}"
        if traces_dir:
            a_ranks = [r["augmented_rank"] for r in recs]
            line += (f"  {mrr(a_ranks):>10.4f}"
                     f"  {recall_at_k(b_ranks, 5):>10.1%}"
                     f"  {recall_at_k(a_ranks, 5):>10.1%}")
        print(line)

    # ---- Save ----
    out_path = tests_dir / "kg_ranking_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
