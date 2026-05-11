#!/usr/bin/env python3
"""
Rank-based evaluation of KG fault localization candidates.

For each instance, ranks KG candidates by similarity score and reports:
  - Recall@k  (k = 1, 3, 5, 10, 15, 20, 25)
  - MRR       (Mean Reciprocal Rank)

Optionally augments KG candidates with top-N trace candidates (union approach):
  trace candidates not already in the KG are appended after KG candidates.

Ground truth: functions from @@ hunk headers of the golden patch.

Usage:
    python scripts/eval_kg_ranking.py --tests-dir kgCompass/tmp/tests
    python scripts/eval_kg_ranking.py --tests-dir kgCompass/tmp/tests \\
        --traces-dir mini-swe-agent/outputs/stack-traces --trace-top-k 5
"""

import argparse
import json
import re
from pathlib import Path


RECALL_K = [1, 3, 5, 10, 15, 20, 25]


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


def load_trace_candidates(instance_id: str, traces_dir: Path, top_k: int) -> list[str]:
    """Return top_k function short-names from all_calls, sorted by call_depth ascending."""
    traj_path = traces_dir / instance_id / f"{instance_id}.traj.json"
    if not traj_path.exists():
        return []
    info = json.loads(traj_path.read_text()).get("info", {})
    settrace = info.get("settrace_traces", {})
    if not settrace:
        return []
    seen: dict[str, int] = {}
    for entry in settrace.values():
        calls = entry.get("all_calls", []) if isinstance(entry, dict) else entry
        for f in calls:
            fn = f.get("func")
            d = f.get("call_depth", 999) or 999
            if fn and (fn not in seen or d < seen[fn]):
                seen[fn] = d
    return [fn for fn, _ in sorted(seen.items(), key=lambda x: x[1])[:top_k]]


def load_kg_candidates(instance_id: str, tests_dir: Path) -> list[dict]:
    path = tests_dir / f"{instance_id}_gemini" / "kg_locations" / f"{instance_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    entities = data.get("related_entities", {})
    candidates = entities.get("methods", []) + entities.get("classes", [])
    candidates.sort(key=lambda e: e.get("similarity", 0.0), reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def first_hit_rank(candidates: list[dict], truth: set[str]) -> int | None:
    for rank, entity in enumerate(candidates, 1):
        if _short(entity.get("name", "")) in truth:
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    rr = [1.0 / r for r in ranks if r is not None]
    return sum(rr) / len(ranks) if ranks else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-dir", default="kgCompass/tmp/tests")
    parser.add_argument("--traces-dir", default=None,
                        help="Path to stack-traces dir. If set, unions top-K trace candidates with KG.")
    parser.add_argument("--trace-top-k", type=int, default=5,
                        help="Number of top trace candidates to union in (default: 5)")
    args, _ = parser.parse_known_args()

    tests_dir = Path(args.tests_dir)
    traces_dir = Path(args.traces_dir) if args.traces_dir else None

    print("Loading SWE-bench dataset...")
    from datasets import load_dataset
    instances = {inst["instance_id"]: inst
                 for inst in load_dataset("princeton-nlp/SWE-bench_Lite", split="test")}
    print(f"Loaded {len(instances)} instances.\n")

    ranks: list[int | None] = []
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

        kg_candidates = candidates[:20]
        kg_rank = first_hit_rank(kg_candidates, truth)

        if traces_dir is not None:
            kg_short_names = {_short(c.get("name", "")) for c in kg_candidates}
            hybrid_candidates = kg_candidates + [
                {"name": fn, "similarity": 0.0}
                for fn in load_trace_candidates(instance_id, traces_dir, args.trace_top_k)
                if fn not in kg_short_names
            ]
            hybrid_rank = first_hit_rank(hybrid_candidates, truth)
        else:
            hybrid_rank = kg_rank

        ranks.append((kg_rank, hybrid_rank))
        results.append({
            "instance_id":  instance_id,
            "repo":         instances[instance_id].get("repo", ""),
            "truth_funcs":  sorted(truth),
            "n_candidates": len(kg_candidates),
            "kg_rank":      kg_rank,
            "hybrid_rank":  hybrid_rank,
        })

    n = len(results)
    kg_ranks    = [kg_r   for kg_r, _       in ranks]
    hybrid_ranks= [hyb_r  for _,    hyb_r   in ranks]

    kg_hits     = sum(1 for r in kg_ranks     if r is not None)
    hybrid_hits = sum(1 for r in hybrid_ranks if r is not None)

    print(f"{'='*50}")
    print(f"Instances evaluated : {n}")
    for k, v in skipped.items():
        if v: print(f"  Skipped ({k}): {v}")
    print(f"\nKG-only  coverage (top-20)  : {kg_hits}/{n}  ({kg_hits/n:.1%})" if n else "")
    if traces_dir:
        print(f"Hybrid   coverage (top-25)  : {hybrid_hits}/{n}  ({hybrid_hits/n:.1%})" if n else "")
    print(f"\nMRR (KG only)          : {mrr(kg_ranks):.4f}")
    for k in RECALL_K:
        ranks_to_use = hybrid_ranks if (traces_dir and k == 25) else kg_ranks
        label = f"Recall@{k:<3}" + (" [KG+trace]" if (traces_dir and k == 25) else "           ")
        print(f"{label}: {recall_at_k(ranks_to_use, k):.1%}")
    print(f"{'='*50}")

    # ---- Per-repo breakdown ----
    by_repo: dict[str, list] = {}
    for r in results:
        by_repo.setdefault(r["repo"], []).append(r)

    print(f"\n  {'Repo':<43}  {'n':>4}  {'MRR(KG)':>8}  {'R@5':>6}  {'R@10':>6}  {'R@15':>6}  {'R@20':>6}  {'R@25+T':>8}")
    print("  " + "-" * 100)
    for repo, recs in sorted(by_repo.items()):
        rkg  = [r["kg_rank"]     for r in recs]
        rhyb = [r["hybrid_rank"] for r in recs]
        print(f"  {repo:<43}  {len(recs):>4}  {mrr(rkg):>8.4f}"
              f"  {recall_at_k(rkg, 5):>6.1%}"
              f"  {recall_at_k(rkg, 10):>6.1%}"
              f"  {recall_at_k(rkg, 15):>6.1%}"
              f"  {recall_at_k(rkg, 20):>6.1%}"
              f"  {recall_at_k(rhyb, 25):>8.1%}")

    # ---- Rank distribution (KG ranks) ----
    print("\nKG rank distribution (top-20):")
    buckets = {"1": 0, "2-5": 0, "6-10": 0, "11-15": 0, "16-20": 0, "not found": 0}
    for r in kg_ranks:
        if r is None:    buckets["not found"] += 1
        elif r == 1:     buckets["1"]         += 1
        elif r <= 5:     buckets["2-5"]        += 1
        elif r <= 10:    buckets["6-10"]       += 1
        elif r <= 15:    buckets["11-15"]      += 1
        else:            buckets["16-20"]      += 1
    for label, count in buckets.items():
        print(f"  {label:>10}: {count:>4}  ({count/n:.1%})" if n else f"  {label:>10}: {count:>4}")

    out_path = tests_dir / "kg_ranking_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to {out_path}")



if __name__ == "__main__":
    main()
