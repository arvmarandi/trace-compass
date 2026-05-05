#!/usr/bin/env python3
"""
Rank-based evaluation of KG fault localization candidates.

For each instance, ranks KG candidates by similarity score and reports:
  - Recall@k  (k = 1, 3, 5, 10, 15, 20)
  - MRR       (Mean Reciprocal Rank)

Ground truth: functions from @@ hunk headers of the golden patch.

Usage:
    python scripts/eval_kg_ranking.py --tests-dir kgCompass/tmp/tests
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
    args, _ = parser.parse_known_args()

    tests_dir = Path(args.tests_dir)

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

        rank = first_hit_rank(candidates, truth)
        ranks.append(rank)
        results.append({
            "instance_id": instance_id,
            "repo":        instances[instance_id].get("repo", ""),
            "truth_funcs": sorted(truth),
            "n_candidates": len(candidates),
            "rank":        rank,
        })

    n = len(results)
    hits = sum(1 for r in ranks if r is not None)

    print(f"{'='*50}")
    print(f"Instances evaluated : {n}")
    for k, v in skipped.items():
        if v: print(f"  Skipped ({k}): {v}")
    cov = hits / n if n else 0.0
    print(f"\nCoverage (hit anywhere): {hits}/{n}  ({cov:.1%})")
    print(f"MRR                    : {mrr(ranks):.4f}")
    for k in RECALL_K:
        print(f"Recall@{k:<3}             : {recall_at_k(ranks, k):.1%}")
    print(f"{'='*50}")

    # ---- Per-repo breakdown ----
    by_repo: dict[str, list] = {}
    for r in results:
        by_repo.setdefault(r["repo"], []).append(r)

    print(f"\n  {'Repo':<43}  {'n':>4}  {'MRR':>7}  {'R@5':>6}  {'R@10':>6}  {'R@15':>6}  {'R@20':>6}  {'R@25':>6}")
    print("  " + "-" * 95)
    for repo, recs in sorted(by_repo.items()):
        repo_ranks = [r["rank"] for r in recs]
        print(f"  {repo:<43}  {len(recs):>4}  {mrr(repo_ranks):>7.4f}"
              f"  {recall_at_k(repo_ranks, 5):>6.1%}"
              f"  {recall_at_k(repo_ranks, 10):>6.1%}"
              f"  {recall_at_k(repo_ranks, 15):>6.1%}"
              f"  {recall_at_k(repo_ranks, 20):>6.1%}"
              f"  {recall_at_k(repo_ranks, 25):>6.1%}")

    # ---- Rank distribution ----
    print("\nRank distribution:")
    found = [r for r in ranks if r is not None]
    buckets = {"1": 0, "2-5": 0, "6-10": 0, "11-15": 0, "16-20": 0, "21-25": 0, ">25": 0, "not found": 0}
    for r in ranks:
        if r is None:        buckets["not found"] += 1
        elif r == 1:         buckets["1"]         += 1
        elif r <= 5:         buckets["2-5"]        += 1
        elif r <= 10:        buckets["6-10"]       += 1
        elif r <= 15:        buckets["11-15"]      += 1
        elif r <= 20:        buckets["16-20"]      += 1
        elif r <= 25:        buckets["21-25"]      += 1
        else:                buckets[">25"]        += 1
    for label, count in buckets.items():
        print(f"  {label:>10}: {count:>4}  ({count/n:.1%})" if n else f"  {label:>10}: {count:>4}")

    out_path = tests_dir / "kg_ranking_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
