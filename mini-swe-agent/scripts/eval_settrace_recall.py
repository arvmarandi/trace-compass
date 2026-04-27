#!/usr/bin/env python3
"""
Evaluate whether the buggy functions (derived from the golden patch) appear
in the settrace_traces produced by our stack-trace generation pipeline.

Methodology (following arxiv 2503.21710):
  - Extract the set of functions/classes modified in the golden patch by
    parsing @@ hunk headers (which include the enclosing function as context)
    and any `def`/`class` lines added or removed in the diff body.
  - Check whether any of those names appear in the settrace_traces frames.
  - Report per-instance results and aggregate recall.

Usage:
    python scripts/eval_settrace_recall.py \
        --traces-dir outputs/stack-traces \
        --subset lite --split test
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def extract_buggy_functions_from_patch(patch: str) -> set[str]:
    """Return function/class names whose bodies were modified in the golden patch.

    Uses only the function-context field on @@ hunk headers:
        @@ -L,N +L,N @@ def foo(self, x):
    This reliably identifies functions that already existed and were changed.

    We intentionally exclude `+def` / `-def` lines from the diff body because
    those represent functions being added or renamed — the generated test was
    written against the buggy codebase and couldn't have called a function that
    didn't exist yet, so including them would unfairly penalise recall.
    """
    names = set()
    for m in re.finditer(r'^@@[^@]+@@\s*(.*)', patch, re.MULTILINE):
        ctx = m.group(1).strip()
        name_m = re.match(r'(?:async\s+)?(?:def|class)\s+(\w+)', ctx)
        if name_m:
            names.add(name_m.group(1))
    return names


def check_recall(settrace_traces: dict, buggy_functions: set[str]) -> tuple[bool, set[str], dict]:
    """Return (hit, matched_functions, location_info) checking all_calls and exception_frames.

    location_info keys per matched function:
      - "in_exception_frames": bool
      - "depth_from_innermost": int or None  (0 = innermost frame at exception)
      - "all_calls_rank": int or None         (0-indexed position in all_calls)
      - "all_calls_total": int                (total functions in all_calls)
    """
    matched = set()
    location: dict[str, dict] = {}

    for entry in settrace_traces.values():
        if isinstance(entry, dict):
            exc_frames = entry.get("exception_frames", [])
            all_calls  = entry.get("all_calls", [])
        else:
            exc_frames = entry
            all_calls  = entry

        exc_funcs = [f.get("func") for f in exc_frames]
        all_funcs = [f.get("func") for f in all_calls]

        for func in buggy_functions:
            if func in exc_funcs or func in all_funcs:
                matched.add(func)

            if func not in location:
                in_exc = func in exc_funcs
                depth = (len(exc_funcs) - 1 - exc_funcs.index(func)) if in_exc else None
                rank  = all_funcs.index(func) if func in all_funcs else None
                call_depth = all_calls[all_funcs.index(func)].get("call_depth") if rank is not None else None
                location[func] = {
                    "in_exception_frames": in_exc,
                    "depth_from_innermost": depth,
                    "all_calls_rank": rank,
                    "all_calls_total": len(all_funcs),
                    "call_depth": call_depth,
                }

    return bool(matched), matched, location


def main():
    parser = argparse.ArgumentParser(description="Evaluate settrace recall against golden patches")
    parser.add_argument("--traces-dir", default="outputs/stack-traces",
                        help="Directory containing per-instance trajectory files")
    parser.add_argument("--subset", default="lite",
                        help="SWE-bench subset (lite, verified, full)")
    parser.add_argument("--split", default="test",
                        help="Dataset split")
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)

    print("Loading SWE-bench dataset...")
    from datasets import load_dataset
    DATASET_MAPPING = {
        "lite": "princeton-nlp/SWE-Bench_Lite",
        "verified": "princeton-nlp/SWE-Bench_Verified",
        "full": "princeton-nlp/SWE-Bench",
    }
    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    instances = {inst["instance_id"]: inst for inst in load_dataset(dataset_path, split=args.split)}
    print(f"Loaded {len(instances)} instances from dataset.")

    results = []
    skipped_no_traj = 0
    skipped_no_settrace = 0
    skipped_no_patch = 0
    skipped_no_buggy_funcs = 0

    for traj_file in sorted(traces_dir.glob("*/*.traj.json")):
        instance_id = traj_file.parent.name

        if instance_id not in instances:
            skipped_no_traj += 1
            continue

        info = json.loads(traj_file.read_text()).get("info", {})
        settrace_traces = info.get("settrace_traces")

        if not settrace_traces:
            skipped_no_settrace += 1
            continue

        golden_patch = instances[instance_id].get("patch", "")
        if not golden_patch:
            skipped_no_patch += 1
            continue

        buggy_functions = extract_buggy_functions_from_patch(golden_patch)
        if not buggy_functions:
            skipped_no_buggy_funcs += 1
            continue

        hit, matched, location = check_recall(settrace_traces, buggy_functions)
        results.append({
            "instance_id": instance_id,
            "repo": instances[instance_id].get("repo", ""),
            "hit": hit,
            "buggy_functions": sorted(buggy_functions),
            "matched_functions": sorted(matched),
            "location": location,
            "settrace_test_count": len(settrace_traces),
        })

    # ---- Summary ----
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    recall = hits / total if total else 0.0

    print(f"\n{'='*60}")
    print(f"Instances with settrace + golden patch + buggy funcs: {total}")
    print(f"Skipped (no traj):          {skipped_no_traj}")
    print(f"Skipped (no settrace):      {skipped_no_settrace}")
    print(f"Skipped (no golden patch):  {skipped_no_patch}")
    print(f"Skipped (no buggy funcs):   {skipped_no_buggy_funcs}")
    print(f"\nHits (buggy func in trace): {hits} / {total}  ({recall:.1%})")
    print(f"{'='*60}\n")

    # ---- Per-repo breakdown ----
    by_repo: dict[str, list] = {}
    for r in results:
        by_repo.setdefault(r["repo"], []).append(r)
    print("Per-repo recall:")
    for repo, recs in sorted(by_repo.items()):
        repo_hits = sum(1 for r in recs if r["hit"])
        print(f"  {repo:<45} {repo_hits}/{len(recs)}  ({repo_hits/len(recs):.0%})")

    # ---- Location analysis for hits ----
    # Categorise each matched buggy function by where it appears.
    in_exc_only, in_calls_only, in_both = 0, 0, 0
    depths_from_innermost = []   # depth in exception_frames (0 = innermost)
    norm_ranks = []              # position in all_calls as fraction of total
    buggy_call_depths = []       # call stack depth at invocation for buggy funcs
    all_call_depths_pool = []    # same but for ALL functions (baseline)

    for r in results:
        # Baseline: collect call_depth for every function in this instance's traces
        traj = traces_dir / r["instance_id"] / f"{r['instance_id']}.traj.json"
        info = json.loads(traj.read_text()).get("info", {})
        for entry in info.get("settrace_traces", {}).values():
            calls = entry.get("all_calls", entry) if isinstance(entry, dict) else entry
            for f in calls:
                if f.get("call_depth") is not None:
                    all_call_depths_pool.append(f["call_depth"])

        for func, loc in r["location"].items():
            if func not in r["matched_functions"]:
                continue
            in_exc  = loc["in_exception_frames"]
            in_all  = loc["all_calls_rank"] is not None
            if in_exc and in_all:
                in_both += 1
            elif in_exc:
                in_exc_only += 1
            elif in_all:
                in_calls_only += 1

            if loc["depth_from_innermost"] is not None:
                depths_from_innermost.append(loc["depth_from_innermost"])
            if loc["all_calls_rank"] is not None and loc["all_calls_total"] > 0:
                norm_ranks.append(loc["all_calls_rank"] / loc["all_calls_total"])
            if loc["call_depth"] is not None:
                buggy_call_depths.append(loc["call_depth"])

    total_matched = in_exc_only + in_calls_only + in_both
    print("Location of matched buggy functions:")
    print(f"  In exception_frames only : {in_exc_only}")
    print(f"  In all_calls only        : {in_calls_only}")
    print(f"  In both                  : {in_both}")
    print(f"  Total matched functions  : {total_matched}")

    if depths_from_innermost:
        avg_depth = sum(depths_from_innermost) / len(depths_from_innermost)
        print(f"\n  exception_frames depth from innermost (lower = closer to exception):")
        print(f"    mean={avg_depth:.1f}  min={min(depths_from_innermost)}  max={max(depths_from_innermost)}")
        dist = {}
        for d in depths_from_innermost:
            dist[d] = dist.get(d, 0) + 1
        print(f"    distribution: { {k: dist[k] for k in sorted(dist)} }")

    if norm_ranks:
        avg_rank = sum(norm_ranks) / len(norm_ranks)
        print(f"\n  all_calls normalised rank (0=first called, 1=last called):")
        print(f"    mean={avg_rank:.2f}  min={min(norm_ranks):.2f}  max={max(norm_ranks):.2f}")
        buckets = {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0}
        for nr in norm_ranks:
            if nr < 0.25:   buckets["0-25%"]   += 1
            elif nr < 0.50: buckets["25-50%"]  += 1
            elif nr < 0.75: buckets["50-75%"]  += 1
            else:           buckets["75-100%"] += 1
        print(f"    buckets: {buckets}")

    if buggy_call_depths:
        def median(xs):
            s = sorted(xs)
            n = len(s)
            return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

        avg_buggy = sum(buggy_call_depths) / len(buggy_call_depths)
        med_buggy = median(buggy_call_depths)
        avg_all   = sum(all_call_depths_pool) / len(all_call_depths_pool) if all_call_depths_pool else 0
        med_all   = median(all_call_depths_pool) if all_call_depths_pool else 0
        print(f"\n  call_depth at invocation (0=called directly from test, higher=deeper):")
        print(f"    buggy funcs : mean={avg_buggy:.1f}  median={med_buggy:.1f}  min={min(buggy_call_depths)}  max={max(buggy_call_depths)}")
        print(f"    all funcs   : mean={avg_all:.1f}  median={med_all:.1f}  min={min(all_call_depths_pool)}  max={max(all_call_depths_pool)}")
        print(f"    → buggy funcs are {'deeper' if avg_buggy > avg_all else 'shallower'} than average by {abs(avg_buggy - avg_all):.1f} levels (mean), {abs(med_buggy - med_all):.1f} levels (median)")
    print()

    # ---- Misses ----
    misses = [r for r in results if not r["hit"]]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for r in misses:
            print(f"  {r['instance_id']}")
            print(f"    buggy:   {r['buggy_functions']}")
            frame_funcs = set()
            traj = traces_dir / r["instance_id"] / f"{r['instance_id']}.traj.json"
            info = json.loads(traj.read_text()).get("info", {})
            for entry in info.get("settrace_traces", {}).values():
                calls = entry.get("all_calls", entry) if isinstance(entry, dict) else entry
                for f in calls:
                    frame_funcs.add(f.get("func"))
            print(f"    in trace: {sorted(frame_funcs)[:10]}")

    # ---- Frequency table and cumulative coverage by call depth ----
    if buggy_call_depths:
        freq = {}
        for d in buggy_call_depths:
            freq[d] = freq.get(d, 0) + 1
        total_buggy = len(buggy_call_depths)
        print("Call depth frequency (buggy functions):")
        print(f"  {'depth':>6}  {'count':>6}  {'cumulative':>10}  {'cum %':>7}")
        cumulative = 0
        for depth in sorted(freq):
            cumulative += freq[depth]
            print(f"  {depth:>6}  {freq[depth]:>6}  {cumulative:>10}  {cumulative/total_buggy:>7.1%}")
        print()
        for k in [1, 3, 5, 10, 15, 20]:
            covered = sum(v for d, v in freq.items() if d < k)
            print(f"  top-{k:<3} (depth < {k:<3}): {covered}/{total_buggy}  ({covered/total_buggy:.1%})")
        print()

    # ---- Plot: buggy function call depth frequency ----
    if buggy_call_depths:
        cap = 30
        capped = [min(d, cap) for d in buggy_call_depths]
        bins = range(0, cap + 2)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(capped, bins=bins, color="tomato", edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(buggy_call_depths), color="darkred", linestyle="--", linewidth=1.5,
                   label=f"Mean ({np.mean(buggy_call_depths):.1f})")
        ax.axvline(np.median(buggy_call_depths), color="firebrick", linestyle=":", linewidth=1.5,
                   label=f"Median ({np.median(buggy_call_depths):.0f})")
        ax.set_xlabel("Call depth at first invocation (capped at 30)")
        ax.set_ylabel("Frequency")
        ax.set_title("Buggy Function Call Depth Distribution")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plot_path = traces_dir / "buggy_call_depth_freq.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
        plt.show()

    # ---- Save detailed results ----
    out_path = traces_dir / "settrace_recall.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
