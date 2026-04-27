#!/usr/bin/env python3
"""
Plot call depth distributions for buggy vs. all functions from settrace_recall.json.

Usage:
    python scripts/plot_call_depth.py --traces-dir outputs/stack-traces
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_depths(traces_dir: Path):
    recall = json.loads((traces_dir / "settrace_recall.json").read_text())

    buggy_depths = []
    all_depths = []

    for r in recall:
        iid = r["instance_id"]
        traj_path = traces_dir / iid / f"{iid}.traj.json"
        if not traj_path.exists():
            continue

        info = json.loads(traj_path.read_text()).get("info", {})
        settrace = info.get("settrace_traces", {})

        for entry in settrace.values():
            calls = entry.get("all_calls", entry) if isinstance(entry, dict) else entry
            for frame in calls:
                d = frame.get("call_depth")
                if d is not None:
                    all_depths.append(d)

        for func, loc in r.get("location", {}).items():
            if func in r.get("matched_functions", []):
                d = loc.get("call_depth")
                if d is not None:
                    buggy_depths.append(d)

    return buggy_depths, all_depths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default="outputs/stack-traces")
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)
    buggy_depths, all_depths = load_depths(traces_dir)

    print(f"Buggy funcs: n={len(buggy_depths)}, mean={np.mean(buggy_depths):.1f}, median={np.median(buggy_depths):.1f}")
    print(f"All funcs:   n={len(all_depths)},  mean={np.mean(all_depths):.1f}, median={np.median(all_depths):.1f}")

    cap = 30  # cap display at depth 30 to keep charts readable
    buggy_capped = [min(d, cap) for d in buggy_depths]
    all_capped   = [min(d, cap) for d in all_depths]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Call Depth at First Invocation: Buggy Functions vs. All Functions", fontsize=13)

    bins = range(0, cap + 2)

    # ── Chart 1: overlapping histograms (normalised to density) ──
    ax = axes[0]
    ax.hist(all_capped,   bins=bins, density=True, alpha=0.5, color="steelblue", label="All functions")
    ax.hist(buggy_capped, bins=bins, density=True, alpha=0.7, color="tomato",    label="Buggy functions")
    ax.axvline(np.mean(all_depths),   color="steelblue", linestyle="--", linewidth=1.5, label=f"All mean ({np.mean(all_depths):.1f})")
    ax.axvline(np.mean(buggy_depths), color="tomato",    linestyle="--", linewidth=1.5, label=f"Buggy mean ({np.mean(buggy_depths):.1f})")
    ax.set_xlabel("Call depth (capped at 30)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution (overlapping)")
    ax.legend(fontsize=8)

    # ── Chart 2: side-by-side bar chart ──
    ax = axes[1]
    counts_all,   _ = np.histogram(all_capped,   bins=bins)
    counts_buggy, _ = np.histogram(buggy_capped, bins=bins)
    norm_all   = counts_all   / counts_all.sum()
    norm_buggy = counts_buggy / counts_buggy.sum()
    x = np.array(bins[:-1])
    width = 0.4
    ax.bar(x - width/2, norm_all,   width=width, color="steelblue", alpha=0.8, label="All functions")
    ax.bar(x + width/2, norm_buggy, width=width, color="tomato",    alpha=0.8, label="Buggy functions")
    ax.set_xlabel("Call depth (capped at 30)")
    ax.set_ylabel("Proportion")
    ax.set_title("Side-by-side proportion")
    ax.legend(fontsize=8)

    # ── Chart 3: CDF ──
    ax = axes[2]
    for depths, color, label in [
        (all_depths,   "steelblue", f"All functions (median={np.median(all_depths):.0f})"),
        (buggy_depths, "tomato",    f"Buggy functions (median={np.median(buggy_depths):.0f})"),
    ]:
        sorted_d = np.sort(depths)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.plot(np.minimum(sorted_d, cap), cdf, color=color, linewidth=2, label=label)
    ax.axvline(np.median(all_depths),   color="steelblue", linestyle=":", linewidth=1.5)
    ax.axvline(np.median(buggy_depths), color="tomato",    linestyle=":", linewidth=1.5)
    ax.set_xlabel("Call depth (capped at 30)")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title("CDF")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = traces_dir / "call_depth_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
