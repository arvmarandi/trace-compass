#!/usr/bin/env python3
"""List all SWE-bench instance IDs into a text file, optionally split into N parts."""

import argparse
from datasets import load_dataset
from pathlib import Path

DATASET_MAP = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
}

parser = argparse.ArgumentParser(description="List SWE-bench instance IDs")
parser.add_argument("--subset", default="verified", choices=list(DATASET_MAP), help="Dataset subset (default: verified)")
parser.add_argument("--split", default=1, type=int, metavar="N", help="Split into N roughly equal parts (default: 1)")
parser.add_argument("--output", default="instances.txt", help="Output file name (default: instances.txt)")
args = parser.parse_args()

dataset = DATASET_MAP[args.subset]
print(f"Loading {dataset}...")
ds = load_dataset(dataset, split="test")
ids = [row["instance_id"] for row in ds]
print(f"Found {len(ids)} instances")

if args.split == 1:
    out = Path(args.output)
    out.write_text("\n".join(ids) + "\n")
    print(f"Written to {out}")
else:
    chunk = (len(ids) + args.split - 1) // args.split  # ceiling division
    stem = Path(args.output).stem
    suffix = Path(args.output).suffix or ".txt"
    for i in range(args.split):
        part = ids[i * chunk : (i + 1) * chunk]
        out = Path(f"{stem}_part{i+1}{suffix}")
        out.write_text("\n".join(part) + "\n")
        print(f"Part {i+1}: {len(part)} instances → {out}")
