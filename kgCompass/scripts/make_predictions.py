#!/usr/bin/env python3
"""
Collect generated patches from tests/ into a predictions.jsonl file
for SWE-bench evaluation.

Each instance may have multiple diff files (one per modified file);
they are concatenated into a single patch string.

Usage:
    python scripts/make_predictions.py --tests-dir tests --output predictions.jsonl
    python scripts/make_predictions.py --tests-dir tests --output predictions.jsonl \
        --model-name kgcompass-deepseek
"""

import argparse
import json
from pathlib import Path


def collect_predictions(tests_dir: Path, model_name: str) -> list[dict]:
    predictions = []
    for run_dir in sorted(tests_dir.glob("*_gemini")):
        instance_id = run_dir.name.replace("_gemini", "")
        diff_dir = run_dir / "patches" / "diff_patches"
        if not diff_dir.exists():
            print(f"  SKIP {instance_id}: no diff_patches directory")
            continue

        diff_files = sorted(diff_dir.glob("*.diff"))
        if not diff_files:
            print(f"  SKIP {instance_id}: no .diff files")
            continue

        patch = "\n".join(f.read_text() for f in diff_files)
        if not patch.strip():
            print(f"  SKIP {instance_id}: empty patch")
            continue

        predictions.append({
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": model_name,
        })
        print(f"  OK   {instance_id} ({len(diff_files)} file(s) patched)")

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-dir", default="tests")
    parser.add_argument("--output", default="predictions.jsonl")
    parser.add_argument("--model-name", default="kgcompass-deepseek")
    args = parser.parse_args()

    tests_dir = Path(args.tests_dir)
    predictions = collect_predictions(tests_dir, args.model_name)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"\nWrote {len(predictions)} predictions to {out_path}")


if __name__ == "__main__":
    main()
