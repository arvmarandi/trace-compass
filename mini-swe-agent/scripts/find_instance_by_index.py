#!/usr/bin/env python3
"""Find instance IDs by index in the SWE-Bench dataset."""

from datasets import load_dataset

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "rebench": "nebius/SWE-rebench",
}


def find_instances_by_repo(repo_name: str, subset: str = "lite", split: str = "dev") -> None:
    """Find all instances for a given repository."""
    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    matches = []
    for i, instance in enumerate(instances):
        if repo_name.lower() in instance["instance_id"].lower():
            matches.append((i, instance["instance_id"]))

    if matches:
        print(f"\nFound {len(matches)} instances for '{repo_name}':")
        for idx, instance_id in matches:
            print(f"  Index {idx}: {instance_id}")
    else:
        print(f"\nNo instances found for '{repo_name}'")


def print_instances_by_index(indices: list[int], subset: str = "lite", split: str = "dev") -> None:
    """Print instance IDs at specific indices."""
    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    print("\nInstances at specified indices:")
    for idx in indices:
        if 0 <= idx < len(instances):
            print(f"  Index {idx}: {instances[idx]['instance_id']}")
        else:
            print(f"  Index {idx}: OUT OF RANGE (dataset has {len(instances)} instances)")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Find SWE-Bench instances by repository name or index")
    parser.add_argument("--repo", help="Repository name to search for")
    parser.add_argument("--index", nargs="+", type=int, help="Indices to look up")
    parser.add_argument("--subset", default="lite", help="Dataset subset (default: lite)")
    parser.add_argument("--split", default="dev", help="Dataset split (default: dev)")

    args = parser.parse_args()

    if not args.repo and not args.index:
        parser.print_help()
        sys.exit(1)

    if args.repo:
        find_instances_by_repo(args.repo, subset=args.subset, split=args.split)
    elif args.index:
        print_instances_by_index(args.index, subset=args.subset, split=args.split)
