#!/usr/bin/env python3
"""Script to convert swt_bench_compatible.json (array format) to preds.json (object format)."""

import json
from pathlib import Path
from typing import Optional

import typer


def convert_swt_to_preds(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Convert SWTBench compatible JSON array to preds JSON object format.

    Args:
        input_path: Path to the input JSON file (array format).
        output_path: Path to the output JSON file (object format).
    """
    if input_path is None:
        input_path = Path(__file__).parent.parent / "outputs" / "swt_bench_compatible.json"
    if output_path is None:
        output_path = Path(__file__).parent.parent / "outputs" / "preds.json"

    # Load the input data
    with input_path.open() as f:
        data_list = json.load(f)

    # Transform to object format with instance_id as keys
    preds_data = {}
    for item in data_list:
        instance_id = item["instance_id"]
        preds_data[instance_id] = item

    # Write the output data
    with output_path.open("w") as f:
        json.dump(preds_data, f, indent=2)

    typer.echo(f"Converted {len(data_list)} items to preds format in {output_path}")


if __name__ == "__main__":
    typer.run(convert_swt_to_preds)