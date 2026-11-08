#!/usr/bin/env python3
"""Write the paper ablations from a score_table main_table.tsv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROWS = (
    ("P1: w/o Task Goal", "F1"),
    ("P1: w/o Per-step Content Goal", "F2"),
    ("P2: w/o Shared Visual Style", "F4"),
    ("P2: w/ Merged Image Prompt and Content", "F3"),
    ("P3: References Fixed before the Text", "O32"),
    ("P3: w/o Per-step Reference Routing", "F5"),
    ("P3: w/o All Reference Images", "F5b"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the ablation table from main_table.tsv."
    )
    parser.add_argument(
        "table",
        nargs="?",
        default="main_table.tsv",
        type=Path,
        help="table written by scoring/scripts/score_table.py",
    )
    parser.add_argument(
        "--full-object",
        required=True,
        help="object id or display name of the full UniWeaver run",
    )
    return parser.parse_args()


def variant_id(object_name: str) -> str:
    prefix, separator, variant = object_name.partition("/")
    if separator and prefix == "ablation":
        return variant
    return ""


def load_scores(path: Path, full_object: str | None) -> dict[str, float]:
    if not path.is_file():
        raise SystemExit(f"score table not found: {path}")

    scores: dict[str, float] = {}
    full_candidates: list[tuple[str, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            object_id = row.get("object_id", "")
            display_name = row.get("display_name", "")
            value = float(row["Overall|Total"])
            variant = variant_id(object_id)
            if variant:
                scores[variant] = value
            else:
                full_candidates.extend([
                    (object_id, value),
                    (display_name, value),
                ])

    matches = [value for name, value in full_candidates if name == full_object]
    if len(matches) != 1:
        unique = {round(value, 6) for value in matches}
        if len(unique) != 1:
            raise SystemExit(
                f"full object {full_object!r} must appear exactly once"
            )
    full = matches[0]

    missing = [variant for _label, variant in ROWS if variant not in scores]
    if missing:
        raise SystemExit(f"missing ablation rows: {', '.join(missing)}")
    return {"__full__": full, **scores}


def main() -> int:
    args = parse_arguments()
    scores = load_scores(args.table, args.full_object)
    full = scores["__full__"]
    print(f"UniWeaver (w/ BAGEL backbone) & {full:.2f} & --- \\\\")
    print("\\midrule")
    for label, variant in ROWS:
        value = scores[variant]
        print(f"{label:<42} & {value:.2f} & ${value - full:+.2f}$ \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
