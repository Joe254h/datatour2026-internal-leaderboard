#!/usr/bin/env python3
"""Validate internal leaderboard submission format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--reference", required=True, help="internal_test_public.csv or .csv.gz")
    args = parser.parse_args()

    sub_path = Path(args.submission)
    ref_path = Path(args.reference)

    sub = pd.read_csv(sub_path)
    ref = pd.read_csv(ref_path, usecols=["id"])

    expected_cols = ["id", "target"]
    if list(sub.columns) != expected_cols:
        raise ValueError(f"Submission must have exactly columns {expected_cols}. Found {list(sub.columns)}")

    if len(sub) != len(ref):
        raise ValueError(f"Wrong number of rows. Expected {len(ref)}, found {len(sub)}")

    if sub["id"].duplicated().any():
        raise ValueError("Submission contains duplicate id values.")

    missing_ids = set(ref["id"]) - set(sub["id"])
    extra_ids = set(sub["id"]) - set(ref["id"])

    if missing_ids:
        raise ValueError(f"Submission is missing {len(missing_ids)} ids.")
    if extra_ids:
        raise ValueError(f"Submission has {len(extra_ids)} unexpected ids.")

    if sub["target"].isna().any():
        raise ValueError("Submission contains missing target values.")

    if not sub["target"].between(0, 1).all():
        raise ValueError("All target values must be between 0 and 1.")

    print("Submission format is valid.")


if __name__ == "__main__":
    main()
