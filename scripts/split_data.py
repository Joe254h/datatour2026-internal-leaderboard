#!/usr/bin/env python3
"""Create a 90/5/5 internal split for DataTour 2026.

Default strategy: period-tail split.
- Train: earliest periods
- Validation: next latest periods
- Hidden internal test: latest periods

This better matches the official competition structure because the official test periods
come after the training periods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGET_COL = "fraud_flag"
ID_COL = "id"
PERIOD_COL = "period"


def choose_tail_periods(df: pd.DataFrame, frac: float, excluded_periods: set[int] | None = None) -> set[int]:
    if excluded_periods is None:
        excluded_periods = set()

    total_rows = len(df)
    target_rows = total_rows * frac

    period_counts = (
        df.loc[~df[PERIOD_COL].isin(excluded_periods)]
        .groupby(PERIOD_COL)
        .size()
        .sort_index(ascending=False)
    )

    chosen: set[int] = set()
    running = 0

    for period, count in period_counts.items():
        chosen.add(int(period))
        running += int(count)
        if running >= target_rows:
            break

    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to original train.csv")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--valid-size", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--compression", default="gzip", choices=["gzip", "none"])
    args = parser.parse_args()

    train_path = Path(args.train)
    out_dir = Path(args.out_dir)
    public_dir = out_dir / "public"
    private_dir = out_dir / "private"

    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)

    required = {ID_COL, PERIOD_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values([PERIOD_COL, ID_COL]).reset_index(drop=True)

    test_periods = choose_tail_periods(df, args.test_size)
    valid_periods = choose_tail_periods(df, args.valid_size, excluded_periods=test_periods)

    internal_test = df[df[PERIOD_COL].isin(test_periods)].copy()
    valid = df[df[PERIOD_COL].isin(valid_periods)].copy()
    train_public = df[~df[PERIOD_COL].isin(test_periods | valid_periods)].copy()

    internal_test_labels = internal_test[[ID_COL, TARGET_COL]].copy()
    internal_test_public = internal_test.drop(columns=[TARGET_COL])

    suffix = ".csv.gz" if args.compression == "gzip" else ".csv"
    compression = "gzip" if args.compression == "gzip" else None

    train_public.to_csv(public_dir / f"train_public{suffix}", index=False, compression=compression)
    valid.to_csv(public_dir / f"valid_public{suffix}", index=False, compression=compression)
    internal_test_public.to_csv(public_dir / f"internal_test_public{suffix}", index=False, compression=compression)
    internal_test_labels.to_csv(private_dir / "internal_test_labels.csv", index=False)

    metadata = {
        "split_strategy": "period_tail",
        "n_total": int(len(df)),
        "n_train_public": int(len(train_public)),
        "n_valid_public": int(len(valid)),
        "n_internal_test": int(len(internal_test_public)),
        "train_period_min": int(train_public[PERIOD_COL].min()),
        "train_period_max": int(train_public[PERIOD_COL].max()),
        "valid_periods": sorted(map(int, valid_periods)),
        "internal_test_periods": sorted(map(int, test_periods)),
        "target_rate_total": float(df[TARGET_COL].mean()),
        "target_rate_train_public": float(train_public[TARGET_COL].mean()),
        "target_rate_valid_public": float(valid[TARGET_COL].mean()),
        "target_rate_internal_test": float(internal_test_labels[TARGET_COL].mean()),
    }

    (out_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Split completed.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
