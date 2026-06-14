#!/usr/bin/env python3
"""Simple operation-rate baseline for the internal hidden-test split.

This predicts the historical fraud rate for each operation using train_public.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/public/train_public.csv.gz")
    parser.add_argument("--test", default="data/public/internal_test_public.csv.gz")
    parser.add_argument("--out", default="submissions/operation_rate_baseline.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    global_rate = train["fraud_flag"].mean()
    op_rates = train.groupby("operation")["fraud_flag"].mean()

    preds = test["operation"].map(op_rates).fillna(global_rate).clip(0, 1)

    out = pd.DataFrame({
        "id": test["id"],
        "target": preds,
    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
