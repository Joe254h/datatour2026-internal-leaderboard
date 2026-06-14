#!/usr/bin/env python3
"""Evaluate a submission against hidden internal-test labels and update leaderboard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss


def write_markdown_leaderboard(df: pd.DataFrame, path: Path) -> None:
    display_cols = [
        "rank",
        "member",
        "model_name",
        "average_precision",
        "roc_auc",
        "log_loss",
        "submission",
        "notes",
        "timestamp_utc",
    ]

    lines = []
    lines.append("# Internal Leaderboard")
    lines.append("")
    lines.append("Metric for ranking: **Average Precision / PR-AUC**.")
    lines.append("")
    lines.append("| Rank | Member | Model | AP | ROC-AUC | Log Loss | Submission | Notes | Time UTC |")
    lines.append("|---:|---|---|---:|---:|---:|---|---|---|")

    for _, row in df[display_cols].iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['member']} | {row['model_name']} | "
            f"{row['average_precision']:.6f} | {row['roc_auc']:.6f} | {row['log_loss']:.6f} | "
            f"`{row['submission']}` | {row['notes']} | {row['timestamp_utc']} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--labels", default="data/private/internal_test_labels.csv")
    parser.add_argument("--leaderboard-csv", default="leaderboard/leaderboard.csv")
    parser.add_argument("--leaderboard-md", default="leaderboard/leaderboard.md")
    parser.add_argument("--member", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    sub_path = Path(args.submission)
    labels_path = Path(args.labels)
    leaderboard_csv = Path(args.leaderboard_csv)
    leaderboard_md = Path(args.leaderboard_md)

    sub = pd.read_csv(sub_path)
    labels = pd.read_csv(labels_path)

    if list(sub.columns) != ["id", "target"]:
        raise ValueError("Submission must have exactly two columns: id,target")

    merged = labels.merge(sub, on="id", how="left", validate="one_to_one")

    if merged["target"].isna().any():
        raise ValueError("Submission is missing predictions for some hidden-test ids.")

    if not merged["target"].between(0, 1).all():
        raise ValueError("All target values must be probabilities between 0 and 1.")

    y_true = merged["fraud_flag"].astype(int)
    y_pred = merged["target"].astype(float).clip(1e-15, 1 - 1e-15)

    ap = average_precision_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "member": args.member,
        "model_name": args.model_name,
        "submission": str(sub_path),
        "average_precision": ap,
        "roc_auc": roc,
        "log_loss": ll,
        "notes": args.notes,
    }

    if leaderboard_csv.exists():
        lb = pd.read_csv(leaderboard_csv)
        lb = pd.concat([lb, pd.DataFrame([record])], ignore_index=True)
    else:
        lb = pd.DataFrame([record])

    lb = lb.sort_values(
        ["average_precision", "roc_auc", "log_loss"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    lb["rank"] = range(1, len(lb) + 1)

    leaderboard_csv.parent.mkdir(parents=True, exist_ok=True)
    lb.to_csv(leaderboard_csv, index=False)
    write_markdown_leaderboard(lb, leaderboard_md)

    print(f"Average Precision: {ap:.6f}")
    print(f"ROC-AUC: {roc:.6f}")
    print(f"Log Loss: {ll:.6f}")
    print(f"Updated {leaderboard_csv} and {leaderboard_md}")


if __name__ == "__main__":
    main()
