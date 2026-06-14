# DataTour 2026 — Internal Team Leaderboard

This repository is for managing a private group leaderboard before submitting to the official DataTour 2026 platform.

## Core rule

Do **not** push the official competition `test.csv` or hidden internal-test labels to GitHub.

Recommended structure:

```text
data/raw/train.csv                         # leader/local only
data/raw/test.csv                          # leader/local only, official test, not for internal leaderboard
data/public/train_public.csv.gz            # optional shared split, labels visible
data/public/valid_public.csv.gz            # optional shared split, labels visible
data/public/internal_test_public.csv.gz     # optional shared split, no fraud_flag
data/private/internal_test_labels.csv       # leader only, hidden labels
submissions/                               # team predictions for internal_test_public
leaderboard/leaderboard.csv                # scored results
leaderboard/leaderboard.md                 # readable leaderboard
```

## Split protocol

We split the original training data into:

- 90% public training data with labels
- 5% public validation data with labels
- 5% hidden internal-test data without labels

Because this competition has a `period` column and the official test starts after the train periods, the default split is **period-tail split**, not random split.

That means:
- earlier periods go to training,
- later periods go to validation,
- latest periods go to internal hidden test.

This is more realistic than random splitting.

## Step 1 — Put the real train file locally

```bash
mkdir -p data/raw
cp /path/to/train.csv data/raw/train.csv
```

Do not commit `data/raw/train.csv`.

## Step 2 — Create the internal split

```bash
python scripts/split_data.py \
  --train data/raw/train.csv \
  --out-dir data \
  --valid-size 0.05 \
  --test-size 0.05
```

This creates:

```text
data/public/train_public.csv.gz
data/public/valid_public.csv.gz
data/public/internal_test_public.csv.gz
data/private/internal_test_labels.csv
data/split_metadata.json
```

Only the leader should keep `data/private/internal_test_labels.csv`.

## Step 3 — Team members generate submissions

Each submission must have exactly two columns:

```text
id,target
```

Example path:

```text
submissions/joel_lgbm_v1.csv
```

The `target` column must contain probabilities between 0 and 1.

## Step 4 — Validate submission format

```bash
python scripts/validate_submission.py \
  --submission submissions/joel_lgbm_v1.csv \
  --reference data/public/internal_test_public.csv.gz
```

## Step 5 — Leader evaluates the hidden-test score

Only the leader runs this because it requires hidden labels:

```bash
python scripts/evaluate_submission.py \
  --submission submissions/joel_lgbm_v1.csv \
  --labels data/private/internal_test_labels.csv \
  --member "Joel" \
  --model-name "LightGBM v1" \
  --notes "basic balance features"
```

This updates:

```text
leaderboard/leaderboard.csv
leaderboard/leaderboard.md
```

## Team policy

1. Use `valid_public.csv.gz` for development.
2. Use `internal_test_public.csv.gz` only to generate leaderboard submissions.
3. Do not share `internal_test_labels.csv`.
4. Do not overfit to the internal leaderboard.
5. Submit to the official platform only when a model improves both public validation and internal hidden-test performance.

## Recommended submission budget

The official competition allows 5 submissions per day. Use them like this:

1. Best stable baseline
2. Best LightGBM
3. Best CatBoost
4. Best specialist model for `op_03`
5. Best blend, only if clearly better
