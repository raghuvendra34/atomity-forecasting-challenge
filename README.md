# Cloud Workload Forecasting — Submission

AI/ML take-home submission for the Atomity forecasting challenge, built from
the Microsoft Azure Public Dataset V2 VM trace (10,000 VMs, 134 five-minute
CPU readings each).

## What this submission does

Builds two aggregate target series — `active_vm_count` and
`aggregate_cpu_utilization` — and forecasts the final 2-hour holdout
(24 timestamps) with a Ridge regression on lag/rolling features, selected
by rolling-origin validation inside the training window. Baselines
(persistence, 12-step moving average, 12-step seasonal-naive) are evaluated
under the same protocol. Full methodology, results, and the honest
validation-vs-holdout comparison are in `reports/final_report.md`.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs numpy, pandas, scikit-learn (the Ridge model), and pytest.

## Dataset (not bundled)

The challenge dataset is **not** included in this submission and must be
downloaded separately. It is distributed as the versioned `data-v1`
GitHub Release of the challenge repository — do not substitute another
dataset or version.

1. Download `data-v1.tar.gz` from the `data-v1` release:
   https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1
2. Extract it into this repository as `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

3. Verify the dataset before running anything:

```bash
cd challenge_data
sha256sum -c SHA256SUMS
```

You should see `OK` for `vm_cpu_readings.csv.gz`, `vm_metadata.csv.gz`,
and `DATA_CARD.md`. The expected layout is:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

`challenge_data/` and `data-v1.tar.gz` are excluded from version control
(see `.gitignore`) and are never part of the submitted files.

## Generate predictions (one command)

From the repository root, with the dataset in place:

```bash
python3 src/predict.py --data challenge_data --output predictions.csv
```

This writes:

- `predictions.csv` — exactly 24 rows, columns
  `timestamp,active_vm_count,aggregate_cpu_utilization`, no index column,
  timestamps matching the holdout in chronological order, counts >= 0,
  CPU clipped to [0, 100].
- `reports/metrics.json` — rolling-origin validation folds, the selected
  model, and post-freeze holdout metrics (computed after freezing, for
  reporting only).
- `reports/analysis_stats.json` — every workload-analysis number quoted
  in the final report.

Run the test suite with:

```bash
python3 -m pytest -q
```

## Main modelling decisions

- **Targets (DATASET.md contract).** `active_vm_count` = distinct VMs per
  timestamp; `aggregate_cpu_utilization` = mean of VM-level `avg_cpu`.
  A timestamp with no readings would be recorded as count 0 / CPU NaN and
  forward-filled with the gap logged; in `data-v1` no such timestamp
  exists, so no imputation occurs.
- **Split.** Chronological only: first 110 timestamps train, final 24
  holdout. The holdout is never used for fitting, feature selection,
  tuning, or model selection.
- **Validation.** Expanding-window rolling-origin evaluation inside the
  training window only (origins at 55%, 67%, 78% of training, 24-step
  horizon, recursive forecasts) — the same protocol for baselines and
  model. No random splits anywhere.
- **Model.** Ridge regression (alpha 10, standardized features) on lags
  1/2/3/6/12, rolling means over 6/12 steps, rolling std over 12 steps,
  and the last one-step change. No time-trend feature: on validation it
  made forecasts strictly worse (MAE 0.249 with trend vs 0.136 without),
  because a trend term extrapolates level drift that does not persist on
  this short flat series. Alpha 10 beat alpha 1 on the same folds
  (MAE 0.136 vs 0.184). Both choices come from validation folds only.
- **Forecasting.** One-step model run recursively for 24 steps, feeding
  each prediction back as history. The final model is refit on the full
  110-timestamp training window before forecasting.
- **Active VM count.** The series is constant at 10,000 in this release
  (all VMs created at timestamp 0, earliest deletion after the window),
  so persistence is exact and no model is fitted for it.
- **Honest headline.** Ridge wins on validation (MAE 0.136 vs 0.223 for
  the best baseline) but trails the moving-average baseline on the
  holdout (MAE 0.181 vs 0.123). The submitted predictions are the Ridge
  model's — switching after seeing the holdout would be holdout-driven
  selection. The failure analysis (recursive drift, late-holdout rebound,
  an unforeseeable spike) is in `reports/final_report.md`, Section 6.

## Layout

```text
src/predict.py            Main pipeline (targets, validation, forecast)
src/cwfc/                 Starter utilities (baselines, metrics, validation)
reports/final_report.md   Full write-up (4–6 page equivalent)
reports/metrics.json      Validation + post-freeze holdout metrics
reports/analysis_stats.json  Workload-analysis numbers
predictions.csv           Required 24-row submission file
tests/                    Starter tests + predictions schema test
```

## Original challenge documents

`ASSIGNMENT.md`, `DATASET.md`, and `SUBMISSION.md` in this folder are the
challenge's own specification and are unchanged.
