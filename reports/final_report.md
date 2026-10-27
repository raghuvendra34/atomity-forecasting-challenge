# Cloud Workload Forecasting — Final Report

## Executive summary

**Question.** How predictable is this Azure VM workload, how accurately can
near-term aggregate demand be forecast, and what should an infrastructure
team do with the forecast?

**Answer in one paragraph.** The workload is highly predictable in level but
not in detail. Aggregate CPU utilization sits at 9.39% on average with a
standard deviation of only 0.19 percentage points, and the active VM count
never changes at all in this window — all 10,000 VMs are present at every
timestamp. A Ridge regression on lag and rolling-mean features, selected by
rolling-origin validation inside the training window, beats simple baselines
on validation (MAE 0.136 vs 0.223 for persistence, a 38.8% improvement).
On the final 2-hour holdout it does not: it trails the 12-step moving
average (MAE 0.181 vs 0.123). The reason is instructive and reported
honestly below — recursive forecasts accumulate a small downward drift,
and the holdout ends with a rebound the model cannot foresee. The
operational conclusion does not depend on squeezing the last decimal of
accuracy: with mean utilization under 10%, the binding problem is spike
headroom, not average capacity, and forecasts should be refreshed hourly
because second-half error is roughly 3x first-half error.

**Headline numbers (CPU utilization, percentage points).**

| Model | Validation MAE | Validation RMSE | Holdout MAE | Holdout RMSE |
|---|---:|---:|---:|---:|
| Ridge (selected model) | 0.136 | 0.167 | 0.181 | 0.224 |
| Persistence | 0.223 | 0.256 | 0.147 | 0.185 |
| Moving average (12 steps) | 0.224 | 0.249 | 0.123 | 0.161 |
| Seasonal-naive (lag 12) | 0.250 | 0.278 | 0.140 | 0.159 |

Active VM count: predicted exactly (MAE 0) because the series is constant
at 10,000 — see Section 2. Holdout metrics were computed only after the
model was frozen and played no part in any modelling choice.

## 1. Data understanding and preprocessing

**Source.** The `data-v1` release of this repository: a curated, real subset
of Microsoft Azure Public Dataset V2. Two files: `vm_cpu_readings.csv.gz`
(1,340,000 rows: 10,000 VMs x 134 observations) and `vm_metadata.csv.gz`
(10,000 rows, one per VM). Timestamps are seconds relative to the source
trace, not wall-clock times, so no calendar (hour-of-day / day-of-week)
features are available — a genuine limitation of this window.

**Integrity checks (all computed, none assumed).**

| Check | Result |
|---|---:|
| Timestamps | 134, from 0 to 39,900 s, exact 300 s cadence, no gaps |
| Duplicate (vm_id, timestamp) pairs | 0 |
| Missing values, all columns | 0 |
| Rows per timestamp | exactly 10,000 at every timestamp |
| VMs in readings vs metadata | 10,000 = 10,000, exact match |
| avg_cpu outside [0, 100] | 0 (range 0.007–99.488) |
| min_cpu > avg_cpu or avg_cpu > max_cpu | 0 |

**Target construction (DATASET.md contract).** At each timestamp:
`active_vm_count` = distinct `vm_id` values with a reading (each VM counted
at most once); `aggregate_cpu_utilization` = mean of VM-level `avg_cpu`,
in percent. Missing-value handling: a timestamp with no readings at all
would be recorded as count 0 and CPU NaN, then forward-filled for modelling
with the gap logged. In this release no such timestamp exists, so no
imputation was performed — the fallback is documented, not exercised.

**Split.** The 134-timestamp series is split chronologically: first 110
timestamps (0–32,700 s) for training, final 24 timestamps (33,000–39,900 s)
as the 2-hour holdout. The holdout observations were never used for
fitting, feature selection, hyperparameter tuning, or model selection.
Model selection used rolling-origin validation inside the 110-timestamp
training window only (Section 4).

## 2. Workload analysis

Each finding ends with its operational meaning, as the assignment requires.

**The fleet never changes size in this window.** `active_vm_count` is
10,000 at all 134 timestamps. Metadata confirms why: every VM was created
at timestamp 0 and the earliest deletion timestamp is 40,500 s, after the
window ends at 39,900 s. The curated selection (complete traces only)
removes in-window churn by construction. *Operational meaning: capacity
planning here is purely a CPU-intensity problem, not a fleet-size problem —
but that is an artefact of curation, and a production system must not
assume it.*

**Aggregate CPU is low and flat.** Mean 9.39%, standard deviation 0.19 pp,
median 9.37%, 90th percentile 9.66%, range 8.98–9.91%. Mean absolute
step-to-step change is 0.12 pp. Lag-1 autocorrelation is 0.68 and lag-12
(1 hour) is 0.44, so the series has exploitable short memory but no strong
structure. *Operational meaning: the fleet is mostly idle on average, so
average capacity is not the constraint — absorbing spikes is.*

**No daily cycle is detectable.** Hourly block means over the 11.1-hour
window are 9.41, 9.40, 9.37, 9.36, 9.62, 9.71, 9.37, 9.30, 9.24, 9.14,
9.35%. The mid-window elevation (hours 4–5, up to 9.71%) is a level shift,
not a repeating season, and 11 hours cannot establish a daily pattern
anyway. *Operational meaning: do not build a seasonal model on this
window; a seasonal-naive baseline is correspondingly the weakest
performer (validation MAE 0.250).*

**The distribution is extremely skewed at VM level.** Per-VM mean CPU has
median 5.48% but a 90th percentile of 19.52%. The top 10% of VMs account for
43.6% of all CPU consumed; 582 VMs average below 1% and 221 average above
50%. *Operational meaning: aggregate forecasts hide concentration risk —
a small set of hot VMs drives demand, so per-VM or segment monitoring
matters more than the aggregate mean suggests.*

**Segments behave differently.** Mean CPU by `vm_category`: Interactive
11.49% (3,091 VMs), Unknown 10.23% (729), Delay-insensitive 8.24%
(6,180). Correlation of each segment's series with the overall series:
Delay-insensitive 0.91, Interactive 0.77, Unknown 0.37. *Operational
meaning: Interactive workloads run ~3.3 pp hotter than Delay-insensitive
ones, so scale interactive pools first and treat the Unknown segment's
divergent pattern as a monitoring flag, not a forecast input.*

**Peak and low periods.** The series peak is 9.91% at index 55 and the
trough is 8.98% at index 111 (inside the holdout). The full peak-to-trough
range is under 1 pp. *Operational meaning: "peak" here is a relative term
— headroom policy should be set in tenths of a point, not in multiples.*

## 3. Forecasting approach

**Baselines (all evaluated under the same recursive protocol).**
Persistence (repeat the last value), moving average over the last 12 steps
(1 hour), and seasonal-naive with a 12-step season (1 hour, implemented in
`src/cwfc/baselines.py`). These are deliberately strong for a flat,
autocorrelated series, and the results confirm it.

**Main model: Ridge regression on lag/rolling features.** Features, all
computable at the forecast origin from history only: lags 1, 2, 3, 6, 12;
rolling means over 6 and 12 steps; rolling standard deviation over 12
steps; and the last one-step change. Features are standardized with a
scaler fit on training data only. The model is fit for one-step-ahead
prediction and then run recursively: each forecast step's prediction is
appended to the history before forecasting the next step, so a 24-step
forecast never touches observed holdout values. Predictions are clipped to
[0, 100] for CPU and rounded to non-negative integers for VM count.

**Why Ridge, and why no trend feature.** With 110 training points, a
regularized linear model is the right complexity: it cannot memorize the
series, it is stable under recursion, and it is fully reproducible. Two
choices were settled on validation folds, not on the holdout: dropping the
time-trend feature (validation MAE 0.249 with trend vs 0.136 without it —
a trend term extrapolates level drift that does not persist on this short
flat series) and raising Ridge's alpha from 1.0 to 10.0 (validation MAE
0.184 → 0.136). Both decisions predate the final fit and use no holdout
information.

**Why not something fancier.** Gradient boosting and neural models were
considered and rejected for this window: 110 training points cannot
support them without overfitting, and the assignment states that
complexity alone scores nothing. A simple, well-validated model is the
better submission here.

## 4. Validation methodology

**Design.** Expanding-window rolling-origin evaluation inside the training
window only, at the same 24-step horizon as the final task, with the same
recursive forecasting protocol for every method. Origins at 55%, 67%, and
78% of the training window (indices 60, 73, 85), giving validation windows
60–83, 73–96, and 85–108 — all strictly before the holdout. This mirrors
the production problem: forecasting the next 2 hours from history only,
repeated as new data arrives.

**Why this design and not a random split.** Randomly shuffling
time-series rows would let the model train on the future and validate on
the past, which is meaningless for forecasting and explicitly prohibited.
The rolling-origin design is the honest analogue of deployment.

**Selection rule.** The final model is the method with the lowest mean
validation MAE across the three folds. That is Ridge (0.136), ahead of
persistence (0.223), moving average (0.224), and seasonal-naive (0.250).
Ridge is also best or tied-best in every individual fold
(0.156 / 0.172 / 0.082 vs the next-best 0.180 / 0.257 / 0.098).

## 5. Results versus baseline

**Validation (the selection basis).** Ridge improves on the best baseline
(persistence) by 38.8% in MAE (0.136 vs 0.223) and by 34.6% in RMSE
(0.167 vs 0.256). Peak-period validation MAE — timestamps at or above the
90th percentile of each fold's own training window — is 0.097 for Ridge
vs 0.098 for persistence, essentially tied; but this rests on only 9 peak
timestamps in a single fold (later folds' validation windows contain no
timestamps above their training 90th percentile), so it is thin evidence
and is reported as such, not as a claim.

**Holdout (post-freeze, report-only).** Ridge: MAE 0.181, RMSE 0.224.
Persistence: 0.147 / 0.185. Moving average: 0.123 / 0.161.
Seasonal-naive: 0.140 / 0.159. Active VM count: exact for all methods
(MAE 0), because the series is constant. Stated plainly: the validation
winner is not the holdout winner. The moving average beats the Ridge model
on this holdout by 47.2% in MAE. Switching to the moving average now would
be holdout-driven selection and is not done; the submitted predictions are
the Ridge model's, and this section reports the comparison honestly rather
than cherry-picking the validation table.

**Scale check.** All errors are in percentage points of CPU utilization.
A holdout MAE of 0.18 pp on a series averaging 9.27 pp is roughly a 2%
relative error — small in absolute terms, which is why the operational
recommendation below is about headroom and refresh cadence rather than
about model choice.

## 6. Error and failure analysis

**Error grows with horizon.** First-half holdout MAE is 0.094; second-half
is 0.268 — nearly 3x. This is the signature of recursive forecasting: each
step feeds the previous prediction back in, so a small bias compounds.
The practical consequence is in Section 7: never trust a 2-hour-old
forecast when a fresh one costs seconds to compute.

**The model drifts down while the holdout rebounds.** Predicted CPU
declines gently from 9.20 to 9.07 across the holdout (mean 9.14) while
actuals average 9.27 and end at 9.31–9.50. Late-training one-step changes
were slightly negative, and recursion turns that into a persistent
downward slope. The model therefore undershoots systematically rather
than randomly — a bias problem, not a variance problem.

**The worst miss is unforeseeable.** At timestamp 37,500 the actual value
spikes to 9.636% (the holdout maximum) against a prediction of 9.122, an
error of 0.514 — nearly 3x the model's holdout MAE. Nothing in the
preceding history signals a spike of that size; no history-only model
would catch it. The next-largest errors are the final three timestamps
(9.50, 9.42, 9.43 actual vs ~9.07–9.09 predicted), i.e. the end-of-holdout
rebound. Both failure modes point the same way: the forecast needs an
uncertainty band and a fast refresh, not a more complex model.

**What did not fail.** The VM-count forecast is exact; the baselines
behave as theory predicts (persistence is strong on a flat series,
seasonal-naive is weak without seasonality); and no leakage occurred at
any stage — features use history only, validation precedes the holdout,
and holdout metrics were computed after freezing.

## 7. Capacity-planning recommendation

**Policy (three rules, deliberately simple).**

1. **Provision to forecast plus a safety margin.** Set target capacity at
   forecast + 0.5 pp of CPU headroom. The margin is sized from validation
   error: validation RMSE is 0.167 pp, so ~3x RMSE covers the observed
   error distribution including the 0.51 pp spike miss, while remaining
   small against the ~9.4% operating level.
2. **Peak-risk trigger.** If the trailing 1-hour mean or the forecast
   reaches the training 90th percentile (9.67%), pre-scale before the
   next hour rather than waiting for the spike. The trigger is set from
   training data only and is deliberately conservative, because the
   observed failure mode is under-forecasting rebounds, not
   over-forecasting peaks.
3. **Refresh hourly, not every 2 hours.** Re-run the forecast each hour
   with the latest data. Second-half error is ~3x first-half error, so a
   2-hour-old forecast is the worst version of this system; an hourly
   refresh keeps every decision inside the accurate first half of the
   horizon. Segment rule: apply the trigger to the Interactive pool
   first — it runs ~3.3 pp hotter than Delay-insensitive and drives
   disproportionate demand.

**Trade-off, stated honestly.** Under-provisioning risks latency and SLO
breaches during spikes like the 9.636% event — the cost is user-visible
and concentrated. Over-provisioning wastes capacity, but on a fleet
averaging 9.4% CPU the fleet is already mostly idle, so the marginal cost
of 0.5 pp of headroom is low relative to the SLO risk. The asymmetry
favours erring toward headroom, which is why the margin is set at ~3x
RMSE rather than 1x. The counter-consideration is scale: at 10,000 VMs,
even 0.5 pp of headroom is real hardware, so the margin should be
re-estimated from live error tracking, not treated as permanent.

## 8. Limitations and future work

- **An 11.1-hour window cannot support seasonality.** No daily or weekly
  pattern is learnable here; a longer trace would change the model class
  entirely (seasonal models, calendar features).
- **Timestamps are relative, not wall-clock.** No hour-of-day or
  day-of-week features exist for this data, which removes the most
  useful predictors for real cloud demand.
- **The count target is degenerate.** Active VM count is constant by
  curation; nothing about fleet-size forecasting can be learned or
  demonstrated from this release.
- **Three validation folds is thin evidence**, and the peak-period
  comparison rests on 9 timestamps in one fold. More folds would require
  a longer training window than 110 points allow.
- **Recursive multi-step forecasting drifts.** A direct multi-horizon
  model (one model per horizon step) or a prediction-interval model
  would address the two documented failure modes and is the natural next
  step, along with segment-level forecasting (Interactive vs
  Delay-insensitive) and prediction intervals feeding the headroom
  policy.
- **No external validity claim.** Results describe this curated
  population only; the full 30-day Azure trace, other clouds, and live
  production traffic may behave differently.

## Reproducibility

Environment: Python 3.11+, `pip install -e ".[dev]"` (adds
scikit-learn for the Ridge model). Dataset: download `data-v1.tar.gz`
from the `data-v1` release of this repository, extract into
`challenge_data/`, verify with `sha256sum -c SHA256SUMS` inside
`challenge_data/`. The dataset and tarball are excluded from version
control via `.gitignore` and are never bundled with the submission.

One command, from the repository root, regenerates `predictions.csv`:

```bash
python3 src/predict.py --data challenge_data --output predictions.csv
```

The script also writes `reports/metrics.json` (validation folds,
selection, and post-freeze holdout metrics) and
`reports/analysis_stats.json` (every analysis number quoted above).
Tests: `python3 -m pytest -q` (starter utility tests plus a schema test
for `predictions.csv`).
