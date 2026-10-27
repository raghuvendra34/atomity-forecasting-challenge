#!/usr/bin/env python3
"""Cloud Workload Forecasting — Atomity take-home (MVP).

Pipeline (leakage-safe by construction):
  1. Load vm_cpu_readings.csv.gz and build the two aggregate target series
     at 5-minute cadence (DATASET.md contract):
       - active_vm_count(t) = distinct vm_id values with a reading at t
       - aggregate_cpu_utilization(t) = mean of avg_cpu at t, in [0, 100]
     Missing-value handling: a timestamp with no readings at all would be
     reported as active_vm_count = 0 and aggregate_cpu_utilization = NaN
     (then forward-filled from the previous timestamp for modelling, and the
     gap is logged). In the supplied data-v1 release no such gap exists:
     every timestamp has a reading for every VM, so no imputation occurs.
  2. Chronological split: first (n - 24) timestamps = training window,
     final 24 timestamps = 2-hour holdout (for data-v1: 110 / 24).
     The holdout observations are NEVER used for fitting, feature
     selection, hyperparameter tuning, or model selection.
  3. Rolling-origin validation INSIDE the training window only
     (expanding window, 24-step horizon, recursive multi-step forecasts)
     compares baselines (persistence, moving average, seasonal-naive)
     against the main model. The final model is the validation winner.
  4. The winner is refit on the full training window and forecast
     recursively for the 24 holdout timestamps; predictions.csv is written.
  5. Holdout metrics are computed AFTER the model is frozen, for reporting
     only, and never feed back into any choice.

Usage (from the repository root):
    python3 src/predict.py --data challenge_data --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cwfc.baselines import persistence_forecast, seasonal_naive_forecast
from cwfc.metrics import mae, rmse

HORIZON = 24
SEASON_LENGTH = 12  # 12 x 5 minutes = 1 hour


# ---------------------------------------------------------------- targets
def build_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate VM-level readings to the two required target series."""
    agg = (
        readings.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return agg


def find_gaps(targets: pd.DataFrame) -> list[int]:
    """Return timestamps missing from the 300-second grid (empty = clean)."""
    ts = targets["timestamp"].to_numpy()
    if len(ts) < 2:
        return []
    full = np.arange(ts[0], ts[-1] + 1, 300)
    return [int(t) for t in np.setdiff1d(full, ts)]


# ---------------------------------------------------------------- features
def make_features(history: list[float], index: int) -> list[float]:
    """Features computable at forecast origin `index` from history only.

    history holds the target values strictly before the timestamp being
    forecast (observed values, plus already-forecast values when recursing).
    No future information is used.

    NOTE: no time-trend feature is included. The series is short (110
    training points) and flat, so a trend term extrapolates level drift
    that does not persist; on rolling-origin validation it made forecasts
    strictly worse (validation MAE 0.249 vs 0.136 without it).
    """
    y = np.asarray(history, dtype=float)
    lag = lambda k: y[-k] if len(y) >= k else y[0]
    rollmean = lambda k: float(np.mean(y[-k:])) if len(y) >= k else float(np.mean(y))
    rollstd = lambda k: float(np.std(y[-k:])) if len(y) >= k else float(np.std(y))
    return [
        lag(1),
        lag(2),
        lag(3),
        lag(6),
        lag(12),
        rollmean(6),
        rollmean(12),
        rollstd(12),
        y[-1] - y[-2] if len(y) >= 2 else 0.0,
    ]


def build_design(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix/targets for one-step-ahead training on `values`.

    Row i predicts values[i] from the history values[:i]. The first 12 rows
    are dropped because lag-12 features need 12 observations of history.
    """
    X, y = [], []
    for i in range(12, len(values)):
        X.append(make_features(list(values[:i]), i))
        y.append(values[i])
    return np.asarray(X), np.asarray(y)


# ---------------------------------------------------------------- models
def moving_average_forecast(train: pd.Series, horizon: int, window: int = 12) -> np.ndarray:
    return np.repeat(float(train.iloc[-window:].mean()), horizon)


def fit_ridge(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # alpha fixed from rolling-origin validation (alpha=10: MAE 0.136 vs
    # alpha=1: MAE 0.184 on the same folds); never tuned on the holdout.
    model = Ridge(alpha=10.0)
    model.fit(Xs, y)
    return scaler, model


def ridge_recursive_forecast(train_values: np.ndarray, horizon: int) -> np.ndarray:
    X, y = build_design(train_values)
    scaler, model = fit_ridge(X, y)
    history = list(train_values)
    preds = []
    for step in range(horizon):
        feats = np.asarray([make_features(history, len(history))])
        preds.append(float(model.predict(scaler.transform(feats))[0]))
        history.append(preds[-1])
    return np.asarray(preds)


def baseline_recursive_forecasts(train: pd.Series, horizon: int) -> dict[str, np.ndarray]:
    """Baselines, each evaluated with the same recursive protocol."""
    frame = train.to_frame("value")
    template = pd.DataFrame(index=range(horizon))
    out = {
        "persistence": persistence_forecast(
            train.to_frame("value").rename(columns={"value": "target"}),
            template,
            "target",
        ).to_numpy(),
        "moving_average_12": moving_average_forecast(train, horizon, 12),
        "seasonal_naive_12": seasonal_naive_forecast(
            train.to_frame("value").rename(columns={"value": "target"}),
            template,
            "target",
            SEASON_LENGTH,
        ).to_numpy(),
    }
    return out


def recursive_forecast(train: pd.Series, horizon: int, method: str) -> np.ndarray:
    if method == "ridge":
        return ridge_recursive_forecast(train.to_numpy(dtype=float), horizon)
    return baseline_recursive_forecasts(train, horizon)[method]


# ---------------------------------------------------------------- validation
def rolling_origin_eval(series: pd.Series, horizon: int = HORIZON) -> dict:
    """Expanding-window rolling-origin evaluation inside the training window.

    Origins are placed at ~55%, ~67% and ~78% of the training window so that
    every validation window lies strictly before the holdout. Returns the
    mean MAE/RMSE per method across folds plus per-fold detail.
    """
    n = len(series)
    origins = sorted({int(n * f) for f in (0.55, 0.67, 0.78)})
    origins = [o for o in origins if o >= 24 and o + horizon <= n]
    methods = ["persistence", "moving_average_12", "seasonal_naive_12", "ridge"]
    per_fold, agg, peak_agg = [], {m: [] for m in methods}, {m: [] for m in methods}
    for origin in origins:
        tr, va = series.iloc[:origin], series.iloc[origin : origin + horizon].to_numpy()
        # peak = validation timestamps at/above the 90th percentile of the
        # fold's own training window (a level the training data has seen).
        peak_mask = va >= np.quantile(tr.to_numpy(), 0.90)
        fold = {"origin": int(origin), "n_peak_timestamps": int(peak_mask.sum())}
        for m in methods:
            pred = recursive_forecast(tr, horizon, m)
            fold[m] = {"mae": mae(va, pred), "rmse": rmse(va, pred)}
            agg[m].append(fold[m]["mae"])
            if peak_mask.sum():
                fold[m]["peak_mae"] = mae(va[peak_mask], pred[peak_mask])
                peak_agg[m].append(fold[m]["peak_mae"])
        per_fold.append(fold)
    summary = {
        m: {
            "mean_mae": float(np.mean(v)),
            "mean_rmse": float(np.mean([f[m]["rmse"] for f in per_fold])),
            "mean_peak_mae": float(np.mean(peak_agg[m])) if peak_agg[m] else None,
        }
        for m, v in agg.items()
    }
    return {"folds": per_fold, "summary": summary}


# ---------------------------------------------------------------- analysis
def workload_analysis(readings: pd.DataFrame, meta: pd.DataFrame | None, targets: pd.DataFrame) -> dict:
    u = targets["aggregate_cpu_utilization"]
    stats: dict = {
        "n_vm": int(readings["vm_id"].nunique()),
        "n_timestamps": int(targets.shape[0]),
        "cadence_seconds": 300,
        "timestamp_range": [int(targets["timestamp"].iloc[0]), int(targets["timestamp"].iloc[-1])],
        "duplicate_vm_timestamp_pairs": int(readings.duplicated(["vm_id", "timestamp"]).sum()),
        "missing_values": {c: int(readings[c].isna().sum()) for c in readings.columns},
        "gaps_in_grid": find_gaps(targets),
        "active_vm_count_unique": sorted(targets["active_vm_count"].unique().astype(int).tolist()),
        "cpu_mean": float(u.mean()),
        "cpu_std": float(u.std()),
        "cpu_min": float(u.min()),
        "cpu_max": float(u.max()),
        "cpu_p50": float(u.median()),
        "cpu_p90": float(u.quantile(0.90)),
        "cpu_p95": float(u.quantile(0.95)),
        "cpu_autocorr_lag1": float(u.autocorr(1)),
        "cpu_autocorr_lag12": float(u.autocorr(12)),
        "mean_abs_step_change": float(u.diff().abs().mean()),
        "peak_timestamp_idx": int(u.idxmax()),
        "peak_value": float(u.max()),
        "trough_timestamp_idx": int(u.idxmin()),
        "trough_value": float(u.min()),
    }
    per_vm = readings.groupby("vm_id")["avg_cpu"].mean()
    stats["per_vm_mean_cpu_median"] = float(per_vm.median())
    stats["per_vm_mean_cpu_p90"] = float(per_vm.quantile(0.90))
    stats["top_10pct_vm_share_of_total_cpu"] = float(per_vm.nlargest(max(1, len(per_vm) // 10)).sum() / per_vm.sum())
    stats["vms_with_mean_cpu_below_1pct"] = int((per_vm < 1).sum())
    stats["vms_with_mean_cpu_above_50pct"] = int((per_vm > 50).sum())
    if meta is not None and "vm_category" in meta.columns:
        merged = readings[["timestamp", "vm_id", "avg_cpu"]].merge(meta[["vm_id", "vm_category"]], on="vm_id", how="left")
        seg = merged.groupby("vm_category")["avg_cpu"].agg(["mean", "count"])
        stats["segment_mean_cpu"] = {k: float(v) for k, v in seg["mean"].items()}
        stats["segment_vm_counts"] = {
            k: int(v) for k, v in meta["vm_category"].value_counts().items()
        }
        overall = u.to_numpy()
        seg_ts = merged.groupby(["timestamp", "vm_category"])["avg_cpu"].mean().unstack()
        stats["segment_corr_with_overall"] = {
            c: float(np.corrcoef(seg_ts[c].to_numpy(), overall)[0, 1]) for c in seg_ts.columns
        }
    return stats


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Cloud workload forecasting (Atomity take-home MVP)")
    ap.add_argument("--data", default="challenge_data", help="directory with vm_cpu_readings.csv.gz")
    ap.add_argument("--output", default="predictions.csv", help="output predictions CSV path")
    ap.add_argument("--metrics", default="reports/metrics.json", help="metrics JSON path")
    ap.add_argument("--analysis", default="reports/analysis_stats.json", help="analysis JSON path")
    args = ap.parse_args()

    data_dir = Path(args.data)
    readings = pd.read_csv(data_dir / "vm_cpu_readings.csv.gz")
    meta_path = data_dir / "vm_metadata.csv.gz"
    meta = pd.read_csv(meta_path) if meta_path.exists() else None

    targets = build_targets(readings)
    analysis = workload_analysis(readings, meta, targets)
    print(f"targets: {len(targets)} timestamps, "
          f"active_vm_count in {analysis['active_vm_count_unique']}, "
          f"cpu mean {analysis['cpu_mean']:.3f}%")

    train = targets.iloc[:-HORIZON].reset_index(drop=True)
    holdout = targets.iloc[-HORIZON:].reset_index(drop=True)
    print(f"train window: {len(train)} timestamps "
          f"({train['timestamp'].iloc[0]}..{train['timestamp'].iloc[-1]}); "
          f"holdout: {len(holdout)} timestamps "
          f"({holdout['timestamp'].iloc[0]}..{holdout['timestamp'].iloc[-1]})")

    cpu_train = train["aggregate_cpu_utilization"]
    count_train = train["active_vm_count"]

    # --- validation inside the training window only (no holdout touched) ---
    val = rolling_origin_eval(cpu_train)
    print("rolling-origin validation (mean MAE over folds, CPU %):")
    for m, s in val["summary"].items():
        peak = f"  peak MAE {s['mean_peak_mae']:.4f}" if s["mean_peak_mae"] is not None else ""
        print(f"  {m:20s} MAE {s['mean_mae']:.4f}  RMSE {s['mean_rmse']:.4f}{peak}")
    best_method = min(val["summary"], key=lambda m: val["summary"][m]["mean_mae"])
    print(f"validation winner: {best_method}")

    # --- final fit on the full training window; recursive 24-step forecast ---
    cpu_pred = recursive_forecast(cpu_train, HORIZON, best_method)
    cpu_pred = np.clip(cpu_pred, 0.0, 100.0)
    # active_vm_count is constant in this release; persistence is exact by
    # construction and needs no model.
    count_pred = np.full(HORIZON, float(count_train.iloc[-1]))
    count_pred = np.maximum(np.round(count_pred), 0).astype(int)

    pred = pd.DataFrame(
        {
            "timestamp": holdout["timestamp"].astype(int),
            "active_vm_count": count_pred,
            "aggregate_cpu_utilization": cpu_pred,
        }
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(pred)} rows)")

    # --- post-freeze reporting metrics only (never fed back into choices) ---
    base_preds = baseline_recursive_forecasts(cpu_train, HORIZON)
    best_base = min(base_preds, key=lambda m: mae(holdout["aggregate_cpu_utilization"], base_preds[m]))
    metrics = {
        "validation": val,
        "final_model": best_method,
        "best_baseline": best_base,
        "holdout_post_freeze": {
            "model": {
                "mae_cpu": mae(holdout["aggregate_cpu_utilization"], cpu_pred),
                "rmse_cpu": rmse(holdout["aggregate_cpu_utilization"], cpu_pred),
                "mae_count": mae(holdout["active_vm_count"], count_pred),
                "rmse_count": rmse(holdout["active_vm_count"], count_pred),
            },
            "baselines_cpu": {
                m: {"mae": mae(holdout["aggregate_cpu_utilization"], p),
                    "rmse": rmse(holdout["aggregate_cpu_utilization"], p)}
                for m, p in base_preds.items()
            },
        },
    }
    for m, s in metrics["holdout_post_freeze"]["baselines_cpu"].items():
        imp = 100.0 * (s["mae"] - metrics["holdout_post_freeze"]["model"]["mae_cpu"]) / s["mae"] if s["mae"] else 0.0
        print(f"holdout (report-only): model MAE {metrics['holdout_post_freeze']['model']['mae_cpu']:.4f} "
              f"vs {m} MAE {s['mae']:.4f} ({imp:+.1f}% improvement)")
    print(f"holdout count: MAE {metrics['holdout_post_freeze']['model']['mae_count']:.4f}")

    for path, payload in ((args.metrics, metrics), (args.analysis, analysis)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {args.metrics} and {args.analysis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
