"""Schema tests for the submission predictions file (SUBMISSION.md contract)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import build_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_build_targets_contract():
    readings = pd.DataFrame(
        {
            "timestamp": [0, 0, 300, 300, 300],
            "vm_id": ["a", "b", "a", "a", "b"],
            "avg_cpu": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    out = build_targets(readings)
    # vm 'a' counted once per timestamp even with two rows at t=300
    assert out["active_vm_count"].tolist() == [2, 2]
    assert out["aggregate_cpu_utilization"].tolist() == pytest.approx([15.0, 40.0])


def test_predictions_csv_schema():
    pred_path = ROOT / "predictions.csv"
    assert pred_path.exists(), "run src/predict.py first"
    pred = pd.read_csv(pred_path)
    assert list(pred.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert len(pred) == 24
    assert pred["timestamp"].is_monotonic_increasing
    assert pred["timestamp"].is_unique
    assert pred["timestamp"].notna().all()
    assert (pred["active_vm_count"] >= 0).all()
    assert pred["aggregate_cpu_utilization"].between(0, 100).all()
    assert pred["aggregate_cpu_utilization"].notna().all()
    # timestamps must match the holdout exactly
    readings = pd.read_csv(ROOT / "challenge_data" / "vm_cpu_readings.csv.gz",
                           usecols=["timestamp"])
    holdout_ts = sorted(readings["timestamp"].unique())[-24:]
    assert pred["timestamp"].tolist() == holdout_ts
