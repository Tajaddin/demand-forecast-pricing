"""Data drift between the window a model was trained on and the newest data, with Evidently."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DRIFT_COLUMNS = ["arrivals", "room_nights", "revenue", "adr_mean", "lead_time_mean", "cancellations"]
# The forecaster consumes the arrivals history and the calendar; these are the columns the promotion gate judges.
GATE_COLUMNS = ["arrivals", "room_nights", "cancellations", "dow"]


def drift_frame(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily[["hotel", *DRIFT_COLUMNS]].copy()
    df["dow"] = daily["arrival_date"].dt.dayofweek
    return df


def _drifted_share(snapshot_dict: dict) -> tuple[float, int, dict[str, dict]]:
    """Reads Evidently 0.7's snapshot: the DriftedColumnsCount value and, per column, the test it ran.

    ValueDrift reports a statistic per column; for p-value methods drift means value < threshold, for
    distance methods value > threshold.
    """
    share, count = 0.0, 0
    per_column: dict[str, dict] = {}
    for m in snapshot_dict.get("metrics", []):
        cfg = m.get("config") or {}
        kind = str(cfg.get("type", ""))
        value = m.get("value")
        if kind.endswith("DriftedColumnsCount") and isinstance(value, dict):
            share = float(value.get("share", share))
            count = int(value.get("count", count))
        elif kind.endswith("ValueDrift") and isinstance(value, (int, float)):
            method = str(cfg.get("method", ""))
            threshold = float(cfg.get("threshold", 0.05))
            drifted = value < threshold if "p_value" in method else value > threshold
            per_column[str(cfg.get("column"))] = {
                "method": method,
                "statistic": float(value),
                "threshold": threshold,
                "drifted": bool(drifted),
            }
    return share, count, per_column


def run_drift(reference: pd.DataFrame, current: pd.DataFrame, out_dir: Path, name: str = "drift") -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{name}.html"
    snapshot.save_html(str(html))
    d = snapshot.dict()
    share, count, per_column = _drifted_share(d)
    result = {
        "drifted_share": share,
        "drifted_columns": count,
        "columns": list(reference.columns),
        "per_column": per_column,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "html": str(html),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
