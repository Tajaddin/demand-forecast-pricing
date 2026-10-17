from __future__ import annotations

import numpy as np


def _arr(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute percentage error over days with non-zero actuals (zeros would make it infinite)."""
    y, yhat = _arr(y), _arr(yhat)
    mask = y != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error: total absolute error over total actuals. Robust to zero days."""
    y, yhat = _arr(y), _arr(yhat)
    denom = np.abs(y).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y - yhat).sum() / denom * 100)


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = _arr(y), _arr(yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100)


def mase(y: np.ndarray, yhat: np.ndarray, y_train: np.ndarray, m: int = 7) -> float:
    """Mean absolute scaled error against the in-sample seasonal-naive error (m = season length)."""
    y, yhat, y_train = _arr(y), _arr(yhat), _arr(y_train)
    if len(y_train) <= m:
        return float("nan")
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y - yhat)) / scale)


def all_metrics(y: np.ndarray, yhat: np.ndarray, y_train: np.ndarray, m: int = 7) -> dict[str, float]:
    return {"mape": mape(y, yhat), "wape": wape(y, yhat), "smape": smape(y, yhat), "mase": mase(y, yhat, y_train, m)}
