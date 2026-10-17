"""
logistic_regression.py
Reusable functions for Phase 5 (Logistic Regression: "is this a high-demand hour?").

Reuses gradient_descent.gd_fit unchanged (through its grad_fn / loss_fn hooks) and the
Phase 1-3 transforms. Nothing here calls an sklearn fit/predict.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from typing import Callable, Optional

import numpy as np
import pandas as pd

import gradient_descent as core
import polynomial_regression as p2

HIGH_QUANTILE = 0.75
GROUP_COLS = ["hr", "workingday"]


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #

def group_thresholds(train_df: pd.DataFrame, q: float = HIGH_QUANTILE) -> dict[tuple[int, int], float]:
    """q-quantile of cnt per (hr, workingday) group, from the rows given (train only).

    Raises ValueError naming any of the 48 expected groups that has no rows.
    Uses pandas' default linear interpolation for the quantile.
    """
    grp = train_df.groupby(GROUP_COLS)["cnt"].quantile(q)
    expected = set(product(range(24), (0, 1)))
    present = {(int(h), int(w)) for h, w in grp.index}
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"No training rows for (hr, workingday) groups: {missing}")
    return {(int(h), int(w)): float(v) for (h, w), v in grp.items()}


def small_groups(train_df: pd.DataFrame, min_n: int = 5) -> dict[tuple[int, int], int]:
    """Groups with fewer than min_n training rows (their quantile is unreliable)."""
    sizes = train_df.groupby(GROUP_COLS).size()
    return {(int(h), int(w)): int(n) for (h, w), n in sizes.items() if n < min_n}


def make_target(df: pd.DataFrame, thresholds: dict[tuple[int, int], float]) -> np.ndarray:
    """1.0 where cnt is strictly greater than the row's (hr, workingday) threshold, else 0.0."""
    thr = np.array([thresholds[(int(h), int(w))] for h, w in zip(df["hr"], df["workingday"])], dtype=np.float64)
    return (df["cnt"].values.astype(np.float64) > thr).astype(np.float64)


def global_target(train_df: pd.DataFrame, df: pd.DataFrame, q: float = HIGH_QUANTILE) -> np.ndarray:
    """Naive target: one global q-quantile of train cnt (the flawed definition, for comparison)."""
    thr = float(train_df["cnt"].quantile(q))
    return (df["cnt"].values.astype(np.float64) > thr).astype(np.float64)


def thresholds_to_json(thresholds: dict[tuple[int, int], float]) -> dict[str, float]:
    return {f"{h}|{w}": v for (h, w), v in thresholds.items()}


def thresholds_from_json(d: dict[str, float]) -> dict[tuple[int, int], float]:
    out = {}
    for k, v in d.items():
        h, w = k.split("|")
        out[(int(h), int(w))] = float(v)
    return out


# --------------------------------------------------------------------------- #
# Features (Phase 1 preprocessing -> Phase 3 expansion -> Phase 4 Lasso subset)
# --------------------------------------------------------------------------- #

def build_features(
    df: pd.DataFrame,
    p1_meta: dict,
    poly_params: dict,
    feature_list: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Stored Phase 1 preprocessing, stored Phase 3 expansion, then keep feature_list in that order.

    Raises KeyError listing any name in feature_list that the expansion does not produce.
    """
    X1, names1 = core.transform(df, p1_meta)
    Xe, names = p2.transform_poly(X1, names1, poly_params)
    missing = [f for f in feature_list if f not in names]
    if missing:
        raise KeyError(f"Features not produced by the Phase 3 expansion: {missing}")
    idx = [names.index(f) for f in feature_list]
    return Xe[:, idx], list(feature_list)


# --------------------------------------------------------------------------- #
# Loss, gradient, step size
# --------------------------------------------------------------------------- #

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic function."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights n / (2 * n_class), so both classes carry equal total weight."""
    y = np.asarray(y)
    n = len(y)
    n_pos = float(np.sum(y == 1))
    n_neg = float(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("balanced_weights needs both classes present")
    return np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))


def bce_loss(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
    l2: float = 0.0, sample_weight: Optional[np.ndarray] = None,
) -> float:
    """(1/n) * sum_i sw_i * [softplus(z_i) - y_i * z_i] + (l2/2) * ||w||^2, with z = Xw + b.

    softplus(z) = max(z, 0) + log1p(exp(-|z|)) never overflows and never takes log(0).
    The intercept b is not penalised.
    """
    z = X @ w + b
    ell = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - y * z
    if sample_weight is not None:
        ell = sample_weight * ell
    return float(np.mean(ell) + 0.5 * l2 * np.sum(w ** 2))


def bce_grad(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
    l2: float = 0.0, sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Gradient of bce_loss: (1/n) X^T [sw * (sigmoid(z) - y)] + l2 * w, and its bias analogue."""
    n = len(y)
    r = sigmoid(X @ w + b) - y
    if sample_weight is not None:
        r = sample_weight * r
    return X.T @ r / n + l2 * w, float(np.mean(r))


def make_fns(
    l2: float = 0.0, sample_weight: Optional[np.ndarray] = None
) -> tuple[Callable, Callable]:
    """(grad_fn, loss_fn) closures with the signature gd_fit expects.

    sample_weight belongs to the training rows. gd_fit also evaluates loss_fn on the validation
    arrays; when the length differs, that call is unweighted (and unpenalised comparison is
    left to the caller), so the weights are never applied to the wrong rows.
    """
    def _sw(y):
        if sample_weight is not None and len(sample_weight) == len(y):
            return sample_weight
        return None

    def grad_fn(X, y, w, b):
        return bce_grad(X, y, w, b, l2=l2, sample_weight=_sw(y))

    def loss_fn(X, y, w, b):
        return bce_loss(X, y, w, b, l2=l2, sample_weight=_sw(y))

    return grad_fn, loss_fn


def lipschitz_logistic(
    X: np.ndarray, l2: float = 0.0, sample_weight: Optional[np.ndarray] = None
) -> float:
    """L = lambda_max(X_aug^T diag(sw) X_aug / n) / 4 + l2.

    The logistic Hessian is X^T diag(s(1-s)) X / n with s(1-s) <= 1/4, so GD is monotone for step <= 1/L.
    """
    n = X.shape[0]
    X_aug = np.hstack([X, np.ones((n, 1))])
    if sample_weight is not None:
        XtX = X_aug.T @ (X_aug * sample_weight[:, None]) / n
    else:
        XtX = X_aug.T @ X_aug / n
    return float(np.max(np.linalg.eigvalsh(XtX)) / 4.0 + l2)


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


# --------------------------------------------------------------------------- #
# Metrics (numpy only)
# --------------------------------------------------------------------------- #

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """[[TN, FP], [FN, TP]]."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall, F1 for the positive class; 0.0 (not an error) when a denominator is 0."""
    (_, fp), (fn, tp) = confusion(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC via the rank-sum (Mann-Whitney) formula with average ranks for ties."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs both classes present in y_true")
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg_rank = ends - (counts - 1) / 2.0   # mean of the 1-based ranks inside each tie block
    ranks = avg_rank[inv]
    return float((np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) at every distinct score, descending, starting at (0, 0)."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    ys = y_true[order]
    ss = np.asarray(score)[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tpr = np.r_[0.0, tps[last] / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps[last] / max(fps[-1], 1)]
    return fpr, tpr


def best_threshold(
    y_true: np.ndarray, proba: np.ndarray, grid: Optional[np.ndarray] = None
) -> dict:
    """Probability threshold (from a grid of >= 99 values) that maximises F1."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    f1s = np.array([precision_recall_f1(y_true, (proba >= t).astype(int))[2] for t in grid])
    k = int(np.argmax(f1s))
    return {"threshold": float(grid[k]), "f1": float(f1s[k]), "grid": grid, "f1_grid": f1s}


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1, AUC and the confusion matrix at one threshold."""
    pred = (proba >= threshold).astype(int)
    prec, rec, f1 = precision_recall_f1(y_true, pred)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy(y_true, pred),
        "precision": prec, "recall": rec, "f1": f1,
        "auc": roc_auc(y_true, proba),
        "confusion": confusion(y_true, pred).tolist(),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(pred)),
    }


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #

def save_artifacts(
    path_stem: str, w: np.ndarray, b: float, meta: dict,
    train_idx: np.ndarray, val_idx: np.ndarray,
) -> str:
    """Write .npz (w, b, indices) and .json (meta + SHA-256 of w). Returns the SHA-256."""
    w_sha256 = hashlib.sha256(w.tobytes()).hexdigest()
    np.savez(path_stem + ".npz", w=w, b=np.array([b]), train_idx=train_idx, val_idx=val_idx)
    out = dict(meta)
    out["w_sha256"] = w_sha256
    with open(path_stem + ".json", "w") as f:
        json.dump(out, f, indent=2)
    return w_sha256


def load_artifacts(path_stem: str) -> tuple[np.ndarray, float, dict, np.ndarray, np.ndarray]:
    """Returns w, b, meta, train_idx, val_idx."""
    data = np.load(path_stem + ".npz", allow_pickle=False)
    with open(path_stem + ".json") as f:
        meta = json.load(f)
    return data["w"], float(data["b"][0]), meta, data["train_idx"], data["val_idx"]
