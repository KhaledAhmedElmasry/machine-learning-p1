"""
phase1_core.py
==============
Reusable functions for Phase 1 (Gradient Descent) and Phase 2.
NO sklearn .fit() / .predict() used anywhere in this module.
All model fitting is via hand-written GD or numpy.linalg.lstsq.

Authors : Team IDs 16002932, 16004208, 16006616
Seed    : 42648
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Seed utility
# ──────────────────────────────────────────────────────────────────────────────

def compute_seed(student_ids: list[str]) -> int:
    """
    Deterministic seed from student IDs.
    seed = int( SHA-256( "_".join(sorted(ids)) ), 16 ) % 100_000
    """
    joined = "_".join(sorted(student_ids))
    hex_digest = hashlib.sha256(joined.encode()).hexdigest()
    return int(hex_digest, 16) % 100_000


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

# Fixed category universes (determined from the full dataset spec,
# not from a particular split — guarantees transform never sees unseen cats
# from the raw encoding even on test).

_HR_CATS       = list(range(24))          # 0–23
_SEASON_CATS   = [1, 2, 3, 4]
_MNTH_CATS     = list(range(1, 13))       # 1–12
_WEEKDAY_CATS  = list(range(7))           # 0–6
_WEATHER_CATS  = [1, 2, 3]               # 4 merged into 3 (very rare)


def fit_preprocessor(train_df: pd.DataFrame, add_windspeed_flag: bool = True) -> dict:
    """
    Fit all preprocessing parameters on train_df only.
    Returns a params dict that can be passed to transform().

    Parameters
    ----------
    train_df : pd.DataFrame
        The training split (rows, not the full CSV).
    add_windspeed_flag : bool
        Whether to add an is_windspeed_zero binary flag.

    Returns
    -------
    params : dict with keys:
        days_origin, scaler_means, scaler_stds,
        numeric_cols, add_windspeed_flag,
        hr_cats, season_cats, mnth_cats, weekday_cats, weather_cats
    """
    df = train_df.copy()

    # Parse dates
    df["dteday"] = pd.to_datetime(df["dteday"])
    days_origin = df["dteday"].min()

    # Derive days_since_start for scaler fitting
    df["days_since_start"] = (df["dteday"] - days_origin).dt.days

    # Numeric columns to standardise
    numeric_cols = ["temp", "atemp", "hum", "windspeed", "days_since_start"]

    scaler_means = {}
    scaler_stds  = {}
    for col in numeric_cols:
        mu  = float(df[col].mean())
        sig = float(df[col].std(ddof=0))  # population std to be deterministic
        if sig == 0:
            sig = 1.0  # guard against constant column
        scaler_means[col] = mu
        scaler_stds[col]  = sig

    params = {
        "days_origin"        : days_origin.strftime("%Y-%m-%d"),
        "scaler_means"       : scaler_means,
        "scaler_stds"        : scaler_stds,
        "numeric_cols"       : numeric_cols,
        "add_windspeed_flag" : add_windspeed_flag,
        "hr_cats"            : _HR_CATS,
        "season_cats"        : _SEASON_CATS,
        "mnth_cats"          : _MNTH_CATS,
        "weekday_cats"       : _WEEKDAY_CATS,
        "weather_cats"       : _WEATHER_CATS,
    }
    return params


def transform(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, list[str]]:
    """
    Apply preprocessing to df using params from fit_preprocessor().
    Never touches any statistics from df itself.

    Returns
    -------
    X     : np.ndarray, shape (n, p)
    names : list[str], length p  — feature names in exact column order
    """
    df = df.copy()

    # ── Date features ──────────────────────────────────────────────────────
    df["dteday"] = pd.to_datetime(df["dteday"])
    origin = pd.Timestamp(params["days_origin"])
    df["days_since_start"] = (df["dteday"] - origin).dt.days

    # ── Recode weathersit=4 → 3 ────────────────────────────────────────────
    df["weathersit"] = df["weathersit"].clip(upper=3)

    # ── Windspeed flag ─────────────────────────────────────────────────────
    add_flag = params["add_windspeed_flag"]
    if add_flag:
        df["is_windspeed_zero"] = (df["windspeed"] == 0.0).astype(float)

    # ── One-hot encode categoricals ────────────────────────────────────────
    def one_hot(series: pd.Series, cats: list, prefix: str, drop_first: bool = True):
        """Return a DataFrame of dummies with fixed category list."""
        col_names = [f"{prefix}_{c}" for c in cats]
        arr = np.zeros((len(series), len(cats)), dtype=float)
        for j, c in enumerate(cats):
            arr[:, j] = (series == c).astype(float)
        dummy_df = pd.DataFrame(arr, columns=col_names, index=series.index)
        if drop_first:
            dummy_df = dummy_df.iloc[:, 1:]  # drop first level (avoid rank deficiency)
        return dummy_df

    hr_dummies      = one_hot(df["hr"],         params["hr_cats"],      "hr",      drop_first=True)
    season_dummies  = one_hot(df["season"],      params["season_cats"],  "season",  drop_first=True)
    mnth_dummies    = one_hot(df["mnth"],        params["mnth_cats"],    "mnth",    drop_first=True)
    weekday_dummies = one_hot(df["weekday"],     params["weekday_cats"], "weekday", drop_first=True)
    weather_dummies = one_hot(df["weathersit"],  params["weather_cats"], "weather", drop_first=True)

    # ── Scale numerics ────────────────────────────────────────────────────
    means = params["scaler_means"]
    stds  = params["scaler_stds"]
    scaled_parts = {}
    for col in params["numeric_cols"]:
        scaled_parts[col] = (df[col] - means[col]) / stds[col]

    # ── Binary pass-through features ──────────────────────────────────────
    # holiday, workingday, yr  (yr=0/1 encodes 2011/2012)
    binary_features = pd.DataFrame({
        "holiday"    : df["holiday"].values.astype(float),
        "workingday" : df["workingday"].values.astype(float),
        "yr"         : df["yr"].values.astype(float),
    }, index=df.index)

    # ── Assemble ─────────────────────────────────────────────────────────
    parts = [
        hr_dummies,
        season_dummies,
        mnth_dummies,
        weekday_dummies,
        weather_dummies,
        binary_features,
        pd.DataFrame(scaled_parts, index=df.index),
    ]
    if add_flag:
        parts.append(pd.DataFrame(
            {"is_windspeed_zero": df["is_windspeed_zero"].values.astype(float)},
            index=df.index
        ))

    X_df = pd.concat(parts, axis=1)
    names = list(X_df.columns)
    X = X_df.values.astype(np.float64)

    return X, names


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Metrics  (no sklearn fit/predict — sklearn.metrics is allowed per rules)
# ──────────────────────────────────────────────────────────────────────────────

def mse_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """J = 1/(2n) * ||Xw + b - y||^2"""
    r = X @ w + b - y
    return float(0.5 * np.mean(r ** 2))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Gradient functions
# ──────────────────────────────────────────────────────────────────────────────

def mse_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
             ) -> tuple[np.ndarray, float]:
    """
    MSE gradients.
    Returns (grad_w, grad_b).
    Loss = 1/(2n) sum (Xw + b - y)^2
    grad_w = (1/n) X^T r
    grad_b = mean(r)
    """
    n = len(y)
    r = X @ w + b - y
    grad_w = X.T @ r / n
    grad_b = float(np.mean(r))
    return grad_w, grad_b


def asymmetric_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                    over_weight: float = 3.0
                    ) -> tuple[np.ndarray, float]:
    """
    Asymmetric loss gradient (bonus).
    L = (1/n) sum c_i * e_i^2,  e_i = y_i - hat_y_i
    c_i = over_weight if e_i > 0 (under-prediction), else 1.
    grad_w = -(2/n) X^T (c * e)
    grad_b = -(2/n) sum(c * e)
    """
    n = len(y)
    e = y - (X @ w + b)           # e_i = y - yhat (positive = under-predicted)
    c = np.where(e > 0, over_weight, 1.0)
    ce = c * e
    grad_w = -2.0 / n * (X.T @ ce)
    grad_b = float(-2.0 / n * np.sum(ce))
    return grad_w, grad_b


def asymmetric_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                    over_weight: float = 3.0) -> float:
    """Asymmetric loss value (for monitoring — not used as GD loss directly)."""
    e = y - (X @ w + b)
    c = np.where(e > 0, over_weight, 1.0)
    return float(np.mean(c * e ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Hand-written Gradient Descent
# ──────────────────────────────────────────────────────────────────────────────

def gd_fit(
    X      : np.ndarray,
    y      : np.ndarray,
    lr     : float,
    max_iter: int,
    tol    : float  = 1e-6,
    gtol   : float  = 1e-8,
    w0     : Optional[np.ndarray] = None,
    b0     : float  = 0.0,
    grad_fn: Optional[Callable]   = None,
    loss_fn: Optional[Callable]   = None,
    X_val  : Optional[np.ndarray] = None,
    y_val  : Optional[np.ndarray] = None,
    record_every: int = 1,
    verbose: bool = False,
) -> dict:
    """
    Hand-written gradient descent for linear regression.

    Parameters
    ----------
    X, y         : Training data (numpy arrays).
    lr           : Learning rate (step size).
    max_iter     : Maximum number of iterations.
    tol          : Relative loss change stopping criterion.
    gtol         : Gradient norm stopping criterion.
    w0           : Initial weights (None → zeros). Used for Phase 2 warm start.
    b0           : Initial bias.
    grad_fn      : Gradient function(X, y, w, b) → (grad_w, grad_b).
                   Defaults to mse_grad.
    loss_fn      : Loss function(X, y, w, b) → float.
                   Defaults to mse_loss.
    X_val, y_val : Validation data for tracking val loss curve.
    record_every : Record loss every N iterations.
    verbose      : Print progress every 1000 iters.

    Returns
    -------
    dict with keys:
        w, b             : Final weights and bias.
        train_losses     : List of (iter, train_loss).
        val_losses       : List of (iter, val_loss) if X_val provided.
        n_iter           : Number of iterations completed.
        status           : 'converged' | 'not_converged' | 'diverged'.
        final_train_loss : float
        final_val_loss   : float or None
    """
    if grad_fn is None:
        grad_fn = mse_grad
    if loss_fn is None:
        loss_fn = mse_loss

    n, p = X.shape
    w = np.zeros(p, dtype=np.float64) if w0 is None else w0.copy().astype(np.float64)
    b = float(b0)

    train_losses = []
    val_losses   = []

    initial_loss = loss_fn(X, y, w, b)

    # Guard against initial loss being 0 (perfect start)
    if initial_loss == 0.0:
        initial_loss = 1.0

    status = "not_converged"
    prev_loss = initial_loss

    for it in range(1, max_iter + 1):
        grad_w, grad_b = grad_fn(X, y, w, b)

        w = w - lr * grad_w
        b = b - lr * grad_b

        current_loss = loss_fn(X, y, w, b)

        if it % record_every == 0 or it == max_iter:
            train_losses.append((it, float(current_loss)))

            if X_val is not None and y_val is not None:
                vl = loss_fn(X_val, y_val, w, b)
                val_losses.append((it, float(vl)))

            if verbose and it % 1000 == 0:
                g_norm_display = float(np.sqrt(np.linalg.norm(grad_w)**2 + grad_b**2))
                print(f"  iter {it:6d}  loss={current_loss:.6f}  |g|={g_norm_display:.2e}  status={status}")

        # Divergence check
        if (not np.isfinite(current_loss) or
                current_loss > 1e6 * initial_loss):
            status = "diverged"
            break

        # Relative loss change
        rel_change = abs(current_loss - prev_loss) / (abs(prev_loss) + 1e-12)
        if rel_change < tol:
            status = "converged"
            break

        # Gradient norm check
        g_norm = float(np.sqrt(np.linalg.norm(grad_w)**2 + grad_b**2))
        if g_norm < gtol:
            status = "converged"
            break

        prev_loss = current_loss

    final_train_loss = loss_fn(X, y, w, b)
    final_val_loss   = (loss_fn(X_val, y_val, w, b)
                        if X_val is not None and y_val is not None else None)

    return {
        "w"               : w,
        "b"               : b,
        "train_losses"    : train_losses,
        "val_losses"      : val_losses,
        "n_iter"          : it,
        "status"          : status,
        "final_train_loss": float(final_train_loss),
        "final_val_loss"  : float(final_val_loss) if final_val_loss is not None else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Lipschitz constant helper
# ──────────────────────────────────────────────────────────────────────────────

def lipschitz_constant(X: np.ndarray) -> float:
    """
    L = lambda_max( X_aug^T X_aug / n )
    GD is monotone for lr <= 1/L, diverges for lr > 2/L.
    """
    n = X.shape[0]
    X_aug = np.hstack([X, np.ones((n, 1))])
    XtX_n = X_aug.T @ X_aug / n
    eigvals = np.linalg.eigvalsh(XtX_n)
    return float(np.max(eigvals))


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Artifact I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_artifacts(
    path_stem : str,
    w         : np.ndarray,
    b         : float,
    feature_names: list[str],
    params    : dict,
    seed      : int,
    lr        : float,
    tol       : float,
    max_iter  : int,
    n_iter    : int,
    final_train_loss: float,
    final_val_loss  : float,
    train_idx : np.ndarray,
    val_idx   : np.ndarray,
) -> str:
    """
    Save w, b, train_idx, val_idx → <path_stem>.npz
    Save all metadata → <path_stem>.json
    Returns the SHA-256 of w (hex string).
    """
    # SHA-256 of weights
    w_sha256 = hashlib.sha256(w.tobytes()).hexdigest()

    # .npz
    np.savez(
        path_stem + ".npz",
        w=w,
        b=np.array([b]),
        train_idx=train_idx,
        val_idx=val_idx,
    )

    # .json  (everything JSON-serialisable)
    meta = {
        "seed"               : seed,
        "feature_names"      : feature_names,
        "scaler_means"       : params["scaler_means"],
        "scaler_stds"        : params["scaler_stds"],
        "numeric_cols"       : params["numeric_cols"],
        "days_origin"        : params["days_origin"],
        "add_windspeed_flag" : params["add_windspeed_flag"],
        "hr_cats"            : params["hr_cats"],
        "season_cats"        : params["season_cats"],
        "mnth_cats"          : params["mnth_cats"],
        "weekday_cats"       : params["weekday_cats"],
        "weather_cats"       : params["weather_cats"],
        "lr"                 : lr,
        "tol"                : tol,
        "max_iter"           : max_iter,
        "iterations_used"    : n_iter,
        "final_train_loss"   : final_train_loss,
        "final_val_loss"     : final_val_loss,
        "w_sha256"           : w_sha256,
    }
    with open(path_stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    return w_sha256


def load_artifacts(path_stem: str) -> tuple[np.ndarray, float, dict, np.ndarray, np.ndarray]:
    """
    Load artifacts saved by save_artifacts().
    Returns w, b, meta_dict, train_idx, val_idx.
    """
    data = np.load(path_stem + ".npz", allow_pickle=False)
    w = data["w"]
    b = float(data["b"][0])
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    with open(path_stem + ".json") as f:
        meta = json.load(f)
    return w, b, meta, train_idx, val_idx


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Gradient finite-difference checker
# ──────────────────────────────────────────────────────────────────────────────

def gradient_check(
    X      : np.ndarray,
    y      : np.ndarray,
    w      : np.ndarray,
    b      : float,
    grad_fn: Callable,
    loss_fn: Callable,
    eps    : float = 1e-5,
    n_check: int   = 10,
    rng    : Optional[np.random.Generator] = None,
) -> dict:
    """
    Central finite-difference gradient check.
    Checks n_check random weight dimensions plus the bias.

    Returns dict with:
        max_rel_error_w, max_rel_error_b, all_pass (bool, threshold 1e-6)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    grad_w_analytic, grad_b_analytic = grad_fn(X, y, w, b)

    # Check random subset of weight dims
    p = len(w)
    indices = rng.choice(p, size=min(n_check, p), replace=False)

    rel_errors_w = []
    for i in indices:
        w_plus  = w.copy(); w_plus[i]  += eps
        w_minus = w.copy(); w_minus[i] -= eps
        fd = (loss_fn(X, y, w_plus, b) - loss_fn(X, y, w_minus, b)) / (2 * eps)
        an = grad_w_analytic[i]
        rel_err = abs(fd - an) / (abs(an) + abs(fd) + 1e-12)
        rel_errors_w.append(rel_err)

    # Check bias
    b_plus  = b + eps
    b_minus = b - eps
    fd_b  = (loss_fn(X, y, w, b_plus) - loss_fn(X, y, w, b_minus)) / (2 * eps)
    an_b  = grad_b_analytic
    rel_err_b = abs(fd_b - an_b) / (abs(an_b) + abs(fd_b) + 1e-12)

    max_rel_w = float(max(rel_errors_w))
    max_rel_b = float(rel_err_b)

    return {
        "max_rel_error_w" : max_rel_w,
        "max_rel_error_b" : max_rel_b,
        "all_pass"        : (max_rel_w < 1e-5) and (max_rel_b < 1e-5),
    }

