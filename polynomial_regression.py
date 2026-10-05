"""
phase2_core.py
==============
Polynomial feature expansion for Phase 2 (Polynomial Regression).
Reuses phase1_core.py functions unchanged.
NO sklearn .fit() / .predict() used anywhere in this module.

Authors : Team IDs 16002932, 16004208, 16006616
Seed    : 42648
"""

from __future__ import annotations

import hashlib
import json
import time
from itertools import combinations_with_replacement
from math import comb
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import gradient_descent as core  # reuse unchanged


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Polynomial term bookkeeping
# ──────────────────────────────────────────────────────────────────────────────

def poly_term_count_expected(k: int, d: int, interactions: bool) -> int:
    """
    Expected number of NEW polynomial columns (degree 2..d) to be appended.

    interactions=False (pure powers): k * (d-1)   [x^2 .. x^d for each source]
    interactions=True  (full):        C(k+d,d) - k - 1
                                      [all monomials of degree 2..d]

    Note: total monomials of degree 1..d =
       k*d             (pure powers, counting degree 1)
       C(k+d,d) - 1   (full, counting degree 1)
    These totals include the degree-1 terms already in Phase 1.
    """
    if interactions:
        return comb(k + d, d) - k - 1
    else:
        return k * (d - 1)


def get_poly_term_specs(
    source_cols: list[str],
    degree: int,
    interactions: bool = False,
) -> list[tuple[str, dict]]:
    """
    Returns list of (term_name, exponent_dict) for all monomials of degree 2..degree.

    Parameters
    ----------
    source_cols  : names of the source continuous columns (in Phase 1 feature space).
    degree       : maximum degree.
    interactions : if True, include all cross terms; if False, pure powers only.

    Returns
    -------
    terms : list of (name: str, exp_dict: {col: power})
            in a deterministic order (degree-ascending, then lex).
    """
    k = len(source_cols)
    terms = []

    if interactions:
        for d in range(2, degree + 1):
            # All multisets of size d from {0..k-1} indices
            for combo in combinations_with_replacement(range(k), d):
                exp_dict: dict[str, int] = {}
                for idx in combo:
                    col = source_cols[idx]
                    exp_dict[col] = exp_dict.get(col, 0) + 1
                # Build name in source_cols order (deterministic)
                name_parts = []
                for col in source_cols:
                    if col in exp_dict:
                        e = exp_dict[col]
                        name_parts.append(f"{col}^{e}" if e > 1 else col)
                name = "*".join(name_parts)
                terms.append((name, dict(exp_dict)))
    else:
        # Pure powers only: col^2, col^3, ..., col^d
        for d in range(2, degree + 1):
            for col in source_cols:
                name = f"{col}^{d}"
                terms.append((name, {col: d}))

    return terms


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Expansion module
# ──────────────────────────────────────────────────────────────────────────────

def fit_poly(
    X_phase1_train   : np.ndarray,
    phase1_feat_names: list[str],
    source_cols      : list[str],
    degree           : int,
    interactions     : bool = False,
) -> dict:
    """
    Fit polynomial expansion parameters on training data only.
    Input X_phase1_train must be the Phase-1-standardised matrix (unchanged).

    Returns
    -------
    poly_params : dict with keys:
        term_specs     – list of (name, exp_dict)
        source_cols    – as given
        source_indices – column positions in phase1_feat_names
        degree, interactions
        poly_means, poly_stds  – per new column (train-only)
        n_new_cols
    """
    # Validate source columns
    for col in source_cols:
        if col not in phase1_feat_names:
            raise ValueError(f"Source column '{col}' not in phase1_feat_names")

    term_specs = get_poly_term_specs(source_cols, degree, interactions)
    source_indices = [phase1_feat_names.index(c) for c in source_cols]

    n = X_phase1_train.shape[0]
    poly_means = []
    poly_stds  = []

    for name, exp_dict in term_specs:
        raw_col = _compute_term(X_phase1_train, phase1_feat_names, exp_dict)
        mu  = float(raw_col.mean())
        sig = float(raw_col.std(ddof=0))
        if sig == 0.0:
            sig = 1.0
        poly_means.append(mu)
        poly_stds.append(sig)

    return {
        "term_specs"     : term_specs,
        "source_cols"    : source_cols,
        "source_indices" : source_indices,
        "degree"         : degree,
        "interactions"   : interactions,
        "poly_means"     : poly_means,
        "poly_stds"      : poly_stds,
        "n_new_cols"     : len(term_specs),
    }


def transform_poly(
    X_phase1        : np.ndarray,
    phase1_feat_names: list[str],
    poly_params     : dict,
) -> tuple[np.ndarray, list[str]]:
    """
    Apply polynomial expansion.

    The Phase 1 block is passed through UNCHANGED (bit-identical).
    New standardised polynomial columns are appended after.

    Returns
    -------
    X_expanded     : np.ndarray  shape (n, p1 + n_new)
    expanded_names : list[str]   length p1 + n_new
    """
    term_specs  = poly_params["term_specs"]
    poly_means  = poly_params["poly_means"]
    poly_stds   = poly_params["poly_stds"]

    n = X_phase1.shape[0]
    new_cols = np.empty((n, len(term_specs)), dtype=np.float64)

    for j, (name, exp_dict) in enumerate(term_specs):
        raw_col = _compute_term(X_phase1, phase1_feat_names, exp_dict)
        new_cols[:, j] = (raw_col - poly_means[j]) / poly_stds[j]

    X_expanded = np.hstack([X_phase1, new_cols])
    new_names  = [name for name, _ in term_specs]
    return X_expanded, list(phase1_feat_names) + new_names


def _compute_term(
    X               : np.ndarray,
    feat_names      : list[str],
    exp_dict        : dict[str, int],
) -> np.ndarray:
    """Compute one monomial column from X (product of powers)."""
    n   = X.shape[0]
    val = np.ones(n, dtype=np.float64)
    for col, power in exp_dict.items():
        idx  = feat_names.index(col)
        val *= X[:, idx] ** power
    return val


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Warm-start helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_warm_start(
    w_phase1 : np.ndarray,
    b_phase1 : float,
    n_new    : int,
) -> tuple[np.ndarray, float]:
    """
    Construct w_init = [w_phase1, 0, ..., 0],  b_init = b_phase1.

    At iteration 0:
        X_expanded @ w_init + b_init
        = X_phase1 @ w_phase1 + X_new @ 0 + b_phase1
        = Phase 1 predictions   (exactly)
    """
    w_init = np.concatenate([w_phase1, np.zeros(n_new, dtype=np.float64)])
    return w_init, float(b_phase1)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Convenience: full pipeline fit
# ──────────────────────────────────────────────────────────────────────────────

def fit_expanded(
    X_train          : np.ndarray,
    y_train          : np.ndarray,
    X_val            : np.ndarray,
    y_val            : np.ndarray,
    phase1_feat_names: list[str],
    source_cols      : list[str],
    degree           : int,
    interactions     : bool,
    w_phase1         : np.ndarray,
    b_phase1         : float,
    max_iter         : int   = 200_000,
    tol              : float = 1e-6,
    record_every     : int   = 500,
    verbose          : bool  = False,
) -> dict:
    """
    Fit the polynomial GD model for one (degree, interactions) setting.

    Returns a results dict with:
        poly_params, X_train_exp, X_val_exp, expanded_names,
        w, b, n_iter, status,
        train_r2, val_r2, train_rmse, val_rmse,
        final_train_loss, final_val_loss,
        kappa, L_d, lr_used,
        gd_result   (full dict from gd_fit),
        lstsq_train_loss, lstsq_val_r2,
        opt_gap_relative, optimization_limited,
        wall_time_s,
    """
    # ── Build expanded matrices ───────────────────────────────────────────
    poly_params = fit_poly(X_train, phase1_feat_names, source_cols, degree, interactions)
    X_tr_exp, exp_names = transform_poly(X_train, phase1_feat_names, poly_params)
    X_vl_exp, _        = transform_poly(X_val,   phase1_feat_names, poly_params)

    p_exp = X_tr_exp.shape[1]

    # ── Lipschitz constant & LR ───────────────────────────────────────────
    L_d   = core.lipschitz_constant(X_tr_exp)
    lr    = 0.9 / L_d

    # ── Condition number ──────────────────────────────────────────────────
    XtX      = X_tr_exp.T @ X_tr_exp
    eigvals  = np.linalg.eigvalsh(XtX)
    pos_eig  = eigvals[eigvals > 1e-10]
    kappa    = float(pos_eig.max() / pos_eig.min()) if len(pos_eig) > 0 else float("inf")

    # ── Warm start ────────────────────────────────────────────────────────
    w_init, b_init = build_warm_start(w_phase1, b_phase1, poly_params["n_new_cols"])

    # ── GD fit ────────────────────────────────────────────────────────────
    t0 = time.time()
    gd_res = core.gd_fit(
        X_tr_exp, y_train,
        lr=lr, max_iter=max_iter, tol=tol,
        w0=w_init, b0=b_init,
        X_val=X_vl_exp, y_val=y_val,
        record_every=record_every,
        verbose=verbose,
    )
    wall = time.time() - t0

    w_gd = gd_res["w"]
    b_gd = gd_res["b"]

    # ── Closed-form reference ─────────────────────────────────────────────
    X_aug   = np.hstack([X_tr_exp, np.ones((X_tr_exp.shape[0], 1))])
    theta_cf, _, _, _ = np.linalg.lstsq(X_aug, y_train, rcond=None)
    w_cf  = theta_cf[:-1]
    b_cf  = float(theta_cf[-1])

    cf_train_loss = core.mse_loss(X_tr_exp, y_train, w_cf, b_cf)
    cf_val_r2     = core.r2_score_np(y_val, core.predict(X_vl_exp, w_cf, b_cf))

    gd_train_loss = gd_res["final_train_loss"]
    opt_gap       = abs(gd_train_loss - cf_train_loss) / (abs(cf_train_loss) + 1e-12)
    opt_limited   = opt_gap > 1e-3

    # ── Metrics ───────────────────────────────────────────────────────────
    y_tr_pred = core.predict(X_tr_exp, w_gd, b_gd)
    y_vl_pred = core.predict(X_vl_exp, w_gd, b_gd)
    tr_r2     = core.r2_score_np(y_train, y_tr_pred)
    vl_r2     = core.r2_score_np(y_val,   y_vl_pred)
    tr_rmse   = core.rmse_np(y_train, y_tr_pred)
    vl_rmse   = core.rmse_np(y_val,   y_vl_pred)

    return {
        "poly_params"          : poly_params,
        "X_train_exp"          : X_tr_exp,
        "X_val_exp"            : X_vl_exp,
        "expanded_names"       : exp_names,
        "w"                    : w_gd,
        "b"                    : b_gd,
        "n_iter"               : gd_res["n_iter"],
        "status"               : gd_res["status"],
        "train_r2"             : float(tr_r2),
        "val_r2"               : float(vl_r2),
        "train_rmse"           : float(tr_rmse),
        "val_rmse"             : float(vl_rmse),
        "final_train_loss"     : float(gd_train_loss),
        "final_val_loss"       : float(gd_res["final_val_loss"]),
        "kappa"                : float(kappa),
        "L_d"                  : float(L_d),
        "lr_used"              : float(lr),
        "gd_result"            : gd_res,
        "lstsq_train_loss"     : float(cf_train_loss),
        "lstsq_val_r2"         : float(cf_val_r2),
        "opt_gap_relative"     : float(opt_gap),
        "optimization_limited" : bool(opt_limited),
        "wall_time_s"          : float(wall),
        "n_new_cols"           : poly_params["n_new_cols"],
        "n_total_cols"         : int(p_exp),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Diagnostics helpers
# ──────────────────────────────────────────────────────────────────────────────

def partial_dependence(
    X_phase1        : np.ndarray,
    w               : np.ndarray,
    b               : float,
    feat_idx        : int,
    phase1_feat_names: list[str],
    poly_params     : dict,
    n_grid          : int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partial-dependence curve: vary feature feat_idx over its range in X_phase1,
    hold all others at their column mean, then apply transform_poly to get X_exp.

    Returns (grid, pd_values).
    """
    X_mean = X_phase1.mean(axis=0, keepdims=True).repeat(n_grid, axis=0)
    grid   = np.linspace(X_phase1[:, feat_idx].min(), X_phase1[:, feat_idx].max(), n_grid)
    X_mean[:, feat_idx] = grid
    X_exp, _ = transform_poly(X_mean, phase1_feat_names, poly_params)
    pd_vals = X_exp @ w + b
    return grid, pd_vals


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Artifact I/O for Phase 2
# ──────────────────────────────────────────────────────────────────────────────

def save_artifacts(
    path_stem        : str,
    w                : np.ndarray,
    b                : float,
    expanded_names   : list[str],
    poly_params      : dict,
    phase1_meta      : dict,
    seed             : int,
    chosen_degree    : int,
    interactions     : bool,
    source_cols      : list[str],
    lr               : float,
    tol              : float,
    max_iter         : int,
    n_iter           : int,
    status           : str,
    kappa            : float,
    train_r2         : float,
    val_r2           : float,
    train_rmse       : float,
    val_rmse         : float,
    final_train_loss : float,
    final_val_loss   : float,
    train_idx        : np.ndarray,
    val_idx          : np.ndarray,
    degree_table     : Optional[list] = None,
) -> str:
    """
    Save w, b, train_idx, val_idx → <path_stem>.npz
    Save metadata → <path_stem>.json
    Returns SHA-256 of w.
    """
    w_sha256 = hashlib.sha256(w.tobytes()).hexdigest()

    np.savez(
        path_stem + ".npz",
        w=w,
        b=np.array([b]),
        train_idx=train_idx,
        val_idx=val_idx,
    )

    # Serialise term_specs: exp_dict values are plain ints
    term_specs_serial = [
        [name, exp_dict]
        for name, exp_dict in poly_params["term_specs"]
    ]

    meta = {
        "phase1_w_sha256"   : phase1_meta["w_sha256"],
        "phase2_w_sha256"   : w_sha256,
        "seed"              : seed,
        "chosen_degree"     : chosen_degree,
        "interactions"      : interactions,
        "source_cols"       : source_cols,
        "phase1_feature_names": phase1_meta["feature_names"],
        "expanded_feature_names": expanded_names,
        "poly_means"        : poly_params["poly_means"],
        "poly_stds"         : poly_params["poly_stds"],
        "term_specs"        : term_specs_serial,
        "n_new_cols"        : poly_params["n_new_cols"],
        "phase1_meta"       : {k: v for k, v in phase1_meta.items()
                               if k not in ("feature_names",)},
        "lr"                : lr,
        "tol"               : tol,
        "max_iter"          : max_iter,
        "iterations_used"   : n_iter,
        "status"            : status,
        "kappa"             : kappa,
        "train_r2"          : train_r2,
        "val_r2"            : val_r2,
        "train_rmse"        : train_rmse,
        "val_rmse"          : val_rmse,
        "final_train_loss"  : final_train_loss,
        "final_val_loss"    : final_val_loss,
        "degree_table"      : degree_table,
    }

    with open(path_stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    return w_sha256


def load_artifacts(path_stem: str) -> tuple[np.ndarray, float, dict]:
    """Load Phase 2 artifacts. Returns w, b, meta_dict."""
    data = np.load(path_stem + ".npz", allow_pickle=False)
    w    = data["w"]
    b    = float(data["b"][0])
    with open(path_stem + ".json") as f:
        meta = json.load(f)
    return w, b, meta
