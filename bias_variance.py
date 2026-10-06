"""
bias_variance.py
==============
Phase 3 Infrastructure (Bias-Variance tradeoff).
Reuses gradient_descent.py functions unchanged.
NO sklearn .fit() / .predict() used anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np

import gradient_descent as core


def get_poly_term_specs(
    phase1_feat_names: list[str],
    complexity_spec: dict,
) -> list[tuple[str, dict]]:
    """
    Returns list of (term_name, exponent_dict) for NEW columns based on complexity_spec.

    complexity_spec = {
        "degrees": {"temp": 3, "hum": 2},
        "interactions": [("hr", "workingday"), ("hr", "weather")]
    }
    """
    terms = []
    degrees = complexity_spec.get("degrees", {})
    interactions = complexity_spec.get("interactions", [])

    # Deterministic order via sorted keys
    for col in sorted(degrees.keys()):
        d = degrees[col]
        if col in phase1_feat_names:
            for power in range(2, d + 1):
                name = f"{col}^{power}"
                terms.append((name, {col: power}))

    # Interactions via prefix matching
    for p1, p2 in interactions:
        cols1 = [c for c in phase1_feat_names if c == p1 or c.startswith(p1 + "_")]
        cols2 = [c for c in phase1_feat_names if c == p2 or c.startswith(p2 + "_")]
        for c1 in cols1:
            for c2 in cols2:
                if c1 == c2:
                    continue
                s1, s2 = sorted([c1, c2])
                name = f"{s1}*{s2}"
                exp_dict = {s1: 1, s2: 1}
                if (name, exp_dict) not in terms:
                    terms.append((name, exp_dict))

    return terms


def fit_poly(
    X_phase1_train: np.ndarray,
    phase1_feat_names: list[str],
    complexity_spec: dict,
) -> dict:
    """Fit polynomial expansion parameters on training data only."""
    term_specs = get_poly_term_specs(phase1_feat_names, complexity_spec)

    poly_means = []
    poly_stds = []

    for name, exp_dict in term_specs:
        raw_col = _compute_term(X_phase1_train, phase1_feat_names, exp_dict)
        mu = float(raw_col.mean())
        sig = float(raw_col.std(ddof=0))
        if sig == 0.0:
            sig = 1.0
        poly_means.append(mu)
        poly_stds.append(sig)

    return {
        "term_specs": term_specs,
        "complexity_spec": complexity_spec,
        "poly_means": poly_means,
        "poly_stds": poly_stds,
        "n_new_cols": len(term_specs),
    }


def _compute_term(X: np.ndarray, feat_names: list[str], exp_dict: dict[str, int]) -> np.ndarray:
    """Compute one monomial column from X (product of powers)."""
    n = X.shape[0]
    out = np.ones(n, dtype=np.float64)
    for col_name, power in exp_dict.items():
        idx = feat_names.index(col_name)
        if power == 1:
            out *= X[:, idx]
        else:
            out *= X[:, idx] ** power
    return out


def transform_poly(
    X_phase1: np.ndarray,
    phase1_feat_names: list[str],
    params: dict,
) -> tuple[np.ndarray, list[str]]:
    """Apply polynomial expansion using fitted params."""
    term_specs = params["term_specs"]
    poly_means = params["poly_means"]
    poly_stds = params["poly_stds"]
    n_new = params["n_new_cols"]

    n = X_phase1.shape[0]
    if n_new == 0:
        return X_phase1.copy(), phase1_feat_names.copy()

    X_new_raw = np.empty((n, n_new), dtype=np.float64)
    expanded_names = list(phase1_feat_names)

    for i, ((name, exp_dict), mu, sig) in enumerate(zip(term_specs, poly_means, poly_stds)):
        raw_col = _compute_term(X_phase1, phase1_feat_names, exp_dict)
        X_new_raw[:, i] = (raw_col - mu) / sig
        expanded_names.append(name)

    X_expanded = np.hstack([X_phase1, X_new_raw])
    return X_expanded, expanded_names


def pad_weights_for_warm_start(
    w_prev: np.ndarray,
    n_old_cols: int,
    n_new_cols: int,
) -> np.ndarray:
    """Pad the previous weight vector with zeros for newly expanded columns."""
    if len(w_prev) != n_old_cols:
        raise ValueError("w_prev length does not match n_old_cols")
    w_new = np.zeros(n_old_cols + n_new_cols, dtype=np.float64)
    w_new[:n_old_cols] = w_prev
    return w_new


def fit_expanded(
    X_tr_exp: np.ndarray,
    y_tr: np.ndarray,
    X_vl_exp: Optional[np.ndarray],
    y_vl: Optional[np.ndarray],
    w_init: np.ndarray,
    b_init: float,
    lr: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, float, float, dict]:
    """Runs GD on the expanded dataset. Returns (w, b, kappa, gd_result)."""
    XtX = X_tr_exp.T @ X_tr_exp
    eigvals = np.linalg.eigvalsh(XtX)
    pos_eig = eigvals[eigvals > 1e-10]
    kappa = float(pos_eig.max() / pos_eig.min()) if len(pos_eig) > 0 else float("inf")

    gd_res = core.gd_fit(
        X_tr_exp, y_tr,
        lr=lr, max_iter=max_iter, tol=tol,
        w0=w_init, b0=b_init,
        X_val=X_vl_exp, y_val=y_vl,
        record_every=100,
    )
    w_opt = gd_res["w"]
    b_opt = gd_res["b"]

    return w_opt, b_opt, kappa, gd_res


def save_artifacts(
    path_stem: str,
    w: np.ndarray,
    b: float,
    expanded_names: list[str],
    poly_params: dict,
    phase2_meta: dict,
    complexity_spec: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> str:
    """Save w, b, indices -> .npz and metadata -> .json. Returns SHA-256 of w."""
    w_sha256 = hashlib.sha256(w.tobytes()).hexdigest()

    np.savez(
        path_stem + ".npz",
        w=w, b=np.array([b]),
        train_idx=train_idx, val_idx=val_idx,
    )

    term_specs_serial = [[name, exp_dict] for name, exp_dict in poly_params["term_specs"]]

    meta = {
        "phase2_w_sha256": phase2_meta.get("phase2_w_sha256", "UNKNOWN"),
        "phase3_w_sha256": w_sha256,
        "expanded_feature_names": expanded_names,
        "complexity_spec": complexity_spec,
        "poly_means": poly_params["poly_means"],
        "poly_stds": poly_params["poly_stds"],
        "term_specs": term_specs_serial,
        "n_new_cols": poly_params["n_new_cols"],
    }

    with open(path_stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    return w_sha256


def load_artifacts(path_stem: str) -> tuple[np.ndarray, float, dict]:
    """Load Phase 3 artifacts. Returns w, b, meta_dict."""
    data = np.load(path_stem + ".npz", allow_pickle=False)
    w = data["w"]
    b = float(data["b"][0])
    with open(path_stem + ".json") as f:
        meta = json.load(f)
    return w, b, meta
