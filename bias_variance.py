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
import polynomial_regression as p2


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
    for a_col, b_col in interactions:
        cols1 = [c for c in phase1_feat_names if c == a_col or c.startswith(a_col + "_")]
        cols2 = [c for c in phase1_feat_names if c == b_col or c.startswith(b_col + "_")]
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


# Reuse _compute_term from polynomial_regression module
_compute_term = p2._compute_term


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
        raise ValueError(f"w_prev length ({len(w_prev)}) does not match n_old_cols ({n_old_cols})")
    w_new = np.zeros(n_old_cols + n_new_cols, dtype=np.float64)
    w_new[:n_old_cols] = w_prev
    return w_new


def warm_start_by_name(
    w_prev: np.ndarray,
    names_prev: list[str],
    names_new: list[str],
) -> np.ndarray:
    """Copy each previous weight to the column with the SAME NAME in the new design; new columns start at 0.

    Unlike pad_weights_for_warm_start (which copies by position), this stays correct when the new
    design orders or selects its columns differently from the previous one.
    """
    if len(w_prev) != len(names_prev):
        raise ValueError(f"w_prev has {len(w_prev)} entries but names_prev has {len(names_prev)}")
    index = {n: i for i, n in enumerate(names_prev)}
    w_new = np.zeros(len(names_new), dtype=np.float64)
    for j, n in enumerate(names_new):
        if n in index:
            w_new[j] = w_prev[index[n]]
    return w_new


def select_parsimonious_complexity(rows: list[dict], tol: float = 0.005, score_key: str = "cv_r2",
                                   size_key: str = "p") -> dict:
    """Least complex row whose score is within `tol` of the best score."""
    best = max(r[score_key] for r in rows)
    ok = [r for r in rows if r[score_key] >= best - tol]
    return min(ok, key=lambda r: r[size_key])


def fit_expanded(
    X_tr_exp: np.ndarray,
    y_tr: np.ndarray,
    X_vl_exp: Optional[np.ndarray],
    y_vl: Optional[np.ndarray],
    w_init: np.ndarray,
    b_init: float,
    lr: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 200_000,
    momentum: bool = False,
) -> tuple[np.ndarray, float, float, dict]:
    """Runs GD on the expanded dataset. Uses adaptive lr = 0.9/L_d if lr is None.

    momentum=True uses the accelerated update of gradient_descent.gd_fit (Nesterov with restart).
    """
    L_d = core.lipschitz_constant(X_tr_exp)
    lr_used = (0.9 / L_d) if lr is None else lr
    kappa = p2.compute_condition_number(X_tr_exp)

    gd_res = core.gd_fit(
        X_tr_exp, y_tr,
        lr=lr_used, max_iter=max_iter, tol=tol,
        w0=w_init, b0=b_init,
        X_val=X_vl_exp, y_val=y_vl,
        record_every=100, momentum=momentum,
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
    extra: Optional[dict] = None,
) -> str:
    """Save w, b, indices -> .npz and metadata -> .json. Returns SHA-256 of w.

    `extra` (JSON-serialisable) is merged into the metadata, e.g. the complexity ladder and diagnosis.
    """
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
    if extra:
        meta.update(extra)

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
