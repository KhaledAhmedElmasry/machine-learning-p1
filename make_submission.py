"""
make_submission.py
Fill sample_submission.csv for test.csv with the unregularised Phase 3 interaction model,
refit with our own gradient descent on ALL labelled rows and floored at 0.

Spec: SUBMISSION_SPEC.md.  Run:  python make_submission.py
Nothing is written until every check has passed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold   # fold assignment only; no sklearn model is fitted

import bias_variance as p3
import gradient_descent as core

SEED = 42648
IDS = ["16002932", "16006616", "16004208"]
DATA_DIR = Path("Datasets")
P3_STEM = "bias_variance_artifacts"
OUT_CSV = Path("sample_submission.csv")
OUT_JSON = Path("submission_fit.json")

FLOOR = 0.0
GD_TOL = 1e-9   # momentum (accelerated) GD stopped on a tight relative loss change
GD_MAX_ITER = 200_000
CV_MIN_R2 = 0.80
RUNTIME_BUDGET_S = 900   # 15 minutes: a warning, never a failure
LOSS_GAP_MAX = 2e-3
R2_GAP_MAX = 1e-3
HOUR_FLAG = 100.0
REFERENCE_CV_R2 = (0.8394, 0.8555)   # raw, floored: the PREVIOUS 127-feature Phase 3 model, printed for comparison only


# --------------------------------------------------------------------------- #
# Inputs and recipe
# --------------------------------------------------------------------------- #

def check_seed() -> None:
    assert core.compute_seed(IDS) == SEED, "team seed mismatch"
    print(f"[PASS] seed {SEED} recomputed from team IDs")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert train.shape[0] == 10886 and test.shape[0] == 574 and template.shape[0] == 574, \
        (train.shape, test.shape, template.shape)
    assert list(template["instant"]) == list(test["instant"]), "template instant order differs from test.csv"
    print(f"[PASS] inputs: train {train.shape}, test {test.shape}, template {template.shape}; instant order matches")
    return train, test, template


def load_spec() -> tuple[dict, list[str], str]:
    with open(P3_STEM + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["complexity_spec"], meta["expanded_feature_names"], meta["phase3_w_sha256"]


def fit_design(fit_df: pd.DataFrame, spec: dict) -> tuple[dict, list[str], dict]:
    """Phase 1 preprocessor and Phase 3 expansion, fitted on fit_df only."""
    pre = core.fit_preprocessor(fit_df)
    X1, names1 = core.transform(fit_df, pre)
    poly = p3.fit_poly(X1, names1, spec)
    return pre, names1, poly


def apply_design(df: pd.DataFrame, pre: dict, names1: list[str], poly: dict,
                 expected_names: list[str]) -> np.ndarray:
    """Transform df with already-fitted parameters; the column names must equal the Phase 3 artifact's."""
    X1, _ = core.transform(df, pre)
    Xe, names = p3.transform_poly(X1, names1, poly)
    assert names == expected_names, "design columns differ from the Phase 3 artifact"
    return Xe


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def fit_gd(X: np.ndarray, y: np.ndarray) -> dict:
    """Own gradient descent from zero weights at step 0.9/L; raises unless converged."""
    L = core.lipschitz_constant(X)
    res = core.gd_fit(X, y, lr=0.9 / L, max_iter=GD_MAX_ITER, tol=GD_TOL, record_every=500, momentum=True)
    if res["status"] != "converged":
        raise RuntimeError(f"GD status {res['status']} after {res['n_iter']} iterations")
    res["L"], res["lr"] = L, 0.9 / L
    return res


def fit_reference(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form least squares, only as a convergence reference."""
    A = np.hstack([X, np.ones((len(X), 1))])
    th = np.linalg.lstsq(A, y, rcond=None)[0]
    return th[:-1], float(th[-1])


def check_reference(X: np.ndarray, y: np.ndarray, gd: dict) -> dict:
    w_cf, b_cf = fit_reference(X, y)
    loss_gd = core.mse_loss(X, y, gd["w"], gd["b"])
    loss_cf = core.mse_loss(X, y, w_cf, b_cf)
    gap = abs(loss_gd - loss_cf) / (abs(loss_cf) + 1e-12)
    r2_gd = core.r2_score_np(y, core.predict(X, gd["w"], gd["b"]))
    r2_cf = core.r2_score_np(y, core.predict(X, w_cf, b_cf))
    assert gap < LOSS_GAP_MAX, f"GD vs lstsq training-loss gap {gap:.2e} >= {LOSS_GAP_MAX}"
    assert abs(r2_gd - r2_cf) < R2_GAP_MAX, f"train R2 differs: {r2_gd:.6f} vs {r2_cf:.6f}"
    print(f"[PASS] GD vs lstsq: relative loss gap {gap:.2e}, train R2 {r2_gd:.6f} vs {r2_cf:.6f}")
    return {"loss_gap": float(gap), "r2_gd": float(r2_gd), "r2_lstsq": float(r2_cf)}


def cv_day_grouped(train: pd.DataFrame, spec: dict, names: list[str]) -> dict:
    """5-fold GroupKFold by day; every fold refits preprocessing, expansion and GD."""
    groups = train["dteday"].values
    rows = []
    for k, (a, v) in enumerate(GroupKFold(n_splits=5).split(train, groups=groups), 1):
        fa, fv = train.iloc[a], train.iloc[v]
        pre, n1, poly = fit_design(fa, spec)
        Xa = apply_design(fa, pre, n1, poly, names)
        Xv = apply_design(fv, pre, n1, poly, names)
        ya, yv = fa["cnt"].values.astype(float), fv["cnt"].values.astype(float)
        res = fit_gd(Xa, ya)
        p = core.predict(Xv, res["w"], res["b"])
        pc = np.clip(p, FLOOR, None)
        row = dict(fold=k, n_val=len(v), iters=res["n_iter"],
                   r2=core.r2_score_np(yv, p), rmse=core.rmse_np(yv, p),
                   r2_floor=core.r2_score_np(yv, pc), rmse_floor=core.rmse_np(yv, pc))
        rows.append(row)
        print(f"  fold {k}: val n={len(v)}  iters {res['n_iter']}  R2 {row['r2']:.4f} / floored {row['r2_floor']:.4f}  "
              f"RMSE {row['rmse']:.2f} / floored {row['rmse_floor']:.2f}")
    df = pd.DataFrame(rows)
    out = {"folds": rows,
           "mean_r2": float(df["r2"].mean()), "std_r2": float(df["r2"].std(ddof=0)),
           "mean_rmse": float(df["rmse"].mean()),
           "mean_r2_floor": float(df["r2_floor"].mean()), "std_r2_floor": float(df["r2_floor"].std(ddof=0)),
           "mean_rmse_floor": float(df["rmse_floor"].mean())}
    print(f"[CV] day-grouped 5-fold: raw R2 {out['mean_r2']:.4f} +-{out['std_r2']:.4f} (RMSE {out['mean_rmse']:.2f}); "
          f"floored R2 {out['mean_r2_floor']:.4f} +-{out['std_r2_floor']:.4f} (RMSE {out['mean_rmse_floor']:.2f})")
    print(f"     previous 127-feature model, for comparison: raw {REFERENCE_CV_R2[0]}, floored {REFERENCE_CV_R2[1]}")
    assert out["mean_r2_floor"] >= CV_MIN_R2, f"mean floored CV R2 {out['mean_r2_floor']:.4f} < {CV_MIN_R2}"
    return out


# --------------------------------------------------------------------------- #
# Leakage test, prediction, report, outputs
# --------------------------------------------------------------------------- #

def leakage_test(train: pd.DataFrame, test: pd.DataFrame, pre: dict) -> None:
    before = copy.deepcopy(pre)
    core.transform(test, pre)                      # transform must not refit anything
    assert pre == before, "transform changed the fitted preprocessor"
    pre_all = core.fit_preprocessor(pd.concat([train, test], ignore_index=True))
    differs = any(abs(pre["scaler_means"][c] - pre_all["scaler_means"][c]) > 1e-9 for c in pre["numeric_cols"])
    assert differs, "fitting on train + test gave identical parameters: the test is not detecting leakage"
    print("[PASS] no test leakage: transform leaves the fitted parameters unchanged and they differ from train + test parameters")


def predict_floor(X_test: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, dict]:
    raw = core.predict(X_test, w, b)
    assert raw.shape[0] == 574 and np.isfinite(raw).all(), "test predictions are not 574 finite values"
    pred = np.clip(raw, FLOOR, None)
    n_clip = int((raw < FLOOR).sum())
    assert (pred >= 0).all()
    info = {"n_clipped": n_clip, "share_clipped": n_clip / len(raw), "raw_min": float(raw.min())}
    print(f"[PASS] 574 finite predictions; clipped at {FLOOR}: {n_clip} values ({100*info['share_clipped']:.1f}%), raw min {raw.min():.2f}")
    return pred, info


def plausibility_report(train: pd.DataFrame, test: pd.DataFrame, pred: np.ndarray) -> dict:
    print(f"\nprediction min {pred.min():.2f}  mean {pred.mean():.2f}  max {pred.max():.2f}   |   train cnt mean {train['cnt'].mean():.2f}")
    hourly_train = train.groupby("hr")["cnt"].mean()
    hourly_pred = pd.Series(pred, index=test.index).groupby(test["hr"].values).mean()
    comp = pd.DataFrame({"train_mean": hourly_train, "test_pred_mean": hourly_pred})
    comp["diff"] = comp["test_pred_mean"] - comp["train_mean"]
    print(comp.round(1).to_string())
    worst = float(comp["diff"].abs().max())
    flag = " (FLAG: above %.0f; test days are only the 20th of each month, so a gap is not an error by itself)" % HOUR_FLAG if worst > HOUR_FLAG else ""
    print(f"max |hourly mean difference| = {worst:.1f}{flag}")
    return {"max_hourly_abs_diff": worst, "flagged": bool(worst > HOUR_FLAG)}


def write_submission(test: pd.DataFrame, pred: np.ndarray) -> None:
    sub = pd.DataFrame({"instant": test["instant"].values, "cnt": pred})
    sub.to_csv(OUT_CSV, index=False, float_format="%.6f")
    chk = pd.read_csv(OUT_CSV)
    assert list(chk.columns) == ["instant", "cnt"] and len(chk) == 574
    assert list(chk["instant"]) == list(test["instant"]) and chk["cnt"].notna().all() and (chk["cnt"] >= 0).all()
    print(f"[SAVED] {OUT_CSV}  ({len(chk)} rows)")


def write_audit(meta: dict) -> None:
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVED] {OUT_JSON}")


# --------------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()
    check_seed()
    train, test, _ = load_inputs()
    spec, names, p3_sha = load_spec()
    y = train["cnt"].values.astype(float)

    print("\nDay-grouped cross-validation of the exact recipe (refits everything per fold):")
    cv = cv_day_grouped(train, spec, names)

    print("\nFinal fit on all labelled rows:")
    pre, n1, poly = fit_design(train, spec)
    X = apply_design(train, pre, n1, poly, names)
    X_test = apply_design(test, pre, n1, poly, names)
    assert X.shape == (10886, len(names)) and X_test.shape == (574, len(names)), (X.shape, X_test.shape)
    gd = fit_gd(X, y)
    p_tr = core.predict(X, gd["w"], gd["b"])
    train_r2, train_rmse = core.r2_score_np(y, p_tr), core.rmse_np(y, p_tr)
    print(f"GD: {gd['n_iter']} iterations, status {gd['status']}, step {gd['lr']:.5f} (0.9/L, L={gd['L']:.4f}); "
          f"train R2 {train_r2:.4f}, RMSE {train_rmse:.2f}")
    ref = check_reference(X, y, gd)
    leakage_test(train, test, pre)

    gd2 = fit_gd(X, y)
    assert np.array_equal(gd["w"], gd2["w"]) and gd["b"] == gd2["b"], "GD is not deterministic"
    w_sha = hashlib.sha256(gd["w"].tobytes()).hexdigest()
    print(f"[PASS] second fit gives bit-identical weights; SHA-256(w) = {w_sha}")

    pred, clip_info = predict_floor(X_test, gd["w"], gd["b"])
    rep = plausibility_report(train, test, pred)

    runtime = time.time() - t0
    meta = {
        "seed": SEED, "ids": sorted(IDS), "rows_used_for_fit": int(len(train)), "n_features": int(X.shape[1]),
        "model": "unregularised Phase 3 interaction model (own GD), refit on all labelled rows",
        "complexity_spec": spec, "floor": FLOOR,
        "gd": {"momentum": True, "lr": gd["lr"], "lipschitz": gd["L"], "tol": GD_TOL, "max_iter": GD_MAX_ITER,
               "iterations": gd["n_iter"], "status": gd["status"]},
        "train_r2": float(train_r2), "train_rmse": float(train_rmse), "reference_check": ref,
        "cv_day_grouped": cv, "clipping": clip_info, "plausibility": rep,
        "weights_sha256": w_sha, "phase3_artifact_w_sha256_for_reference": p3_sha,
        "prediction_summary": {"min": float(pred.min()), "mean": float(pred.mean()), "max": float(pred.max())},
        "runtime_seconds": round(runtime, 1),
    }
    write_audit(meta)
    write_submission(test, pred)
    print(f"\ntotal runtime {time.time() - t0:.0f}s")
    elapsed = time.time() - t0
    if elapsed > RUNTIME_BUDGET_S:
        print(f"WARNING: runtime {elapsed:.0f}s exceeded the {RUNTIME_BUDGET_S // 60}-minute budget (the output above is still valid)")


if __name__ == "__main__":
    main()
