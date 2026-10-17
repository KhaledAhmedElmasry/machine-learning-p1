from __future__ import annotations
import numpy as np
import gradient_descent as core

def ridge_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lam: float) -> float:
    base = core.mse_loss(X, y, w, b)
    return base + (lam / 2.0) * float(np.sum(w ** 2))

def ridge_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lam: float) -> tuple[np.ndarray, float]:
    n = len(y)
    r = X @ w + b - y
    grad_w = (X.T @ r) / n + lam * w
    grad_b = float(np.sum(r) / n)
    return grad_w, grad_b

def enet_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lam: float, alpha: float) -> float:
    base = core.mse_loss(X, y, w, b)
    l1 = alpha * float(np.sum(np.abs(w)))
    l2 = 0.5 * (1 - alpha) * float(np.sum(w ** 2))
    return base + lam * (l1 + l2)

def proximal_gd_fit(
    X: np.ndarray, y: np.ndarray, lr: float, max_iter: int, lam: float, alpha: float,
    tol: float = 1e-6, w0: np.ndarray | None = None, b0: float = 0.0,
    X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
    record_every: int = 100, verbose: bool = False, momentum: bool = False
) -> dict:
    """Proximal gradient descent for Elastic Net (alpha=1 is Lasso, alpha=0 is Ridge but handled by soft thresholding).

    momentum=True gives the accelerated version (FISTA): the smooth gradient is taken at the look-ahead
    point z = w + k/(k+3) * (w - w_prev), and k restarts at 0 whenever the objective goes up.
    momentum=False is the plain proximal step, unchanged.
    """
    n, p = X.shape
    w = np.zeros(p, dtype=np.float64) if w0 is None else w0.copy().astype(np.float64)
    b = float(b0)
    
    train_losses = []
    val_losses = []
    
    initial_loss = enet_loss(X, y, w, b, lam, alpha)
    if initial_loss == 0.0: initial_loss = 1.0
    
    status = "not_converged"
    prev_loss = initial_loss
    w_prev, b_prev = w, b      # previous iterate (momentum only)
    k = 0                      # momentum counter

    for it in range(1, max_iter + 1):
        if momentum:
            beta = k / (k + 3.0)
            zw = w + beta * (w - w_prev)
            zb = b + beta * (b - b_prev)
        else:
            zw, zb = w, b

        # 1. Gradient of smooth part: MSE + L2
        r = X @ zw + zb - y
        grad_w = (X.T @ r) / n + lam * (1 - alpha) * zw
        grad_b = float(np.sum(r) / n)

        # 2. Gradient step
        z = zw - lr * grad_w
        w_prev, b_prev = w, b
        b = zb - lr * grad_b

        # 3. Proximal step (soft thresholding for L1 part)
        threshold = lr * lam * alpha
        w = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)

        current_loss = enet_loss(X, y, w, b, lam, alpha)
        if momentum:
            k = 0 if current_loss > prev_loss else k + 1     # restart when the objective goes up
        
        if it % record_every == 0 or it == max_iter:
            train_losses.append((it, float(current_loss)))
            if X_val is not None and y_val is not None:
                vl = enet_loss(X_val, y_val, w, b, lam, alpha)
                val_losses.append((it, float(vl)))
            
            if verbose and it % 1000 == 0:
                print(f"  iter {it:6d}  loss={current_loss:.6f}  status={status}")
                
        if not np.isfinite(current_loss) or current_loss > 1e6 * initial_loss:
            status = "diverged"
            break
            
        rel_change = abs(current_loss - prev_loss) / (abs(prev_loss) + 1e-12)
        if rel_change < tol:
            status = "converged"
            break
            
        # Optional: check if w changes minimally
        # We'll stick to loss change here
        prev_loss = current_loss
        
    return {
        "w": w,
        "b": b,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "n_iter": it,
        "status": status,
        "final_train_loss": float(current_loss),
        "final_val_loss": float(enet_loss(X_val, y_val, w, b, lam, alpha)) if X_val is not None else None,
    }

def get_original_col_mapping(expanded_names: list[str]) -> dict[str, list[str]]:
    """Map original column names to their derived features."""
    original_cols = ['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 
                     'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    mapping = {col: [] for col in original_cols}
    
    # Days since start is derived from dteday
    # is_windspeed_zero is derived from windspeed
    for name in expanded_names:
        # Check components split by *
        parts = name.split("*")
        for part in parts:
            base = part.split("^")[0]
            if base.startswith("hr_"): mapping["hr"].append(name)
            elif base.startswith("season_"): mapping["season"].append(name)
            elif base.startswith("mnth_"): mapping["mnth"].append(name)
            elif base.startswith("weekday_"): mapping["weekday"].append(name)
            elif base.startswith("weather_"): mapping["weathersit"].append(name)
            elif base == "days_since_start": mapping["dteday"].append(name)
            elif base == "is_windspeed_zero": mapping["windspeed"].append(name)
            elif base in mapping: mapping[base].append(name)
            
    # Deduplicate lists
    for col in mapping:
        mapping[col] = list(set(mapping[col]))
    return mapping

def safe_lr(L: float, lam: float, alpha: float) -> float:
    """Stable step size 0.9/(L + lam*(1-alpha)).

    The smooth part of the penalised loss has curvature L + lam*(1-alpha)
    (alpha = L1 share; 0 for Ridge, 1 for Lasso), so a step of 0.9/L alone
    diverges once lam*(1-alpha) is large.
    """
    return 0.9 / (L + lam * (1.0 - alpha))


def select_parsimonious(results: list[dict], tol: float = 0.005) -> dict:
    """Largest lambda whose val R2 is within `tol` of the best val R2.

    Diverged fits must not be passed in. Ties on lambda (Elastic Net across
    l1_ratios) are broken by the higher val R2.
    """
    best = max(r["val_r2"] for r in results)
    ok = [r for r in results if r["val_r2"] >= best - tol]
    return max(ok, key=lambda r: (r["lam"], r["val_r2"]))


def partner_table(df, cols_map: dict[str, str]) -> dict[str, tuple[str, float]]:
    """For each original column, its most correlated other original column.

    cols_map: original column name -> numeric Series name in `df`
    (dteday is represented by a numeric day counter).
    """
    import pandas as pd
    names = list(cols_map)
    mat = pd.DataFrame({k: df[v].astype(float) for k, v in cols_map.items()}).corr().abs()
    out = {}
    for k in names:
        others = mat[k].drop(k)
        out[k] = (str(others.idxmax()), float(others.max()))
    return out


def column_verdict(delta_r2: float, partner: str, partner_corr: float,
                   delta_thr: float = 0.001, corr_thr: float = 0.9) -> str:
    """Useful / Redundant / Uninformative from drop-family delta R2 and correlation."""
    if delta_r2 >= delta_thr:
        return "Useful"
    if partner_corr >= corr_thr:
        return f"Redundant (with {partner}, |r|={partner_corr:.2f})"
    return "Uninformative"


def fit_ridge(X, y, lr, max_iter, lam, tol=1e-6, w0=None, b0=0.0, X_val=None, y_val=None, momentum=False):
    return core.gd_fit(
        X, y, lr=lr, max_iter=max_iter, tol=tol,
        w0=w0, b0=b0,
        grad_fn=lambda X, y, w, b: ridge_grad(X, y, w, b, lam),
        loss_fn=lambda X, y, w, b: ridge_loss(X, y, w, b, lam),
        X_val=X_val, y_val=y_val, record_every=100, momentum=momentum
    )
