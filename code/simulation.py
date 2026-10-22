#!/usr/bin/env python3
"""Monte Carlo simulation of Meza & Singh (2025) CCA / PCA / Whiten IV.

Uses stored text (covariates) and image (instruments) embeddings.

Pipeline
--------
1. Load embeddings (one row per product: period == 0).
2. Post-process in two ways (separate simulation designs):
     (i)  demean columns          (as in process_embeddings.py)
     (ii) demean then / column std
3. PCA-truncate: replace text by its rank-k reconstruction and images by
   their rank-l reconstruction in the original embedding dimension.
   These low-rank matrices are the clean (X, W). Images overidentify when l > k.
4. For j = 1..J (default 1000):
     - Infuse correlated Gaussian noise (Meza & Singh Appendix B) -> (Z_X, Z_W)
     - Build endogenous Y = X beta* + eps with eps orthogonal to span(W)
     - Save artifacts under data/simulation_run/<design>/run{j}/
     - Split 50/50; fit Oracle, PCA-2SLS, Whiten-2SLS, CCA-2SLS on train
     - OOS MSE on test: (1/n_test) sum (Y - X beta_hat)^2

Noise process follows Appendix B of Meza & Singh (2025), "Canonical correlation
regression with noisy data" (the synthetic design in Isaac's CCA-IV experiments).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from paths import PROJECT_DIR
except ImportError:
    PROJECT_DIR = Path(__file__).resolve().parent.parent


K_DEFAULT = 10
L_DEFAULT = 15
N_REPS_DEFAULT = 1000
ALPHA = 1.5
RHO = 0.9
C1 = 2.0
SIGMA_EPS = 1.25
SEED = 42


# ---------------------------------------------------------------------------
# Data loading / post-processing / PCA truncation
# ---------------------------------------------------------------------------
def _embedding_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if re.fullmatch(rf"{prefix}_\d+", c)]


def load_clean_embeddings(project_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Full text/image matrices, one row per product (period 0)."""
    text = pd.read_parquet(project_dir / "data/embeddings/text_embeddings.parquet")
    image = pd.read_parquet(project_dir / "data/embeddings/image_embeddings.parquet")

    mask = text["period"].to_numpy() == 0
    text_cols = _embedding_cols(text, "text")
    image_cols = _embedding_cols(image, "image")

    X_full = text.loc[mask, text_cols].to_numpy(dtype=np.float64)
    W_full = image.loc[mask, image_cols].to_numpy(dtype=np.float64)
    if X_full.shape[0] != W_full.shape[0]:
        raise ValueError(
            f"row mismatch after period filter: text {X_full.shape}, image {W_full.shape}"
        )
    return X_full, W_full


def post_process(M: np.ndarray, kind: str) -> np.ndarray:
    """Column-wise demean / demean_scale (same as process_embeddings.py)."""
    if kind == "demean":
        return M - M.mean(axis=0, keepdims=True)
    if kind == "demean_scale":
        mu = M.mean(axis=0, keepdims=True)
        sd = M.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (M - mu) / sd
    raise ValueError(f"unknown post-processing: {kind}")


def pca_reconstruct(M: np.ndarray, rank: int) -> np.ndarray:
    """Rank-r SVD reconstruction in the original column space (n x p)."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, U.shape[1])
    return (U[:, :r] * S[:r]) @ Vt[:r, :]


def make_beta_star(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """beta* in the row-space of X (unit direction along top right singular vector)."""
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    beta = Vt[0, :].copy()
    beta /= np.linalg.norm(beta)
    # small jitter so it is not exactly a single PC (still in row space)
    noise = rng.normal(size=beta.shape)
    noise = noise - Vt.T @ (Vt @ noise)  # orthogonal to row space -> 0 if full rank row space
    # keep pure unit top-right-singular direction
    return beta


# ---------------------------------------------------------------------------
# Noise infusion (Meza & Singh Appendix B)
# ---------------------------------------------------------------------------
def add_correlated_noise(
    X: np.ndarray,
    W: np.ndarray,
    k: int,
    ell: int,
    rng: np.random.Generator,
    alpha: float = ALPHA,
    rho: float = RHO,
    c1: float = C1,
) -> tuple[np.ndarray, np.ndarray]:
    """Z_X = X + H_X, Z_W = W + H_W with shared Gaussian noise on common columns."""
    n, p = X.shape
    _, p_w = W.shape

    sigma_hx = c1 / ((k + 1) ** alpha * np.sqrt(p))
    sigma_hw = c1 / ((ell + 1) ** alpha * np.sqrt(p_w))

    H_X = rng.normal(0.0, sigma_hx, size=(n, p))
    H_W = rng.normal(0.0, sigma_hw, size=(n, p_w))

    p_common = min(p, p_w)
    if p_common > 0 and abs(rho) > 0:
        z1 = rng.normal(size=(n, p_common))
        z2 = rng.normal(size=(n, p_common))
        H_X[:, :p_common] = sigma_hx * z1
        H_W[:, :p_common] = sigma_hw * (
            rho * z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * z2
        )

    return X + H_X, W + H_W


def simulate_outcome(
    X: np.ndarray,
    W: np.ndarray,
    beta_star: np.ndarray,
    ell: int,
    rng: np.random.Generator,
    sigma_eps: float = SIGMA_EPS,
) -> np.ndarray:
    """Y = X beta* + eps; eps correlated with X but orthogonal to span(W)."""
    n, p = X.shape
    # Endogeneity direction inside the row space of X.
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(p, Vt.shape[0])
    coef = np.linspace(1.0, 0.2, r)
    gamma = Vt[:r, :].T @ (coef / np.linalg.norm(coef))

    eta = rng.normal(0.0, 1.0, size=n)
    eps0 = X @ gamma + eta

    Uw, _, _ = np.linalg.svd(W, full_matrices=False)
    Uw = Uw[:, : min(ell, Uw.shape[1])]
    eps = eps0 - Uw @ (Uw.T @ eps0)

    scale = np.sqrt(np.mean(eps**2))
    if scale > 0:
        eps = eps * (sigma_eps / scale)
    return X @ beta_star + eps


# ---------------------------------------------------------------------------
# Estimators (Meza & Singh Examples 1–4)
# ---------------------------------------------------------------------------
def _svd_truncate(Z: np.ndarray, rank: int):
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    r = min(rank, U.shape[1])
    return U[:, :r], S[:r], Vt[:r, :]


def _beta_from_P(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(P, Y, rcond=None)
    return beta


def _to_x_beta(P: np.ndarray, beta_p: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Convert a second-stage coef on P into a structural coef on X.

    We want beta_X such that X beta_X ≈ P beta_P (the IV fitted values), so
    predictions of the form X_test @ beta_X are on the structural scale.
    """
    fitted = P @ beta_p
    beta_x, *_ = np.linalg.lstsq(X, fitted, rcond=None)
    return beta_x


def fit_oracle_2sls(X: np.ndarray, W: np.ndarray, Y: np.ndarray, ell: int) -> np.ndarray:
    # P = proj_W X lives in the same coordinate system as X, so P^+ Y is structural.
    Uw, _, _ = _svd_truncate(W, ell)
    P = Uw @ (Uw.T @ X)
    return _beta_from_P(P, Y)


def fit_pca_2sls(Z_X: np.ndarray, Z_W: np.ndarray, Y: np.ndarray, k: int, ell: int) -> np.ndarray:
    U, S, Vt = _svd_truncate(Z_X, k)
    Uw, _, _ = _svd_truncate(Z_W, ell)
    Xhat = (U * S) @ Vt
    P = Uw @ (Uw.T @ Xhat)
    return _beta_from_P(P, Y)


def fit_whiten_2sls(
    Z_X: np.ndarray, Z_W: np.ndarray, Y: np.ndarray, k: int, ell: int
) -> np.ndarray:
    U, S, Vt = _svd_truncate(Z_X, k)
    Uw, _, _ = _svd_truncate(Z_W, ell)
    X_under = U @ Vt  # U_k V_k^\top
    P = Uw @ (Uw.T @ X_under)
    beta_p = _beta_from_P(P, Y)
    # If P ≈ U V' and X ≈ U diag(S) V', then beta_P ≈ V S V' beta_X.
    return Vt.T @ ((Vt @ beta_p) / np.maximum(S, 1e-12))


def fit_cca_2sls(Z_X: np.ndarray, Z_W: np.ndarray, Y: np.ndarray, k: int, ell: int) -> np.ndarray:
    U, S, Vt = _svd_truncate(Z_X, k)
    Uw, Sw, _ = _svd_truncate(Z_W, ell)
    Sw_safe = np.maximum(Sw, Sw.max() * 1e-8)
    Xhat = (U * S) @ Vt
    cross = Uw.T @ U
    P = (Uw * (1.0 / Sw_safe)) @ cross @ Vt
    # Map IV fitted values P beta_P back into the column space of Xhat so that
    # predictions of the form X @ beta remain well-defined / comparable.
    return _to_x_beta(P, _beta_from_P(P, Y), Xhat)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _to_parquet(path: Path, M: np.ndarray, prefix: str) -> None:
    cols = [f"{prefix}_{i:04d}" for i in range(M.shape[1])]
    pd.DataFrame(M, columns=cols).to_parquet(path, index=False)


def save_run_artifacts(
    run_dir: Path,
    X: np.ndarray,
    W: np.ndarray,
    Z_X: np.ndarray,
    Z_W: np.ndarray,
    Y: np.ndarray,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _to_parquet(run_dir / "X_text.parquet", X, "text")
    _to_parquet(run_dir / "W_image.parquet", W, "image")
    _to_parquet(run_dir / "Z_text.parquet", Z_X, "text")
    _to_parquet(run_dir / "Z_image.parquet", Z_W, "image")
    pd.DataFrame({"Y": Y}).to_parquet(run_dir / "Y.parquet", index=False)


# ---------------------------------------------------------------------------
# One Monte Carlo replication
# ---------------------------------------------------------------------------
def one_replication(
    X: np.ndarray,
    W: np.ndarray,
    beta_star: np.ndarray,
    k: int,
    ell: int,
    rng: np.random.Generator,
    run_dir: Path | None,
) -> dict[str, float]:
    Z_X, Z_W = add_correlated_noise(X, W, k, ell, rng)
    Y = simulate_outcome(X, W, beta_star, ell, rng)

    if run_dir is not None:
        save_run_artifacts(run_dir, X, W, Z_X, Z_W, Y)

    n = X.shape[0]
    idx = rng.permutation(n)
    half = n // 2
    train, test = idx[:half], idx[half:]

    X_tr, X_te = X[train], X[test]
    W_tr = W[train]
    Zx_tr, Zw_tr = Z_X[train], Z_W[train]
    Y_tr, Y_te = Y[train], Y[test]

    betas = {
        "oracle": fit_oracle_2sls(X_tr, W_tr, Y_tr, ell),
        "pca": fit_pca_2sls(Zx_tr, Zw_tr, Y_tr, k, ell),
        "whiten": fit_whiten_2sls(Zx_tr, Zw_tr, Y_tr, k, ell),
        "cca": fit_cca_2sls(Zx_tr, Zw_tr, Y_tr, k, ell),
    }

    out: dict[str, float] = {}
    for name, beta in betas.items():
        y_hat = X_te @ beta
        out[f"mse_{name}"] = float(np.mean((Y_te - y_hat) ** 2))
        out[f"beta_err_{name}"] = float(np.mean((beta - beta_star) ** 2))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_design(
    X_full: np.ndarray,
    W_full: np.ndarray,
    design: str,
    k: int,
    ell: int,
    n_reps: int,
    seed: int,
    out_root: Path,
    save_runs: bool,
) -> pd.DataFrame:
    Xp = post_process(X_full, design)
    Wp = post_process(W_full, design)
    X = pca_reconstruct(Xp, k)
    W = pca_reconstruct(Wp, ell)

    # Fixed beta* across replications (depends on clean X only).
    beta_star = make_beta_star(X, np.random.default_rng(seed))

    design_dir = out_root / design
    design_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"beta_star": beta_star}).to_csv(design_dir / "beta_star.csv", index=False)

    rows = []
    for j in range(1, n_reps + 1):
        rng = np.random.default_rng(seed + j)
        run_dir = design_dir / f"run{j}" if save_runs else None
        stats = one_replication(X, W, beta_star, k, ell, rng, run_dir)
        stats.update({"rep": j, "design": design, "k": k, "l": ell})
        rows.append(stats)
        if j % 50 == 0 or j == n_reps:
            print(f"[{design}] finished replication {j}/{n_reps}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(design_dir / "mse_by_replication.csv", index=False)

    metric_cols = [
        "mse_oracle",
        "mse_pca",
        "mse_whiten",
        "mse_cca",
        "beta_err_oracle",
        "beta_err_pca",
        "beta_err_whiten",
        "beta_err_cca",
    ]
    summary = (
        results[metric_cols]
        .agg(["mean", "std", "median"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
    summary.insert(0, "design", design)
    summary.to_csv(design_dir / "mse_summary.csv", index=False)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=K_DEFAULT, help="Text PCA rank / IV truncation")
    p.add_argument("--l", type=int, default=L_DEFAULT, help="Image PCA rank / IV truncation")
    p.add_argument("--n-reps", type=int, default=N_REPS_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--designs",
        nargs="+",
        default=["demean", "demean_scale"],
        choices=["demean", "demean_scale"],
    )
    p.add_argument(
        "--no-save-runs",
        action="store_true",
        help="Only write summary CSVs (skip per-rep parquet folders).",
    )
    p.add_argument("--smoke", action="store_true", help="2 replications for a quick test.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_reps = 2

    project_dir = args.project_dir.resolve()
    out_root = (args.out_dir or (project_dir / "data" / "simulation_run")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"project_dir: {project_dir}")
    print(f"out_dir:     {out_root}")
    print(f"k={args.k}, l={args.l}, n_reps={args.n_reps}, designs={args.designs}")

    X_full, W_full = load_clean_embeddings(project_dir)
    print(f"loaded embeddings: text {X_full.shape}, image {W_full.shape}")

    all_results = []
    for design in args.designs:
        res = run_design(
            X_full=X_full,
            W_full=W_full,
            design=design,
            k=args.k,
            ell=args.l,
            n_reps=args.n_reps,
            seed=args.seed,
            out_root=out_root,
            save_runs=not args.no_save_runs,
        )
        all_results.append(res)
        means = res[["mse_oracle", "mse_pca", "mse_whiten", "mse_cca"]].mean()
        print(f"\n=== {design}: mean OOS MSE over {args.n_reps} reps ===")
        print(means.to_string())

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(out_root / "mse_by_replication_all.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "design": design,
                **res[["mse_oracle", "mse_pca", "mse_whiten", "mse_cca"]].mean().to_dict(),
            }
            for design, res in zip(args.designs, all_results)
        ]
    )
    summary.to_csv(out_root / "mse_summary_all.csv", index=False)
    print("\n=== overall mean MSE ===")
    print(summary.to_string(index=False))
    print(f"\nwrote summaries under {out_root}")


if __name__ == "__main__":
    main()
