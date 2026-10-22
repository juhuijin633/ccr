#!/usr/bin/env python3
"""Monte Carlo simulation of Meza & Singh (2025) CCA / PCA / Whiten IV.

Uses stored text (covariates X) and image (instruments W) embeddings.

Pipeline
--------
1. Load embeddings (one row per product: period == 0).
2. Post-process: demean OR demean_scale (as in process_embeddings.py).
3. PCA-truncate to scores: text -> n x k, image -> n x l  (default k=10, l=15).
4. For j = 1..J:
     - Add Appendix-B correlated Gaussian noise -> (Z_X, Z_W)
     - Build endogenous Y = X beta* + eps (eps ⊥ span(W))
     - Optionally save under data/simulation_run/<design>/run{j}/
     - 50/50 split; fit Oracle / PCA / Whiten / CCA 2SLS on train
     - OOS MSE: mean((Y - X @ beta_hat)^2) on test

Noise follows Meza & Singh (2025) Appendix B.
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


def _embedding_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if re.fullmatch(rf"{prefix}_\d+", c)]


def load_clean_embeddings(project_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    text = pd.read_parquet(project_dir / "data/embeddings/text_embeddings.parquet")
    image = pd.read_parquet(project_dir / "data/embeddings/image_embeddings.parquet")
    mask = text["period"].to_numpy() == 0
    X = text.loc[mask, _embedding_cols(text, "text")].to_numpy(np.float64)
    W = image.loc[mask, _embedding_cols(image, "image")].to_numpy(np.float64)
    return X, W


def post_process(M: np.ndarray, kind: str) -> np.ndarray:
    if kind == "demean":
        return M - M.mean(axis=0, keepdims=True)
    if kind == "demean_scale":
        sd = M.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (M - M.mean(axis=0, keepdims=True)) / sd
    raise ValueError(kind)


def pca_scores(M: np.ndarray, rank: int) -> np.ndarray:
    """n x rank PCA scores U_r * S_r."""
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    r = min(rank, U.shape[1])
    return U[:, :r] * S[:r]


def add_noise(
    X: np.ndarray, W: np.ndarray, rng: np.random.Generator,
    alpha=ALPHA, rho=RHO, c1=C1,
) -> tuple[np.ndarray, np.ndarray]:
    """Appendix B: correlated Gaussian measurement error."""
    n, p = X.shape
    _, q = W.shape
    sig_x = c1 / ((p + 1) ** alpha * np.sqrt(p))
    sig_w = c1 / ((q + 1) ** alpha * np.sqrt(q))
    Hx = rng.normal(0.0, sig_x, size=(n, p))
    Hw = rng.normal(0.0, sig_w, size=(n, q))
    m = min(p, q)
    z1 = rng.normal(size=(n, m))
    z2 = rng.normal(size=(n, m))
    Hx[:, :m] = sig_x * z1
    Hw[:, :m] = sig_w * (rho * z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * z2)
    return X + Hx, W + Hw


def make_outcome(
    X: np.ndarray, W: np.ndarray, beta: np.ndarray,
    rng: np.random.Generator, sigma_eps=SIGMA_EPS,
) -> np.ndarray:
    """Y = X beta + eps, with eps correlated with X but orthogonal to W."""
    n, p = X.shape
    gamma = np.linspace(1.0, 0.2, p)
    gamma /= np.linalg.norm(gamma)
    eps0 = X @ gamma + rng.normal(0.0, 1.0, n)
    Uw, _, _ = np.linalg.svd(W, full_matrices=False)
    eps = eps0 - Uw @ (Uw.T @ eps0)
    eps *= sigma_eps / (np.sqrt(np.mean(eps**2)) + 1e-15)
    return X @ beta + eps


def _svd(Z: np.ndarray, r: int):
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    r = min(r, U.shape[1])
    return U[:, :r], S[:r], Vt[:r]


def _lstsq(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(P, Y, rcond=None)[0]


def fit_oracle(X, W, Y, ell):
    Uw, _, _ = _svd(W, ell)
    return _lstsq(Uw @ (Uw.T @ X), Y)


def fit_pca(Zx, Zw, Y, k, ell):
    U, S, Vt = _svd(Zx, k)
    Uw, _, _ = _svd(Zw, ell)
    Xhat = (U * S) @ Vt
    return _lstsq(Uw @ (Uw.T @ Xhat), Y)


def fit_whiten(Zx, Zw, Y, k, ell):
    U, S, Vt = _svd(Zx, k)
    Uw, _, _ = _svd(Zw, ell)
    P = Uw @ (Uw.T @ (U @ Vt))
    beta_p = _lstsq(P, Y)
    # Map whitened coef -> structural X-coef: beta_P ≈ V S V' beta_X
    return Vt.T @ ((Vt @ beta_p) / np.maximum(S, 1e-12))


def fit_cca(Zx, Zw, Y, k, ell):
    U, S, Vt = _svd(Zx, k)
    Uw, Sw, _ = _svd(Zw, ell)
    Sw = np.maximum(Sw, Sw.max() * 1e-8)
    Xhat = (U * S) @ Vt
    P = (Uw * (1.0 / Sw)) @ (Uw.T @ U) @ Vt
    fitted = P @ _lstsq(P, Y)
    return _lstsq(Xhat, fitted)


def one_rep(X, W, beta, k, ell, rng, run_dir=None):
    Zx, Zw = add_noise(X, W, rng)
    Y = make_outcome(X, W, beta, rng)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X, columns=[f"text_{i:04d}" for i in range(X.shape[1])]).to_parquet(run_dir / "X_text.parquet", index=False)
        pd.DataFrame(W, columns=[f"image_{i:04d}" for i in range(W.shape[1])]).to_parquet(run_dir / "W_image.parquet", index=False)
        pd.DataFrame(Zx, columns=[f"text_{i:04d}" for i in range(Zx.shape[1])]).to_parquet(run_dir / "Z_text.parquet", index=False)
        pd.DataFrame(Zw, columns=[f"image_{i:04d}" for i in range(Zw.shape[1])]).to_parquet(run_dir / "Z_image.parquet", index=False)
        pd.DataFrame({"Y": Y}).to_parquet(run_dir / "Y.parquet", index=False)

    n = len(Y)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2 :]

    betas = {
        "oracle": fit_oracle(X[tr], W[tr], Y[tr], ell),
        "pca": fit_pca(Zx[tr], Zw[tr], Y[tr], k, ell),
        "whiten": fit_whiten(Zx[tr], Zw[tr], Y[tr], k, ell),
        "cca": fit_cca(Zx[tr], Zw[tr], Y[tr], k, ell),
    }
    out = {}
    for name, b in betas.items():
        out[f"mse_{name}"] = float(np.mean((Y[te] - X[te] @ b) ** 2))
    return out


def run_design(X_full, W_full, design, k, ell, n_reps, seed, out_root, save_runs):
    X = pca_scores(post_process(X_full, design), k)
    W = pca_scores(post_process(W_full, design), ell)
    beta = np.ones(k) / np.sqrt(k)

    design_dir = out_root / design
    design_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for j in range(1, n_reps + 1):
        rng = np.random.default_rng(seed + j)
        run_dir = (design_dir / f"run{j}") if save_runs else None
        stats = one_rep(X, W, beta, k, ell, rng, run_dir)
        stats.update(rep=j, design=design, k=k, l=ell)
        rows.append(stats)
        if j % 200 == 0 or j == n_reps:
            print(f"[{design}] {j}/{n_reps}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(design_dir / "mse_by_replication.csv", index=False)
    summary = results[["mse_oracle", "mse_pca", "mse_whiten", "mse_cca"]].agg(["mean", "std", "median"]).T
    summary.to_csv(design_dir / "mse_summary.csv")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--l", type=int, default=L_DEFAULT)
    ap.add_argument("--n-reps", type=int, default=N_REPS_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--designs", nargs="+", default=["demean", "demean_scale"])
    ap.add_argument("--no-save-runs", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_reps = 2

    project = args.project_dir.resolve()
    out = (args.out_dir or project / "data" / "simulation_run").resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"project={project}  out={out}  k={args.k} l={args.l} reps={args.n_reps}")
    X_full, W_full = load_clean_embeddings(project)
    print(f"loaded text {X_full.shape}, image {W_full.shape}")

    all_res = []
    for design in args.designs:
        res = run_design(
            X_full, W_full, design, args.k, args.l, args.n_reps, args.seed, out,
            save_runs=not args.no_save_runs,
        )
        all_res.append(res)
        print(res[["mse_oracle", "mse_pca", "mse_whiten", "mse_cca"]].mean().to_string())
        print()

    combined = pd.concat(all_res, ignore_index=True)
    combined.to_csv(out / "mse_by_replication_all.csv", index=False)
    summary = pd.DataFrame([
        {"design": d, **r[["mse_oracle", "mse_pca", "mse_whiten", "mse_cca"]].mean().to_dict()}
        for d, r in zip(args.designs, all_res)
    ])
    summary.to_csv(out / "mse_summary_all.csv", index=False)
    print(summary.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
