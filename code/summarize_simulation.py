#!/usr/bin/env python3
"""Aggregate Monte Carlo simulation results and plot mean MSE by method.

Reads data/simulation_run/mse_by_replication_all.csv (or per-design CSVs),
writes a summary table, and saves two bar charts — one per post-processing
design (demean, demean_scale) — comparing Oracle / PCA / Whiten / CCA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from paths import PROJECT_DIR
except ImportError:
    try:
        PROJECT_DIR = Path(__file__).resolve().parent.parent
    except NameError:
        cwd = Path.cwd()
        PROJECT_DIR = cwd.parent if cwd.name == "code" else cwd

METHODS = [
    ("mse_oracle", "Oracle"),
    ("mse_pca", "PCA"),
    ("mse_whiten", "Whiten"),
    ("mse_cca", "CCA"),
]
DESIGNS = ("demean", "demean_scale")


def load_results(sim_dir: Path) -> pd.DataFrame:
    all_path = sim_dir / "mse_by_replication_all.csv"
    if all_path.exists():
        return pd.read_csv(all_path)

    frames = []
    for design in DESIGNS:
        p = sim_dir / design / "mse_by_replication.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(
            f"No simulation results under {sim_dir}. "
            "Run: python simulation.py --no-save-runs"
        )
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c, _ in METHODS if c in df.columns]
    rows = []
    for design, g in df.groupby("design"):
        for col in metric_cols:
            rows.append(
                {
                    "design": design,
                    "method": dict(METHODS)[col],
                    "mean_mse": g[col].mean(),
                    "std_mse": g[col].std(ddof=1),
                    "median_mse": g[col].median(),
                    "n_reps": len(g),
                }
            )
    return pd.DataFrame(rows)


def plot_design(summary: pd.DataFrame, design: str, out_path: Path) -> None:
    sub = summary[summary["design"] == design].copy()
    if sub.empty:
        print(f"skip plot: no rows for design={design}")
        return

    # Keep a stable method order
    order = [label for _, label in METHODS]
    sub["method"] = pd.Categorical(sub["method"], categories=order, ordered=True)
    sub = sub.sort_values("method")

    x = np.arange(len(sub))
    means = sub["mean_mse"].to_numpy()
    stds = sub["std_mse"].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["method"].tolist())
    ax.set_ylabel("Mean OOS MSE")
    ax.set_xlabel("Method")
    title = "demean" if design == "demean" else "demean + scale"
    ax.set_title(f"Average Monte Carlo MSE — {title}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate bar heights
    for rect, m in zip(bars, means):
        ax.annotate(
            f"{m:.3g}",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    ap.add_argument(
        "--sim-dir",
        type=Path,
        default=None,
        help="Default: <project>/data/simulation_run",
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Default: <project>/figures",
    )
    args = ap.parse_args()

    project = args.project_dir.resolve()
    sim_dir = (args.sim_dir or project / "data" / "simulation_run").resolve()
    fig_dir = (args.fig_dir or project / "figures").resolve()

    df = load_results(sim_dir)
    summary = summarize(df)
    out_csv = sim_dir / "mse_method_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    print(summary.to_string(index=False))

    plot_design(summary, "demean", fig_dir / "sim_mse_demean.png")
    plot_design(summary, "demean_scale", fig_dir / "sim_mse_demean_scale.png")


if __name__ == "__main__":
    main()
