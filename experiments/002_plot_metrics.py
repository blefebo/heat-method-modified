#!/usr/bin/env python3
"""
02_plot_metrics.py

Plot error/roughness vs t_coef from one or more CSVs produced by 01_sweep_tcoef.py.

Example:
  python experiments/02_plot_metrics.py --csv outputs/metrics/bunny.csv --outdir outputs/figures
  python experiments/02_plot_metrics.py --csv outputs/metrics/*.csv --outdir outputs/figures
"""

import argparse
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_csvs(patterns: list[str]) -> pd.DataFrame:
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No CSVs matched: {patterns}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="CSV path(s) or glob(s)")
    ap.add_argument("--outdir", required=True, help="Output directory for figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csvs(args.csv)

    # Group by mesh + noise + robust flag + sources (so plots are interpretable)
    group_cols = ["mesh", "noise_frac", "use_robust", "sources", "k_sources"]
    for keys, g in df.groupby(group_cols, dropna=False):
        mesh, noise_frac, use_robust, sources, k_sources = keys
        g = g.sort_values("t_coef")

        t = g["t_coef"].to_numpy()
        rel_L2 = g["rel_L2"].to_numpy()
        rel_Linf = g["rel_Linf"].to_numpy()
        rough = g["roughness"].to_numpy()

        # --- Plot 1: error vs t_coef ---
        plt.figure()
        plt.plot(t, rel_L2, marker="o", label="Relative L2 error (vs FMM)")
        plt.plot(t, rel_Linf, marker="o", label="Relative L∞ error (vs FMM)")
        plt.xscale("log")
        plt.xlabel("t_coef (log scale)")
        plt.ylabel("Error")
        plt.title(f"{mesh} | noise={noise_frac} | robust={use_robust} | k={k_sources}")
        plt.legend()
        plt.tight_layout()

        # EXERCISE: annotate the minimizer (best t_coef) on the plot.
        # Hint: idx = np.argmin(rel_L2); then plt.scatter(t[idx], rel_L2[idx], ...)

        fname1 = outdir / f"{mesh}_noise{noise_frac}_robust{use_robust}_k{k_sources}_error.png"
        plt.savefig(fname1, dpi=200)
        plt.close()

        # --- Plot 2: roughness vs t_coef ---
        plt.figure()
        plt.plot(t, rough, marker="o")
        plt.xscale("log")
        plt.xlabel("t_coef (log scale)")
        plt.ylabel("Roughness proxy |d^T L d|")
        plt.title(f"{mesh} | noise={noise_frac} (optimal choice) | robust={use_robust} | k={k_sources}")
        plt.tight_layout()

        fname2 = outdir / f"{mesh}_noise{noise_frac}_robust{use_robust}_k{k_sources}_roughness.png"
        plt.savefig(fname2, dpi=200)
        plt.close()

        print("Saved:", fname1.name, "and", fname2.name)

    # Optional: global summary table
    summary = (
        df.sort_values("rel_L2")
          .groupby(group_cols, dropna=False)
          .head(1)[group_cols + ["t_coef", "rel_L2", "rel_Linf", "roughness"]]
          .reset_index(drop=True)
    )
    summary_path = outdir / "best_tcoef_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSaved summary:", summary_path)


if __name__ == "__main__":
    main()
