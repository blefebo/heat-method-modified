#!/usr/bin/env python3
"""
03_visualize_fields.py

Visualize heat-method distance fields for a few representative t_coef values.

Example:
  python experiments/03_visualize_fields.py --mesh data/meshes/bunny.obj --csv outputs/metrics/bunny.csv --outdir outputs/figures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import potpourri3d as pp3d

# Optional polyscope
try:
    import polyscope as ps
    HAS_PS = True
except Exception:
    HAS_PS = False

# Optional matplotlib fallback
try:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def load_mesh(mesh_path: str):
    V, F = pp3d.read_mesh(mesh_path)
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int32)


def heat_distance(V, F, sources, t_coef, use_robust):
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F, t_coef=float(t_coef), use_robust=use_robust)
    if len(sources) == 1:
        return np.asarray(solver.compute_distance(int(sources[0])), dtype=np.float64)
    return np.asarray(solver.compute_distance_multisource([int(s) for s in sources]), dtype=np.float64)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x - np.min(x)
    denom = np.max(x)
    return x / denom if denom > 0 else x


def parse_sources(sources_str: str) -> list[int]:
    # sources stored as "12;987;33"
    return [int(s) for s in sources_str.split(";") if s.strip()]


def visualize_polyscope(V, F, fields, outdir: Path, base_name: str):
    ps.init()
    ps.set_up_dir("z_up")
    m = ps.register_surface_mesh("mesh", V, F)

    # Add all fields so you can toggle them; also screenshot each
    for label, values in fields.items():
        m.add_scalar_quantity(label, values, enabled=False)

    # EXERCISE: set a consistent camera view across screenshots.
    # Hint: ps.look_at(...) / ps.set_view_projection_mode(...) etc.
    # For now, just show and let you move camera once.

    ps.show()  # interact; close window when ready

    # After closing, also save screenshots automatically (optional)
    # NOTE: polyscope screenshots depend on window context; if you want purely automated,
    # we can switch to ps.screenshot() workflow after you set a camera pose.


def visualize_matplotlib(V, F, fields, outdir: Path, base_name: str):
    if not HAS_MPL:
        raise RuntimeError("matplotlib not available, and polyscope not installed. Install one of them.")

    tri = Triangulation(V[:, 0], V[:, 1], triangles=F)

    for label, values in fields.items():
        plt.figure()
        # Simple 2D projection using x-y; for some meshes, you may want a better view.
        plt.tripcolor(tri, values, shading="gouraud")
        plt.gca().set_aspect("equal", "box")
        plt.title(label)
        plt.axis("off")
        outpath = outdir / f"{base_name}_{label}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
        print("Saved:", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--csv", required=True, help="CSV from 01_sweep_tcoef.py (used to pick best t_coef + sources)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--t_small", type=float, default=0.1)
    ap.add_argument("--t_large", type=float, default=10.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    V, F = load_mesh(args.mesh)

    df = pd.read_csv(args.csv)

    # Pick the best row by rel_L2 (same mesh, smallest error)
    # If your CSV contains multiple meshes, filter first:
    mesh_name = Path(args.mesh).name
    df_m = df[df["mesh"] == mesh_name].copy()
    if df_m.empty:
        # fallback: use all rows
        df_m = df

    best_row = df_m.loc[df_m["rel_L2"].idxmin()]
    t_best = float(best_row["t_coef"])
    use_robust = bool(best_row["use_robust"])
    sources = parse_sources(str(best_row["sources"]))

    # Compute three fields
    fields = {}
    for tc, tag in [(args.t_small, "small"), (t_best, "best"), (args.t_large, "large")]:
        d = heat_distance(V, F, sources, tc, use_robust=use_robust)
        fields[f"heat_tcoef_{tag}_{tc:g}"] = normalize01(d)

    base_name = f"{mesh_name.replace('.', '_')}"

    print(f"Using sources={sources}, use_robust={use_robust}, t_best={t_best:g}")
    print("Generating visualizations...")

    if HAS_PS:
        visualize_polyscope(V, F, fields, outdir, base_name)
    else:
        visualize_matplotlib(V, F, fields, outdir, base_name)


if __name__ == "__main__":
    main()
