#!/usr/bin/env python3
"""
01_sweep_tcoef.py

Sweep potpourri3d heat-method time parameter t_coef and compare to
fast marching distances as a reference. Writes CSV of metrics.

Example:
  python experiments/01_sweep_tcoef.py --mesh data/meshes/bunny.obj --out outputs/metrics/bunny.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import potpourri3d as pp3d
from scipy.sparse import csr_matrix


def load_mesh(mesh_path: str):
    V, F = pp3d.read_mesh(mesh_path)  # returns numpy arrays
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    return V, F


def bbox_diag(V: np.ndarray) -> float:
    mn = V.min(axis=0)
    mx = V.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def add_vertex_noise(V: np.ndarray, sigma_frac: float, seed: int) -> np.ndarray:
    """Add iid Gaussian noise to vertices; sigma_frac is fraction of bbox diagonal."""
    if sigma_frac <= 0:
        return V
    rng = np.random.default_rng(seed)
    sigma = sigma_frac * bbox_diag(V)
    return V + rng.normal(scale=sigma, size=V.shape)


def pick_sources(V: np.ndarray, k: int, seed: int) -> list[int]:
    """
    Pick k source vertices.

    EXERCISE (optional improvement):
      Replace this with "farthest point sampling" on the mesh graph or Euclidean FPS,
      so sources are well-separated.
    """
    rng = np.random.default_rng(seed)
    n = V.shape[0]
    return [int(i) for i in rng.choice(n, size=k, replace=False)]


def fmm_distance(V: np.ndarray, F: np.ndarray, sources: list[int]) -> np.ndarray:
    """
    Fast marching reference. potpourri3d expects a list of points; a vertex source
    is represented as [(vertex_index, [])] where [] is empty barycentric coords.
    """
    solver = pp3d.MeshFastMarchingDistanceSolver(V, F)
    points = [[(int(s), [])] for s in sources]
    d = solver.compute_distance(points, sign=False)
    return np.asarray(d, dtype=np.float64)


def heat_distance(V: np.ndarray, F: np.ndarray, sources: list[int], t_coef: float, use_robust: bool) -> np.ndarray:
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F, t_coef=float(t_coef), use_robust=use_robust)
    if len(sources) == 1:
        d = solver.compute_distance(int(sources[0]))
    else:
        d = solver.compute_distance_multisource([int(s) for s in sources])
    return np.asarray(d, dtype=np.float64)


def dirichlet_energy(d: np.ndarray, L: csr_matrix) -> float:
    # Many Laplacian conventions are negative semidefinite; take absolute value for a simple roughness proxy.
    val = float(d.T @ (L @ d))
    return abs(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="Path to mesh (.obj/.ply/etc.)")
    ap.add_argument("--out", required=True, help="Output CSV path (e.g., outputs/metrics/bunny.csv)")
    ap.add_argument("--k_sources", type=int, default=1, help="Number of source vertices")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for source selection/noise")
    ap.add_argument("--noise_frac", type=float, default=0.0, help="Gaussian noise as fraction of bbox diagonal")
    ap.add_argument("--use_robust", action="store_true", help="Use intrinsic triangulation robustness")
    ap.add_argument("--tcoef_min", type=float, default=0.05)
    ap.add_argument("--tcoef_max", type=float, default=10.0)
    ap.add_argument("--tcoef_n", type=int, default=12)
    args = ap.parse_args()

    mesh_path = args.mesh
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    V, F = load_mesh(mesh_path)
    V = add_vertex_noise(V, args.noise_frac, args.seed)

    sources = pick_sources(V, args.k_sources, args.seed)

    # Reference: Fast marching distance
    d_ref = fmm_distance(V, F, sources)

    # Optional: Laplacian for roughness proxy
    L = pp3d.cotan_laplacian(V, F, denom_eps=1e-6)

    # Sweep t_coef (log spaced)
    tcoefs = np.geomspace(args.tcoef_min, args.tcoef_max, args.tcoef_n)

    rows = []
    for tc in tcoefs:
        d_h = heat_distance(V, F, sources, tc, args.use_robust)

        # Metrics
        rel_L2 = float(np.linalg.norm(d_h - d_ref) / (np.linalg.norm(d_ref) + 1e-12))
        rel_Linf = float(np.max(np.abs(d_h - d_ref)) / (np.max(np.abs(d_ref)) + 1e-12))
        rough = dirichlet_energy(d_h, L)

        rows.append(
            dict(
                mesh=os.path.basename(mesh_path),
                mesh_path=str(mesh_path),
                noise_frac=args.noise_frac,
                k_sources=args.k_sources,
                sources=";".join(map(str, sources)),
                use_robust=bool(args.use_robust),
                t_coef=float(tc),
                rel_L2=rel_L2,
                rel_Linf=rel_Linf,
                roughness=rough,
            )
        )

        print(f"t_coef={tc:>7.4g}  rel_L2={rel_L2:>9.3e}  rel_Linf={rel_Linf:>9.3e}  rough={rough:>9.3e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Quick summary
    best_idx = int(df["rel_L2"].idxmin())
    best_tc = df.loc[best_idx, "t_coef"]
    print("\nSaved:", out_csv)
    print(f"Best by rel_L2: t_coef = {best_tc:.4g}  (rel_L2={df.loc[best_idx,'rel_L2']:.3e})")
    print(f"Sources: {sources}")


if __name__ == "__main__":
    main()

