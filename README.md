

---

# Heat Method (Modified): Empirical Sensitivity of the Diffusion Time Parameter

This repository contains an empirical study of the **heat method for geodesic distance computation on triangle meshes**, with a focus on understanding how the diffusion time parameter (`t_coef`) affects numerical stability and geometric fidelity. Using `potpourri3d`, we sweep `t_coef` across multiple meshes of varying morphology and compare heat-method distances to **fast marching** distances as a reference. The results reveal a clear **stability–smoothing tradeoff** whose optimal regime depends on mesh geometry and sampling quality.

**TL;DR:**
Small `t_coef` → noisy/unstable distances.
Large `t_coef` → oversmoothed distances.
Intermediate `t_coef` → best accuracy (morphology-dependent).

---

## Contents

```
heat-method-modified/
  README.md
  requirements.txt
  data/
    meshes/                 # input meshes (.obj/.ply)
  experiments/
    01_sweep_tcoef.py       # generate CSV metrics vs t_coef
    02_plot_metrics.py      # plot error/roughness vs t_coef
    03_visualize_fields.py  # qualitative distance field visualizations
  outputs/
    metrics/                # CSV outputs from sweeps
    figures/                # PNG figures for plots and visualizations
    reports/                # compiled PDFs (optional)
  paper/
    draft.tex               # paper writeup
```

---

## Key Results (Examples)

**Error vs `t_coef` (multiple meshes)**
Plots show a non-monotonic error curve on meshes with thin features and irregular triangulation (e.g., bunny and statue scans), with a clear minimum near an intermediate `t_coef`. Smooth, uniformly sampled meshes (sphere) show monotonic improvement with increasing `t_coef`.

**Qualitative distance fields**
Side-by-side visualizations confirm failure modes:

* small `t_coef`: noisy artifacts
* large `t_coef`: oversmoothing / shortcutting
* intermediate `t_coef`: best geometric fidelity

(See `outputs/figures/` and `paper/draft.pdf` for full figures.)

---

## Methods (Short Summary)

* **Algorithm:** Heat method for geodesic distance on meshes (via `potpourri3d`)
* **Parameter of interest:** diffusion time coefficient `t_coef`
* **Reference baseline:** Fast marching distance on meshes
* **Metrics:** relative L2 error, relative L∞ error, roughness proxy ( |d^T L d| )
* **Meshes:** sphere, Stanford bunny, scanned statues, and geometries with thin features
* **Protocol:** sweep `t_coef` on a log scale; hold sources fixed per mesh

---

## Reproducibility

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
potpourri3d
numpy
scipy
pandas
matplotlib
polyscope   # optional (for interactive visualization)
```

### 2) Add meshes

Place `.obj` or `.ply` files in:

```
data/meshes/
```

### 3) Run experiments

**Sweep `t_coef` and generate CSV metrics**

```bash
python experiments/01_sweep_tcoef.py \
  --mesh data/meshes/bunny.obj \
  --out outputs/metrics/bunny.csv \
  --use_robust
```

**Generate plots**

```bash
python experiments/02_plot_metrics.py \
  --csv outputs/metrics/bunny.csv \
  --outdir outputs/figures
```

**Generate qualitative visualizations**

```bash
python experiments/03_visualize_fields.py \
  --mesh data/meshes/bunny.obj \
  --csv outputs/metrics/bunny.csv \
  --outdir outputs/figures
```

---

## Interpretation Guide

* If **error decreases monotonically** with `t_coef`: the mesh is smooth and uniform (oversmoothing is less harmful).
* If **error is U-shaped**: there is a clear stability–smoothing tradeoff; pick `t_coef` near the minimum.
* Thin features and noisy scans typically require **smaller optimal `t_coef`** than smooth synthetic meshes.

---

## Future Directions

* Morphology-aware heuristics for selecting `t_coef`
* Spatially varying diffusion times
* Robustness to mesh noise and decimation
* Extensions to vector heat method and other PDE-based geometry processing tools

---

## Citation / Acknowledgments

This project uses the `potpourri3d` library by Nicholas Sharp for geometry processing and geodesic distance computation. The heat method originates from Crane et al., *Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow*.

---

## Contact

Beamlak Lefebo
Email: beamlak-at-beam-dot-rip
This repository accompanies an application to the MIT Summer Geometry Initiative and is intended as a small, reproducible research study in geometry processing.



