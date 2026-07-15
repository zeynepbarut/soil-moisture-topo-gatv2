# Topo-GATv2-Concat: Spatially Transferable Soil Moisture Estimation

## Title
Topography-Aware Graph Attention Networks for Spatially Transferable Soil Moisture Estimation: A Spatial-Holdout Benchmark Study — Code and Analysis Scripts

## Description
This repository contains the full analysis pipeline used to train and evaluate Topo-GATv2-Concat, a topography-aware graph attention network for daily soil-moisture prediction under a strict spatial (not temporal) holdout. The code covers data preprocessing, k-nearest-neighbor graph construction, model training (the proposed model and 17 baseline methods), physics-informed loss sensitivity analysis, topological/feature ablations, sensor-failure robustness testing, five-seed reproducibility checks, and independent out-of-network validation on the NASA SoilSCAPE dataset. It accompanies the manuscript submitted to *PeerJ Computer Science*.

## Dataset Information
Two data sources are used, both third-party and publicly available (see Data Availability in the manuscript for full citations). Each corresponds to one of the subfolders expected under `SOIL_DATA_DIR` (see Usage below):

- **International Soil Moisture Network (ISMN)** (`ismn_data/`) — in-situ daily soil moisture observations, 343 stations across the contiguous United States (26 Feb 2016–20 Dec 2017). Source: https://ismn.earth
- **NASA SoilSCAPE** (`soilscape_data/`) — 6 independent co-located sensors at a single instrumented field site, used exclusively for out-of-network validation (14 Jan 2016–29 May 2017). Source: Moghaddam et al. (2017), ORNL DAAC, DOI: https://doi.org/10.3334/ORNLDAAC/1339
- **SRTM GL1 Digital Elevation Model** (`dem_data/`, ~30 m resolution) — used to derive per-station elevation, slope, and aspect covariates. Distributed by OpenTopography, DOI: https://doi.org/10.5069/G9445JDF

Raw data are not redistributed in this repository; each account requires registration with the respective provider before download.

## Code Information
The pipeline is implemented as a single analysis script (run cell-by-cell or as a whole in a GPU-enabled environment, e.g. Google Colab Pro+ / A100), covering:
- Data loading and preprocessing (quality filtering, resampling, delta-target formulation)
- k-nearest-neighbor graph construction (Algorithm 1 in the manuscript)
- Topo-GATv2-Concat and all 17 baseline model definitions
- Training loop with physics-informed loss and early stopping (Algorithms 2–3)
- Evaluation (MAE/RMSE/$R^2$, Wilcoxon signed-rank tests, bootstrap confidence intervals)
- Topological/feature ablations, sensor-failure robustness, five-seed reproducibility
- Figure generation (reproduces Figures 1–13 and Tables 1–7 of the manuscript)

```
├── soil_moisture_topo_gatv2.py   # (or .ipynb) main analysis script, run top-to-bottom
├── requirements.txt              # pinned package versions
├── LICENSE
└── README.md
```

## Usage Instructions
1. Clone the repository and install dependencies (see Requirements below).
2. Set the `SOIL_DATA_DIR` environment variable to point to a local folder containing three subfolders:
   - `dem_data/` — SRTM GL1 DEM tiles covering the study area
   - `ismn_data/` — raw ISMN station downloads (eastern/central US, California)
   - `soilscape_data/` — NASA SoilSCAPE station files
3. Run the script cell by cell (or as a whole) in a GPU-enabled environment. All outputs (trained model checkpoints, evaluation tables, figures) are written to a results folder relative to the script location.
4. To reproduce a specific baseline or ablation, see the corresponding named section/cell in the script (baseline and ablation identifiers match the model names used in Tables 3–5 of the manuscript).

## Requirements
- Python 3.10+
- PyTorch 2.11.0 (CUDA 12.8 build)
- PyTorch Geometric 2.8.0
- scikit-learn 1.6.1, XGBoost 3.3.0
- pandas 2.2.2, NumPy 2.0.2, NetworkX 3.6.1
- rasterio, xarray (DEM/NetCDF I/O)
- Recommended: an NVIDIA GPU (e.g., A100, as used for the manuscript results via Google Colab Pro+, CUDA 12.8) for mixed-precision training; the code also runs on CPU at reduced speed.

Exact pinned versions are listed in `requirements.txt`. Install with:
```
pip install -r requirements.txt
```

## Methodology
Full methodological details (spatial-holdout partitioning, graph construction, model architecture, physics-informed loss, training protocol, and statistical testing) are described in the Materials & Methods section of the accompanying manuscript. The code implements Algorithms 1–3 exactly as specified there.

## Citations
If you use this code or the derived results, please cite:

Barut, Z. (2026). Topo-GATv2-Concat: Spatially Transferable Soil Moisture Estimation — Code and Analysis Scripts (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.21243364

Please also cite the original data sources listed under Dataset Information above.

## License & Contribution Guidelines
This repository is released under the MIT License (see `LICENSE`). Issues and pull requests are welcome; please open an issue before submitting substantial changes.
