# Topo-GATv2-Concat: Spatially Transferable Soil Moisture Estimation

Code accompanying the manuscript:
"Topography-Aware Graph Attention Networks for Spatially Transferable
Soil Moisture Estimation: A Spatial-Holdout Benchmark Study"
(Zeynep Barut, submitted to PeerJ Computer Science).

## Data Sources
- ISMN (International Soil Moisture Network): https://ismn.earth
- NASA SoilSCAPE: https://catalog.data.gov/dataset/soilscape
- SRTM GL1 DEM: standard DEM data portals

## Requirements
See `requirements.txt`. Recommended: NVIDIA A100 GPU
(e.g., Google Colab Pro+, CUDA 12.8).

## Usage
Set the `SOIL_DATA_DIR` environment variable to point to a local folder
containing `dem_data/`, `ismn_data/`, and `soilscape_data/`, then run
the script cell by cell (or as a whole) in a GPU-enabled environment.

## Citation
Citation details and DOI to be added upon publication.
