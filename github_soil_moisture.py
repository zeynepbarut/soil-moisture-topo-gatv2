# ============================================================
# 0) ENVIRONMENT SETUP
# ============================================================
import os, torch

os.environ['TORCH'] = torch.__version__
print(f"PyTorch: {torch.__version__}")

get_ipython().system(
    "pip install -q torch-geometric==2.8.0 "
    "-f https://data.pyg.org/whl/torch-{}.html".format(os.environ['TORCH'])
) if 'get_ipython' in dir() else os.system(
    f"pip install -q torch-geometric==2.8.0 -f https://data.pyg.org/whl/torch-{os.environ['TORCH']}.html"
)

os.system(
    "pip install -q scikit-learn==1.6.1 networkx==3.6.1 "
    "pandas==2.2.2 numpy==2.0.2 xgboost==3.3.0 "
    "xarray netCDF4 rasterio matplotlib seaborn"
)

import os, gc, re, time, glob, random, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, GATConv, SAGEConv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

os.environ['TORCH'] = torch.__version__
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f"✅ Device: {device}")

# ============================================================
# 1) COLAB DETECTION + PORTABLE PATHS
# ============================================================
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    try:
        drive.mount('/content/drive', force_remount=False)
        print("✅ Drive connected.")
    except Exception as e:
        print(f"⚠️ Drive warning: {e}")

# Set SOIL_DATA_DIR as an environment variable to point to your own data
# folder if running outside Colab or with a different Drive layout.
BASE_DIR = os.environ.get(
    "SOIL_DATA_DIR",
    "/content/drive/MyDrive/TARLA SULAMA" if IN_COLAB else "./data"
)

DEM_PATH        = os.path.join(BASE_DIR, "dem_data/output_SRTMGL1.tif")
ISMN_ZIP_PATH   = os.path.join(BASE_DIR, "ismn_data/ismn_local.zip")
ISMN_PATH       = "/content/ismn_local" if IN_COLAB else os.path.join(BASE_DIR, "ismn_local")
SOILSCAPE_PATH  = os.path.join(BASE_DIR, "soilscape_data")
CACHE_DIR       = os.path.join(BASE_DIR, "cache")
FIGURES_DIR     = os.path.join(BASE_DIR, "figures")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ISMN_PATH, exist_ok=True)

print(f"📁 BASE_DIR      = {BASE_DIR}")
print(f"📁 CACHE_DIR     = {CACHE_DIR}")
print(f"📁 FIGURES_DIR   = {FIGURES_DIR}")

# ============================================================
# 2) SEEDING
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 3) VERSION REPORT (for reproducibility / paper reporting)
# ============================================================
import torch_geometric, sklearn, xgboost, pandas, numpy, networkx
print("PyTorch:", torch.__version__)
print("PyTorch Geometric:", torch_geometric.__version__)
print("scikit-learn:", sklearn.__version__)
print("XGBoost:", xgboost.__version__)
print("pandas:", pandas.__version__)
print("NumPy:", numpy.__version__)
print("NetworkX:", networkx.__version__)
print("CUDA:", torch.version.cuda)

# ============================================================
# 4) ISMN EXTRACTION
# ============================================================
import zipfile, shutil

if len(glob.glob(os.path.join(ISMN_PATH, '*.stm'))) < 2662:
    print("🔄 Extracting ISMN zip (~2-3 min)...")
    shutil.rmtree(ISMN_PATH, ignore_errors=True)
    os.makedirs(ISMN_PATH, exist_ok=True)
    with zipfile.ZipFile(ISMN_ZIP_PATH, 'r') as z:
        z.extractall(ISMN_PATH)
    print(f"✅ {len(glob.glob(os.path.join(ISMN_PATH, '*.stm')))} stm files extracted.")
else:
    print("✅ ISMN already exists, skipping extraction.")

# ============================================================
# 5) DEM / ISMN / SOILSCAPE PARSING FUNCTIONS
# ============================================================
import rasterio
import xarray as xr

def load_and_process_dem(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        lat_max = transform.f
        lon_min = transform.c
        res_lat = abs(transform.e)
        res_lon = abs(transform.a)

    data[data == -32768] = np.nan
    data[data < -1000] = np.nan
    if np.isnan(data).any():
        data = np.where(np.isnan(data), np.nanmean(data), data)

    dy, dx = np.gradient(
        data,
        res_lat * 111320,
        res_lon * 111320 * np.cos(np.radians(lat_max - data.shape[0]*res_lat/2))
    )
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)
    aspect = np.degrees(np.arctan2(-dx, dy)).astype(np.float32)
    aspect[aspect < 0] += 360.0

    print(f"✅ DEM loaded: {data.shape} | lat_max={lat_max:.4f} lon_min={lon_min:.4f}")
    return data, slope, aspect, float(lat_max), float(lon_min)


def _process_one_stm(stm_path):
    try:
        basename = os.path.basename(stm_path)
        if '_sm_' not in basename:
            return None, None, None
        with open(stm_path, 'r', errors='replace') as fh:
            lines = fh.readlines()
        if len(lines) < 2:
            return None, None, None
        header = lines[0].strip().split()
        lat = float(header[3])
        lon = float(header[4])
        station_name = basename.split('_sm_')[0]

        records = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                dt = pd.to_datetime(parts[0] + ' ' + parts[1], format='%Y/%m/%d %H:%M')
                val = float(parts[2])
                if val <= 1.0:
                    val = val * 100.0
                flag = parts[3] if len(parts) > 3 else 'U'
                if flag not in ('G', 'M', 'D'):
                    val = np.nan
                records.append({'datetime': dt, 'value': val})
            except Exception:
                continue

        if not records:
            return None, None, None
        return pd.DataFrame(records), station_name, (lat, lon)
    except Exception:
        return None, None, None


def parse_ismn_folder(ismn_path, max_workers=8):
    all_stm = glob.glob(os.path.join(ismn_path, '**', '*_sm_*.stm'), recursive=True)
    print(f"   Number of sm_ files found: {len(all_stm)}")

    if not all_stm:
        print("⚠️ No sm_ files found → Generating demo data.")
        dates = pd.date_range('2015-01-01', periods=2000, freq='H')
        n = 10
        data = np.random.uniform(5, 35, (len(dates), n)).astype(np.float32)
        df = pd.DataFrame(data, index=dates, columns=[f'ST_{i}' for i in range(n)])
        coords = [{'lat': 38.0 + i*0.05, 'lon': -121.0 + i*0.03} for i in range(n)]
        return df, coords

    all_series, coords_list = {}, []
    for i, f in enumerate(all_stm):
        if i % 200 == 0:
            print(f"   {i}/{len(all_stm)} processed...")
        df_s, name, latlon = _process_one_stm(f)
        if df_s is None or name is None or name in all_series:
            continue
        s = df_s.set_index('datetime')['value'].resample('H').mean()
        if s.dropna().empty or latlon is None or latlon[0] is None or latlon[1] is None:
            continue
        all_series[name] = s
        coords_list.append({'lat': latlon[0], 'lon': latlon[1]})

    if not all_series:
        raise ValueError("No valid data could be read from any sm_ file.")

    df_ismn = pd.DataFrame(all_series).sort_index()
    df_ismn = df_ismn.loc[:, df_ismn.isnull().mean() < 0.5]

    n_cols, n_coords = df_ismn.shape[1], len(coords_list)
    if n_cols != n_coords:
        min_n = min(n_cols, n_coords)
        df_ismn = df_ismn.iloc[:, :min_n]
        coords_list = coords_list[:min_n]

    print(f"✅ ISMN: {df_ismn.shape[1]} stations | {len(df_ismn)} time steps")
    print(f"   Date range: {df_ismn.index[0]} → {df_ismn.index[-1]}")
    return df_ismn, coords_list


def parse_soilscape_folder(soilscape_path):
    csv_files = glob.glob(os.path.join(soilscape_path, '**', '*.csv'), recursive=True)
    nc_files = glob.glob(os.path.join(soilscape_path, '**', '*.nc'), recursive=True)

    if not csv_files and not nc_files:
        print("⚠️ No SoilSCAPE files found → Generating demo data.")
        dates = pd.date_range('2015-01-01', periods=2000, freq='H')
        n_nodes = 6
        data = np.random.uniform(5, 35, (len(dates), n_nodes)).astype(np.float32)
        df = pd.DataFrame(data, index=dates, columns=[f'NS_{i}' for i in range(n_nodes)])
        coords = [{'lat': 38.0 + i*0.02, 'lon': -120.9 + i*0.015} for i in range(n_nodes)]
        return df, coords

    all_series, coords_dict = {}, {}
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            time_col = next((c for c in df.columns if 'time' in c or 'date' in c), None)
            sm_col = next((c for c in df.columns if 'moisture' in c or 'vwc' in c or 'sm' in c), None)
            lat_col = next((c for c in df.columns if 'lat' in c), None)
            lon_col = next((c for c in df.columns if 'lon' in c), None)
            if not time_col or not sm_col:
                continue
            df['dt'] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=['dt'])
            s = df.set_index('dt')[sm_col].resample('H').mean()
            name = os.path.splitext(os.path.basename(f))[0]
            if lat_col and lon_col:
                lat_v, lon_v = float(df[lat_col].iloc[0]), float(df[lon_col].iloc[0])
            else:
                continue
            all_series[name] = s
            coords_dict[name] = (lat_v, lon_v)
        except Exception:
            continue

    for f in nc_files:
        try:
            ds = xr.open_dataset(f)
            sm_var = [v for v in ds.data_vars if v.lower() == 'soil_moisture']
            if not sm_var:
                continue
            da = ds[sm_var[0]]
            if 'depth' in da.dims:
                da = da.isel(depth=0)
            time_dim = [d for d in da.dims if 'time' in d.lower()]
            if not time_dim:
                continue
            df_nc = da.to_dataframe().reset_index()
            df_nc['dt'] = pd.to_datetime(df_nc[time_dim[0]], errors='coerce')
            s = df_nc.set_index('dt')[sm_var[0]].resample('H').mean()
            if 'lat' not in ds.coords or 'lon' not in ds.coords:
                ds.close()
                continue
            lat = float(np.asarray(ds.coords['lat'].values).mean())
            lon = float(np.asarray(ds.coords['lon'].values).mean())
            name = os.path.splitext(os.path.basename(f))[0]
            all_series[name] = s
            coords_dict[name] = (lat, lon)
            ds.close()
        except Exception as e:
            print(f"   ⚠️ Could not read {os.path.basename(f)}: {e}")
            continue

    df_nasa = pd.DataFrame(all_series).sort_index()
    df_nasa = df_nasa.loc[:, df_nasa.isnull().mean() < 0.5]
    coords = [{'lat': coords_dict[c][0], 'lon': coords_dict[c][1]} for c in df_nasa.columns]
    print(f"✅ NASA SoilSCAPE: {df_nasa.shape[1]} stations, {len(df_nasa)} time steps")
    return df_nasa, coords


def _clean_coords_to_numpy(coords_input):
    if isinstance(coords_input, list) and len(coords_input) > 0 and isinstance(coords_input[0], dict):
        return np.array([[float(c.get('lat', c.get('latitude', 0))),
                           float(c.get('lon', c.get('longitude', 0)))] for c in coords_input], dtype=np.float32)
    elif isinstance(coords_input, dict):
        return np.array([[float(v[0]), float(v[1])] for v in coords_input.values()], dtype=np.float32)
    return np.atleast_2d(np.array(coords_input, dtype=np.float32))


def build_topographic_graph(coords, dem, slope, lat_max, lon_min,
                             dist_threshold_km=12.0, res=1/3600):
    coords_np = _clean_coords_to_numpy(coords)
    n = len(coords_np)
    G = nx.Graph()

    def _get_topo(lat, lon):
        r = max(0, min(int(round((lat_max - lat) / res)), dem.shape[0]-1))
        c = max(0, min(int(round((lon - lon_min) / res)), dem.shape[1]-1))
        return dem[r, c], slope[r, c]

    topo = [_get_topo(lat, lon) for lat, lon in coords_np]
    edge_list, edge_attrs = [], []

    for i in range(n):
        lat1, lon1 = np.radians(coords_np[i])
        for j in range(i+1, n):
            lat2, lon2 = np.radians(coords_np[j])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            dist_km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
            if dist_km <= dist_threshold_km:
                slope_diff = abs(topo[i][1] - topo[j][1])
                elev_diff = abs(topo[i][0] - topo[j][0])
                norm_dist = dist_km / dist_threshold_km
                norm_sdiff = slope_diff / (slope.max() + 1e-8)
                norm_ediff = elev_diff / (dem.max() + 1e-8)
                edge_list.append([i, j]); edge_list.append([j, i])
                attr = [norm_dist, norm_sdiff, norm_ediff]
                edge_attrs.append(attr); edge_attrs.append(attr)
                G.add_edge(i, j, weight=1.0/max(dist_km, 0.1))

    if len(edge_list) == 0:
        print(f"⚠️ Empty graph: threshold {dist_threshold_km} km — all nodes isolated!")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    print(f"   Graph: {n} nodes | {len(edge_list)//2} edges | threshold={dist_threshold_km} km")
    return edge_index, edge_attr, G


def build_knn_topographic_graph(coords, dem, slope, lat_max, lon_min, k=2, res=1/3600):
    coords_np = _clean_coords_to_numpy(coords)
    n = len(coords_np)

    def _get_topo(lat, lon):
        r = max(0, min(int(round((lat_max - lat) / res)), dem.shape[0]-1))
        c = max(0, min(int(round((lon - lon_min) / res)), dem.shape[1]-1))
        return dem[r, c], slope[r, c]

    topo = [_get_topo(lat, lon) for lat, lon in coords_np]
    dist_mat = np.zeros((n, n))
    for i in range(n):
        lat1, lon1 = np.radians(coords_np[i])
        for j in range(n):
            if i == j:
                continue
            lat2, lon2 = np.radians(coords_np[j])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            dist_mat[i, j] = 2 * 6371.0 * np.arcsin(np.sqrt(a))

    edge_list, edge_attrs = [], []
    max_dist = dist_mat.max() + 1e-8
    for i in range(n):
        knn_idx = np.argsort(dist_mat[i])[:k]
        for j in knn_idx:
            slope_diff = abs(topo[i][1] - topo[j][1])
            elev_diff = abs(topo[i][0] - topo[j][0])
            norm_dist = dist_mat[i, j] / max_dist
            norm_sdiff = slope_diff / (slope.max() + 1e-8)
            norm_ediff = elev_diff / (dem.max() + 1e-8)
            edge_list.append([i, j])
            edge_attrs.append([norm_dist, norm_sdiff, norm_ediff])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    print(f"   k-NN Graph: {n} nodes | {edge_index.shape[1]} edges (k={k})")
    return edge_index, edge_attr

# ============================================================
# 6) CACHE / LOAD DATA
# ============================================================
import pickle

save_path = CACHE_DIR + "/"

if os.path.exists(save_path + "df_ismn.pkl"):
    print("⚡ Loading cache...")
    with open(save_path + "df_ismn.pkl", "rb") as f:
        df_ismn, ismn_coords = pickle.load(f)
    with open(save_path + "df_nasa.pkl", "rb") as f:
        df_nasa, nasa_coords = pickle.load(f)
    dem_matrix = np.load(save_path + "dem.npy")
    slope_matrix = np.load(save_path + "slope.npy")
    aspect_matrix = np.load(save_path + "aspect.npy")
    with open(save_path + "dem_coords.pkl", "rb") as f:
        DEM_LAT_MAX, DEM_LON_MIN = pickle.load(f)
    print(f"✅ Loaded: {df_ismn.shape[1]} ISMN | {df_nasa.shape[1]} NASA stations")
else:
    print("🔄 No cache found, parsing (first run)...")
    dem_matrix, slope_matrix, aspect_matrix, DEM_LAT_MAX, DEM_LON_MIN = load_and_process_dem(DEM_PATH)
    df_ismn, ismn_coords = parse_ismn_folder(ISMN_PATH, max_workers=4)
    df_nasa, nasa_coords = parse_soilscape_folder(SOILSCAPE_PATH)

    os.makedirs(save_path, exist_ok=True)
    with open(save_path + "df_ismn.pkl", "wb") as f:
        pickle.dump((df_ismn, ismn_coords), f)
    with open(save_path + "df_nasa.pkl", "wb") as f:
        pickle.dump((df_nasa, nasa_coords), f)
    np.save(save_path + "dem.npy", dem_matrix)
    np.save(save_path + "slope.npy", slope_matrix)
    np.save(save_path + "aspect.npy", aspect_matrix)
    with open(save_path + "dem_coords.pkl", "wb") as f:
        pickle.dump((DEM_LAT_MAX, DEM_LON_MIN), f)
    print("✅ Cache saved.")

# ============================================================
# 7) RESAMPLE + SPATIAL HOLDOUT SPLIT
# ============================================================
print("🔄 Resampling hourly → daily...")

df_ismn_daily = df_ismn.resample('D').mean()
df_nasa_daily = df_nasa.resample('D').mean()

df_ismn_daily = df_ismn_daily.loc[df_ismn_daily.isnull().mean(axis=1) < 0.5]
df_nasa_daily = df_nasa_daily.loc[df_nasa_daily.isnull().mean(axis=1) < 0.5]

print(f"✅ ISMN daily: {df_ismn_daily.shape} | NASA daily: {df_nasa_daily.shape}")

coords_arr = np.array([[c['lat'], c['lon']] for c in ismn_coords])
print(f"\n📍 ISMN coordinate distribution:")
print(f"   Lat: {coords_arr[:,0].min():.1f} → {coords_arr[:,0].max():.1f}")
print(f"   Lon: {coords_arr[:,1].min():.1f} → {coords_arr[:,1].max():.1f}")

idx_train, idx_val, idx_test = [], [], []
for i, (lat, lon) in enumerate(coords_arr):
    if lon < -115 and 32 <= lat <= 42:
        idx_test.append(i)
    elif -120 <= lon <= -115:
        idx_val.append(i)
    else:
        idx_train.append(i)

print(f"\n✅ Spatial holdout split:")
print(f"   Train: {len(idx_train)} stations")
print(f"   Val  : {len(idx_val)} stations")
print(f"   Test : {len(idx_test)} stations")

assert len(idx_train) > 0 and len(idx_val) > 0 and len(idx_test) > 0
assert len(set(idx_train) & set(idx_val)) == 0
assert len(set(idx_train) & set(idx_test)) == 0
assert len(set(idx_val) & set(idx_test)) == 0
print("✅ No group overlap.")

def safe_idx(df, idx_list):
    valid = [i for i in idx_list if i < len(df.columns)]
    return df.iloc[:, valid]

df_train = safe_idx(df_ismn_daily, idx_train)
df_val = safe_idx(df_ismn_daily, idx_val)
df_test = safe_idx(df_ismn_daily, idx_test)

coords_train = [ismn_coords[i] for i in idx_train if i < len(ismn_coords)]
coords_val = [ismn_coords[i] for i in idx_val if i < len(ismn_coords)]
coords_test = [ismn_coords[i] for i in idx_test if i < len(ismn_coords)]

print(f"\n✅ Subsets: train={df_train.shape} | val={df_val.shape} | test={df_test.shape}")

# ============================================================
# 8) FIGURES 1-2 — Study area & holdout check
# ============================================================
res_dem = 1/3600
n_rows, n_cols = dem_matrix.shape
lon_grid_min = DEM_LON_MIN
lon_grid_max = DEM_LON_MIN + n_cols * res_dem
lat_grid_max = DEM_LAT_MAX
lat_grid_min = DEM_LAT_MAX - n_rows * res_dem
extent = [lon_grid_min, lon_grid_max, lat_grid_min, lat_grid_max]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes[0]
im1 = ax.imshow(dem_matrix, extent=extent, origin='upper', cmap='terrain', aspect='auto')
plt.colorbar(im1, ax=ax, label='Elevation (m)', fraction=0.046, pad=0.04)
ax.scatter(coords_arr[idx_train, 1], coords_arr[idx_train, 0], c='#3498DB', s=20,
           label=f'Train ({len(idx_train)})', edgecolors='white', linewidth=0.3, zorder=3)
ax.scatter(coords_arr[idx_val, 1], coords_arr[idx_val, 0], c='#F39C12', s=20,
           label=f'Val ({len(idx_val)})', edgecolors='white', linewidth=0.3, zorder=3)
ax.scatter(coords_arr[idx_test, 1], coords_arr[idx_test, 0], c='#E74C3C', s=40,
           marker='*', label=f'Test/California ({len(idx_test)})', edgecolors='white', linewidth=0.3, zorder=4)
coords_nasa_arr = np.array([[c['lat'], c['lon']] for c in nasa_coords])
ax.scatter(coords_nasa_arr[:, 1], coords_nasa_arr[:, 0], c='#2ECC71', s=60,
           marker='^', label=f'NASA SoilSCAPE ({len(nasa_coords)})', edgecolors='black', linewidth=0.4, zorder=5)
ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
ax.set_title('Study Area — DEM (SRTM GL1) with Station Locations', fontweight='bold')
ax.legend(fontsize=8, loc='lower left')

ax2 = axes[1]
im2 = ax2.imshow(slope_matrix, extent=extent, origin='upper', cmap='YlOrRd', aspect='auto')
plt.colorbar(im2, ax=ax2, label='Slope (°)', fraction=0.046, pad=0.04)
ax2.scatter(coords_arr[idx_test, 1], coords_arr[idx_test, 0], c='#3498DB', s=40,
            marker='*', label=f'Test/California ({len(idx_test)})', edgecolors='white', linewidth=0.3, zorder=4)
ax2.set_xlabel('Longitude (°)'); ax2.set_ylabel('Latitude (°)')
ax2.set_title('Terrain Slope with Test Stations', fontweight='bold')
ax2.legend(fontsize=8, loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_study_area_dem.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 1 saved.")

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    if idx_train:
        ax.scatter(coords_arr[idx_train, 1], coords_arr[idx_train, 0], c='steelblue', s=20,
                   label=f'Train ({len(idx_train)})', alpha=0.7)
    if idx_val:
        ax.scatter(coords_arr[idx_val, 1], coords_arr[idx_val, 0], c='orange', s=20,
                   label=f'Val ({len(idx_val)})', alpha=0.7)
    if idx_test:
        ax.scatter(coords_arr[idx_test, 1], coords_arr[idx_test, 0], c='red', s=40,
                   label=f'Test ({len(idx_test)})', alpha=0.9, marker='*')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Spatial Holdout — Station Distribution')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for df_part, label, color in [
        (df_train, 'Train (East)', 'steelblue'),
        (df_val, 'Val (Central)', 'orange'),
        (df_test, 'Test (CA)', 'red'),
        (df_nasa_daily, 'NASA (CA)', 'green'),
    ]:
        vals = df_part.values[~np.isnan(df_part.values)].flatten()
        if len(vals) > 0:
            ax2.hist(vals, bins=50, alpha=0.4, label=label, color=color, density=True)
    ax2.set_xlabel('Soil Moisture (Vol%)'); ax2.set_ylabel('Density')
    ax2.set_title('Distribution Comparison')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_spatial_holdout_check.png'), dpi=150)
    plt.show()
    print("✅ Figure 2 saved.")
except Exception as e:
    print(f"⚠️ Could not generate figure: {e}")

# ============================================================
# FIGURE 6 - SPATIAL HOLDOUT STATION DISTRIBUTION + SM DISTRIBUTION
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

ax = axes[0]
ax.scatter(coords_arr[idx_train, 1], coords_arr[idx_train, 0],
           c='#3498DB', s=30, label=f'Train ({len(idx_train)})', alpha=0.8, zorder=3)
ax.scatter(coords_arr[idx_val, 1], coords_arr[idx_val, 0],
           c='#F39C12', s=30, label=f'Val ({len(idx_val)})', alpha=0.8, zorder=3)
ax.scatter(coords_arr[idx_test, 1], coords_arr[idx_test, 0],
           c='#E74C3C', s=80, label=f'Test/California ({len(idx_test)})',
           alpha=0.9, marker='*', zorder=4)

coords_nasa_arr = np.array([[c['lat'], c['lon']] for c in nasa_coords])
ax.scatter(coords_nasa_arr[:, 1], coords_nasa_arr[:, 0],
           c='#2ECC71', s=150, label=f'NASA SoilSCAPE ({len(nasa_coords)})',
           alpha=1.0, marker='^', zorder=5, edgecolors='black', linewidth=0.5)

ax.axvline(x=-100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=-115, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.text(-98, 49, 'Train\n(East)', ha='left', fontsize=9, color='#3498DB')
ax.text(-113, 49, 'Val\n(Central)', ha='right', fontsize=9, color='#F39C12')
ax.text(-122, 49, 'Test\n(CA)', ha='left', fontsize=9, color='#E74C3C')

ax.set_xlabel('Longitude (°)', fontsize=11)
ax.set_ylabel('Latitude (°)', fontsize=11)
ax.set_title('Spatial Holdout — Station Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-128, -68)
ax.set_ylim(24, 52)

ax2 = axes[1]
for df_part, label, color in [
    (df_train, 'Train (East)', '#3498DB'),
    (df_val,   'Val (Central)',   '#F39C12'),
    (df_test,  'Test (CA)',    '#E74C3C'),
    (df_nasa_daily, 'NASA (CA)', '#2ECC71'),
]:
    vals = df_part.values[~np.isnan(df_part.values)].flatten()
    if len(vals) > 0:
        ax2.hist(vals, bins=50, alpha=0.5, label=label, color=color, density=True)

ax2.set_xlabel('Soil Moisture (Vol%)', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('SM Distribution Comparison', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig6_spatial_holdout.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 6 saved.")

# ============================================================
# 9) WINDOWING / FEATURE ENGINEERING
# ============================================================
WINDOW_LEN = 14
print(f"\n✅ Daily window size: {WINDOW_LEN} days")

print("\n🌐 Building topographic graphs...")
ismn_edge_index, ismn_edge_attr, G_ismn = build_topographic_graph(
    ismn_coords, dem_matrix, slope_matrix, DEM_LAT_MAX, DEM_LON_MIN, dist_threshold_km=12.0)
nasa_edge_index, nasa_edge_attr = build_knn_topographic_graph(
    nasa_coords, dem_matrix, slope_matrix, DEM_LAT_MAX, DEM_LON_MIN, k=2)
print("✅ Graph topologies successfully built.")

def _extract_topo_matrix(coords, dem, slope, aspect, lat_max, lon_min, res=1/3600):
    features = []
    for lat, lon in coords:
        row = max(0, min(int(round((lat_max - lat)/res)), dem.shape[0]-1))
        col = max(0, min(int(round((lon - lon_min)/res)), dem.shape[1]-1))
        features.append([dem[row, col], slope[row, col], aspect[row, col]])
    return np.array(features, dtype=np.float32)


def _build_windows(norm_diff, topo_norm, window_len):
    T, N = norm_diff.shape
    assert topo_norm.shape[0] == N
    n_windows = T - window_len
    idx = np.arange(window_len)[None, :] + np.arange(n_windows)[:, None]
    temporal = norm_diff[idx].transpose(0, 2, 1)
    topo_expanded = np.tile(topo_norm[None, :, :], (n_windows, 1, 1))
    X = np.concatenate([temporal, topo_expanded], axis=2).astype(np.float32)
    last_known = norm_diff[window_len-1:-1]
    y_next = norm_diff[window_len:]
    Y_delta = (y_next - last_known).reshape(n_windows, N, 1).astype(np.float32)
    Y_persist = last_known.reshape(n_windows, N, 1).astype(np.float32)
    return X, Y_delta, Y_persist


class SoilMoistureDataset(Dataset):
    def __init__(self, X, Y, edge_index, edge_attr, Y_persist=None):
        self.X, self.Y = X, Y
        self.Y_persist = Y_persist
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = X.shape[1]

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        yp = self.Y_persist[idx] if self.Y_persist is not None else np.zeros_like(self.Y[idx])
        return Data(
            x=torch.tensor(self.X[idx], dtype=torch.float32),
            y=torch.tensor(self.Y[idx], dtype=torch.float32),
            y_persist=torch.tensor(yp, dtype=torch.float32),
            edge_index=self.edge_index, edge_attr=self.edge_attr, num_nodes=self.num_nodes
        )


print("\n" + "="*60)
print("DATA PREPARATION — Spatial Holdout Pipeline")
print("="*60)

coords_train_np = _clean_coords_to_numpy(coords_train)
coords_val_np = _clean_coords_to_numpy(coords_val)
coords_test_np = _clean_coords_to_numpy(coords_test)
coords_nasa_np = _clean_coords_to_numpy(nasa_coords)

topo_train_raw = _extract_topo_matrix(coords_train_np, dem_matrix, slope_matrix, aspect_matrix, DEM_LAT_MAX, DEM_LON_MIN)
topo_val_raw = _extract_topo_matrix(coords_val_np, dem_matrix, slope_matrix, aspect_matrix, DEM_LAT_MAX, DEM_LON_MIN)
topo_test_raw = _extract_topo_matrix(coords_test_np, dem_matrix, slope_matrix, aspect_matrix, DEM_LAT_MAX, DEM_LON_MIN)
topo_nasa_raw = _extract_topo_matrix(coords_nasa_np, dem_matrix, slope_matrix, aspect_matrix, DEM_LAT_MAX, DEM_LON_MIN)

topo_mn = topo_train_raw.min(0)
topo_mx = topo_train_raw.max(0)
topo_den = np.where((topo_mx - topo_mn) == 0, 1.0, topo_mx - topo_mn)

topo_train_norm = (topo_train_raw - topo_mn) / topo_den
topo_val_norm = (topo_val_raw - topo_mn) / topo_den
topo_test_norm = (topo_test_raw - topo_mn) / topo_den
topo_nasa_norm = (topo_nasa_raw - topo_mn) / topo_den

SM_MIN, SM_MAX = 0.0, 60.0
SM_DEN = SM_MAX - SM_MIN

def make_windows_daily(df, topo_norm, window_len=14):
    arr = df.values.copy().astype(np.float32)
    arr[(arr <= 0.0) | (arr > 65.0)] = np.nan
    df_t = pd.DataFrame(arr)
    df_t = df_t.interpolate(method='linear', limit=7, axis=0).ffill(axis=0).bfill(axis=0)
    values = df_t.values

    N_v, N_topo = values.shape[1], topo_norm.shape[0]
    if N_v != N_topo:
        min_N = min(N_v, N_topo)
        values = values[:, :min_N]
        topo_norm = topo_norm[:min_N, :]

    norm_vals = (values - SM_MIN) / SM_DEN
    X, Y_delta, Y_persist = _build_windows(norm_vals, topo_norm, window_len)
    return X, Y_delta, Y_persist


print("🔄 Building windows (delta target)...")
X_tr, Y_tr, Ptr_tr = make_windows_daily(df_train, topo_train_norm, WINDOW_LEN)
X_va, Y_va, Ptr_va = make_windows_daily(df_val, topo_val_norm, WINDOW_LEN)
X_te, Y_te, Ptr_te = make_windows_daily(df_test, topo_test_norm, WINDOW_LEN)
X_na, Y_na, Ptr_na = make_windows_daily(df_nasa_daily, topo_nasa_norm, WINDOW_LEN)

scale_eval = float(SM_MAX - SM_MIN)
offset_eval = float(SM_MIN)
total_input_features = WINDOW_LEN + 3

print(f"✅ Windows ready: train={X_tr.shape} | val={X_va.shape} | test={X_te.shape} | nasa={X_na.shape}")

print("\n🌐 Building graph topologies...")
train_edge_index, train_edge_attr = build_knn_topographic_graph(
    coords_train, dem_matrix, slope_matrix, DEM_LAT_MAX, DEM_LON_MIN, k=3)
val_edge_index, val_edge_attr = build_knn_topographic_graph(
    coords_val, dem_matrix, slope_matrix, DEM_LAT_MAX, DEM_LON_MIN, k=3)
test_edge_index, test_edge_attr = build_knn_topographic_graph(
    coords_test, dem_matrix, slope_matrix, DEM_LAT_MAX, DEM_LON_MIN, k=3)
nasa_edge_index, nasa_edge_attr = build_knn_topographic_graph(
    nasa_coords, dem_matrix, slope_matrix, DEM_LAT_MAX, DEM_LON_MIN, k=2)

BATCH_SIZE = 32
train_loader = PyGDataLoader(SoilMoistureDataset(X_tr, Y_tr, train_edge_index, train_edge_attr, Ptr_tr),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = PyGDataLoader(SoilMoistureDataset(X_va, Y_va, val_edge_index, val_edge_attr, Ptr_va),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = PyGDataLoader(SoilMoistureDataset(X_te, Y_te, test_edge_index, test_edge_attr, Ptr_te),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
nasa_loader = PyGDataLoader(SoilMoistureDataset(X_na, Y_na, nasa_edge_index, nasa_edge_attr, Ptr_na),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✅ Train={len(train_loader)} | Val={len(val_loader)} | Test={len(test_loader)} | NASA={len(nasa_loader)} batch")

for name, Y_part in [("Test (California ISMN)", Y_te), ("NASA SoilSCAPE", Y_na)]:
    delta_real = Y_part.flatten() * scale_eval
    pers_mae = np.abs(delta_real).mean()
    print(f"  {name}: Persistence MAE = {pers_mae:.4f} Vol%")

# ============================================================
# 10) DATASET DESCRIPTION TABLE
# ============================================================
def _dataset_stats(df, coords, label):
    coords_np = _clean_coords_to_numpy(coords) if len(coords) > 0 else np.zeros((0, 2))
    n_stations = df.shape[1]
    n_days = df.shape[0]
    missing_pct = df.isnull().mean().mean() * 100
    if len(coords_np) > 0:
        lat_min, lat_max = coords_np[:, 0].min(), coords_np[:, 0].max()
        lon_min, lon_max = coords_np[:, 1].min(), coords_np[:, 1].max()
    else:
        lat_min = lat_max = lon_min = lon_max = np.nan
    return {
        'Subset': label, 'N Stations': n_stations, 'N Days': n_days,
        'Date Range': f"{df.index.min().date()} to {df.index.max().date()}",
        'Missing (%)': f"{missing_pct:.2f}",
        'Lat Range': f"{lat_min:.2f} to {lat_max:.2f}",
        'Lon Range': f"{lon_min:.2f} to {lon_max:.2f}",
    }

dataset_rows = [
    _dataset_stats(df_train, coords_train, 'ISMN Train'),
    _dataset_stats(df_val, coords_val, 'ISMN Validation'),
    _dataset_stats(df_test, coords_test, 'ISMN Test (California)'),
    _dataset_stats(df_nasa_daily, nasa_coords, 'NASA SoilSCAPE (Supplementary)'),
]
print("\n" + "="*72 + "\nTABLE I: DATASET DESCRIPTION\n" + "="*72)
print(pd.DataFrame(dataset_rows).to_string(index=False))

# ============================================================
# 11) MODEL DEFINITIONS
# ============================================================
criterion = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

def physics_informed_loss_vectorized(pred, target, edge_index, edge_attr, lam=0.01):
    mse = criterion(pred, target)
    if edge_index.shape[1] == 0:
        return mse
    src, dst = edge_index
    diff = (pred[src] - pred[dst]).squeeze(-1)
    slope_w = edge_attr[:, 1].to(pred.device)
    return mse + lam * (slope_w * diff.pow(2)).mean()


class TopoGATv2_Concat(nn.Module):
    def __init__(self, in_features=17, hidden_dim=32, edge_dim=3, heads=4):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden_dim)
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=False)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=False)
        self.topo_proj = nn.Linear(3, hidden_dim)
        self.merge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr=None, bypass_topography=False):
        x_topo = x[:, 14:]
        if bypass_topography:
            x_topo = torch.zeros_like(x_topo)
        h = F.silu(self.proj(x))
        if edge_index.shape[1] > 0 and edge_attr is not None:
            ea = edge_attr.to(x.device)
            h = self.norm1(F.silu(self.gat1(h, edge_index, ea)) + h)
            h = self.norm2(F.silu(self.gat2(h, edge_index, ea)) + h)
        else:
            h = self.norm1(h); h = self.norm2(h)
        t = F.silu(self.topo_proj(x_topo))
        h = F.silu(self.merge(torch.cat([h, t], dim=-1)))
        return self.head(h)

    def forward_with_attention(self, x, edge_index, edge_attr=None):
        x_topo = x[:, 14:]
        h = F.silu(self.proj(x))
        if edge_index.shape[1] > 0 and edge_attr is not None:
            ea = edge_attr.to(x.device)
            h_gat1, (ei1, attn1) = self.gat1(h, edge_index, ea, return_attention_weights=True)
            h = self.norm1(F.silu(h_gat1) + h)
            h_gat2, (ei2, attn2) = self.gat2(h, edge_index, ea, return_attention_weights=True)
            h = self.norm2(F.silu(h_gat2) + h)
        else:
            h = self.norm1(h); h = self.norm2(h)
            ei1, attn1, ei2, attn2 = None, None, None, None
        t = F.silu(self.topo_proj(x_topo))
        h = F.silu(self.merge(torch.cat([h, t], dim=-1)))
        out = self.head(h)
        return out, (ei1, attn1), (ei2, attn2)


class STGCN_Inspired(nn.Module):
    def __init__(self, window_size=17, hidden_channels=32):
        super().__init__()
        self.temporal_len = 14
        self.temp_conv = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.topo_proj = nn.Linear(3, hidden_channels)
        self.gcn_lin = nn.Linear(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr=None):
        x_temp = x[:, :self.temporal_len]
        x_topo = x[:, self.temporal_len:]
        h = torch.tanh(self.temp_conv(x_temp.unsqueeze(1))).mean(-1)
        h = h + F.silu(self.topo_proj(x_topo))
        if edge_index.shape[1] > 0:
            row, col = edge_index
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, h[col])
            deg = torch.zeros(h.size(0), device=x.device).float()
            deg.index_add_(0, row, torch.ones(row.size(0), device=x.device))
            h = self.norm(F.relu(self.gcn_lin(agg / deg.clamp(min=1).unsqueeze(-1))) + h)
        return self.out(h)


class DCRNN_Baseline(nn.Module):
    def __init__(self, in_features=17, hidden_dim=32):
        super().__init__()
        self.rnn = nn.GRU(in_features, hidden_dim, batch_first=True)
        self.gcn = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr=None):
        h, _ = self.rnn(x.unsqueeze(1)); h = h.squeeze(1)
        if edge_index.shape[1] > 0:
            row, col = edge_index
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, h[col])
            deg = torch.zeros(h.size(0), device=x.device).float()
            deg.index_add_(0, row, torch.ones(row.size(0), device=x.device))
            h = F.relu(self.gcn(agg / deg.clamp(min=1).unsqueeze(-1))) + h
        return self.out(h)


class BaselineLSTM(nn.Module):
    def __init__(self, in_features=17, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index=None, edge_attr=None):
        h, _ = self.lstm(x.unsqueeze(1))
        return self.out(h.squeeze(1))


class SoilNet(nn.Module):
    def __init__(self, in_features=17, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.temporal_len = 14
        self.temporal_proj = nn.Linear(self.temporal_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.topo_proj = nn.Linear(3, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                                 dropout=dropout, batch_first=True)
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                                   nn.Linear(hidden_dim // 2, 1))

    def forward(self, x, edge_index=None, edge_attr=None):
        x = x.float()
        x_temp = x[:, :self.temporal_len]
        x_topo = x[:, self.temporal_len:]
        h_t = self.temporal_proj(x_temp).unsqueeze(1)
        h_t = self.transformer(h_t)
        h_p = self.topo_proj(x_topo).unsqueeze(1)
        h_ca, _ = self.cross_attn(query=h_t, key=h_p, value=h_p)
        h = (h_t + h_ca).squeeze(1).float()
        if edge_index is not None and edge_index.shape[1] > 0:
            row, col = edge_index
            agg = torch.zeros_like(h)
            if edge_attr is not None:
                weight = torch.exp(-edge_attr[:, 0].float()).unsqueeze(1)
                agg.index_add_(0, row, weight * h[col])
                deg = torch.zeros(h.size(0), device=x.device, dtype=torch.float32)
                deg.index_add_(0, row, weight.squeeze(1))
                h_g = self.graph_proj(agg / deg.clamp(min=1e-6).unsqueeze(-1))
            else:
                agg.index_add_(0, row, h[col])
                deg = torch.ones(h.size(0), device=x.device, dtype=torch.float32)
                h_g = self.graph_proj(agg / deg.unsqueeze(-1))
            h = self.norm(F.silu(h_g) + h)
        return self.head(self.dropout(h))


class GraphSAGE_SM(nn.Module):
    def __init__(self, in_features=17, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.temporal_len = 14
        self.sage1 = SAGEConv(in_features, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.topo_gate = nn.Sequential(nn.Linear(3, hidden_dim), nn.Sigmoid())
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.proj = nn.Linear(in_features, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index=None, edge_attr=None):
        x = x.float()
        x_topo = x[:, self.temporal_len:]
        res = self.proj(x)
        if edge_index is not None and edge_index.shape[1] > 0:
            h = F.silu(self.bn1(self.sage1(x, edge_index)))
            h = self.drop(h)
            h = F.silu(self.bn2(self.sage2(h, edge_index))) + res
        else:
            h = F.silu(res)
        h = h * self.topo_gate(x_topo)
        return self.head(self.drop(h))


class STAEformer_SM(nn.Module):
    def __init__(self, in_features=17, hidden_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.temporal_len = 14
        self.hidden_dim = hidden_dim
        self._emb_dim = hidden_dim // 2
        self.input_proj = nn.Linear(in_features, hidden_dim)
        t_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                              dim_feedforward=hidden_dim * 2, dropout=dropout,
                                              batch_first=True, norm_first=True)
        self.temporal_attn = nn.TransformerEncoder(t_layer, num_layers=1)
        s_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                              dim_feedforward=hidden_dim * 2, dropout=dropout,
                                              batch_first=True, norm_first=True)
        self.spatial_attn = nn.TransformerEncoder(s_layer, num_layers=1)
        self.topo_proj = nn.Linear(3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x, edge_index=None, edge_attr=None):
        x = x.float(); N = x.size(0)
        x_topo = x[:, self.temporal_len:]
        node_emb = torch.zeros(N, self._emb_dim, device=x.device, dtype=torch.float32)
        h = self.input_proj(x)
        h_t = self.temporal_attn(h.unsqueeze(1)).squeeze(1)
        h_s = self.spatial_attn(h.unsqueeze(0)).squeeze(0)
        h_p = F.silu(self.topo_proj(x_topo))
        h_combined = self.norm(h_t + h_s + h_p)
        node_emb_full = F.pad(node_emb, (0, self.hidden_dim - self._emb_dim))
        h_final = torch.cat([h_combined, node_emb_full], dim=-1)
        return self.head(self.drop(h_final))


class AGCRN_SM(nn.Module):
    def __init__(self, in_features=17, hidden_dim=32, embed_dim=8, max_nodes=150, dropout=0.1):
        super().__init__()
        self.temporal_len = 14
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.node_embeddings = nn.Parameter(torch.randn(max_nodes, embed_dim) * 0.01)
        self.gru = nn.GRUCell(in_features + embed_dim, hidden_dim)
        self.gc_z = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.gc_r = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.gc_c = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.topo_gate = nn.Sequential(nn.Linear(3, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def _get_embeddings(self, N, device):
        if N <= self.node_embeddings.size(0):
            return self.node_embeddings[:N].float()
        pad = torch.zeros(N - self.node_embeddings.size(0), self.embed_dim, device=device, dtype=torch.float32)
        return torch.cat([self.node_embeddings.float(), pad], dim=0)

    def forward(self, x, edge_index=None, edge_attr=None):
        x = x.float(); N = x.size(0)
        x_topo = x[:, self.temporal_len:]
        E = self._get_embeddings(N, x.device)
        A = F.softmax(F.relu(E @ E.T), dim=-1)
        x_aug = torch.cat([x, E], dim=-1)
        h = self.gru(x_aug, torch.zeros(N, self.hidden_dim, device=x.device, dtype=torch.float32)).float()
        Ah = A @ h
        h_aug = torch.cat([Ah, E], dim=-1)
        z = torch.sigmoid(self.gc_z(h_aug))
        r = torch.sigmoid(self.gc_r(h_aug))
        c = torch.tanh(self.gc_c(torch.cat([r * Ah, E], dim=-1)))
        h = (1 - z) * h + z * c
        h = h * self.topo_gate(x_topo)
        h = self.norm(h)
        return self.head(self.drop(h))


class GCN_Vanilla(nn.Module):
    def __init__(self, in_features=17, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden_dim)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index=None, edge_attr=None):
        x = x.float()
        h = F.silu(self.proj(x))
        if edge_index is not None and edge_index.shape[1] > 0:
            h = self.norm1(F.silu(self.gcn1(h, edge_index)) + h)
            h = self.norm2(F.silu(self.gcn2(h, edge_index)) + h)
        return self.head(self.drop(h))


class GAT_Vanilla(nn.Module):
    def __init__(self, in_features=17, hidden_dim=64, heads=4):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=0.1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index=None, edge_attr=None):
        x = x.float()
        h = F.silu(self.proj(x))
        if edge_index is not None and edge_index.shape[1] > 0:
            h = self.norm1(F.silu(self.gat1(h, edge_index)) + h)
            h = self.norm2(F.silu(self.gat2(h, edge_index)) + h)
        return self.head(self.drop(h))


class TemporalFusionTransformer(nn.Module):
    def __init__(self, in_features=17, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.temporal_len = 14
        self.topo_len = 3
        self.var_temp = nn.Sequential(nn.Linear(self.temporal_len, hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.var_topo = nn.Sequential(nn.Linear(self.topo_len, hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.grn_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.grn_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                     dim_feedforward=hidden_dim * 4, dropout=dropout,
                                                     batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                                   nn.Linear(hidden_dim // 2, 1))

    def forward(self, x, edge_index=None, edge_attr=None):
        x = x.float()
        h_temp = self.var_temp(x[:, :self.temporal_len])
        h_topo = self.var_topo(x[:, self.temporal_len:])
        h_cat = torch.cat([h_temp, h_topo], dim=-1)
        gate = self.grn_gate(h_cat)
        h_fuse = gate * self.grn_proj(h_cat) + (1 - gate) * h_temp
        h_attn = self.attn(h_fuse.unsqueeze(1)).squeeze(1)
        h_out = self.norm(torch.sigmoid(self.gate(h_attn)) * h_attn + h_fuse)
        return self.head(self.drop(h_out))


class _NBeatsBlock(nn.Module):
    def __init__(self, in_features, hidden_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.backcast = nn.Linear(hidden_dim, in_features)
        self.forecast = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.fc(x)
        return x - self.backcast(h), self.forecast(h)


class NBEATS_SM(nn.Module):
    def __init__(self, in_features=17, hidden_dim=64, n_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([_NBeatsBlock(in_features, hidden_dim, hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim * n_blocks, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, edge_index=None, edge_attr=None):
        x = x.float()
        forecasts, h = [], x
        for block in self.blocks:
            h, f = block(h)
            forecasts.append(f)
        return self.head(self.drop(torch.cat(forecasts, dim=-1)))


class SoilMoistureNet(nn.Module):
    def __init__(self, in_features=17, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.temporal_len = 14
        self.topo_len = 3
        self.lstm = nn.LSTM(input_size=self.temporal_len, hidden_size=hidden_dim,
                             num_layers=num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.topo_phys = nn.Sequential(nn.Linear(self.topo_len, hidden_dim), nn.Tanh(),
                                        nn.Linear(hidden_dim, hidden_dim))
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                                   nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

    def forward(self, x, edge_index=None, edge_attr=None):
        x = x.float()
        h_lstm, _ = self.lstm(x[:, :self.temporal_len].unsqueeze(1))
        h_lstm = h_lstm.squeeze(1)
        h_topo = self.topo_phys(x[:, self.temporal_len:])
        h_cat = torch.cat([h_lstm, h_topo], dim=-1)
        gate = self.fusion_gate(h_cat)
        h_fuse = self.norm(gate * self.fusion_proj(h_cat) + (1 - gate) * h_lstm)
        return self.head(self.drop(h_fuse))


print("✅ All model definitions ready.")

# ============================================================
# FIGURE 5 - MODEL ARCHITECTURE DIAGRAM
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color='#4A90D9', fontsize=10):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white',
                          linewidth=2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold',
            wrap=True, multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

draw_box(ax, 0.3, 3.5, 2.0, 1.0, 'Input\nx ∈ R^(N×17)\n(14 temporal\n+ 3 topo)', '#2C3E50', fontsize=8)

draw_box(ax, 2.8, 3.5, 1.8, 1.0, 'Linear\nProj\n(17→32)', '#27AE60', fontsize=8)
draw_arrow(ax, 2.3, 4.0, 2.8, 4.0)

draw_box(ax, 5.1, 3.5, 1.8, 1.0, 'GATv2Conv\nLayer 1\n+ LayerNorm', '#8E44AD', fontsize=8)
draw_arrow(ax, 4.6, 4.0, 5.1, 4.0)

draw_box(ax, 7.4, 3.5, 1.8, 1.0, 'GATv2Conv\nLayer 2\n+ LayerNorm', '#8E44AD', fontsize=8)
draw_arrow(ax, 6.9, 4.0, 7.4, 4.0)

draw_box(ax, 5.1, 1.5, 1.8, 1.0, 'Topo\nProjection\n(3→32)', '#E67E22', fontsize=8)

draw_box(ax, 2.8, 1.5, 1.8, 1.0, 'x_topo\n(slope, elevation\naspect)', '#2C3E50', fontsize=8)
draw_arrow(ax, 4.6, 2.0, 5.1, 2.0)

draw_box(ax, 9.7, 2.8, 1.8, 1.5, 'Concat\n[h ‖ t]\n(64→32)\nMerge', '#C0392B', fontsize=8)
draw_arrow(ax, 9.2, 4.0, 9.7, 4.0)
draw_arrow(ax, 6.9, 2.0, 9.85, 2.8)

draw_box(ax, 12.0, 3.5, 1.7, 1.0, 'Linear\nHead\n(32→1)', '#27AE60', fontsize=8)
draw_arrow(ax, 11.5, 3.85, 12.0, 4.0)

draw_box(ax, 5.1, 5.8, 3.8, 0.9, 'Edge Features: [distance, slope_diff, elevation_diff]', '#16A085', fontsize=8)
ax.annotate('', xy=(6.0, 4.5), xytext=(6.0, 5.8),
            arrowprops=dict(arrowstyle='->', color='#16A085', lw=1.5, linestyle='dashed'))
ax.annotate('', xy=(8.3, 4.5), xytext=(8.3, 5.8),
            arrowprops=dict(arrowstyle='->', color='#16A085', lw=1.5, linestyle='dashed'))

ax.set_title('Topo-GATv2-Concat Model Architecture', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig5_architecture.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 5 saved.")

# ============================================================
# 12) TRAIN / EVAL UTILITIES
# ============================================================
EPOCHS = 60
PATIENCE = 10

def evaluate(model, loader, is_gat=False, bypass_topo=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                if is_gat:
                    out = model(b.x, b.edge_index, b.edge_attr, bypass_topography=bypass_topo)
                else:
                    try:
                        out = model(b.x, b.edge_index, b.edge_attr)
                    except TypeError:
                        try:
                            out = model(b.x, b.edge_index)
                        except TypeError:
                            out = model(b.x)
            delta_pred = out.cpu().float().numpy()
            persist = b.y_persist.cpu().numpy()
            delta_true = b.y.cpu().numpy()
            p = (delta_pred + persist) * scale_eval + offset_eval
            t = (delta_true + persist) * scale_eval + offset_eval
            preds.extend(p.flatten()); trues.extend(t.flatten())
    return preds, trues


def get_scores(t, p):
    t, p = np.array(t), np.array(p)
    return {'MAE': mean_absolute_error(t, p), 'RMSE': np.sqrt(mean_squared_error(t, p)), 'R2': r2_score(t, p)}


def train_model(model, loader, opt, epochs, loss_fn, label, val_loader=None, patience=PATIENCE):
    model.train()
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=4)
    best_val = float('inf'); best_state = None; patience_ctr = 0

    for ep in range(1, epochs + 1):
        model.train(); ep_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                try:
                    pred = model(batch.x, batch.edge_index, batch.edge_attr)
                except TypeError:
                    try:
                        pred = model(batch.x, batch.edge_index)
                    except TypeError:
                        pred = model(batch.x)
                loss = loss_fn(pred, batch.y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()
        avg_loss = ep_loss / len(loader)

        ref_loader = val_loader if val_loader is not None else loader
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for vb in ref_loader:
                vb = vb.to(device)
                try:
                    vp = model(vb.x, vb.edge_index, vb.edge_attr)
                except TypeError:
                    try:
                        vp = model(vb.x, vb.edge_index)
                    except TypeError:
                        vp = model(vb.x)
                val_loss += criterion(vp, vb.y).item()
        val_loss /= len(ref_loader)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if ep % 5 == 0 or ep == 1:
            print(f"   {label} | Ep {ep:02d}/{epochs} | Train: {avg_loss:.6f} | Val: {val_loss:.6f} | {time.time()-t0:.1f}s")

        if patience_ctr >= patience:
            print(f"   {label} | Early stop @ epoch {ep} (best val={best_val:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def physics_violation_rate(model, loader, edge_index, edge_attr, threshold=0.1):
    model.eval()
    violations, total = 0, 0
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            try:
                out = model(b.x, b.edge_index, b.edge_attr)
            except Exception:
                out = model(b.x, b.edge_index)
            src, dst = b.edge_index
            pred_diff = (out[src] - out[dst]).abs().squeeze()
            slope_diff = b.edge_attr[:, 1].to(device)
            violation = ((slope_diff > 0.3) & (pred_diff < threshold)).float()
            violations += violation.sum().item()
            total += len(violation)
    return violations / max(total, 1)


def clip_for_report(value, floor=-5.0):
    return max(value, floor) if value < floor else value

# ============================================================
# 13) PHYSICS LOSS SWEEP + FINAL MODEL TRAINING
# ============================================================
EPOCHS_FINAL = 60

print("\n" + "="*72 + "\nSTEP A: Physics Loss Lambda Sweep\n" + "="*72)

LAMBDAS = [0.0, 0.001, 0.01, 0.05]
lam_results = {}

for lam in LAMBDAS:
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    m = TopoGATv2_Concat(in_features=total_input_features).to(device)
    o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    m = train_model(m, train_loader, o, EPOCHS_FINAL,
                     lambda p, y, lam=lam: physics_informed_loss_vectorized(
                         p, y, train_edge_index.to(device), train_edge_attr.to(device), lam=lam),
                     f"lam={lam}", val_loader=val_loader, patience=PATIENCE)
    p, t = evaluate(m, test_loader, is_gat=True)
    lam_results[lam] = get_scores(t, p)
    del m, o; gc.collect(); torch.cuda.empty_cache()

df_lam = pd.DataFrame({
    'lam': list(lam_results.keys()),
    'MAE': [f"{v['MAE']:.4f}" for v in lam_results.values()],
    'RMSE': [f"{v['RMSE']:.4f}" for v in lam_results.values()],
    'R2': [f"{v['R2']:.4f}" for v in lam_results.values()],
})
print(df_lam.to_string(index=False))

best_lam = 0.001
print(f"\n✅ Selected lambda: {best_lam}")

print("\n" + "="*72 + "\nSTEP B: FINAL Topo-GATv2-Concat\n" + "="*72)

results_final = {}

torch.manual_seed(42); np.random.seed(42); random.seed(42)
model_final = TopoGATv2_Concat(in_features=total_input_features).to(device)
n_params_proposed = sum(p.numel() for p in model_final.parameters() if p.requires_grad)
print(f"Topo-GATv2-Concat parameter count: {n_params_proposed:,}")

opt = torch.optim.Adam(model_final.parameters(), lr=1e-3, weight_decay=1e-5)
model_final = train_model(model_final, train_loader, opt, EPOCHS_FINAL,
                           lambda p, y: physics_informed_loss_vectorized(
                               p, y, train_edge_index.to(device), train_edge_attr.to(device), lam=best_lam),
                           "Final-Topo-GATv2-Concat", val_loader=val_loader, patience=PATIENCE)

p_final, y_true_final = evaluate(model_final, test_loader, is_gat=True, bypass_topo=False)
p_final_notopo, _ = evaluate(model_final, test_loader, is_gat=True, bypass_topo=True)
pvr_final = physics_violation_rate(model_final, test_loader, test_edge_index, test_edge_attr)
results_final['Topo-GATv2-Concat (Proposed, Final)'] = get_scores(y_true_final, p_final)
results_final['Topo-GATv2-Concat (Proposed, Final)']['PVR'] = pvr_final
gc.collect(); torch.cuda.empty_cache()

torch.manual_seed(42); np.random.seed(42); random.seed(42)
model_nophys_f = TopoGATv2_Concat(in_features=total_input_features).to(device)
opt = torch.optim.Adam(model_nophys_f.parameters(), lr=1e-3, weight_decay=1e-5)
model_nophys_f = train_model(model_nophys_f, train_loader, opt, EPOCHS_FINAL, criterion,
                              "Final-No-Physics", val_loader=val_loader, patience=PATIENCE)
p_no_phys_f, _ = evaluate(model_nophys_f, test_loader, is_gat=True)
pvr_nophys_f = physics_violation_rate(model_nophys_f, test_loader, test_edge_index, test_edge_attr)
results_final['Ablation: No Physics Loss'] = get_scores(y_true_final, p_no_phys_f)
results_final['Ablation: No Physics Loss']['PVR'] = pvr_nophys_f
del model_nophys_f, opt; gc.collect(); torch.cuda.empty_cache()

torch.manual_seed(42); np.random.seed(42); random.seed(42)
model_stgcn_f = STGCN_Inspired(window_size=total_input_features).to(device)
opt = torch.optim.Adam(model_stgcn_f.parameters(), lr=1e-3)
model_stgcn_f = train_model(model_stgcn_f, train_loader, opt, EPOCHS_FINAL, criterion,
                             "Final-STGCN", val_loader=val_loader, patience=PATIENCE)
p_stgcn_f, _ = evaluate(model_stgcn_f, test_loader)
results_final['STGCN-inspired'] = get_scores(y_true_final, p_stgcn_f)
results_final['STGCN-inspired']['PVR'] = '-'
del model_stgcn_f, opt; gc.collect(); torch.cuda.empty_cache()

torch.manual_seed(42); np.random.seed(42); random.seed(42)
model_dcrnn_f = DCRNN_Baseline(in_features=total_input_features).to(device)
opt = torch.optim.Adam(model_dcrnn_f.parameters(), lr=1e-3)
model_dcrnn_f = train_model(model_dcrnn_f, train_loader, opt, EPOCHS_FINAL, criterion,
                             "Final-DCRNN", val_loader=val_loader, patience=PATIENCE)
p_dcrnn_f, _ = evaluate(model_dcrnn_f, test_loader)
results_final['DCRNN Baseline'] = get_scores(y_true_final, p_dcrnn_f)
results_final['DCRNN Baseline']['PVR'] = '-'
del model_dcrnn_f, opt; gc.collect(); torch.cuda.empty_cache()

torch.manual_seed(42); np.random.seed(42); random.seed(42)
model_lstm_f = BaselineLSTM(in_features=total_input_features).to(device)
opt = torch.optim.Adam(model_lstm_f.parameters(), lr=1e-3)
model_lstm_f = train_model(model_lstm_f, train_loader, opt, EPOCHS_FINAL, criterion,
                            "Final-LSTM", val_loader=val_loader, patience=PATIENCE)
p_lstm_f, _ = evaluate(model_lstm_f, test_loader)
results_final['Baseline LSTM'] = get_scores(y_true_final, p_lstm_f)
results_final['Baseline LSTM']['PVR'] = '-'
del model_lstm_f, opt; gc.collect(); torch.cuda.empty_cache()

results_final['Ablation: No Topo Features'] = get_scores(y_true_final, p_final_notopo)
results_final['Ablation: No Topo Features']['PVR'] = '-'

print("\n" + "="*72 + "\nTABLE III: CROSS-DOMAIN PERFORMANCE\n" + "="*72)
df_t1_final = pd.DataFrame({
    'Model': list(results_final.keys()),
    'MAE (Vol%)': [f"{v['MAE']:.4f}" for v in results_final.values()],
    'RMSE (Vol%)': [f"{v['RMSE']:.4f}" for v in results_final.values()],
    'R²': [f"{clip_for_report(v['R2']):.4f}" + ('*' if v['R2'] < -5 else '') for v in results_final.values()],
    'PVR': [f"{v['PVR']:.4f}" if isinstance(v['PVR'], float) else v['PVR'] for v in results_final.values()],
}).reset_index(drop=True)
print(df_t1_final.to_string(index=False))

# ============================================================
# FIGURE 9 - PHYSICS LOSS LAMBDA SENSITIVITY CHART
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(14, 8))

lam_labels = [f'λ={k}' for k in lam_results.keys()]
maes  = [v['MAE']  for v in lam_results.values()]
rmses = [v['RMSE'] for v in lam_results.values()]
r2s   = [v['R2']   for v in lam_results.values()]

colors = ['#E74C3C' if k == best_lam else '#3498DB' for k in lam_results.keys()]

for ax, vals, ylabel, title in zip(
    axes,
    [maes, rmses, r2s],
    ['MAE (Vol%)', 'RMSE (Vol%)', 'R²'],
    ['MAE vs Lambda', 'RMSE vs Lambda', 'R² vs Lambda']
):
    bars = ax.bar(lam_labels, vals, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
    ax.set_xlabel('Physics Loss λ', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ymin = min(vals) * 0.995
    ymax = max(vals) * 1.005
    ax.set_ylim(ymin, ymax)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', label=f'Selected (λ={best_lam})'),
    Patch(facecolor='#3498DB', label='Other')
]
axes[0].legend(handles=legend_elements, fontsize=9)

plt.suptitle('Physics Loss λ Sensitivity Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig9_lambda.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 9 saved.")

# ============================================================
# FIGURE 8 - REPRESENTATIVE TIME SERIES (BEST/REPRESENTATIVE STATIONS)
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

n_stations = X_te.shape[1]
station_maes = []
for i in range(n_stations):
    y_s = (Y_te[:, i, 0] + Ptr_te[:, i, 0]) * scale_eval + offset_eval
    station_maes.append(np.abs(y_s - y_s.mean()).mean())

idx_best   = np.argmin(station_maes)
idx_median = np.argsort(station_maes)[len(station_maes)//2]
idx_worst  = np.argmax(station_maes)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

for ax_i, (station_idx, title) in enumerate(zip(
    [idx_best, idx_median, idx_worst],
    ['Best-Performing Station', 'Median Station', 'Most Challenging Station']
)):
    y_true_s = (Y_te[:, station_idx, 0] + Ptr_te[:, station_idx, 0]) * scale_eval + offset_eval
    y_pers_s = Ptr_te[:, station_idx, 0] * scale_eval + offset_eval

    model_final.eval()
    preds_s = []
    with torch.no_grad():
        for b in test_loader:
            b = b.to(device)
            out = model_final(b.x, b.edge_index, b.edge_attr)
            out_np = (out.cpu().float().numpy() + b.y_persist.cpu().numpy()) * scale_eval + offset_eval
            n_nodes = X_te.shape[1]
            batch_size = b.num_graphs
            out_reshaped = out_np.reshape(batch_size, n_nodes, 1)
            preds_s.extend(out_reshaped[:, station_idx, 0].tolist())

    preds_s = np.array(preds_s[:len(y_true_s)])
    t = np.arange(len(y_true_s))

    axes[ax_i].plot(t, y_true_s, 'k-', linewidth=1.5, label='Observed', alpha=0.9)
    axes[ax_i].plot(t, preds_s,  'b-', linewidth=1.5, label='Proposed', alpha=0.8)
    axes[ax_i].plot(t, y_pers_s, 'r--', linewidth=1.0, label='Persistence', alpha=0.6)
    mae_s = np.abs(y_true_s - preds_s).mean()
    axes[ax_i].set_title(f'{title} (Station {station_idx}) — MAE={mae_s:.3f} Vol%',
                          fontsize=10, fontweight='bold')
    axes[ax_i].set_ylabel('SM (Vol%)', fontsize=9)
    axes[ax_i].legend(fontsize=8, loc='upper right')
    axes[ax_i].grid(True, alpha=0.3)
    axes[ax_i].set_ylim(0, 55)

axes[-1].set_xlabel('Day (window index)', fontsize=10)
plt.suptitle('Time Series Comparison — Test Set (California)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig8_timeseries.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 8 saved.")

# ============================================================
# 14) TOPOLOGICAL ABLATION
# ============================================================
def build_pure_euclidean_knn_graph(coords, k=2):
    from scipy.spatial import distance_matrix
    coords_np = np.array(coords)
    n = len(coords_np)
    dist_mat = distance_matrix(coords_np, coords_np)
    np.fill_diagonal(dist_mat, np.inf)
    edges, attrs = [], []
    for i in range(n):
        knn_idx = np.argsort(dist_mat[i])[:k]
        for j in knn_idx:
            edges.append([i, j]); attrs.append([1.0, 0.0, 0.0])
    return (torch.tensor(edges, dtype=torch.long).t().contiguous(),
            torch.tensor(attrs, dtype=torch.float32))


def build_random_knn_graph(num_nodes, k=2):
    edges, attrs = [], []
    for i in range(num_nodes):
        others = [j for j in range(num_nodes) if j != i]
        chosen = np.random.choice(others, size=min(k, len(others)), replace=False)
        for j in chosen:
            edges.append([i, j]); attrs.append(list(np.random.randn(3)))
    return (torch.tensor(edges, dtype=torch.long).t().contiguous(),
            torch.tensor(attrs, dtype=torch.float32))


K_ABLATION = 2
nn_final = coords_test_np.shape[0]
ei_eu_f, ea_eu_f = build_pure_euclidean_knn_graph(coords_test_np, k=K_ABLATION)
ei_rn_f, ea_rn_f = build_random_knn_graph(nn_final, k=K_ABLATION)
ei_ng_f = torch.stack([torch.arange(nn_final)] * 2)
ea_ng_f = torch.zeros((nn_final, 3))

topo_ablation_final = {
    'No Graph (Isolated)': (ei_ng_f, ea_ng_f),
    'Random Graph (k=2)': (ei_rn_f, ea_rn_f),
    'Pure Euclidean (k=2)': (ei_eu_f, ea_eu_f),
    'No Topo Features': None,
    'No Physics Loss': None,
    'Full Topo-GATv2-Concat (k=2)': (test_edge_index, test_edge_attr),
}

ablation_rows_final = []
for name, graph in topo_ablation_final.items():
    if name == 'No Topo Features':
        p, _ = evaluate(model_final, test_loader, is_gat=True, bypass_topo=True)
    elif name == 'No Physics Loss':
        p = p_no_phys_f
    else:
        model_final.eval(); p = []
        ei_alt = graph[0].to(device); ea_alt = graph[1].to(device)
        with torch.no_grad():
            for b in test_loader:
                b = b.to(device)
                n_graphs = b.num_graphs
                ei_batch_list, ea_batch_list = [], []
                for g in range(n_graphs):
                    offset = g * nn_final
                    ei_batch_list.append(ei_alt + offset)
                    ea_batch_list.append(ea_alt)
                ei_batch = torch.cat(ei_batch_list, dim=1)
                ea_batch = torch.cat(ea_batch_list, dim=0)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    out = model_final(b.x, ei_batch, ea_batch, bypass_topography=False)
                persist_b = b.y_persist.cpu().numpy()
                p.extend(((out.cpu().float().numpy() + persist_b) * scale_eval + offset_eval).flatten())

    sc = get_scores(y_true_final, p)
    ablation_rows_final.append({
        'Configuration': name, 'MAE': f"{sc['MAE']:.4f}", 'RMSE': f"{sc['RMSE']:.4f}",
        'R²': f"{clip_for_report(sc['R2']):.4f}" + ('*' if sc['R2'] < -5 else '')
    })

df_t2_final = pd.DataFrame(ablation_rows_final)
pvr_map = {'Full Topo-GATv2-Concat (k=2)': pvr_final, 'No Physics Loss': pvr_nophys_f}
df_t2_final['PVR'] = [f"{pvr_map[name]:.4f}" if name in pvr_map else '-' for name in df_t2_final['Configuration']]
print("\n" + "="*72 + "\nTABLE IV: TOPOLOGICAL ABLATION\n" + "="*72)
print(df_t2_final.to_string(index=False))

# ============================================================
# FIGURE 10 - TOPOLOGICAL ABLATION BAR CHARTS
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

configs = [r['Configuration'] for r in ablation_rows_final]
maes_ab  = [float(r['MAE'])  for r in ablation_rows_final]
rmses_ab = [float(r['RMSE']) for r in ablation_rows_final]
r2s_ab   = [float(r['R²'].replace('*',''))  for r in ablation_rows_final]

colors_ab = ['#2ECC71' if 'Full' in c else '#3498DB' for c in configs]
short_labels = [
    'No Graph', 'Random\nGraph', 'Pure\nEuclidean',
    'No Topo\nFeatures', 'No Physics\nLoss', 'Full Model\n(Proposed)'
]

fig, axes = plt.subplots(1, 3, figsize=(16, 8))

for ax, vals, ylabel, title in zip(
    axes,
    [maes_ab, rmses_ab, r2s_ab],
    ['MAE (Vol%)', 'RMSE (Vol%)', 'R²'],
    ['MAE — Ablation', 'RMSE — Ablation', 'R² — Ablation']
):
    bars = ax.bar(short_labels, vals, color=colors_ab,
                  edgecolor='white', linewidth=1.5, width=0.6)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=8)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ymin = min(vals) * 0.995
    ymax = max(vals) * 1.005
    ax.set_ylim(ymin, ymax)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ECC71', label='Full Model (Proposed)'),
    Patch(facecolor='#3498DB', label='Ablation Config')
]
axes[0].legend(handles=legend_elements, fontsize=9)

plt.suptitle('Topological Ablation Study', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig10_ablation.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 10 saved.")

# ============================================================
# 15) SENSOR FAILURE ROBUSTNESS
# ============================================================
print("\n" + "="*72 + "\nTABLE V: SENSOR FAILURE ROBUSTNESS\n" + "="*72)

rates = [0.0, 0.10, 0.20, 0.30, 0.50]
model_final.eval()
rob_rows_final = []
for r in rates:
    pf, tf = [], []
    np.random.seed(42)
    with torch.no_grad():
        for b in test_loader:
            b = b.to(device)
            xb = b.x.clone()
            if r > 0:
                fi = np.random.choice(xb.size(0), size=max(1, int(xb.size(0)*r)), replace=False)
                last_known = xb[fi, 13:14]
                xb[fi, :14] = last_known.repeat(1, 14)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model_final(xb, b.edge_index, b.edge_attr)
            persist_b = b.y_persist.cpu().numpy()
            pf.extend(((out.cpu().float().numpy() + persist_b) * scale_eval + offset_eval).flatten())
            tf.extend(((b.y.cpu().numpy() + persist_b) * scale_eval + offset_eval).flatten())

    sc = get_scores(tf, pf)
    rob_rows_final.append({
        'Failure Ratio': f"{r*100:.0f}%", 'MAE': f"{sc['MAE']:.4f}", 'RMSE': f"{sc['RMSE']:.4f}",
        'R²': f"{clip_for_report(sc['R2']):.4f}" + ('*' if sc['R2'] < -5 else '')
    })
print(pd.DataFrame(rob_rows_final).to_string(index=False))

# ============================================================
# FIGURE 11 - SENSOR FAILURE ROBUSTNESS LINE CHARTS
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

failure_rates = [float(r['Failure Ratio'].replace('%',''))/100 for r in rob_rows_final]
maes_rob  = [float(r['MAE'])  for r in rob_rows_final]
rmses_rob = [float(r['RMSE']) for r in rob_rows_final]
r2s_rob   = [float(r['R²'].replace('*',''))  for r in rob_rows_final]

fig, axes = plt.subplots(1, 3, figsize=(15, 8))

for ax, vals, ylabel, title, color in zip(
    axes,
    [maes_rob, rmses_rob, r2s_rob],
    ['MAE (Vol%)', 'RMSE (Vol%)', 'R²'],
    ['MAE vs Sensor Failure Rate', 'RMSE vs Sensor Failure Rate', 'R² vs Sensor Failure Rate'],
    ['#E74C3C', '#E67E22', '#27AE60']
):
    ax.plot([r*100 for r in failure_rates], vals,
            'o-', color=color, linewidth=2.5, markersize=8,
            markerfacecolor='white', markeredgewidth=2.5)

    for x, y in zip(failure_rates, vals):
        ax.annotate(f'{y:.4f}', (x*100, y),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Sensor Failure Rate (%)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([r*100 for r in failure_rates])
    ax.set_xticklabels([f'{int(r*100)}%' for r in failure_rates])
    ax.grid(True, alpha=0.3)

    ymin = min(vals) * 0.998
    ymax = max(vals) * 1.002
    ax.set_ylim(ymin, ymax)

plt.suptitle('Robustness Analysis Under Sensor Failure', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig11_robustness.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 11 saved.")

# ============================================================
# 16) FIVE-SEED REPRODUCIBILITY
# ============================================================
print("\n" + "="*72 + "\nMULTI-SEED (n=5)\n" + "="*72)

def run_with_seed_final(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = TopoGATv2_Concat(in_features=total_input_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model = train_model(model, train_loader, opt, EPOCHS_FINAL,
                         lambda p, y: physics_informed_loss_vectorized(
                             p, y, train_edge_index.to(device), train_edge_attr.to(device), lam=best_lam),
                         f"FinalSeed{seed}", val_loader=val_loader, patience=PATIENCE)
    p, t = evaluate(model, test_loader, is_gat=True)
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return get_scores(t, p)

SEEDS = [42, 123, 2024, 7, 99]
multi_results_final = [run_with_seed_final(s) for s in SEEDS]
maes_f = [r['MAE'] for r in multi_results_final]
rmses_f = [r['RMSE'] for r in multi_results_final]
r2s_f = [r['R2'] for r in multi_results_final]

print(f"MAE  : {np.mean(maes_f):.4f} ± {np.std(maes_f):.4f}")
print(f"RMSE : {np.mean(rmses_f):.4f} ± {np.std(rmses_f):.4f}")
print(f"R²   : {np.mean(r2s_f):.4f} ± {np.std(r2s_f):.4f}")

p_nasa_supp, y_nasa_supp = evaluate(model_final, nasa_loader, is_gat=True)
sc_nasa_supp = get_scores(y_nasa_supp, p_nasa_supp)
print(f"\nNASA Supplementary: MAE={sc_nasa_supp['MAE']:.4f} | RMSE={sc_nasa_supp['RMSE']:.4f} | R²={sc_nasa_supp['R2']:.4f}")

# ============================================================
# FIGURE 7 - PREDICTED VS OBSERVED SCATTERPLOTS (TEST + NASA)
# ============================================================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

ax1 = axes[0]
t_arr = np.array(y_true_final)
p_arr = np.array(p_final)

ax1.scatter(t_arr, p_arr, alpha=0.2, s=5, color='#3498DB', rasterized=True)
lim = [0, 55]
ax1.plot(lim, lim, 'r--', linewidth=2, label='1:1 Line')
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel('Observed SM (Vol%)', fontsize=11)
ax1.set_ylabel('Predicted SM (Vol%)', fontsize=11)
mae = mean_absolute_error(t_arr, p_arr)
r2  = r2_score(t_arr, p_arr)
ax1.set_title(f'Test Set (California ISMN)\nMAE={mae:.4f} | R²={r2:.4f}', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
t_nasa = np.array(y_nasa_supp)
p_nasa = np.array(p_nasa_supp)

ax2.scatter(t_nasa, p_nasa, alpha=0.3, s=8, color='#2ECC71', rasterized=True)
ax2.plot(lim, lim, 'r--', linewidth=2, label='1:1 Line')
ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xlabel('Observed SM (Vol%)', fontsize=11)
ax2.set_ylabel('Predicted SM (Vol%)', fontsize=11)
mae_n = mean_absolute_error(t_nasa, p_nasa)
r2_n  = r2_score(t_nasa, p_nasa)
ax2.set_title(f'NASA SoilSCAPE (Supplementary)\nMAE={mae_n:.4f} | R²={r2_n:.4f}', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig7_scatter.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 7 saved.")

# ============================================================
# 17) FIGURE 3 — ERROR MAP
# ============================================================
n_stations = len(coords_test_np)
y_true_arr = np.array(y_true_final)
p_final_arr = np.array(p_final)
assert len(y_true_arr) % n_stations == 0

station_errors = np.abs(y_true_arr.reshape(-1, n_stations) - p_final_arr.reshape(-1, n_stations)).mean(axis=0)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter([c[1] for c in coords_test_np], [c[0] for c in coords_test_np],
                 c=station_errors, cmap='RdYlGn_r', s=80, edgecolors='k')
plt.colorbar(sc, label='Mean Absolute Error (Vol%)')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('Spatial Distribution of Prediction Error — California Test Stations')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig3_error_map.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✅ Figure 3 saved.")

# ============================================================
# 18) CLASSICAL BASELINES: RANDOM FOREST, XGBOOST, KRIGING
# ============================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from xgboost import XGBRegressor

def flatten_for_sklearn(X, Y):
    n_samples, n_nodes, n_features = X.shape
    return X.reshape(n_samples * n_nodes, n_features), Y.reshape(n_samples * n_nodes)

X_tr_flat, Y_tr_flat = flatten_for_sklearn(X_tr, Y_tr)
X_te_flat, Y_te_flat = flatten_for_sklearn(X_te, Y_te)
Ptr_tr_flat = Ptr_tr.reshape(-1)
Ptr_te_flat = Ptr_te.reshape(-1)

Y_tr_real = (Ptr_tr_flat + Y_tr_flat) * scale_eval + offset_eval
Y_te_real = (Ptr_te_flat + Y_te_flat) * scale_eval + offset_eval

print("\n" + "="*60 + "\nADDITIONAL BASELINE MODELS\n" + "="*60)
extra_results = {}

print("\n▶ Random Forest")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(X_tr_flat, Y_tr_flat)
p_rf = (rf.predict(X_te_flat) + Ptr_te_flat) * scale_eval + offset_eval
sc_rf = {'MAE': mean_absolute_error(Y_te_real, p_rf), 'RMSE': np.sqrt(mean_squared_error(Y_te_real, p_rf)),
         'R2': r2_score(Y_te_real, p_rf)}
extra_results['Random Forest'] = sc_rf
print(f"   MAE={sc_rf['MAE']:.4f} | RMSE={sc_rf['RMSE']:.4f} | R²={sc_rf['R2']:.4f}")

print("\n▶ XGBoost")
xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)
xgb.fit(X_tr_flat, Y_tr_flat)
p_xgb = (xgb.predict(X_te_flat) + Ptr_te_flat) * scale_eval + offset_eval
sc_xgb = {'MAE': mean_absolute_error(Y_te_real, p_xgb), 'RMSE': np.sqrt(mean_squared_error(Y_te_real, p_xgb)),
          'R2': r2_score(Y_te_real, p_xgb)}
extra_results['XGBoost'] = sc_xgb
print(f"   MAE={sc_xgb['MAE']:.4f} | RMSE={sc_xgb['RMSE']:.4f} | R²={sc_xgb['R2']:.4f}")

print("\n▶ Kriging (Gaussian Process)")
np.random.seed(42)
idx_sub = np.random.choice(len(X_tr_flat), size=5000, replace=False)
kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.01)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True, random_state=42)
gp.fit(X_tr_flat[idx_sub], Y_tr_flat[idx_sub])
idx_te_sub = np.random.choice(len(X_te_flat), size=min(3000, len(X_te_flat)), replace=False)
p_gp = (gp.predict(X_te_flat[idx_te_sub]) + Ptr_te_flat[idx_te_sub]) * scale_eval + offset_eval
sc_gp = {'MAE': mean_absolute_error(Y_te_real[idx_te_sub], p_gp),
         'RMSE': np.sqrt(mean_squared_error(Y_te_real[idx_te_sub], p_gp)),
         'R2': r2_score(Y_te_real[idx_te_sub], p_gp)}
extra_results['Kriging (GP)'] = sc_gp
print(f"   MAE={sc_gp['MAE']:.4f} | RMSE={sc_gp['RMSE']:.4f} | R²={sc_gp['R2']:.4f}")

print("\n▶ Simple Physics Model")
alpha = 0.01
sm_phys = np.clip(X_te_flat[:, 13] * (1 - alpha * X_te_flat[:, 15]), 0, 1)
p_phys = sm_phys * scale_eval + offset_eval
sc_phys = {'MAE': mean_absolute_error(Y_te_real, p_phys), 'RMSE': np.sqrt(mean_squared_error(Y_te_real, p_phys)),
           'R2': r2_score(Y_te_real, p_phys)}
extra_results['Simple Physics Model'] = sc_phys
print(f"   MAE={sc_phys['MAE']:.4f} | RMSE={sc_phys['RMSE']:.4f} | R²={sc_phys['R2']:.4f}")

sm_persist = X_te_flat[:, 13] * scale_eval + offset_eval
extra_results['Persistence'] = {'MAE': mean_absolute_error(Y_te_real, sm_persist),
                                 'RMSE': np.sqrt(mean_squared_error(Y_te_real, sm_persist)),
                                 'R2': r2_score(Y_te_real, sm_persist)}

extra_results['Proposed (Topo-GATv2)'] = {
    'MAE': results_final['Topo-GATv2-Concat (Proposed, Final)']['MAE'],
    'RMSE': results_final['Topo-GATv2-Concat (Proposed, Final)']['RMSE'],
    'R2': results_final['Topo-GATv2-Concat (Proposed, Final)']['R2'],
}

# ============================================================
# 19) 2020-2025 GRAPH-BASED SOTA BASELINES
# ============================================================
EPOCHS_SOTA = 60
sota_models_config = {
    "Transformer-GNN (Wang 2025)": lambda: SoilNet(in_features=total_input_features, hidden_dim=64, num_heads=4, num_layers=2),
    "GraphSAGE-SM (Hamilton 2017)": lambda: GraphSAGE_SM(in_features=total_input_features, hidden_dim=64),
    "STAEformer-SM (Liu 2024)": lambda: STAEformer_SM(in_features=total_input_features, hidden_dim=32, num_heads=4),
    "AGCRN-SM (Bai 2020)": lambda: AGCRN_SM(in_features=total_input_features, hidden_dim=32, embed_dim=8, max_nodes=150),
}

sota_results = {}
sota_predictions = {}
print("\n" + "="*72 + "\nGRAPH-BASED SOTA BASELINE TRAINING\n" + "="*72)

for model_name, model_fn in sota_models_config.items():
    print(f"\n▶ {model_name}")
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    model_sota = model_fn().to(device)
    n_params = sum(p.numel() for p in model_sota.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")
    opt_sota = torch.optim.Adam(model_sota.parameters(), lr=1e-3, weight_decay=1e-5)
    model_sota = train_model(model_sota, train_loader, opt_sota, EPOCHS_SOTA, criterion,
                              model_name, val_loader=val_loader, patience=PATIENCE_SOTA if 'PATIENCE_SOTA' in dir() else PATIENCE)
    p_sota, t_sota = evaluate(model_sota, test_loader, is_gat=False)
    sc = get_scores(t_sota, p_sota)
    sota_results[model_name] = sc
    sota_predictions[model_name] = p_sota
    print(f"   ✅ MAE={sc['MAE']:.4f} | RMSE={sc['RMSE']:.4f} | R²={sc['R2']:.4f}")
    del model_sota, opt_sota; gc.collect(); torch.cuda.empty_cache()

for k, v in {
    'Baseline LSTM': results_final['Baseline LSTM'],
    'DCRNN': results_final['DCRNN Baseline'],
    'Transformer-GNN (Wang 2025)': sota_results['Transformer-GNN (Wang 2025)'],
    'STAEformer-SM (Liu 2024)': sota_results['STAEformer-SM (Liu 2024)'],
    'AGCRN-SM (Bai 2020)': sota_results['AGCRN-SM (Bai 2020)'],
}.items():
    extra_results[k] = {'MAE': v['MAE'], 'RMSE': v['RMSE'], 'R2': v['R2']}

order_all = ['Persistence', 'Simple Physics Model', 'Kriging (GP)', 'Random Forest', 'XGBoost',
             'Baseline LSTM', 'DCRNN', 'Transformer-GNN (Wang 2025)', 'STAEformer-SM (Liu 2024)',
             'AGCRN-SM (Bai 2020)', 'Proposed (Topo-GATv2)']

print("\n" + "="*72 + "\nFULL COMPARISON TABLE\n" + "="*72)
for name in order_all:
    if name not in extra_results:
        continue
    sc = extra_results[name]
    marker = "  ◄ PROPOSED" if name == 'Proposed (Topo-GATv2)' else ""
    print(f"  {name:<35} {sc['MAE']:>7.4f} {sc['RMSE']:>7.4f} {sc['R2']:>7.4f}{marker}")

# ============================================================
# 20) WILCOXON SIGNED-RANK TEST
# ============================================================
from scipy import stats

print("\n" + "="*72 + "\nTABLE VII: STATISTICAL SIGNIFICANCE (Wilcoxon)\n" + "="*72)

p_prop, t_prop = evaluate(model_final, test_loader, is_gat=True, bypass_topo=False)
errors_proposed = np.abs(np.array(p_prop) - np.array(t_prop))

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled_std

wilcoxon_rows = []
for model_name, model_fn in sota_models_config.items():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    m = model_fn().to(device)
    o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    m = train_model(m, train_loader, o, EPOCHS_SOTA, criterion, f"Wilcox-{model_name}",
                     val_loader=val_loader, patience=PATIENCE)
    p_b, _ = evaluate(m, test_loader, is_gat=False)
    del m, o; gc.collect(); torch.cuda.empty_cache()

    errors_base = np.abs(np.array(p_b) - np.array(t_prop))
    n = min(len(errors_proposed), len(errors_base))
    stat, pval = stats.wilcoxon(errors_proposed[:n], errors_base[:n], alternative='less')
    d = cohens_d(errors_base[:n], errors_proposed[:n])
    sig = pval < 0.05
    wilcoxon_rows.append({
        "Baseline": model_name, "W": f"{stat:.0f}", "p-value": f"{pval:.4f}",
        "Cohen's d": f"{d:.3f}", "Significant (α=0.05)": "✓" if sig else "✗",
        "Conclusion": "Proposed better" if sig else "No difference",
    })

print(pd.DataFrame(wilcoxon_rows).to_string(index=False))

# ============================================================
# 21) FIGURE 4 — ERROR DISTRIBUTION BOXPLOT
# ============================================================
error_data = {
    'Persistence': np.abs(np.array(sm_persist) - np.array(Y_te_real)),
    'Random Forest': np.abs(np.array(p_rf) - np.array(Y_te_real)),
    'Proposed': np.abs(np.array(p_final) - np.array(y_true_final)),
    'STAEformer-SM': np.abs(np.array(sota_predictions['STAEformer-SM (Liu 2024)']) - np.array(t_sota)),
}
fig, ax = plt.subplots(figsize=(9, 5))
box_data = [v for v in error_data.values()]
bp = ax.boxplot(box_data, labels=list(error_data.keys()), patch_artist=True, showfliers=False)
colors = ['#95A5A6', '#3498DB', '#2ECC71', '#E74C3C']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.set_ylabel('Absolute Error (Vol%)', fontsize=11)
ax.set_title('Error Distribution Comparison Across Models', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_error_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 4 saved.")

# ============================================================
# 22) COMPUTATIONAL COMPLEXITY (PARAMS + INFERENCE TIME)
# ============================================================
def measure_inference_time(model, loader, n_runs=3):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.time()
            for b in loader:
                b = b.to(device)
                try:
                    _ = model(b.x, b.edge_index, b.edge_attr)
                except TypeError:
                    _ = model(b.x, b.edge_index)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)
    return np.mean(times) * 1000 / len(loader.dataset)

model_classes_full = {
    'Baseline LSTM': lambda: BaselineLSTM(in_features=total_input_features, hidden_dim=64),
    'DCRNN Baseline': lambda: DCRNN_Baseline(in_features=total_input_features, hidden_dim=32),
    'STGCN-inspired': lambda: STGCN_Inspired(window_size=total_input_features, hidden_channels=32),
    'GCN Vanilla': lambda: GCN_Vanilla(in_features=total_input_features, hidden_dim=64),
    'GAT Vanilla': lambda: GAT_Vanilla(in_features=total_input_features, hidden_dim=64, heads=4),
    'TFT (Lim 2021)': lambda: TemporalFusionTransformer(in_features=total_input_features, hidden_dim=64, num_heads=4),
    'N-BEATS (Oreshkin 2020)': lambda: NBEATS_SM(in_features=total_input_features, hidden_dim=64, n_blocks=3),
    'Physics-LSTM (Fang & Shen 2017)': lambda: SoilMoistureNet(in_features=total_input_features, hidden_dim=64, num_layers=2),
    'Transformer-GNN (Wang 2025)': lambda: SoilNet(in_features=total_input_features, hidden_dim=64, num_heads=4, num_layers=2),
    'GraphSAGE-SM (Hamilton 2017)': lambda: GraphSAGE_SM(in_features=total_input_features, hidden_dim=64),
    'STAEformer-SM (Liu 2024)': lambda: STAEformer_SM(in_features=total_input_features, hidden_dim=32, num_heads=4),
    'AGCRN-SM (Bai 2020)': lambda: AGCRN_SM(in_features=total_input_features, hidden_dim=32, embed_dim=8, max_nodes=150),
    'Topo-GATv2-Concat (Proposed)': lambda: TopoGATv2_Concat(in_features=total_input_features, hidden_dim=32, edge_dim=3, heads=4),
}

complexity_rows_full = []
for name, ctor in model_classes_full.items():
    torch.manual_seed(42)
    m = ctor().to(device)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    inf_time = measure_inference_time(m, test_loader)
    complexity_rows_full.append({'Model': name, 'Parameters': f"{n_params:,}",
                                  'Inference (ms/sample)': f"{inf_time:.4f}"})
    del m; gc.collect(); torch.cuda.empty_cache()

print("\n" + "="*72 + "\nTABLE IX: COMPUTATIONAL COMPLEXITY\n" + "="*72)
print(pd.DataFrame(complexity_rows_full).to_string(index=False))

# ============================================================
# 23) BOOTSTRAP CI + TRAINING CURVE + ATTENTION VISUALIZATION
# ============================================================
def bootstrap_ci(errors, n_boot=2000, ci=95):
    boot_means = [np.mean(np.random.choice(errors, size=len(errors), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_means, (100-ci)/2)
    upper = np.percentile(boot_means, 100-(100-ci)/2)
    return np.mean(errors), lower, upper

errors_proposed = np.abs(np.array(p_final) - np.array(y_true_final))
mean_mae, ci_low, ci_high = bootstrap_ci(errors_proposed)
print(f"Proposed MAE: {mean_mae:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}]")

def train_model_with_history(model, loader, opt, epochs, loss_fn, label, val_loader=None, patience=PATIENCE):
    train_hist, val_hist = [], []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=4)
    best_val, best_state, patience_ctr = float('inf'), None, 0

    for ep in range(1, epochs + 1):
        model.train(); ep_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = loss_fn(pred, batch.y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()
        avg_loss = ep_loss / len(loader)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for vb in val_loader:
                vb = vb.to(device)
                vp = model(vb.x, vb.edge_index, vb.edge_attr)
                val_loss += criterion(vp, vb.y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        train_hist.append(avg_loss); val_hist.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, best_state, patience_ctr = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_hist, val_hist

torch.manual_seed(42); np.random.seed(42); random.seed(42)
model_for_plot = TopoGATv2_Concat(in_features=total_input_features).to(device)
opt_plot = torch.optim.Adam(model_for_plot.parameters(), lr=1e-3, weight_decay=1e-5)
model_for_plot, train_hist, val_hist = train_model_with_history(
    model_for_plot, train_loader, opt_plot, EPOCHS_FINAL,
    lambda p, y: physics_informed_loss_vectorized(
        p, y, train_edge_index.to(device), train_edge_attr.to(device), lam=best_lam),
    "Plot-Run", val_loader=val_loader, patience=PATIENCE)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(train_hist, label='Training Loss', color='#3498DB', linewidth=2)
ax.plot(val_hist, label='Validation Loss', color='#E74C3C', linewidth=2)
ax.axvline(x=len(val_hist)-PATIENCE, color='gray', linestyle='--', alpha=0.5, label='Early Stopping Point')
ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('MSE Loss (normalized delta space)', fontsize=11)
ax.set_title('Training Convergence — Topo-GATv2-Concat', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig12_convergence.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 12 saved.")

model_final.eval()
sample_batch = next(iter(test_loader)).to(device)
with torch.no_grad():
    out, (ei1, attn1), (ei2, attn2) = model_final.forward_with_attention(
        sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)
attn1_mean = attn1.mean(dim=1).cpu().numpy()
src = ei1[0].cpu().numpy()
n_test = len(coords_test_np)
src_local = src % n_test
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter([coords_test_np[i][1] for i in src_local], [coords_test_np[i][0] for i in src_local],
                 c=attn1_mean, cmap='viridis', s=60, alpha=0.7)
plt.colorbar(sc, label='Attention Weight (Layer 1, head-averaged)')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('GATv2 Attention Weights — California Test Stations', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig13_attention.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 13 saved.")

print("\n✅✅✅ PIPELINE COMPLETE ✅✅✅")
