import xarray as xr
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
import os

SRC_DIR = "/scratch/tm70/ag9761/iceds_datasets"
OUT_DIR = "/scratch/tm70/ag9761/iceds_datasets/heatmaps_cartopy"
os.makedirs(OUT_DIR, exist_ok=True)

REGIONS = [3] # [0, 1, 2]
LEAD = [1, 2, 3, 4, 5, 6] # [1, 2, 3, 4, 5, 6]
FEATURE = ["cbrt"]
MODELS = ["XGBoost", "ecmwf", "ukmo", "access"]
METRICS = ["mae", "rmse"]

# --- reference grid from silo (defines the true lat/lon coordinates) ---
silo = xr.open_dataset(f"/scratch/tm70/ag9761/iceds_datasets/silo_monthly_rain_iceds_-1_processed.nc")
lat_grid = silo["pr"]["latitude"].values
lon_grid = silo["pr"]["longitude"].values


def get_queensland_geometry():
    """Fetch the Queensland state boundary polygon from Natural Earth (admin_1)."""
    shp_path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_1_states_provinces"
    )
    reader = shpreader.Reader(shp_path)
    qld = [
        rec.geometry
        for rec in reader.records()
        if rec.attributes.get("name") == "Queensland"
    ]
    if not qld:
        raise RuntimeError("Queensland geometry not found in Natural Earth shapefile")
    return qld


def qld_extent(geoms, pad=0.5):
    """Bounding box covering the full Queensland geometry, with padding."""
    minx, miny, maxx, maxy = unary_union(geoms).bounds
    return [minx - pad, maxx + pad, miny - pad, maxy + pad]


QLD_GEOM = get_queensland_geometry()
QLD_EXTENT = qld_extent(QLD_GEOM)


def parse_point(s):
    nums = re.findall(r"float64\((-?\d+\.?\d*)\)", s)
    lat, lon = float(nums[0]), float(nums[1])
    return lat, lon


def load_region(feature, lead, region):
    src = f"{SRC_DIR}/output_region{region}_lead{lead}_feature{feature}_perc95.0.csv"
    data = pd.read_csv(src)
    data["lat"], data["lon"] = zip(*data["Point"].map(parse_point))
    return data


def nearest_idx(grid, val):
    return int(np.argmin(np.abs(grid - val)))


def build_full_grid(region_data, metric, model):
    """Place every region's values onto the full silo lat/lon grid (NaN elsewhere)."""
    full = np.full((len(lat_grid), len(lon_grid)), np.nan)
    col = f"{model}_{metric}"
    for region, data in region_data.items():
        for _, row in data.iterrows():
            i = nearest_idx(lat_grid, row["lat"])
            j = nearest_idx(lon_grid, row["lon"])
            full[i, j] = row[col]
    return full

def plot_metric_map(region_data, metric, lead, feature, point_size=75):
    # gather all points and values per model, no grid-mapping needed
    all_data = pd.concat(region_data.values(), ignore_index=True)
    cols = {model: f"{model}_{metric}" for model in MODELS}
    vmin = min(all_data[c].min() for c in cols.values())
    vmax = max(all_data[c].max() for c in cols.values())

    fig, axes = plt.subplots(
        nrows=1, ncols=len(MODELS),
        figsize=(5 * len(MODELS), 5.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    sc = None
    for ax, model in zip(axes, MODELS):
        ax.add_geometries(
            QLD_GEOM,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.coastlines(resolution="10m", linewidth=0.8, zorder=4)
        sc = ax.scatter(
            all_data["lon"], all_data["lat"], c=all_data[cols[model]],
            s=point_size, vmin=vmin, vmax=vmax, cmap="viridis",
            edgecolors="black", linewidths=0.4,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        ax.set_extent(QLD_EXTENT, crs=ccrs.PlateCarree())
        ax.set_title(f"{model}\nmean: {all_data[cols[model]].mean():.3f}")

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.4,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.xlocator = mticker.MultipleLocator(1)
        gl.ylocator = mticker.MultipleLocator(1)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

    fig.suptitle(f"{metric.upper()} comparison (lead {lead}, {feature.upper()})", fontsize=16, y=1.02)
    fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label=metric.upper())

    out_path = f"{OUT_DIR}/{metric}_map_{feature}_lead{lead}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


for feature in FEATURE:
    for lead in LEAD:
        region_data = {region: load_region(feature, lead, region) for region in REGIONS}
        # region_data[2] = region_data[2].iloc[:-1]
        # region_data[2] = pd.concat([region_data[2].iloc[:-4], region_data[2].iloc[-3:]])
        for metric in METRICS:
            plot_metric_map(region_data, metric, lead, feature)