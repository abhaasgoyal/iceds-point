import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

SRC_DIR = "/scratch/tm70/ag9761/iceds_datasets"
OUT_DIR = "/scratch/tm70/ag9761/iceds_datasets/heatmaps_queensland"
os.makedirs(OUT_DIR, exist_ok=True)

REGIONS = [3]  # Queensland region
LEAD = [1, 2, 3, 4, 5, 6]
FEATURE = ["cbrt"]

MODELS = ["XGBoost", "ecmwf", "ukmo", "access"]
METRICS = ["mae", "rmse"]


def parse_point(s):
    nums = re.findall(r"float64\((-?\d+\.?\d*)\)", s)
    lat, lon = float(nums[0]), float(nums[1])
    return lat, lon


def load_region(feature, lead, region):
    src = f"{SRC_DIR}/output_region{region}_lead{lead}_feature{feature}_perc95.0.csv"
    data = pd.read_csv(src)
    data["lat"], data["lon"] = zip(*data["Point"].map(parse_point))
    return data


def plot_metric(data, region, metric, lead, feature):
    cols = [f"{m}_{metric}" for m in MODELS]
    result = data.set_index(["lat", "lon"])[cols]
    result.columns = MODELS
    result = result.astype(float)

    model_mean = result.mean()

    # size the figure based on how many unique lat/lon points there are,
    # so each grid cell is big enough for its annotation to fit cleanly
    n_lats = result.index.get_level_values("lat").nunique()
    n_lons = result.index.get_level_values("lon").nunique()
    cell_size = 0.6  # inches per grid cell, per subplot
    fig_w = max(10, n_lons * cell_size * 2)
    fig_h = max(10, n_lats * cell_size * 2)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_w, fig_h), sharey=True)
    fig.suptitle(
        f"{metric.upper()} comparison between model predictions "
        f"(Queensland - Region {region}, lead {lead}, {feature.upper()})"
    )
    vmin = result.min().min()
    vmax = result.max().max()
    for idx, model in enumerate(MODELS):
        j = idx % 2
        i = idx // 2
        pivot = result[model].unstack()
        pivot = pivot.sort_index(axis=0, ascending=False)  # flip so north (higher lat) is on top
        pivot = pivot.sort_index(axis=1)  # sort lon left-to-right so cells aren't jumbled
        ax = sns.heatmap(
            pivot, annot=True, fmt=".3f", annot_kws={"size": 8},
            ax=axes[i, j], vmin=vmin, vmax=vmax, cmap="viridis",
            square=True, linewidths=0.5, linecolor="white",
        )
        ax.set_title(f"{model}: {model_mean[model]:.3f}")

    plt.tight_layout()
    out_path = f"{OUT_DIR}/{metric}_heatmap_{feature}_lead{lead}_region{region}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_diff(data, region, metric, lead, feature):
    """Plot the difference between XGBoost and ECMWF for a given metric."""
    xgb_col = f"XGBoost_{metric}"
    ecmwf_col = f"ecmwf_{metric}"
    result = data.set_index(["lat", "lon"])[[xgb_col, ecmwf_col]].astype(float)
    diff = (result[xgb_col] - result[ecmwf_col]).rename("diff")

    n_lats = diff.index.get_level_values("lat").nunique()
    n_lons = diff.index.get_level_values("lon").nunique()
    cell_size = 0.6
    fig_w = max(6, n_lons * cell_size)
    fig_h = max(6, n_lats * cell_size)

    pivot = diff.unstack()
    pivot = pivot.sort_index(axis=0, ascending=False)
    pivot = pivot.sort_index(axis=1)

    # diverging colormap centered at 0 so we can see which model does better
    max_abs = np.nanmax(np.abs(pivot.values))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", annot_kws={"size": 8},
        ax=ax, vmin=-max_abs, vmax=max_abs, cmap="RdBu_r",
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"label": f"XGBoost - ECMWF ({metric.upper()})"},
    )
    ax.set_title(
        f"XGBoost vs ECMWF {metric.upper()} difference "
        f"(Queensland - Region {region}, lead {lead}, {feature.upper()})\n"
        f"mean diff: {diff.mean():.3f}"
    )

    plt.tight_layout()
    out_path = f"{OUT_DIR}/diff_{metric}_heatmap_{feature}_lead{lead}_region{region}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


for feature in FEATURE:
    for lead in LEAD:
        for region in REGIONS:
            data = load_region(feature, lead, region)
            for metric in METRICS:
                plot_metric(data, region, metric, lead, feature)
                plot_diff(data, region, metric, lead, feature)