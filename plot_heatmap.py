import argparse
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

HOME_DIR = "/scratch/tm70/ag9761/iceds_datasets"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Queensland metric and XGBoost vs best-of-3 diff heatmaps."
    )
    parser.add_argument(
        "--season",
        required=True,
        help="Season subfolder to read from, e.g. dry or wet",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Use forecast data instead of hindcast (default: hindcast)",
    )
    parser.add_argument(
        "--bias",
        choices=["before", "after"],
        default="after",
        help="Bias-correction stage to plot (default: after)",
    )
    return parser.parse_args()


args = parse_args()
is_forecast = args.forecast
season = args.season
bias = args.bias

out_dir_cast = "forecast" if is_forecast else "hindcast"
SRC_DIR = f"{HOME_DIR}/{out_dir_cast}/{season}/bias_analysis/{bias}"
OUT_DIR = f"{HOME_DIR}/{out_dir_cast}/{season}/bias_analysis/{bias}/heatmaps_queensland_diff"

os.makedirs(OUT_DIR, exist_ok=True)

REGIONS = [3]  # Queensland region
LEAD = [1, 2, 3, 4, 5, 6]
FEATURE = ["cbrt"]

MODELS = ["XGBoost", "ecmwf", "ukmo", "access"]
OTHER_MODELS = ["ecmwf", "ukmo", "access"]  # models to compare XGBoost against
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


def plot_diff(data, region, metric, lead, feature):
    """Plot XGBoost vs best-of-3 diff, alongside a categorical map of
    which model (ecmwf/ukmo/access) was best at each point."""
    xgb_col = f"XGBoost_{metric}"
    other_cols = [f"{m}_{metric}" for m in OTHER_MODELS]

    result = data.set_index(["lat", "lon"])[[xgb_col] + other_cols].astype(float)
    result.columns = ["XGBoost"] + OTHER_MODELS

    # best (lowest error) of the 3 non-XGBoost models at each point
    best_other = result[OTHER_MODELS].min(axis=1)
    best_other_name = result[OTHER_MODELS].idxmin(axis=1)  # which model won, per point

    diff = (result["XGBoost"] - best_other).rename("diff")

    n_lats = diff.index.get_level_values("lat").nunique()
    n_lons = diff.index.get_level_values("lon").nunique()
    cell_size = 0.6
    fig_w = max(12, n_lons * cell_size * 2)
    fig_h = max(6, n_lats * cell_size)

    diff_pivot = diff.unstack()
    diff_pivot = diff_pivot.sort_index(axis=0, ascending=False)  # north on top
    diff_pivot = diff_pivot.sort_index(axis=1)  # lon left-to-right

    best_pivot = best_other_name.unstack()
    best_pivot = best_pivot.reindex_like(diff_pivot)  # match row/col order

    # map model names -> integer codes for the categorical heatmap
    model_codes = {m: i for i, m in enumerate(OTHER_MODELS)}
    code_pivot = best_pivot.replace(model_codes).astype(float)

    colors = ["#4C72B0", "#DD8452", "#55A868"]  # one distinct color per model
    cmap = ListedColormap(colors)
    bounds = list(range(len(OTHER_MODELS) + 1))
    norm = BoundaryNorm(bounds, cmap.N)

    max_abs = np.nanmax(np.abs(diff_pivot.values))

    fig, axes = plt.subplots(ncols=2, figsize=(fig_w, fig_h), sharey=True)

    # left panel: diff heatmap
    sns.heatmap(
        diff_pivot, annot=True, fmt=".3f", annot_kws={"size": 8},
        ax=axes[0], vmin=-max_abs, vmax=max_abs, cmap="RdBu_r",
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"label": f"XGBoost - best-of-3 ({metric.upper()})"},
    )
    axes[0].set_title(f"XGBoost - best-of-3 {metric.upper()}\nmean diff: {diff.mean():.3f}")

    # right panel: categorical "who won" heatmap
    sns.heatmap(
        code_pivot, cmap=cmap, norm=norm,
        ax=axes[1], square=True, linewidths=0.5, linecolor="white",
        cbar=False,
    )
    axes[1].set_title("Best of the 3 models at each point")
    legend_handles = [Patch(color=colors[i], label=OTHER_MODELS[i].upper()) for i in range(len(OTHER_MODELS))]
    axes[1].legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    fig.suptitle(
        f"XGBoost vs best-of-3 ({'/'.join(m.upper() for m in OTHER_MODELS)}) {metric.upper()} "
        f"(Queensland - Region {region}, lead {lead}, {feature.upper()})"
    )

    plt.tight_layout()
    out_path = f"{OUT_DIR}/diff_best3_{metric}_heatmap_{feature}_lead{lead}_region{region}.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")

    # quick summary of which model was the "best other" most often
    print(best_other_name.str.upper().value_counts())


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
        ax.set_title(f"{model.upper()}: {model_mean[model]:.3f}")

    plt.tight_layout()
    out_path = f"{OUT_DIR}/{metric}_heatmap_{feature}_lead{lead}_region{region}.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


for feature in FEATURE:
    for lead in LEAD:
        for region in REGIONS:
            data = load_region(feature, lead, region)
            for metric in METRICS:
                plot_diff(data, region, metric, lead, feature)   # diff shown first
                plot_metric(data, region, metric, lead, feature)