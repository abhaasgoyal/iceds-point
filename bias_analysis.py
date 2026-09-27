import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

is_forecast = True
season = "wet"
forecast = "forecast" if is_forecast else "hindcast"

SRC_DIR = f"/scratch/tm70/ag9761/iceds_datasets/{forecast}/{season}"
BEFORE_DIR = f"{SRC_DIR}/bias_analysis/before"
AFTER_DIR = f"{SRC_DIR}/bias_analysis/after"
OUT_DIR = f"{SRC_DIR}/bias_analysis/histograms"
BOX_OUT_DIR = f"{SRC_DIR}/bias_analysis/boxplots"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(BOX_OUT_DIR, exist_ok=True)

REGIONS = [3]  # Queensland region
LEAD = [1, 2, 3, 4, 5, 6]
FEATURE = ["cbrt"]

OTHER_MODELS = ["ecmwf", "ukmo", "access"]  # the "3 models"
BIN_COUNT = 40


def parse_point(s):
    nums = re.findall(r"float64\((-?\d+\.?\d*)\)", s)
    lat, lon = float(nums[0]), float(nums[1])
    return lat, lon


def load_bias(folder, feature, lead, region):
    src = f"{folder}/output_region{region}_lead{lead}_feature{feature}_perc95.0.csv"
    data = pd.read_csv(src)
    data["lat"], data["lon"] = zip(*data["Point"].map(parse_point))
    return data


def get_bias_data(region, lead, feature):
    """Load before/after bias columns for the 3 models, plus XGBoost after."""
    before = load_bias(BEFORE_DIR, feature, lead, region)
    after = load_bias(AFTER_DIR, feature, lead, region)

    before_bias = {m: before[f"{m}_bias"].astype(float).dropna() for m in OTHER_MODELS}
    after_bias = {m: after[f"{m}_bias"].astype(float).dropna() for m in OTHER_MODELS}
    after_bias["XGBoost"] = after["XGBoost_bias"].astype(float).dropna()
    return before_bias, after_bias


def plot_bias_histograms(region, lead, feature):
    """Two-row histogram of per-point bias.

    Top row (after bias correction): XGBoost, ecmwf, ukmo, access -> 4 panels.
    Bottom row (before bias correction): ecmwf, ukmo, access only -> 3 panels,
    centered under the top row (under the last 3 top-row columns), with the
    leftmost bottom slot left blank.
    """
    before_bias, after_bias = get_bias_data(region, lead, feature)

    # shared bins/x-range across all panels so the histograms are visually comparable
    all_vals = np.concatenate(
        [v.values for v in before_bias.values()] + [v.values for v in after_bias.values()]
    )
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    bins = np.linspace(vmin, vmax, BIN_COUNT + 1)

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(nrows=2, ncols=4)

    top_order = ["XGBoost", "ecmwf", "ukmo", "access"]
    top_color = "#4C72B0"
    xgb_color = "#C44E52"

    # top row: after bias correction (4 panels)
    for j, model in enumerate(top_order):
        ax = fig.add_subplot(gs[0, j])
        vals = after_bias[model]
        color = xgb_color if model == "XGBoost" else top_color
        ax.hist(vals, bins=bins, color=color, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(vals.mean(), color="black", linewidth=1.2)
        ax.set_title(f"{model if model == "XGBoost" else model.upper()} (after)\nmean bias: {vals.mean():.3f}")
        ax.set_xlim(vmin, vmax)

    # bottom row: before bias correction (3 panels, aligned under the top
    # row's ecmwf/ukmo/access columns; leftmost slot, under XGBoost, is blank)
    for j, model in enumerate(OTHER_MODELS):
        ax = fig.add_subplot(gs[1, j + 1])
        vals = before_bias[model]
        ax.hist(vals, bins=bins, color="#8C8C8C", edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(vals.mean(), color="black", linewidth=1.2)
        ax.set_title(f"{model if model == "XGBoost" else model.upper()} (before)\nmean bias: {vals.mean():.3f}")
        ax.set_xlim(vmin, vmax)

    fig.suptitle(
        f"Bias distribution before vs. after correction "
        f"(Queensland - Region {region}, lead {lead}, {feature.upper()})"
    )
    plt.tight_layout()
    out_path = f"{OUT_DIR}/bias_hist_{feature}_lead{lead}_region{region}.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_bias_boxplots(region, lead, feature):
    """Same panel layout as plot_bias_histograms, but seaborn boxplots
    instead of histograms, saved as a separate figure."""
    before_bias, after_bias = get_bias_data(region, lead, feature)

    # Separate y-axis scales for the "after" and "before" rows: the before
    # panels have much larger outliers than the after panels, and sharing one
    # range across both rows squashes the after panels flat. Each row is
    # scaled to its own values instead.
    after_vals = np.concatenate([v.values for v in after_bias.values()])
    after_vmin, after_vmax = np.nanmin(after_vals), np.nanmax(after_vals)
    after_pad = 0.05 * (after_vmax - after_vmin)

    before_vals = np.concatenate([v.values for v in before_bias.values()])
    before_vmin, before_vmax = np.nanmin(before_vals), np.nanmax(before_vals)
    before_pad = 0.05 * (before_vmax - before_vmin)

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(nrows=2, ncols=4)

    top_order = ["XGBoost", "ecmwf", "ukmo", "access"]
    top_color = "#4C72B0"
    xgb_color = "#C44E52"

    # top row: after bias correction (4 panels)
    for j, model in enumerate(top_order):
        ax = fig.add_subplot(gs[0, j])
        vals = after_bias[model]
        color = xgb_color if model == "XGBoost" else top_color
        sns.boxplot(y=vals, ax=ax, color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{model if model == "XGBoost" else model.upper()} (after)\nmean bias: {vals.mean():.3f}")
        ax.set_ylim(after_vmin - after_pad, after_vmax + after_pad)
        ax.set_ylabel("bias" if j == 0 else "")

    # bottom row: before bias correction (3 panels, aligned under the top
    # row's ecmwf/ukmo/access columns; leftmost slot, under XGBoost, is blank)
    for j, model in enumerate(OTHER_MODELS):
        ax = fig.add_subplot(gs[1, j + 1])
        vals = before_bias[model]
        sns.boxplot(y=vals, ax=ax, color="#8C8C8C")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{model if model == "XGBoost" else model.upper()} (before)\nmean bias: {vals.mean():.3f}")
        ax.set_ylim(before_vmin - before_pad, before_vmax + before_pad)
        ax.set_ylabel("")

    fig.suptitle(
        f"Bias boxplots before vs. after correction "
        f"(Queensland - Region {region}, lead {lead}, {feature.upper()})"
    )
    plt.tight_layout()
    out_path = f"{BOX_OUT_DIR}/bias_box_{feature}_lead{lead}_region{region}.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


for feature in FEATURE:
    for lead in LEAD:
        for region in REGIONS:
            plot_bias_histograms(region, lead, feature)
            plot_bias_boxplots(region, lead, feature)