import argparse
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

HOME_DIR = "/scratch/tm70/ag9761/iceds_datasets"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mean metrics vs lead time for each model."
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
        "--forecast_month",
        default=None,
        help="Single forecast month YYYY-MM; reads from forecast_month/<season>/<YYYY-MM>/ "
             "(implies --forecast)",
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

if args.forecast_month:
    BASE_DIR = f"{HOME_DIR}/forecast_month/{season}/{args.forecast_month}"
    TITLE_SUFFIX = f", forecast {args.forecast_month}"
else:
    out_dir_cast = "forecast" if is_forecast else "hindcast"
    BASE_DIR = f"{HOME_DIR}/{out_dir_cast}/{season}"
    TITLE_SUFFIX = ""

SRC_DIR = f"{BASE_DIR}/bias_analysis/{bias}"
OUT_DIR = f"{BASE_DIR}/bias_analysis/{bias}/lead_plots_all"

os.makedirs(OUT_DIR, exist_ok=True)

REGIONS = [3]
LEAD = [1, 2, 3, 4, 5, 6]
FEATURE = ["cbrt"]
MODELS = ["XGBoost", "ecmwf", "ukmo", "access"]
METRICS = ["mae", "rmse"]

def model_label(model):
    return model if model == "XGBoost" else model.upper()

def parse_point(s):
    nums = re.findall(r"float64\((-?\d+\.?\d*)\)", s)
    lat, lon = float(nums[0]), float(nums[1])
    return lat, lon


def load_region(feature, lead, region):
    src = f"{SRC_DIR}/output_region{region}_lead{lead}_feature{feature}_perc95.0.csv"
    data = pd.read_csv(src)
    data["lat"], data["lon"] = zip(*data["Point"].map(parse_point))
    return data


def get_region_data(feature, lead):
    """Load all regions for a given feature/lead, applying the same
    region-2 row filtering used in the original script."""
    region_data = {region: load_region(feature, lead, region) for region in REGIONS}
    return region_data


def build_summary():
    """Build a tidy dataframe: feature, lead, region, model, metric, mean_value."""
    rows = []
    for feature in FEATURE:
        for lead in LEAD:
            region_data = get_region_data(feature, lead)
            for region, data in region_data.items():
                for metric in METRICS:
                    cols = [f"{m}_{metric}" for m in MODELS]
                    result = data[cols].astype(float)
                    result.columns = MODELS
                    means = result.mean()
                    for model in MODELS:
                        rows.append({
                            "feature": feature,
                            "lead": lead,
                            "region": region,
                            "metric": metric,
                            "model": model,
                            "mean_value": means[model],
                        })
    return pd.DataFrame(rows)


def plot_metric_vs_lead(summary, feature, metric):
    """One figure per feature/metric, with one subplot per region,
    showing mean metric vs lead time, one line per model."""
    sub = summary[(summary["feature"] == feature) & (summary["metric"] == metric)]
    regions_sorted = sorted(sub["region"].unique())
    n_regions = len(regions_sorted)

    fig, axes = plt.subplots(
        nrows=1, ncols=n_regions,
        figsize=(6 * n_regions, 5), sharey=True
    )
    if n_regions == 1:
        axes = [axes]

    fig.suptitle(
        f"Mean {metric.upper()} vs Lead Time ({feature.upper()}{TITLE_SUFFIX})",
        fontsize=16
    )

    for ax, region in zip(axes, regions_sorted):
        region_sub = sub[sub["region"] == region]
        region_sub = region_sub.assign(model=region_sub["model"].map(model_label))
        sns.lineplot(
            data=region_sub, x="lead", y="mean_value", hue="model",
            marker="o", ax=ax
        )
        ax.set_title(f"Region {region}")
        ax.set_xlabel("Lead time")
        ax.set_ylabel(f"Mean {metric.upper()}")
        ax.set_xticks(LEAD)
        ax.legend(title="Model")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = f"{OUT_DIR}/{metric}_vs_lead_{feature}.png"
    plt.savefig(out_path, dpi=300)
    plt.clf()
    plt.close(fig)
    print(f"saved {out_path}")


def plot_metric_vs_lead_combined(summary, feature, metric):
    """Single figure per feature/metric, averaging across all regions,
    one line per model."""
    sub = summary[(summary["feature"] == feature) & (summary["metric"] == metric)]
    overall = sub.groupby(["lead", "model"], as_index=False)["mean_value"].mean()
    overall = overall.assign(model=overall["model"].map(model_label))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=overall, x="lead", y="mean_value", hue="model",
        marker="o", ax=ax
    )
    ax.set_title(f"Mean {metric.upper()} vs Lead Time ({feature.upper()}, all regions{TITLE_SUFFIX})")
    ax.set_xlabel("Lead time")
    ax.set_ylabel(f"Mean {metric.upper()}")
    ax.set_xticks(LEAD)
    ax.legend(title="Model")

    plt.tight_layout()
    out_path = f"{OUT_DIR}/{metric}_vs_lead_{feature}_allregions.png"
    plt.savefig(out_path, dpi=300)
    plt.clf()
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    summary = build_summary()
    summary.to_csv(f"{OUT_DIR}/summary_mean_metrics.csv", index=False)

    for feature in FEATURE:
        for metric in METRICS:
            plot_metric_vs_lead(summary, feature, metric)          # per-region subplots
            if len(REGIONS) > 1:
                plot_metric_vs_lead_combined(summary, feature, metric)  # averaged across regions