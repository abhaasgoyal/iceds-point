#!/bin/bash
#PBS -l wd
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -l walltime=3:00:00
#PBS -q normal
#PBS -P tm70
#PBS -j oe
#PBS -m e
#PBS -l storage=gdata/xp65+gdata/tm70

source ~/.bashrc

module use /g/data/xp65/public/modules
module load conda/analysis3
conda deactivate

conda activate iceds
echo $(which python)

# stop on the first failed step from here on (the conda lines above can
# print harmless errors under PBS, so they're kept outside set -e)
set -e

PYTHON=/scratch/tm70/ag9761/conda/envs/iceds/bin/python

# Wet season = Oct-Mar, dry season = Apr-Sep (matches prepare_dataset) (must match run_sims.sh)
season_for_month() {
    local m=$((10#${1#*-}))
    if [ "$m" -ge 10 ] || [ "$m" -le 3 ]; then echo wet; else echo dry; fi
}

run_all_plots() {
    # $1 = season, remaining args = extra flags (--forecast)
    local season=$1; shift
    $PYTHON bias_analysis.py --season "$season" "$@"
    $PYTHON plot_heatmap.py --season "$season" "$@" --bias "$PLOT_BIAS"
    $PYTHON plot_lead_time_results.py --season "$season" "$@" --bias "$PLOT_BIAS"
}

echo "Bias     $PLOT_BIAS"

if [ "$PLOT_FORECAST_MONTHLY" = "true" ]; then
    # months arrive colon-separated (qsub -v can't carry spaces/commas)
    MONTHS=$(echo "$PLOT_FORECAST_MONTHS" | tr ':' ' ')
    echo "Monthly forecast plots for: $MONTHS"
    for month in $MONTHS
    do
        month_season=$(season_for_month "$month")
        echo "---- $month ($month_season) ----"
        # forecast-month CSVs hold predicted values (no metrics), so only the
        # 2x2 XGBoost-vs-GCM value heatmaps apply here
        $PYTHON plot_heatmap.py --season "$month_season" --forecast_month "$month" --bias "$PLOT_BIAS"
    done
else
    echo "Season   $PLOT_SEASON"
    echo "Forecast $PLOT_FORECAST"

    FORECAST_FLAGS=()
    if [ "$PLOT_FORECAST" = "true" ]; then
        FORECAST_FLAGS=(--forecast)
    fi

    run_all_plots "$PLOT_SEASON" "${FORECAST_FLAGS[@]}"
fi