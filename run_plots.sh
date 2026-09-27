#!/bin/bash
#PBS -l wd
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -l walltime=1:00:00
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

echo "Season   $PLOT_SEASON"
echo "Forecast $PLOT_FORECAST"
echo "Bias     $PLOT_BIAS"

FORECAST_FLAG=""
if [ "$PLOT_FORECAST" = "true" ]; then
    FORECAST_FLAG="--forecast"
fi

$PYTHON bias_analysis.py --season "$PLOT_SEASON" $FORECAST_FLAG
$PYTHON plot_heatmap.py --season "$PLOT_SEASON" $FORECAST_FLAG --bias "$PLOT_BIAS"
$PYTHON plot_lead_time_results.py --season "$PLOT_SEASON" $FORECAST_FLAG --bias "$PLOT_BIAS"