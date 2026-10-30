#!/bin/bash
#PBS -l wd
#PBS -l ncpus=26
#PBS -l mem=80GB
#PBS -l walltime=6:00:00
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

echo "Region $XGBOOST_REGION"
echo "Lead Time $XGBOOST_LEAD"
echo "Feature $XGBOOST_FEATURE"
echo "BC exclude percentile $XGBOOST_BC_EXCLUDE_PERCENTILE"
echo "Season $RUN_SEASON"
echo "Forecast $RUN_FORECAST"
echo "Bias correction $RUN_BIAS_CORRECTION"
echo "Forecast month ${XGBOOST_FORECAST_MONTH:-<all>}"

EXTRA_ARGS=()
[ -n "$RUN_SEASON" ] && EXTRA_ARGS+=(--season "$RUN_SEASON")
[ "$RUN_FORECAST" = "true" ] && EXTRA_ARGS+=(--forecast)
[ "$RUN_BIAS_CORRECTION" = "false" ] && EXTRA_ARGS+=(--no_bias_correction)
[ -n "$XGBOOST_FORECAST_MONTH" ] && EXTRA_ARGS+=(--forecast_month "$XGBOOST_FORECAST_MONTH")

/scratch/tm70/ag9761/conda/envs/iceds/bin/python ensemble_xgboost_nearest.py \
    -r "$XGBOOST_REGION" -l "$XGBOOST_LEAD" -f "$XGBOOST_FEATURE" \
    --bc_extrapolate_extremes \
    --bc_exclude_percentile "$XGBOOST_BC_EXCLUDE_PERCENTILE" \
    "${EXTRA_ARGS[@]}"