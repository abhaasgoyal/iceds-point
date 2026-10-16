#!/bin/bash
# Usage:
#   bash run_sims.sh               # submit sims, then plots once they all finish
#   bash run_sims.sh --plots-only  # submit only the plot job (uses existing results)

# ---- choose params here ----
SEASON=wet            # dry | wet
FORECAST=false         # true = forecast, false = hindcast
BIAS_CORRECTION=false # true | false (false = --no_bias_correction in sims)
# No need for below mostly
BIAS=after            # before | after (for plot_heatmap.py / plot_lead_time_results.py)
# ----------------------------

RUN_VARS="RUN_SEASON=$SEASON,RUN_FORECAST=$FORECAST,RUN_BIAS_CORRECTION=$BIAS_CORRECTION"
PLOT_VARS="PLOT_SEASON=$SEASON,PLOT_FORECAST=$FORECAST,PLOT_BIAS=$BIAS"

if [ "$1" = "--plots-only" ]; then
    plot_id=$(qsub -v $PLOT_VARS run_plots.sh)
    echo "submitted plots $plot_id"
    exit 0
fi

job_ids=""

for region in 3
do
    for lead in {1..6}
    do
        for feature in cbrt # log
        do
            for bc_exclude_percentile in 95 # 97 100
            do
                job_id=$(qsub -v XGBOOST_REGION=$region,XGBOOST_LEAD=$lead,XGBOOST_FEATURE="$feature",XGBOOST_BC_EXCLUDE_PERCENTILE=$bc_exclude_percentile,$RUN_VARS run_xgboost_sim.sh)
                echo "submitted sim $job_id (region $region, lead $lead, $feature, perc $bc_exclude_percentile, $SEASON, forecast=$FORECAST, bc=$BIAS_CORRECTION)"
                job_ids="$job_ids:$job_id"
            done
        done
    done
done

# plots run only after every sim above finishes successfully
plot_id=$(qsub -W depend=afterok$job_ids -v $PLOT_VARS run_plots.sh)
echo "submitted plots $plot_id (waits on all sims)"