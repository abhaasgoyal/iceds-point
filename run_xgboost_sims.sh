#!/bin/bash
# Usage:
#   bash run_sims.sh               # submit sims, then plots once they all finish
#   bash run_sims.sh --plots-only  # submit only the plot job (uses existing results)

# ---- choose params here ----
SEASON=wet            # dry | wet   (ignored when FORECAST_MONTHLY=true)
FORECAST=false        # true = forecast, false = hindcast (ignored when FORECAST_MONTHLY=true)
BIAS_CORRECTION=false # true | false (false = --no_bias_correction in sims)
# Single-month forecasts: one set of sims per target month, output to forecast_month/<season>/<YYYY-MM>/
FORECAST_MONTHLY=true # true | false
FORECAST_MONTHS="2026-10 2026-11 2026-12 2027-01 2027-02 2027-03"
LAST_INIT=2026-10     # last indexing (init) month available in the GCM data
# ----------------------------

# Plots read from the same folder the sims write to:
# BIAS_CORRECTION=false -> bias_analysis/before, true -> bias_analysis/after
if [ "$BIAS_CORRECTION" = "true" ]; then BIAS=after; else BIAS=before; fi

# Wet season = Oct-Mar, dry season = Apr-Sep (matches prepare_dataset)
season_for_month() {
    local m=$((10#${1#*-}))
    if [ "$m" -ge 10 ] || [ "$m" -le 3 ]; then echo wet; else echo dry; fi
}

# YYYY-MM -> months since year 0 (for month arithmetic)
month_index() {
    local y=${1%-*} m=$((10#${1#*-}))
    echo $((y * 12 + m - 1))
}

RUN_VARS="RUN_SEASON=$SEASON,RUN_FORECAST=$FORECAST,RUN_BIAS_CORRECTION=$BIAS_CORRECTION"
if [ "$FORECAST_MONTHLY" = "true" ]; then
    # months passed colon-separated: qsub -v can't carry spaces or commas in a value
    PLOT_VARS="PLOT_FORECAST_MONTHLY=true,PLOT_FORECAST_MONTHS=$(echo $FORECAST_MONTHS | tr ' ' ':'),PLOT_BIAS=$BIAS"
else
    PLOT_VARS="PLOT_FORECAST_MONTHLY=false,PLOT_SEASON=$SEASON,PLOT_FORECAST=$FORECAST,PLOT_BIAS=$BIAS"
fi

if [ "$1" = "--plots-only" ]; then
    plot_id=$(qsub -v $PLOT_VARS run_plots.sh)
    echo "submitted plots $plot_id"
    exit 0
fi

job_ids=""

submit_sims() {
    # $1 = extra -v vars (comma-prefixed), $2 = label for echo,
    # $3 = target month YYYY-MM (optional, single-month forecast mode only)
    for region in 3
    do
        for lead in {1..6}
        do
            if [ -n "$3" ]; then
                # lead L for target T needs a forecast initialised in T - (L-1),
                # so skip leads whose init month is after LAST_INIT
                gap=$(( $(month_index "$3") - $(month_index "$LAST_INIT") ))
                if [ "$lead" -le "$gap" ]; then
                    echo "skipping $3 lead $lead (needs init after $LAST_INIT)"
                    continue
                fi
            fi
            for feature in cbrt # log
            do
                for bc_exclude_percentile in 95 # 97 100
                do
                    job_id=$(qsub -v XGBOOST_REGION=$region,XGBOOST_LEAD=$lead,XGBOOST_FEATURE="$feature",XGBOOST_BC_EXCLUDE_PERCENTILE=$bc_exclude_percentile$1 run_xgboost_sim.sh)
                    echo "submitted sim $job_id (region $region, lead $lead, $feature, perc $bc_exclude_percentile, $2, bc=$BIAS_CORRECTION)"
                    job_ids="$job_ids:$job_id"
                done
            done
        done
    done
}

if [ "$FORECAST_MONTHLY" = "true" ]; then
    # [ -d forecast_month ] && find forecast_month -type f \( -name '*.csv' -o -name '*.png' \) -delete

    for month in $FORECAST_MONTHS
    do
        month_season=$(season_for_month "$month")
        submit_sims ",RUN_SEASON=$month_season,RUN_FORECAST=true,RUN_BIAS_CORRECTION=$BIAS_CORRECTION,XGBOOST_FORECAST_MONTH=$month" \
                    "month $month, $month_season" "$month"
    done

else
    # find hindcast forecast -type f \( -name '*.csv' -o -name '*.png' \) -delete

    submit_sims ",$RUN_VARS" "$SEASON, forecast=$FORECAST"
fi

# plots run only after every sim above finishes successfully
if [ -n "$job_ids" ]; then
    plot_id=$(qsub -W depend=afterok$job_ids -v $PLOT_VARS run_plots.sh)
    echo "submitted plots $plot_id (waits on all sims)"
else
    plot_id=$(qsub -v $PLOT_VARS run_plots.sh)
    echo "submitted plots $plot_id (no sims submitted)"
fi