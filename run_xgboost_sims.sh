#!/bin/bash

for region in {0..2}
do
    for lead in 1 3 6 
    do
	for feature in cbrt log 
	do
	    qsub -v XGBOOST_REGION=$region,XGBOOST_LEAD=$lead,XGBOOST_FEATURE="$feature" run_xgboost_sim.sh
	done
    done
done
