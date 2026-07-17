#!/bin/bash
#PBS -l wd
#PBS -l ncpus=18
#PBS -l mem=30GB
#PBS -l walltime=6:00:00
#PBS -q normal
#PBS -P tm70
#PBS -j oe
#PBS -m e
#PBS -l storage=gdata/xp65+gdata/tm70

source ~/.bashrc

module use /g/data/xp65/public/modules
module load conda/analysis3

conda activate iceds

echo "Region $XGBOOST_REGION"
echo "Lead Time $XGBOOST_LEAD"
echo "Feature $XGBOOST_FEATURE"

python ensemble_xgboost_nearest.py -r $XGBOOST_REGION -l $XGBOOST_LEAD -f $XGBOOST_FEATURE

