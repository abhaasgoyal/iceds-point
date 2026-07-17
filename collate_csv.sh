#!/bin/bash
set -euo pipefail

DATA_DIR="/scratch/tm70/ag9761/iceds_datasets"
OUT_DIR="${DATA_DIR}"   # change if you want outputs elsewhere

leads=(1 3 6)
features=(cbrt log)

for lead in "${leads[@]}"; do
    for feature in "${features[@]}"; do

        pattern="${DATA_DIR}/output_region*_lead${lead}_feature${feature}.csv"
        outfile="${OUT_DIR}/combined_lead${lead}_feature${feature}.csv"

        # Check that at least one matching file exists before running awk
        if compgen -G "${pattern}" > /dev/null; then
            echo "Processing lead=${lead}, feature=${feature} -> ${outfile}"
            awk 'NR == 1 || FNR > 1' ${pattern} > "${outfile}"
        else
            echo "No files found for lead=${lead}, feature=${feature}, skipping."
        fi

    done
done

echo "Done."
