#!/bin/bash

# Define the mapping between file types and scripts
declare -A script_map
script_map[srf]="script_model_temperature.py script_model_polar.py script_model_province.py"
script_map[nonpolar]="script_model_layer.py script_model_layer2.py"
script_map[epi-nonpolar]="script_model_npp.py script_model_no3.py script_model_cflux.py"

matrix_files=(
    "Matrix_MX_srf.tsv"
    "Matrix_M0_srf.tsv"
    "Matrix_M1_srf.tsv"
    "Matrix_guidi_srf.tsv"
    "Matrix_salazar_srf.tsv"
    "Matrix_stress_srf.tsv"
    "Matrix_MX_nonpolar.tsv"
    "Matrix_M0_nonpolar.tsv"
    "Matrix_M1_nonpolar.tsv"
    "Matrix_guidi_nonpolar.tsv"
    "Matrix_salazar_nonpolar.tsv"
    "Matrix_stress_nonpolar.tsv"
    "Matrix_MX_epi-nonpolar.tsv"
    "Matrix_M0_epi-nonpolar.tsv"
    "Matrix_M1_epi-nonpolar.tsv"
    "Matrix_guidi_epi-nonpolar.tsv"
    "Matrix_salazar_epi-nonpolar.tsv"
    "Matrix_stress_epi-nonpolar.tsv"
)

for matrix_file in "${matrix_files[@]}"; do
    # Extract the type from the file name
    if [[ $matrix_file == *"srf.tsv" ]]; then
        matrix_type="srf"
    elif [[ $matrix_file == *"nonpolar.tsv" ]]; then
        matrix_type="nonpolar"
    elif [[ $matrix_file == *"epi-nonpolar.tsv" ]]; then
        matrix_type="epi-nonpolar"
    else
        echo "Unknown matrix type for file: $matrix_file"
        continue
    fi

    # Get the corresponding scripts
    scripts=${script_map[$matrix_type]}

    # Submit the job for each script
    for script in $scripts; do
        sbatch standar.sh $script $matrix_file
    done
done
