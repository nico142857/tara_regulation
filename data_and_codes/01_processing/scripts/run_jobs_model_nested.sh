# Define the mapping between file types and script
declare -A script_map
script_map[srf]="script_model_temperature_nested.py script_model_polar_nested.py script_model_province_nested.py"
script_map[nonpolar]="script_model_layer_nested.py script_model_layer2_nested.py"
script_map[epi-nonpolar]="script_model_npp_nested.py script_model_no3_nested.py script_model_cflux_nested.py"

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

total_jobs=0

for matrix_file in "${matrix_files[@]}"; do
    # Extract the type from the file name
    if [[ $matrix_file == *"_srf.tsv" ]]; then
        matrix_type="srf"
    elif [[ $matrix_file == *"_nonpolar.tsv" ]]; then
        matrix_type="nonpolar"
    elif [[ $matrix_file == *"_epi-nonpolar.tsv" ]]; then
        matrix_type="epi-nonpolar"
    else
        echo "Unknown matrix type for file: $matrix_file"
        continue
    fi

    # Get the corresponding scripts
    scripts=${script_map[$matrix_type]}

    # Submit the job for each script
    for script in $scripts; do
        echo "Submitting job for script: $script with matrix file: $matrix_file"
        sbatch standar_models.sh "$script" "$matrix_file"
        total_jobs=$((total_jobs + 1))
    done
done

echo "Total jobs submitted: $total_jobs"