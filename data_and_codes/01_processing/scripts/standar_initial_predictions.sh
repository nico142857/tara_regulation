#!/bin/bash
#SBATCH --job-name=Job
#SBATCH --partition=largemem
#SBATCH --output=logs7/out.%a.%N.%j.out
#SBATCH --error=logs7/out.%a.%N.%j.err
##SBATCH --mail-user=
##SBATCH --mail-type=ALL
#SBATCH -n 1 # number of jobs
#SBATCH -c 1 # number of cpu
##SBATCH --array=1-24
#SBATCH --mem=5G

# command line process down here.

python initial_prediction_tfs_genes.py
