#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 12:00:00
#SBATCH --job-name ER_anl
#SBATCH --array=1-22
#
#######################################

N=5000

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "parameter files/ER p extra.txt")

echo "Task $SLURM_ARRAY_TASK_ID: p=$x"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/network_anl.py -N $N -p "$x" -m ER --all