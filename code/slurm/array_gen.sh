#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 2:00:00
#SBATCH --job-name ER
#SBATCH --array=1-3
#
#######################################

N=100

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "./parameter_files/ER p mini.txt")

echo "Task $SLURM_ARRAY_TASK_ID: p=$x"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/network_gen.py -N $N -p "$x" -n 1 -m ER --samples --mini