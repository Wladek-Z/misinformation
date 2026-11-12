#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 2:00:00
#SBATCH --job-name networks
#SBATCH --array=1-2
#
#######################################

N=5000

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "./parameter files/REG d.txt")

echo "Task $SLURM_ARRAY_TASK_ID: p=$x"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/network_gen.py -N $N -p "$x" -n 1 -m REG --samples