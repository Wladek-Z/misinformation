#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 12:00:00
#SBATCH --job-name REG
#SBATCH --array=1-20
#
#######################################

N=3500

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "./parameter_files/REG d.txt")

echo "Task $SLURM_ARRAY_TASK_ID: p=$x"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/network_gen.py -N $N -p "$x" -m REG --samples