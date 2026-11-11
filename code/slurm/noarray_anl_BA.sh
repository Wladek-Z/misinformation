#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 8G
#SBATCH --time 12:00:00
#SBATCH --job-name BA_anl
#
#######################################

N=5000

x=500

echo "Task $SLURM_TASK_ID: p=$x"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/network_anl.py -N $N -p "$x" -m BA --all