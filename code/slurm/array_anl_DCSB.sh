#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 12:00:00
#SBATCH --job-name DCSB_anl
#SBATCH --array=1-100
#
#######################################

N=5000

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "parameter files/DCSB m.txt")
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "parameter files/DCSB nu.txt")

echo "Task $SLURM_ARRAY_TASK_ID: p=$x q=$y"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/network_anl.py -N $N -p "$x" -q "$y" -m DCSB --all -K 10