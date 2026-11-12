#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 12:00:00
#SBATCH --job-name DCSB
#SBATCH --array=1-9
#
#######################################

N=100

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "./parameter_files/DCSB m mini.txt")
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "./parameter_files/DCSB nu mini.txt")

echo "Task $SLURM_ARRAY_TASK_ID: p=$x q=$y"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/network_gen.py -N $N -p "$x" -q "$y" -n 1 -m DCSB --samples --mini -K 4