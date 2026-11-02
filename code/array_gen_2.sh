#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 12:00:00
#SBATCH --job-name networks
#SBATCH --array=1-100
#
#######################################

N=5000

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "./parameter files/DCSB m.txt")
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "./parameter files/DCSB nu.txt")

echo "Task $SLURM_ARRAY_TASK_ID: p=$x q=$y"

source activate base
conda activate mphys

python ./code/network_gen.py -N $N -p "$x" -q "$y" -n 10 -m DCSB --samples