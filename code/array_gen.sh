#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 2:00:00
#SBATCH --job-name networks
#SBATCH --array=1-20
#
#######################################

N=5000

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "code/parameter files/ER p.txt")

echo "Task $SLURM_ARRAY_TASK_ID: p=$x"

source activate base
conda activate mphys

python ./code/network_anl.py -N $N -n 10 -p "$x" -m ER --all