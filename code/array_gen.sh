#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 2:00:00
#SBATCH --job-name networks
#SBATCH --array=1-10
#
#######################################

N=5000

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "code/parameter files/BA m.txt")

echo "Task $SLURM_ARRAY_TASK_ID: p=$x"

source activate base
conda activate mphys

python ./code/network_gen.py -N $N -p "$x" -n 10 -m BA --samples