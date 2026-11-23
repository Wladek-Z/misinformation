#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 12:00:00
#SBATCH --job-name BA_RDS
#SBATCH --array=1-10
#
#######################################

N=5000
s=1
x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "parameter_files/BA m.txt")

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/epidemic_sim.py -N $N -p $x -m BA --run -s $s -i 1 -j 30 -n 4999 1 0 0 0