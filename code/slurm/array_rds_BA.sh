#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 12:00:00
#SBATCH --job-name BA_RDS
#SBATCH --array=1-1000
#
#######################################

N=5000
p=25
s=1
x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "parameter_files/1000.txt")

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/epidemic_sim.py -N $N -p $p -m BA --run -s $s -i 10 -j $x --specj -n 4999 1 0 0 0