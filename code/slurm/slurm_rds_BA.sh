#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 12:00:00
#SBATCH --job-name BA_RDS
#
#######################################

N=5000
p=5
s=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/epidemic_sim.py -N $N -p $p -m BA --run -s $s -i 10 -j 10 -n 4999 1 0 0 0