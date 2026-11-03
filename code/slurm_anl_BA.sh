#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 2:00:00
#SBATCH --job-name BA_anl
#
#######################################

N=5000

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/network_anl.py -N $N -p "parameter files/BA m extra.txt" -m BA --mean