#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 4G
#SBATCH --time 2:00:00
#SBATCH --job-name SWSF_anl
#
#######################################

N=5000

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/network_anl.py -N $N -p "parameter files/SWSF m.txt" -m SWSF --mean2d -q "parameter files/SWSF mu.txt"