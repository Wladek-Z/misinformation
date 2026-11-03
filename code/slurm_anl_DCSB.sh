#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 4G
#SBATCH --time 2:00:00
#SBATCH --job-name DCSB_anl
#
#######################################

N=5000

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mphys

python ./code/network_anl.py -N $N -p "parameter files/DCSB m.txt" -m DCSB --mean2d -q "parameter files/DCSB nu.txt" -K 10