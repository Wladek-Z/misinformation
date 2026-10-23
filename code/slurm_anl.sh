#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 2:00:00
#SBATCH --job-name networks
#
#######################################

N=5000

source activate base
conda activate mphys

python ./code/network_anl.py -N $N -p "parameter files/BA m.txt" -m BA --mean