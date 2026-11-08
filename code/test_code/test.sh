#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 2:00:00
#SBATCH --job-name networks
#SBATCH --array=1-20
#
#######################################

echo "Hello"

N=5000
new_dir="graphs/data/ER/uncombined"

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" "code/parameter files/ER p.txt")

echo "Task $SLURM_ARRAY_TASK_ID: p=$x"

mkdir -p $new_dir

source activate base
conda activate mphys

echo "It works!"