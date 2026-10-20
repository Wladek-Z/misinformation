#!/bin/bash
filename=$1

for x in $(seq 0.001 0.001 0.02); do
    echo "$x" >> "$HOME/repos/misinformation/parameter files/${filename}"
done