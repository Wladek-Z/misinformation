#!/bin/bash
filename=$1

for x in $(seq 5 5 50); do
    echo "$x" >> "$HOME/repos/misinformation/parameter files/${filename}"
done