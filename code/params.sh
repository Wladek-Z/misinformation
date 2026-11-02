#!/bin/bash
filename=$1

for x in $(seq 0.04 0.08 0.20); do
    echo "$x" >> "parameter files/ER p mini.txt"
done