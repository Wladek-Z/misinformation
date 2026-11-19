#!/bin/bash

for x in $(seq 0 1 999); do
    echo "$x" >> "parameter_files/1000.txt"
done