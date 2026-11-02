#!/bin/bash

for x in $(seq 5 5 50);
do
    for y in $(seq 0.0 0.1 0.9);
    do
        echo "$x" >> 'parameter files/SWSF m.txt'
        echo "$y" >> 'parameter files/SWSF mu.txt'
    
    done
done