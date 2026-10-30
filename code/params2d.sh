#!/bin/bash

for x in $(seq 5 5 50);
do
    for y in $(seq 0.01 0.09 0.82);
    do
        echo "$x" >> 'parameter files/DCSB m.txt'
        echo "$y" >> 'parameter files/DCSB nu.txt'
    
    done
done