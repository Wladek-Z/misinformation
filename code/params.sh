#!/bin/bash

for x in $(seq 0.25 0.25 4);
do
    for y in $(seq 0.5 0.5 8);
    do
        echo "$x" >> x.txt
        echo "$y" >> y.txt
    
    done
done