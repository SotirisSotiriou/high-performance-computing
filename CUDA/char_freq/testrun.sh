#!/bin/bash

output=./outputs.txt
filename=./input.txt

for blocks in 10 50 100
do
    for threads_per_block in 10 50 100
    do
        ./char_freq.exe $filename $blocks $threads_per_block | tee -a $output
        sleep 1
    done
done
