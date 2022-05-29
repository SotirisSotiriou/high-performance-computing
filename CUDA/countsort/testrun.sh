#!/bin/bash

output=./outputs.txt

for size in 100000 200000 300000 400000
do
    for blocks in 10 50 100
    do
        for threads_per_block in 10 50 100
        do
            ./countsort.exe $size $blocks $threads_per_block | tee -a $output
            sleep 1
        done
    done
done
