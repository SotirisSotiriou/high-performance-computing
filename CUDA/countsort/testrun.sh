#!/bin/bash

output=./outputs

if [ -f $output ]; then rm -f $output

make

for size in 100000 200000 300000 400000
do
    for blocks in 10 50 100
    do
        for threads_per_block in 10 50 100
        do
            ./countsort $size $blocks $threads_per_block | tee -a $output
            sleep 1
        done
    done
done

make clean
