#!/bin/bash

output=./outputs
filename=./input
filesize=1000000000

if [ -f $filename ]; then rm -f $filename
if [ -f $output ]; then rm -f $output

make

./produce $filename $filesize

for blocks in 10 20 30
do
    for threads_per_block in 10 50 100
    do
        ./char_freq $filename $blocks $threads_per_block | tee -a $output
        sleep 1
    done
done

make clean