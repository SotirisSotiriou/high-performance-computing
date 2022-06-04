#!/bin/bash

output=./outputs.txt
filename=./input.txt
pattern=pi
filesize=1000000

make

./produce $filename $filesize

for blocks in 10 50 100
do
    for threads_per_block in 10 50 100
    do
        ./string_matching.exe $filename $pattern $blocks $threads_per_block | tee -a $output
        sleep 1
    done
done

make clean