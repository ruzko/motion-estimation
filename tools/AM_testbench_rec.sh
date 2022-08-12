#!/bin/bash

if [ "$1" = "" ]; then echo "format: `basename $0` [recording time in milliseconds]"; exit; fi

#echo "removing /dev/shm/out.*.raw"
#rm -rf /dev/shm/*

#echo "creating files for timestamp and headers"
#touch /dev/shm/tstamps.csv
#touch /dev/shm/hd0.32k

#echo "capturing frames for ${1}ms with 600fps requested"
raspivid -md 7 -t $1 --fps 180 -w 64 -w 640 -pts $2_tstamps.txt --flush -lev 4.2 -ex off -awb off -ag 6 -dg 0.2 -awbg 1.5,1.9 -ss 5000 --initial pause --signal -o $2.h264

#echo "frame delta time[us] distribution"
#cut -f1 -d, /dev/shm/tstamps.csv | sort -n | uniq -c


