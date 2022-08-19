#!/bin/bash

if [ "$1" = "" ]; then echo "format: `basename $0` [recording time in milliseconds]"; exit; fi


raspivid -md 7 -t $1 -fps 180 -h 64 -w 640 -pts $2_tstamps.txt --flush -lev 4.2 -ex off -awb off -ag 1 -dg 0.2 -awbg 1.5,1.9 -ss 4600 --initial pause --signal --nopreview -o $2.h264

sleep $1

process_id=$(pgrep raspivid)

#tell raspivid to pause recording and quit
kill -SIGUSR1 $process_id
sleep 1
kill -SIGUSR2 $process_id
