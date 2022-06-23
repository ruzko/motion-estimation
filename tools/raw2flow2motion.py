#!/bin/python3
# Copyright Jacob Dybvald Ludvigsen, Christoph Rackwitz, 2022
# This is free software, licenced under BSD-3-Clause
#install dependencies: pip3 install numpy rawpy imageio matplotlib memory-profiler opencv-contrib-python

import subprocess # to execute c-program "double"
import linecache # to read long files line by line efficiently
import random # to choose a random image from image range
import numpy as np # to manipulate images as arrays
import rawpy # to convert from raw image format to viewable RGB format
import imageio # to save and read images in various formats
import argparse # to accept command-line input
import cv2 as cv # to use various image manipulations
import matplotlib.pyplot as plt # for plot functionality
from pathlib import Path # to handle directory paths properly
from memory_profiler import profile # for memory benchmarking

#breakpoint()
# Take input, expand to range, convert to list with leading zeroes and return
#@profile
def retFileList():
    fileList = []
    firstFrame = ''
    lastFrame = ''
    parser = argparse.ArgumentParser()
    parser.add_argument(type=int, nargs = 2, action='store', dest='fileIndex', \
          default=False, help='numbers of first and last image files to be read')
    parser.add_argument('-p', '--path', nargs='?', type=Path, dest="srcDirectory", default='/dev/shm/', \
          help='which directory to read images from. Specify with "-p <path-to-folder>" or "--path <path-to-folder". Leave empty for /dev/shm/')
    parser.add_argument('-d', nargs='?', type=str, dest='doLineDoubling', action='store', \
           help='optionally add "-d" if images were recorded with line skips, to stretch lines.') 
    args = parser.parse_args()
    srcDir = args.srcDirectory
    firstFrame, lastFrame = args.fileIndex
    needsLineDoubling = args.doLineDoubling
    r = range(firstFrame, lastFrame)
    fileList = list([*r])
    fileList.append(lastFrame)
    fileListMap = map(str, fileList)
    numberList = [str(x).zfill(4) for x in list(fileList)]
    fileList = ["out."+str(x)+".raw" for x in list(numberList)]
    return fileList, numberList, srcDir, needsLineDoubling


rawList, numberList, srcDir, needsDoubling = retFileList()
imagePath = str(srcDir)
#breakpoint()

# prepend headers to rawfiles if they don't already have a header
hf = open('/home/Jacob/Dokumenter/03-Skole/01 - UiT universitet/År 3/08-Bacheloroppgave/pi-media/raw/hd0.32k', 'rb')
header = hf.read()
hf.close()
for x in list(rawList):
    with open(imagePath + '/' +x, 'rb') as rawFile: partialRaw = rawFile.read(32) # read first 32 blocks of raw
    if header != partialRaw: # check whether the first 32 blocks of the rawfile is identical to the header
        with open(imagePath  + '/' + x, 'rb') as original: data = original.read()
        with open(imagePath + '/hd.' + x, 'wb') as modified: modified.write(header + data)

# list with modified filenames

# breaking list into chunks
chunk_size = 10 #images held in memory at once
chunked_rawList = [rawList[i:i+chunk_size] for i in range(0, len(rawList), chunk_size)]
chunked_numberList = [numberList[i:i+chunk_size] for i in range(0, len(numberList), chunk_size)]

headedList = [imagePath + '/hd.' + str(x) for x in list(rawList)]
viewableList = []
#breakpoint()

# Convert from raw to viewable format, stretch lines, denoise
# Does denoising of the bayer format image before demosaicing
def convertAndPostProcess():
    grayList = []
    nframes = int(len(headedList))
#    denoiseList = (nframes - 5)
    for (x,y) in zip(headedList, numberList):
        numberIndex = numberList.index(y)
        currentImage = (imagePath + '/img.'+ str(y) +'.tiff')
        viewableList.append(currentImage)
        with rawpy.imread(x) as raw:
            rgb = raw.postprocess(fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full, no_auto_bright=False, output_bps=8)
            grayframe = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
            if needsDoubling == '-d':
                subprocess.Popen(double, currentImage)
        #if numberIndex < 5 or numberIndex > denoiseList: # denoise images individually
        cleanImage = cv.fastNlMeansDenoising(src=grayframe, \
                   h=3, templateWindowSize=7, searchWindowSize=21)
        #else: # I want to use this better denoising method, but I couldn't get it to work
        #    with grayList as grays: # denoise using neighbouring images as template
        #        cleanImage = cv.fastNlMeansDenoisingMulti(srcImgs=grayList, imgToDenoiseIndex=numberIndex, \
        #        temporalWindowSize=3, h=3, templateWindowSize=7, searchWindowSize=21)
        imageio.imwrite(currentImage, cleanImage)
#    frames = np.array(grayList)
    return nframes


# get number of frames and list with viewable filenames, check dimensions
nframes = convertAndPostProcess()
random_frame = random.choice(viewableList)
testFrame = cv.imread(random_frame, cv.IMREAD_GRAYSCALE)
height, width = testFrame.shape


# divide filelist into manageable chunks
chunked_viewableList = [viewableList[i:i+chunk_size] for i in range(0, len(viewableList), chunk_size)]

# counting pixels
max_filament_speed = 140 #mm/s
pixels_per_mm = 611 # estimated by counting pixels between edges of known object
max_filament_speed = pixels_per_mm * max_filament_speed # px/s
max_filament_speed = max_filament_speed / 1000000 # conversion to px/microsecond (px/s *s/1 000 000 us)

def calcGFTTShift(fimg1, fimg2):
    frame1=cv.imread(fimg1, 0)
    frame2=cv.imread(fimg2, 0)
    pts1=cv.goodFeaturesToTrack(frame1, 1000, 0.01, 30)
    pts2=cv.goodFeaturesToTrack(frame2, 1000, 0.01, 30)
    nextPts, status, err = cv.calcOpticalFlowPyrLK(frame1, frame2, pts1, pts2)
    # print status
    pts1Good=pts1[ status==1 ]
    # pts1Good=np.reshape(pts1Good, (pts1Good.shape[0],1,pts1Good.shape[1]))
    nextPtsG=nextPts[ status==1 ]
    # nextPtsG=np.reshape(nextPtsG, (nextPtsG.shape[0],1,nextPtsG.shape[1]))
    # T=cv2.estimateRigidTransform(pts1Good, nextPtsG, True)
    T,msk=cv.findHomography(pts1Good, nextPtsG, cv.USAC_MAGSAC)
    dx,dy=T[0,2],T[1,2]
    dxy=(dx,dy)
    tmp=np.zeros((frame1.shape[0], frame1.shape[1], 3), np.uint8)
    frame2_shift=np.roll(frame2, int(np.floor(-dxy[0])), 1)
    fram2_shift=np.roll(frame2_shift, int(np.floor(-dxy[1])), 0)
    tmp[:,:,2]=frame1
    tmp[:,:,1]=frame2_shift
    tmp[:,:,0]=0
    return dx, dy



for i in list(viewableList):
    nextImg = next(iter(viewableList))
    dx, dy = calcGFTTShift(i, nextImg)
    print ('dx: ' + str(dx), 'dy: ' + str(dy))
