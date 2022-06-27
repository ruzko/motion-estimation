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


# prepend headers to rawfiles if they don't already have a header
def checkRawHeader ():
    hf = open('/home/Jacob/Dokumenter/03-Skole/01 - UiT universitet/År 3/08-Bacheloroppgave/pi-media/raw/hd0.32k', 'rb')
    header = hf.read()
    hf.close()
    for x in list(rawList):
        with open(imagePath + '/' +x, 'rb') as rawFile: partialRaw = rawFile.read(32) # read first 32 blocks of raw
        if header != partialRaw: # check whether the first 32 blocks of the rawfile is identical to the header
            with open(imagePath  + '/' + x, 'rb') as original: data = original.read()
            with open(imagePath + '/hd.' + x, 'wb') as modified: modified.write(header + data)
    return




# breaking list into chunks

chunk_size = 10 #images held in memory at once
chunked_rawList = [rawList[i:i+chunk_size] for i in range(0, len(rawList), chunk_size)]
chunked_numberList = [numberList[i:i+chunk_size] for i in range(0, len(numberList), chunk_size)]


# list with files which have a header
headedList = [imagePath + '/hd.' + str(x) for x in list(rawList)]
viewableList = []



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


# Instantiating stores for values
velocity_list_x = []
velocity_list_y = []
orb_vel_list_x = []
orb_vel_list_y = []


def calc_feature_shift(currentFrame, nextFrame):
    # sanity checks
#    if currentFrame == nextFrame:
#        return None

    frame1 = cv.imread(currentFrame, 0)
    frame2 = cv.imread(nextFrame, 0)
    pts1 = cv.goodFeaturesToTrack(frame1, 600, 0.01, 10)
    pts2 = cv.goodFeaturesToTrack(frame2, 600, 0.01, 10)
    nextPts, status, err = cv.calcOpticalFlowPyrLK(frame1, frame2, pts1, pts2)
    pts1Good = pts1[ status==1 ]
    #pts1Good=np.reshape(pts1Good, (pts1Good.shape[0],1,pts1Good.shape[1]))
    nextPtsG = nextPts[ status==1 ]
    #nextPtsG=np.reshape(nextPtsG, (nextPtsG.shape[0],1,nextPtsG.shape[1]))
    matrixTransform, status = cv.estimateAffinePartial2D(pts1, nextPts)
    #T,msk=cv.findHomography(pts1Good, nextPtsG, cv.USAC_MAGSAC)
    dx, dy = matrixTransform[0,2],matrixTransform[1,2] # get third element of first and second row
#    tmp = np.zeros((frame1.shape[0], frame1.shape[1], 3), np.uint8)
#    frame2_shift = np.roll(frame2, int(np.floor(-dxy[0])), 1)
#    frame2_shift = np.roll(frame2_shift, int(np.floor(-dxy[1])), 0)
#    tmp[:,:,2] = frame1
#    tmp[:,:,1] = frame2_shift
#    tmp[:,:,0] = 0
    return dx, dy




#breakpoint()
def calc_ORB_shift(currentFrame, nextFrame):
    # initialize ORB detector algo
    orb = cv.ORB_create(nfeatures=500, edgeThreshold=3, patchSize=3)

    # Read images
    frame1 = cv.imread(currentFrame, 0)
    frame2 = cv.imread(nextFrame, 0)

    # Detect keypoints and compute descriptors for currentFrame and nextFrame
    kpts1, descriptors1 = orb.detectAndCompute(frame1,None)
    kpts2, descriptors2 = orb.detectAndCompute(frame2,None)

    # initialize matcher for keypoints, then do matching
    matcher = cv.BFMatcher.create(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1,descriptors2)

    # Sort matches by score (distance)
    matches = sorted(matches, key=lambda x:x.distance)
    # Remove bad matches with worse than 15% match
 #   numGoodMatches = int(len(matches) * 0.15
 #   matches = matches[0:numGoodMatches]
#    print (matches)
#    readableMatches = map(str, matches)
#    print(readableMatches)
    # Extract location of good matches
#    points1 = np.zeros((len(matches), 2), dtype=np.float32)
#    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    src_pts  = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
#    for i, match in enumerate(matches):

#        points1[i, :] = keypoints1[match.queryIdx].pt
#        points2[i, :] = keypoints2[match.trainIdx].pt


    matches = np.array(matches)
    # Calculate shift / flow
#    nextPts, status, err = cv.calcOpticalFlowPyrLK(frame1, frame2, kpts1, kpts2)
    matrixTransform, status = cv.estimateAffinePartial2D(src_pts, dst_pts)

    dx, dy = matrixTransform[0,2],matrixTransform[1,2] # get third element of first and second row
    # combine to final image containing matched keypoints
#    final_img = cv.drawMatches(query_img, queryKeypoints,
#    train_img, trainKeypoints, matches[:20],None)
# Draw first 10 matches.
 #   img3 = cv.drawMatches(frame1,kpts1,frame2,kpts2,matches[:60],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    plt.imshow(img3),plt.show()

    return dx, dy


outInformation = []
k = 0

for i in list(viewableList):
    nextImg = next(iter(viewableList))
    dx, dy = calc_feature_shift(i, nextImg)
    odx, ody = calc_ORB_shift(i, nextImg)
    # assosiate timestamps to images
    if k == 0:
        timestamp = 1 #should be zero, but set as 1 to avoid devision by zero. temporary workaround.
    else:
        line = linecache.getline(imagePath + "/tstamps.csv", k+1) # fetch specific line from cached file, an efficient method.
                                                                  # since k is 0-indexed and getline is 1-indexed, we must increment with k+1
        timestamp = line.split(",")[0] # store whatever comes before comma in the specific line as timestamp. microsecond format
#    print (timestamp)
    dx, dy = dx / (int(timestamp)), dy / (int(timestamp)) #converting from non-timebound relative motion to timebound (seconds) relative motion
    xmax = max_filament_speed * (int(timestamp))
    k += 1
    velocity_list_x.append(dx)
    velocity_list_y.append(dy)
    orb_vel_list_x.append(odx)
    orb_vel_list_y.append(ody)

# GFTT_shift
print ('GFTT dx: \n', velocity_list_x, '\n GFTT dy: \n', velocity_list_y)


# ORB_shift

print ('ORB dx: \n', orb_vel_list_x, '\n ORB dy: \n', orb_vel_list_y)


plt.figure(figsize=(12,8))
plt.plot(velocity_list_x, c='red')
plt.xlabel('frame index', fontsize=12)
plt.ylabel('lateral motion, GFTT', fontsize=12)
plt.show()


plt.figure(figsize=(12,8))
plt.plot(velocity_list_y, c='green')
plt.xlabel('frame index', fontsize=12)
plt.ylabel('twisting motion, GFTT', fontsize=12)
plt.show()


plt.figure(figsize=(12,8))
plt.plot(velocity_list_x, c='red')
plt.xlabel('frame index', fontsize=12)
plt.ylabel('lateral motion, ORB', fontsize=12)
plt.show()


plt.figure(figsize=(12,8))
plt.plot(velocity_list_y, c='green')
plt.xlabel('frame index', fontsize=12)
plt.ylabel('twisting motion, ORB', fontsize=12)
plt.show()
