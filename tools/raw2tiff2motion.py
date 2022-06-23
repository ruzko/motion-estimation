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
hf = open('/home/Jacob/pi/raw/hd0.32k', 'rb')
header = hf.read()
hf.close()
for x in list(rawList):
    with open(imagePath + '/' +x, 'rb') as rawFile: partialRaw = rawFile.read(32) # read first 32 blocks of raw
    if header != partialRaw: # check whether the first 32 blocks of the rawfile is identical to the header
        with open(imagePath  + '/' + x, 'rb') as original: data = original.read()
        with open(imagePath + '/hd.' + x, 'wb') as modified: modified.write(header + data)

# list with modified filenames
headedList = [imagePath + '/hd.' + str(x) for x in list(rawList)]
#breakpoint()

# Convert from raw to viewable format, stretch lines, denoise
# Does denoising of the bayer format image before demosaicing
def convertAndPostProcess():
    grayList = []
    viewableList = []
    nframes = int(len(headedList))
    denoiseList = (nframes - 5)
    
    for (x,y) in zip(headedList, numberList):
        numberIndex = numberList.index(y)
        currentImage = (imagePath + '/img.'+ y +'.tiff')
#        viewableList.append(currentImage)
        with rawpy.imread(x) as raw:
            rgb = raw.postprocess(fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full, no_auto_bright=False, output_bps=8)
            grayframe = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
            if needsDoubling == '-d':
                subprocess.Popen(double, currentImage)
#        grayList.append(grayframe)
        #if numberIndex < 5 or numberIndex > denoiseList: # denoise images individually
        cleanImage = cv.fastNlMeansDenoising(src=grayframe, \
                   h=3, templateWindowSize=7, searchWindowSize=21)
        #else:
        #    with grayList as grays: # denoise using neighbouring images as template
        #        cleanImage = cv.fastNlMeansDenoisingMulti(srcImgs=grayList, imgToDenoiseIndex=numberIndex, \
        #        temporalWindowSize=3, h=3, templateWindowSize=7, searchWindowSize=21)
        imageio.imwrite(currentImage, cleanImage)
    frames = np.array(grayList)
    return nframes, viewableList, frames


# split dataset based on filesize


# get number of frames and list with viewable filenames, check dimensions
nframes, viewableList, frames = convertAndPostProcess()
#breakpoint()
random_frame = random.choice(viewableList)
testFrame = cv.imread(random_frame, cv.IMREAD_GRAYSCALE)
height, width = testFrame.shape


# counting pixels
max_filament_speed = 140 #mm/s
pixels_per_mm = 611 # estimated by counting pixels between edges of known object
max_filament_speed = pixels_per_mm * max_filament_speed # px/s
max_filament_speed = max_filament_speed / 1000000 # conversion to px/microsecond (px/s *s/1 000 000 us)
toothx = 160 # pixels, ~4px/tooth, width/4, 640/4

# calculate good points to track
#@profile
def calculate_shift(f1, f2, normalize=True):
    assert f1.shape == f2.shape
    (height, width) = f1.shape[:2]
    if normalize:
        f1 = f1.astype('f4') - f1.mean()
        f2 = f2.astype('f4') - f2.mean()
    scores = cv.filter2D(src=f1, kernel=f2, ddepth=cv.CV_32F)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(scores)
    shift = np.float32(maxLoc) - np.array([width, height])/2.0

    return shift

    print("shift:", shift)

    scores -= scores.min()
    scores /= scores.max()
#    imshow(scores)
    canvas = cv.cvtColor(scores, cv.COLOR_GRAY2BGR)
    cv.circle(canvas, center=maxLoc, radius=3, thickness=cv.FILLED, color=(0,0,1))
#    imshow(canvas)

#upscale frames to have more data
scale = 2
scaled_frames = []
for frame in frames:

    scaled_frame = cv.resize(frame, fx=scale, fy=scale, dsize=None, interpolation=cv.INTER_LINEAR)
    scaled_frame = scaled_frame.astype("f4")
    scaled_frame /= scaled_frame.max()
    scaled_frame -= scaled_frame.mean()
    scaled_frames.append(scaled_frame)
scaled_frames = np.array(scaled_frames)





shiftsum = np.float32([0,0])
shifts = [(0,0)]
for k in range(1, nframes):
    f1 = scaled_frames[k-1]
    f2 = scaled_frames[k]
    shift = calculate_shift(f1, f2, normalize=False) / scale
    shifts.append(shift)
    shiftsum += shift
    print(k, 'shift', shift, 'sum', shiftsum)

print("total:", shiftsum)
cshift = np.cumsum(shifts, axis=0)



#xmax = 1 * toothx
anchor_index = 0
anchor_pos = np.float32([0,0])

breakpoint()
abs_shifts = []
#@profile
def find_good_keyframe(current_frame, tolerance=5):
    # test from k-1 down for frames that didn't move much vs the previous one
    k = current_frame
    while k > 0:
        k -= 1
        if k == 0: break # nothing better

        relshift = abs_shifts[k] - abs_shifts[k-1]
        if np.linalg.norm(relshift) <= tolerance:
            break

    return k
#breakpoint()
k = 0
while k < nframes:
    anchor = scaled_frames[anchor_index]
    frame = scaled_frames[k]
    relshift = calculate_shift(anchor, frame, normalize=False) / scale
    (relx, rely) = relshift
    #assosiate timestamps to images
    if k == 0:
        timestamp = 1 #should be zero, but set as 1 to avoid devision by zero. temporary workaround.
    else:
        line = linecache.getline(imagePath + "/tstamps.csv", k+1) #fetch specific line from cached file, an efficient method.  since k is 0-indexed and getline is 1-indexed, we must increment with k+1
        timestamp = line.split(",")[0] # store whatever comes before comma in the specific line as timestamp. microsecond format
#    print (timestamp)
    relx = relx / (int(timestamp)) #converting from non-timebound relative motion to timebound (seconds) relative motion
    xmax = max_filament_speed * (int(timestamp))
    keep_going = True
    if abs(relx) > xmax:
        assert k > 0 # because that would be ridiculous, autocorrelation not saying "0"
        # new anchor: previous frame, repeat calculation
        print(f"{k}: rel {relshift} + {anchor_pos} = abs {anchor_pos + relshift}, finding new anchor")
        new_anchor_index = find_good_keyframe(k)

        keep_going = (new_anchor_index == anchor_index)
        if not keep_going: # new anchor
            anchor_index = new_anchor_index
            anchor_pos = abs_shifts[anchor_index]
            print(f"new anchor is {anchor_index} at {anchor_pos}")
            # k stays, don't "keep going", recalculate against new anchor

    if keep_going: # ok OR we have to overextend
        absshift = anchor_pos + relshift
        abs_shifts.append(absshift)
        shiftdiff = abs_shifts[-1] - abs_shifts[-2] if k > 0 else [0,0]
        dist = np.linalg.norm(shiftdiff)
        candidate = (dist <= 5)
        print(f"{k}: rel {relshift} + {anchor_pos} = abs {anchor_pos + relshift}, moved by {dist:.1f}", "USELESS" * (not candidate))
        k += 1
        assert len(abs_shifts) == k


assert len(abs_shifts) == nframes
abs_shifts = np.array(abs_shifts)
cshift = abs_shifts

## Code block for associating timestamp with corresponding frame and position,
## to display velocity




## End of code block

plt.figure(figsize=(12,8))
plt.plot(np.array(abs_shifts)[:,0], c='red')
plt.xlabel('frame index', fontsize=12)
plt.ylabel('lateral motion', fontsize=12)
plt.show()


plt.figure(figsize=(12,8))
plt.plot(np.array(abs_shifts)[:,1], c='green')
plt.xlabel('frame index', fontsize=12)
plt.ylabel('twisting motion', fontsize=12)
plt.show()


