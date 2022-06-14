#!/bin/env python3

import numpy as np
import rawpy
import imageio
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from memory_profiler import profile


# Take input, convert to list with leading zeroes and  return
@profile
def retFileList():
    fileList = []
    parser = argparse.ArgumentParser()
    parser.add_argument(type=int, nargs = 2, action='store', dest='fileIndex', default=False, help='numbes of first and last image files to be read')
    parser.add_argument(type=Path, dest="srcDirectory", default='/dev/shm/', help='which directory to read images from. Leave empty for /dev/shm/')
    args = parser.parse_args()
    srcDir = args.srcDirectory
    fileList = args.fileIndex
    fileList.extend(range(*fileList))
    fileList = map(str, fileList)
    numberList = [str(x).zfill(4) for x in list(fileList)]
    fileList = ["out."+str(x)+".raw" for x in list(fileList)]
    return fileList, numberList, srcDir



# prepend headers to rawfiles if they don't already have a header
rawList, numberList, srcDir = retFileList()
imagePath = str(srcDir)

hf = open('/home/Jacob/pi/raw/hd0.32k', 'rb')
header = hf.read()
hf.close()
for x in list(rawList):
    with open(imagePath + '/' +x, 'rb') as rawFile: partialRaw = rawFile.read(32)
    if header != partialRaw:
        with open(imagePath  + '/' + x, 'rb') as original: data = original.read()
        with open(imagePath + '/hd.' + x, 'wb') as modified: modified.write(header + data)

# list with modified filenames
headedList = [imagePath + '/hd.' + str(x) for x in list(rawList)]


# Convert from raw to viewable format and save
#for (x,y) in zip(headedList, numberList):
def rawRaws():
    rawArray = []
    rawArrayWithNoise = []
    for (x,y) in zip(headedList, numberList):
        with rawpy.imread(x) as raw:
            rawArrayWithNoise = raw.raw_image
            newRawArray = raw.raw_image_visible
            rawArray = np.vstack((rawArray, newRawArray))
        rawArray = rawArray.copy()
    return rawArray

rawArray = rawRaws()
rawMean = np.mean(rawArray, axis=0, dtype=np.float64)
info = np.finfo(rawMean.dtype) # Get the information of the incoming image type
rawMean = rawMean.astype('uint8'). # normalize the data to 0 - 1
#rawMean = 255 * 255 # Now scale by 255
with rawpy.imread(rawMean) as raw:
    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=8)
    img_copy = raw.copy()
#img = img_copy.astype(np.uint8)
print (img_copy)
#rgb = img.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=8)
cv2.imshow("Window", rgb)
grayframe = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
        
#    imageio.imsave(imagePath + '/img.'+y+'.tiff', grayframe)

# list with viewable filenames
#viewableList = [imagePath + '/img.'+str(x)+'.tiff' for x in list(numberList)]


# calculate mean of still pictures to assert a "ground truth" image
#noisy = [cv.imread(x) for x in list(viewableList)]

#cleanImage = cv.fastNlMeansDenoisingMulti(srcImgs=noisy, imgToDenoiseIndex=10, temporalWindowSize=5, h=3, templateWindowSize=5, searchWindowSize=17)
imageio.imsave('%s/cleanImage.tiff' % imagePath, grayframe)
