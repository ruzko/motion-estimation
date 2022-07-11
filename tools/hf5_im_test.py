#!/bin/env python3
# Copyright Jacob Dybvald Ludvigsen, 2022
# This is free software, licenced under BSD-3-Clause
#install dependencies:
# python3 -m pip install numpy rawpy imageio matplotlib opencv-contrib-python h5py

#import pydegensac
import h5py # to enable high-performance file handling
#from numba import jit, njit # to compile code for quicker execution
#import multiprocessing # to run multiple instances of time-consuming processes
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
#from memory_profiler import profile # for memory benchmarking




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
    imagePath = str(srcDir)
    return fileList, numberList, imagePath, needsLineDoubling



rawList, numberList, imagePath, needsDoubling = retFileList()

#breakpoint()
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

chunk_size = 200 #images held in memory at once

#numberList = [numberList[i:i+chunk_size] for i in range(0, len(numberList), chunk_size)]
#breakpoint()

# list with files which have a header
headedList = [imagePath + '/hd.' + str(x) for x in list(rawList)]

denoiseNum = len(numberList)

random_frame = random.choice(headedList)
with rawpy.imread(random_frame) as raw:
    height, width = raw.raw_image_visible.shape



hf5_params = dict(shape=(len(numberList), height, width),
                  maxshape=(len(numberList), height, width),
                  chunks = True,
                  dtype = 'uint8')
                  #chunks = (height, width))




# open raw file, denoising of the bayer format image before demosaicing,
# convert raw images to grayscale, save as efficient hdf5 format file
with h5py.File(imagePath + '/images.h5', 'w') as f:
    for n, x in enumerate(headedList):
    # incoming data
        with rawpy.imread(x) as raw:
                rgb = raw.postprocess(
                      fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full,
                      no_auto_bright=False, output_bps=8)
                grayframe = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
                img_array = np.asarray(grayframe)

        # first file; create the dummy dataset with no max shape
        if n == 0:
            noisy_dataset = f.create_dataset("noisy_images", **hf5_params) # compression="lzf", shuffle=True)

        # stack image array in 0-indexed position of third axis
        f['noisy_images'][n,:,:]=img_array

#set attributes for image dataset
    noisy_dataset.attrs['CLASS'] = 'IMAGE'
    noisy_dataset.attrs['IMAGE_VERSION'] = '1.2'
    noisy_dataset.attrs['IMAGE_SUBCLASS'] =  'IMAGE_GRAYSCALE'
    noisy_dataset.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)
    noisy_dataset.attrs['IMAGE_WHITE_IS_ZERO'] =  0
    # print attribute names and values
    for k in f['noisy_images'].attrs.keys():
        attr_value= f['noisy_images'].attrs[k]
        print(k , attr_value)

#breakpoint()

"""
# Convert from raw to viewable format, stretch lines, denoise
# Does denoising of the bayer format image before demosaicing
def convertAndPostProcess():

    if needsDoubling == '-d':
        subprocess.Popen(double, currentImage)
"""



### Increase contrast by equalisizing histogram
def enhance_contrast(image_matrix, bins=256): # https://gist.github.com/msameeruddin/8629aa0bf58521a22bb67ed0fea82fee
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cumulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq







#breakpoint()

def denoising(arrays, numberIndex, num_frames_window):
                      # denoise using some neighbouring images as template
    cleanImageArray = cv.fastNlMeansDenoisingMulti(srcImgs=arrays,
                        imgToDenoiseIndex=numberIndex,     temporalWindowSize=num_frames_window,
                          h=4, templateWindowSize=7,     searchWindowSize=21)
    return cleanImageArray



def denoise_hf5_parallel():
    numberIndex = 0
    # open file read-write
    with h5py.File(imagePath + '/images.h5', 'r+') as f:

        # load a slice containing n images from noisy dataset
        if len(numberList) >= chunk_size:
            noisy_slice = f['noisy_images'][:] #[:chunksize]
        else:
            # get slice with all elements.
            noisy_slice = f['noisy_images'][:]


        print(noisy_slice)
        # iterate over slice's first axis to make images from individual layers
        for z in noisy_slice:

            # increase image contrast
#            cleanImageArray = enhance_contrast(z)


            # denoise images
            if (numberIndex <= 1) or (numberIndex >= (denoiseNum - 2)):
                # denoise two first and last images individually
                cleanImageArray = cv.fastNlMeansDenoising(src=z,
                           h=3, templateWindowSize=7, searchWindowSize=21)

            elif (numberIndex <= 4) or (numberIndex >= (denoiseNum - 4)):
                # denoise using some neighbouring images as template
                cleanImageArray = denoising(noisy_slice, numberIndex, 5)

            elif(numberIndex <= 7) or (numberIndex >= (denoiseNum - 7)):
                # denoise using more neighbouring images as template
                cleanImageArray = denoising(noisy_slice, numberIndex, 9)
            else:
                # denoise using more neighbouring images as template
                cleanImageArray = denoising(noisy_slice, numberIndex, 13)


            # increase image contrast
            cleanImageArray = enhance_contrast(cleanImageArray)


            # make a dataset to hold denoised images, so the images don't bleed     out due to their
            # denoised neighbours.
            # first file; create the dummy dataset with no max shape
            if numberIndex == 0:
                clean_dataset = f.create_dataset("clean_images", **hf5_params) #, compression="lzf", shuffle=True)
#            breakpoint()
#            if numberIndex >= chunk_size or numberIndex == denoiseNum - 1 :
#                # stack image array in 0-indexed position of third axis
#                clean_dataset.write_direct(cleanImageArrays, #np.s_[appendFrom:numberIndex], np.s_[appendFrom:numberIndex]) ##
#                appendFrom = numberIndex + 1
#                del cleanImageArrays
#
#            if "cleanImageArrays" not in locals():
#                cleanImageArrays = np.array([])
#            np.append(cleanImageArrays, cleanImageArray, axis=0)

            f['clean_images'][numberIndex,:,:]=cleanImageArray
            numberIndex += 1

    #set attributes for image dataset
        clean_dataset.attrs['CLASS'] = 'IMAGE'
        clean_dataset.attrs['IMAGE_VERSION'] = '1.2'
        clean_dataset.attrs['IMAGE_SUBCLASS'] =  'IMAGE_GRAYSCALE'
        clean_dataset.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)
        clean_dataset.attrs['IMAGE_WHITE_IS_ZERO'] =  0







### dicts for calc_feature_shift
feature_params = dict ( qualityLevel = 0.1,
                        minDistance = 2,
                        useHarrisDetector = True,
                        k = 0.04,
                        blockSize=33)

LK_params = dict ( winSize = (21, 7),
                   maxLevel = 4)
                   

estimate_affine_params = dict ( refineIters = 400,
                                method = cv.RANSAC,
                                ransacReprojThreshold = 0.5,
                                maxIters = 20000,
                                confidence = 0.998)

maxCorners = 21000



def calc_feature_shift(prevFrame, curFrame):

    pts1 = cv.goodFeaturesToTrack(prevFrame, maxCorners, **feature_params)
    pts2 = cv.goodFeaturesToTrack(curFrame, maxCorners, **feature_params)
    nextPts, status, err = cv.calcOpticalFlowPyrLK(prevFrame, curFrame, pts1, pts2, **LK_params)
    pts1Good = pts1[ status==1 ]
    pts1Good=np.reshape(pts1Good, (pts1Good.shape[0],1,pts1Good.shape[1]))
    nextPtsG = nextPts[ status==1 ]
#    num_good_kpts = len(nextPtsG)
    nextPtsG=np.reshape(nextPtsG, (nextPtsG.shape[0],1,nextPtsG.shape[1]))
    matrixTransform, status = cv.estimateAffinePartial2D(pts1Good, nextPtsG, **estimate_affine_params)
#    print(status)
    if matrixTransform is not None:
        dx, dy = matrixTransform[0,2],matrixTransform[1,2] # get third element of first and second row
    else:
        print("No good points to track")
        dx, dy = (0, 0)
    return dx, dy #, num_good_kpts



### dicts for calc_ORB_shift

# FLANN parameters

###    FLANN_INDEX_LINEAR = 0,
###    FLANN_INDEX_KDTREE = 1,
###    FLANN_INDEX_KMEANS = 2,
###    FLANN_INDEX_COMPOSITE = 3,
###    FLANN_INDEX_KDTREE_SINGLE = 4,
###    FLANN_INDEX_HIERARCHICAL = 5,
###    FLANN_INDEX_LSH = 6,
###    FLANN_INDEX_SAVED = 254,
###    FLANN_INDEX_AUTOTUNED = 255,

FLANN_INDEX_LSH = 6
FLANN_INDEX_AUTOTUNED = 255
FLANN_INDEX_KDTREE = 1
FLANN_DIST_HAMMING = 9

LSH_index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 12, # 12 or 6
                        key_size = 20,     # 20 or 12
                        multi_probe_level = 1,
                        target_precision = 95)

KDTREE_index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                           trees = 16,
                           target_precision = 99)


search_params = dict(checks=100)   # or pass empty dictionary

###    FLANN_DIST_EUCLIDEAN = 1,
###    FLANN_DIST_L2 = 1,
###    FLANN_DIST_MANHATTAN = 2,
###    FLANN_DIST_L1 = 2,
###    FLANN_DIST_MINKOWSKI = 3,
###    FLANN_DIST_MAX   = 4,
###    FLANN_DIST_HIST_INTERSECT   = 5,
###    FLANN_DIST_HELLINGER = 6,
###    FLANN_DIST_CHI_SQUARE = 7,
###    FLANN_DIST_CS         = 7,
###    FLANN_DIST_KULLBACK_LEIBLER  = 8,
###    FLANN_DIST_KL                = 8,
###    FLANN_DIST_HAMMING          = 9,


### pydegensac parameters
pydegensac_params = dict( px_th = 4, # threshold
                          conf = 0.995,
                          max_iters = 2000, #
                          laf_consistensy_coef=0, # check patch for consistency after. Needs special input
                          error_type='sampson',
                          symmetric_error_check=True) # same as crossCheck


ORB_params = dict(nfeatures=10000, # Max number of keypoints detected.
                  edgeThreshold=4, # detecting keypoints along the edge of the image is unstable, therefore we ignore the 4 outermost pixel layers.
                  patchSize=7) # window within to search. Does not overlap at any scale.
#                  WTA_K=4) # number of matches produced for each keypoint. default is 2


HOG_params = dict(_winSize=(32,320), # height x width. winSize is the size of the image cropped to an multiple of the cell size
                  _blockSize=(8,8), # multiple of cellsize
                  _blockStride=(8,8), # equal to blocksize
                  _nbins=9, #
                  _nlevels=128) # max number of detection window increases. default 64


def calc_ORB_shift(prevFrame, curFrame):
    # initialize ORB detector algo
    orb = cv.ORB_create(**ORB_params)

    # initialize Histogram-of-Oriented_gradients descriptor.
#    hog = cv.HOGDescriptor(**HOG_params)
    # Detect keypoints and compute descriptors for currentFrame and nextFrame
    kpts1, descriptors1 = orb.detectAndCompute(prevFrame,None)
    kpts2, descriptors2 = orb.detectAndCompute(curFrame,None)
#    breakpoint()
    flannMatcher = cv.FlannBasedMatcher(KDTREE_index_params, search_params)

    # return k nearest neighbours
    flann_matches = flannMatcher.knnMatch(np.asarray(descriptors1, dtype=np.float32),
                                          np.asarray(descriptors2, dtype=np.float32), k=2)

#    goodmatches = flannMatcher.knnMatch(np.asarray(descriptors1, dtype=np.uint8),
#                                          np.asarray(descriptors2, dtype=np.uint8), k=2)

#    flann_matches = flannMatcher.knnMatch(descriptors1,
#                                         descriptors2, k=2)
    # create BFMatcher object
#    bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True) # HAMMING2 when WTA_K is 3 or 4

    # Match descriptors.
#    bfmatches = bf.match(descriptors1, descriptors2)
#    bfmatches = np.asarray(bfmatches, dtype=np.uint8)
    # Need to draw only good matches, so create a mask
#    breakpoint()

    # Sort them in the order of their distance.
#    bfmatches = sorted(bfmatches, key = lambda x:x.distance)

    goodmatches = []
    # ratio test as per Lowe's paper
    for (m,n) in (flann_matches):
        if m.distance < 0.95 * n.distance:
            goodmatches.append(m)

    first = np.empty((len(goodmatches),2), dtype=np.uint8)
    second = np.empty((len(goodmatches),2), dtype=np.uint8)
#    for i in range(len(goodmatches)):
#        #-- Get the keypoints from the good matches
#        first[i,0] = kpts1[bfmatches[i].queryIdx].pt[0]
#        first[i,1] = kpts1[bfmatches[i].queryIdx].pt[1]
#        second[i,0] = kpts2[bfmatches[i].trainIdx].pt[0]
#        second[i,1] = kpts2[bfmatches[i].trainIdx].pt[1]


#    first = np.empty((len(flann_goodmatches),2), dtype=np.float32)
#    second = np.empty((len(flann_goodmatches),2), dtype=np.float32)
    for i in range(len(goodmatches)):
        #-- Get the keypoints from the good matches
        first[i,0] = kpts1[goodmatches[i].queryIdx].pt[0]
        first[i,1] = kpts1[goodmatches[i].queryIdx].pt[1]
        second[i,0] = kpts2[goodmatches[i].trainIdx].pt[0]
        second[i,1] = kpts2[goodmatches[i].trainIdx].pt[1]

    flann_matrixTransform, fstatus = cv.estimateAffinePartial2D(first, second, **estimate_affine_params)
#    degensac_homography, status = pydegensac.findHomography(first, second, **pydegensac_params)
#    cv.decomposeHomography is needed to extract trabslation from degensac_homography

    if flann_matrixTransform is not None:
        pdx, pdy = flann_matrixTransform[0,2],flann_matrixTransform[1,2] # get third #element of first and second row
#        img3 = cv.drawMatches(prevFrame,kpts1,curFrame,kpts2,goodmatches[:],
#                          None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#        plt.imshow(img3),plt.show()
#    breakpoint()


#    if degensac_homography is not None:
#    degpdx, degpdy = degensac_homography[0,2],degensac_homography[1,2] # get third #element of first and second row
#        img4 = cv.drawMatches(prevFrame,kpts1,curFrame,kpts2,flann_goodmatches[:],
#                          None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#        plt.imshow(img4),plt.show()

    return  pdx, pdy # pdx, pdy



# counting pixels
max_filament_speed = 140 #mm/s
pixels_per_mm = 611 # estimated by counting pixels between edges of known object
max_filament_speed = pixels_per_mm * max_filament_speed # px/s


# Instantiating stores for values
velocity_list_x = []
velocity_list_y = []
orb_vel_list_x = []
orb_vel_list_y = []
old_orb_vx = 0
k = 0
tsList = []

#breakpoint()
denoise_hf5_parallel()

with h5py.File(imagePath + '/images.h5', 'r') as f:
    # load a slice containing n image arrays from clean dataset

    clean_slice = f['clean_images'][:] #[:chunksize]


    print(clean_slice)
    # iterate over slice's first axis to make images from individual layers
    for z in clean_slice:

        if k == 0:
            prevFrame = z
            k += 1
            continue # nothing to do with just the first image array
        else:
            # fetch specific line from cached file,an efficient method.
            # since k is 0-indexed and getline is 1-indexed, we must increment with k+1
            line = linecache.getline(imagePath + "/tstamps.csv", k+1)
            # store whatever comes before comma in the specific line as timestamp. microsecond format
            timestamp = line.split(",")[0]
            timestamp = int(timestamp)/(10E+6)

            tsList.append(timestamp)


#            pdx, pdy = calc_feature_shift(prevFrame, z)
#            mmdx, mmdy = pdx / pixels_per_mm, pdy / pixels_per_mm

            orb_pdx, orb_pdy = calc_ORB_shift(prevFrame, z)
            mm_orb_dx, mm_orb_dy = orb_pdx / pixels_per_mm, orb_pdy / pixels_per_mm

            #converting from non-timebound relative motion to timebound (seconds) relative motion
#            vx, vy = mmdx / timestamp, mmdy / timestamp
            orb_vx, orb_vy = mm_orb_dx / timestamp, mm_orb_dy / timestamp

            xmax = max_filament_speed * timestamp # px/interval
            print("xmax = ", xmax, " pixels for this image interval")





#            velocity_list_x.append(pdx)
#            velocity_list_y.append(pdy)
            orb_vel_list_x.append(orb_vx)
            orb_vel_list_y.append(orb_vy)

            prevFrame = z
            k += 1

"""

            if old_orb_vx != 0:
                if np.abs(orb_vx) > 1.5 * old_orb_vx or np.abs(orb_vx) < 0.5 * old_orb_vx:
                    orb_vx = old_orb_vx

            if np.abs(orb_vx) > xmax:
                orb_vx = old_orb_vx

            old_orb_vx = orb_vx
            print ('ORB vx: \n', orb_vx, '\n ORB vy: \n', orb_vy)


"""




# GFTT_shift
#print ('GFTT dx: \n', velocity_list_x, '\n GFTT dy: \n', velocity_list_y)


# ORB_shift

#print ('ORB vx: \n', orb_vel_list_x, '\n ORB vy: \n', orb_vel_list_y)



# ORB_BEBLID_shift

#print ('ORB + BEBLID vx: \n', orb_beblid_vel_list_x, '\n ORB + BEBLID vy: \n', orb_beblid_vel_list_y)


"""
plt.figure(figsize=(12,8))
plt.plot(velocity_list_x, c='red')
plt.xlabel('timestamp us', fontsize=12)
plt.ylabel('lateral motion, GFTT', fontsize=12)
#plt.xticks(labels=tsList, rotation=45)
plt.show()


plt.figure(figsize=(12,8))
plt.plot(velocity_list_y, c='green')
plt.xlabel('timestamp us', fontsize=12)
plt.ylabel('twisting motion, GFTT', fontsize=12)
#plt.xticks(x, tsList, rotation=45)
plt.show()
"""

plt.figure(figsize=(12,8))
plt.plot(orb_vel_list_x, c='red')
plt.xlabel('timestamp us', fontsize=12)
plt.ylabel('lateral motion, ORB', fontsize=12)
#plt.xticks(x, tsList, rotation=45)
plt.show()


plt.figure(figsize=(12,8))
plt.plot(orb_vel_list_y, c='green')
plt.xlabel('timestamp us', fontsize=12)
plt.ylabel('perpendicular motion, ORB', fontsize=12)
#plt.xticks(x, tsList, rotation=45)
plt.show()

# plt.figure(figsize=(12,8))
# plt.plot(orb_beblid_vel_list_x, c='red')
# plt.xlabel('timestamp us', fontsize=12)
# plt.ylabel('lateral motion, ORB FLANN', fontsize=12)
# #plt.xticks(x, tsList, rotation=45)
# plt.show()
#
#
# plt.figure(figsize=(12,8))
# plt.plot(orb_beblid_vel_list_y, c='green')
# plt.xlabel('timestamp us', fontsize=12)
# plt.ylabel('twisting motion, ORB FLANN', fontsize=12)
# #plt.xticks(x, tsList, rotation=45)
# plt.show()

