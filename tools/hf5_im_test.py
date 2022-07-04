#!/bin/python3
# Copyright Jacob Dybvald Ludvigsen, Christoph Rackwitz, 2022
# This is free software, licenced under BSD-3-Clause
#install dependencies:
# python3 -m pip install numpy rawpy imageio matplotlib opencv-contrib-python 

import h5py
from math import sqrt
from PIL import Image, ImageEnhance
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

chunk_size = 150 #images held in memory at once

#numberList = [numberList[i:i+chunk_size] for i in range(0, len(numberList), chunk_size)]
#breakpoint()

# list with files which have a header
headedList = [imagePath + '/hd.' + str(x) for x in list(rawList)]
#viewableList = []
#grayList= []


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
                img_array = np.array(grayframe)
        total_rows = grayframe.shape[0]
        total_columns = grayframe.shape[1]

        # first file; create the dummy dataset with no max shape
        if n == 0:
            noisy_dataset = f.create_dataset("noisy_images", 
                      shape=(len(numberList), total_rows, total_columns), 
                      maxshape=(len(numberList), total_rows, total_columns),
                      chunks = True, dtype='uint8')

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








# Convert from raw to viewable format, stretch lines, denoise
# Does denoising of the bayer format image before demosaicing
def convertAndPostProcess():
#    grayList = []
    nframes = int(len(headedList))
    denoiseList = (nframes - 5)
    for (x,y) in zip(headedList, numberList):
        numberIndex = numberList.index(y)
        currentImage = (imagePath + '/img.'+ str(y) +'.tiff')
        viewableList.append(currentImage)


        if needsDoubling == '-d':
            subprocess.Popen(double, currentImage)
        if numberIndex < 5 or numberIndex > denoiseList: # denoise images individually
            cleanImage = cv.fastNlMeansDenoising(src=grayframe, \
                   h=3, templateWindowSize=7, searchWindowSize=21)
        else: # I want to use this better denoising method, but I couldn't get it to work
              # denoise using neighbouring images as template
            cleanImage = cv.fastNlMeansDenoisingMulti(srcImgs=grayList, 
                           imgToDenoiseIndex=(numberIndex-3), temporalWindowSize=5, 
                           h=4, templateWindowSize=7, searchWindowSize=21)
        imageio.imwrite(currentImage, cleanImage)
        grayList.append(cleanImage)
    return nframes


# get number of frames and list with viewable filenames, check dimensions
#nframes = convertAndPostProcess()



def downsampling(img):
    with Image.open(img) as big_img:
        # increase contrast with a factor of 2.5
        contrast_img = ImageEnhance.Contrast(big_img).enhance(2)
        # reduce resolution / downsample to remove noise
#        small_img = contrast_img.resize((160, 24), Image.Resampling(1)) # LANCZOS algo
        imageio.imwrite(img, contrast_img)
    return




numberIndex = []
# open file read-write
with h5py.File(imagePath + '/images.h5', 'r+') as f:
    # load a slice containing n images from noisy dataset, as grayscale images using PIL
    if len(numberList) >= chunk_size:
        noisy_slice = f['noisy_images'][:chunksize]
    else:
        # get slice with all elements
        noisy_slice = f['noisy_images'][:]

    print(noisy_slice)
    # iterate over slice's first axis to make images from individual layers
    for z in noisy_slice:


#        im = Image.fromarray(z, mode='L')
        
    # make a dataset to hold denoised images, so the images don't bleed out due to their
    # denoised neighbours.
#    if numberIndex < 5 or numberIndex > denoiseList: # denoise images individually
        cleanImage = cv.fastNlMeansDenoising(src=z, \
                   h=3, templateWindowSize=7, searchWindowSize=21)
#    else: # I want to use this better denoising method, but I couldn't get it to work
              # denoise using neighbouring images as template
#    cleanImage = cv.fastNlMeansDenoisingMulti(srcImgs=noisy_slice,
#                           imgToDenoiseIndex=(numberIndex-3), temporalWindowSize=5,
#                           h=4, templateWindowSize=7, searchWindowSize=21)

"""
    # first file; create the dummy dataset with no max shape
    if n == 0:
        clean_dataset = f.create_dataset("clean_images",
                      shape=(len(numberList), total_rows, total_columns),
                      maxshape=(len(numberList), total_rows, total_columns),
                      chunks = True, dtype='uint8')

        # stack image array in 0-indexed position of third axis
        f['clean_images'][n,:,:]=img_array

#set attributes for image dataset
    img_dataset.attrs['CLASS'] = 'IMAGE'
    img_dataset.attrs['IMAGE_VERSION'] = '1.2'
    img_dataset.attrs['IMAGE_SUBCLASS'] =  'IMAGE_GRAYSCALE'
    img_dataset.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)
    img_dataset.attrs['IMAGE_WHITE_IS_ZERO'] =  0


"""


"""



# divide filelist into manageable chunks
#chunked_viewableList = [viewableList[i:i+chunk_size] for i in range(0, len(viewableList), chunk_size)]

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
orb_beblid_vel_list_x = []
orb_beblid_vel_list_y = []
#inlier_vector = np.empty((2,32),dtype="double")

feature_params = dict ( qualityLevel = 0.1,
                        minDistance = 2,
                        useHarrisDetector = True,
                        k = 0.04,
                        blockSize=33)

LK_params = dict ( winSize = (21, 7),
                   maxLevel = 4)
                   

estimate_affine_params = dict ( refineIters = 200,
                                method = cv.RANSAC,
                                ransacReprojThreshold = 0.9,
                                maxIters = 20000,
                                confidence = 0.995)

maxCorners = 15000
"""
# detect first features and keypoints, and update when keypoints are lost
def GFTT_detect(img1, n_good_kpts):
    frame = cv.imread(img1, 0)
    z = maxCorners # - n_good_kpts
    new_pts = cv.goodFeaturesToTrack(frame, z, **feature_params)
    totalFeatures = len(new_pts)

    return new_pts, totalFeatures
"""
def calc_feature_shift(currentFrame, nextFrame):

    frame1 = cv.imread(currentFrame, 0)
    frame2 = cv.imread(nextFrame, 0)
    pts1 = cv.goodFeaturesToTrack(frame1, maxCorners, **feature_params)
    pts2 = cv.goodFeaturesToTrack(frame2, maxCorners, **feature_params)
    nextPts, status, err = cv.calcOpticalFlowPyrLK(frame1, frame2, pts1, pts2, **LK_params)
    pts1Good = pts1[ status==1 ]
    #pts1Good=np.reshape(pts1Good, (pts1Good.shape[0],1,pts1Good.shape[1]))
    nextPtsG = nextPts[ status==1 ]
#    num_good_kpts = len(nextPtsG)
#    nextPtsG=np.reshape(nextPtsG, (nextPtsG.shape[0],1,nextPtsG.shape[1]))
    matrixTransform, status = cv.estimateAffinePartial2D(pts1Good, nextPtsG, **estimate_affine_params)
    print(status)
    if matrixTransform is not None:
        dx, dy = matrixTransform[0,2],matrixTransform[1,2] # get third element of first and second row
    else:
        print("No good points to track")
        dx, dy = (0, 0)
    return dx, dy #, num_good_kpts





def ORB_detect(img1, num_good_pts):
    # initialize ORB detector algo
    orb = cv.ORB_create(nfeatures=2000, edgeThreshold=3, patchSize=5)

    # Read images
    frame1 = cv.imread(img1, 0)

    # Detect keypoints and compute descriptors for currentFrame
    kpts, descriptors = orb.detectAndCompute(frame1,None)

    z = maxCorners # - n_good_kpts
    new_pts = cv.goodFeaturesToTrack(frame, z, **feature_params)
    totalFeatures = len(new_pts)

    return new_pts, totalFeatures



#breakpoint()
def calc_ORB_shift(currentFrame, nextFrame):
    # initialize ORB detector algo
    orb = cv.ORB_create(nfeatures=2000, edgeThreshold=3, patchSize=5)

    # Read images
    frame1 = cv.imread(currentFrame, 0)
    frame2 = cv.imread(nextFrame, 0)

    # Detect keypoints and compute descriptors for currentFrame and nextFrame
    kpts1, descriptors1 = orb.detectAndCompute(frame1,None)
    kpts2, descriptors2 = orb.detectAndCompute(frame2,None)

#    kpts1Good = kpts1[ status==1 ]
#    kpts1Good = kpts2[ status==1 ]


    # initialize matcher for keypoints, then do matching
    matcher = cv.BFMatcher.create(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1,descriptors2)

    # Sort matches by score (distance)
    matches = sorted(matches, key=lambda x:x.distance)
    # Remove bad matches with worse than 15% match
#    numGoodMatches = int(len(matches) * 0.15
#    matches = matches[0:numGoodMatches]
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
#    img3 = cv.drawMatches(frame1,kpts1,frame2,kpts2,matches[:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    plt.imshow(img3),plt.show()

    return dx, dy



"""
def ORB_shift_BEBLID(currentFrame, nextFrame):
    # initialize ORB detector algo
    detector = cv.ORB_create(nfeatures=2000, edgeThreshold=3, patchSize=5)

    # Read images
    frame1 = cv.imread(currentFrame, 0)
    frame2 = cv.imread(nextFrame, 0)

    # Detect keypoints and compute descriptors for currentFrame and nextFrame
    kpts1 = detector.detect(frame1,None)
    kpts2 = detector.detect(frame2,None)

#    kpts1Good = kpts1[ status==1 ]
#    kpts2Good = kpts2[ status==1 ]


    # Compute descriptors for keypoints with improved BEBLID function
    descriptor = cv.xfeatures2d.BEBLID_create(0.75)
    kpts1, desc1 = descriptor.compute(frame1, kpts1)
    kpts2, desc2 = descriptor.compute(frame2, kpts2)

    # initialize matcher for keypoints, then do matching
#    matcher = cv.BFMatcher.create(cv.NORM_HAMMING, crossCheck=True)
#    matches = matcher.match(descriptors1,descriptors2)

    # find homography
#    homography = cv.estimateAffine2D(kpts1, kpts2)
    homography, status = cv.findHomography(kpts1, kpts2)

    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    matched1 = []
    matched2 = []
    nn_match_ratio = 0.8  # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])

    inliers1 = []
    inliers2 = []
    good_matches = []
    inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check
    for i, m in enumerate(matched1):
        # Create the homogeneous point
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt
        # Project from image 1 to image 2
        col = np.dot(homography, col)
        col /= col[2, 0]
        # Calculate euclidean distance
        dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + \
                pow(col[1, 0] - matched2[i].pt[1], 2))
        if dist < inlier_threshold:
            good_matches.append(cv.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])




    # Sort matches by score (distance)
#    matches = sorted(matches, key=lambda x:x.distance)
    # Remove bad matches with worse than 15% match
#    numGoodMatches = int(len(matches) * 0.15
 #   matches = matches[0:numGoodMatches]
#    print (matches)
#    readableMatches = map(str, matches)
#    print(readableMatches)
    # Extract location of good matches
#    points1 = np.zeros((len(matches), 2), dtype=np.float32)
#    points2 = np.zeros((len(matches), 2), dtype=np.float32)

 #   src_pts  = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
 #   dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
#    for i, match in enumerate(matches):

#        points1[i, :] = keypoints1[match.queryIdx].pt
#        points2[i, :] = keypoints2[match.trainIdx].pt


    matches = np.array(matches)
    # Calculate shift / flow
#    nextPts, status, err = cv.calcOpticalFlowPyrLK(frame1, frame2, kpts1, kpts2)
    matrixTransform, status = cv.estimateAffinePartial2D(inliers1, inliers2)

    dx, dy = matrixTransform[0,2],matrixTransform[1,2] # get third element of first and second row
    # combine to final image containing matched keypoints
#    final_img = cv.drawMatches(query_img, queryKeypoints,
#    train_img, trainKeypoints, matches[:20],None)
# Draw first 10 matches.

#    img3 = cv.drawMatches(frame1,kpts1,frame2,kpts2,matches[:60],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    plt.imshow(img3),plt.show()

    return dx, dy





"""
outInformation = []
k = 0
tsList = []
num_good_kpts = 0

for i in list(viewableList):
    # downsample image to reduce noise
    downsampling(i)
    # detect first keypoints and update if 20% of keypoints disappear
#    if k == 0 or num_good_kpts <= (num_kpts * 0.8):
#        kpts, num_kpts = GFTT_detect(i, num_good_kpts)

    nextImg = next(iter(viewableList))
    dx, dy = calc_feature_shift(i, nextImg)
#    orb_dx, orb_dy = calc_ORB_shift(i, nextImg)
#    orb_beblid_dx, orb_beblid_dy = ORB_shift_BEBLID(i, nextImg)

    # assosiate timestamps to images
    if k == 0:
        timestamp = 1 #should be zero, but set as 1 to avoid devision by zero. temporary workaround.
    else:
        line = linecache.getline(imagePath + "/tstamps.csv", k+1) # fetch specific line from cached file, an efficient method.
                                                                  # since k is 0-indexed and getline is 1-indexed, we must increment with k+1
        timestamp = line.split(",")[0] # store whatever comes before comma in the specific line as timestamp. microsecond format
        tsList.append(timestamp)
#    print (timestamp)
    vx, vy = dx / (int(timestamp)), dy / (int(timestamp)) #converting from non-timebound relative motion to timebound (seconds) relative motion
#    orb_vx, orb_vy = orb_dx / (int(timestamp)), orb_dy / (int(timestamp))
#    orb_beblid_vx, orb_beblid_vy = orb_beblid_dx / (int(timestamp)), orb_beblid_dy / (int(timestamp))

    xmax = max_filament_speed * (int(timestamp))

    k += 1
    velocity_list_x.append(vx)
    velocity_list_y.append(vy)
#    orb_vel_list_x.append(orb_vx)
#    orb_vel_list_y.append(orb_vy)
#    orb_beblid_vel_list_x.append(orb_beblid_vx)
#    orb_beblid_vel_list_y.append(orb_beblid_vy)


# GFTT_shift
print ('GFTT dx: \n', velocity_list_x, '\n GFTT dy: \n', velocity_list_y)


# ORB_shift

#print ('ORB dx: \n', orb_vel_list_x, '\n ORB dy: \n', orb_vel_list_y)



# ORB_BEBLID_shift

#print ('ORB + BEBLID vx: \n', orb_beblid_vel_list_x, '\n ORB + BEBLID vy: \n', orb_beblid_vel_list_y)



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


plt.figure(figsize=(12,8))
plt.plot(velocity_list_x, c='red')
plt.xlabel('timestamp us', fontsize=12)
plt.ylabel('lateral motion, ORB', fontsize=12)
#plt.xticks(x, tsList, rotation=45)
plt.show()


plt.figure(figsize=(12,8))
plt.plot(velocity_list_y, c='green')
plt.xlabel('timestamp us', fontsize=12)
plt.ylabel('twisting motion, ORB', fontsize=12)
#plt.xticks(x, tsList, rotation=45)
plt.show()

"""

