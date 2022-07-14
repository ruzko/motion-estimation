#!/bin/env python3

# Copyright Jacob Dybvald Ludvigsen, 2022
# you may use this software for any purpose, as long as you include the above Copyright notice,
# and follow the conditions of the licence.
# This is free software, licenced under BSD-3-Clause


#install dependencies:
# python3 -m pip install numpy rawpy imageio matplotlib opencv-contrib-python h5py matplotlib

import csv # for output of data
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
    hf = open('../exampleRaws/IMX219_rawHeader/hd0.32k', 'rb')
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

        # first file; create the dummy dataset with no max shape
        if n == 0:
            noisy_dataset = f.create_dataset("noisy_images", **hf5_params) # compression="lzf", shuffle=True)

        # stack image array in 0-indexed position of third axis
        f['noisy_images'][n,:,:]=grayframe

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
    #print(norm)
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')
    #print("uniform: ", uniform_norm)

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq



### Denoise image numberIndex by comparing with other images in num_frames_window
def denoising(arrays, numberIndex, num_frames_window):

    cleanImageArray = cv.fastNlMeansDenoisingMulti(srcImgs=arrays,
                        imgToDenoiseIndex=numberIndex,     temporalWindowSize=num_frames_window,
                          h=4, templateWindowSize=7,     searchWindowSize=21)
    return cleanImageArray




### Main function for denoising and contrast enhancing of image arrays
def denoise_hf5():
    numberIndex = 0
    # open hdf5 file read-write
    with h5py.File(imagePath + '/images.h5', 'r+') as f:

        # load a slice containing n images from noisy dataset, not yet implemented
        if len(numberList) >= chunk_size:
            noisy_slice = f['noisy_images'][:] #[:chunksize]
        else:
            # get slice with all elements.
            noisy_slice = f['noisy_images'][:]


        print(noisy_slice)
        # iterate over slice's first axis to clean individual layered arrays
        for z in noisy_slice:


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
#

#            # Work on a subportion of the images at a time, to conserve memory
#            if numberIndex >= chunk_size or numberIndex == denoiseNum - 1 :
#                # stack image array in 0-indexed position of third axis
#                clean_dataset.write_direct(cleanImageArrays, #np.s_[appendFrom:numberIndex], np.s_[appendFrom:numberIndex]) ##
#                appendFrom = numberIndex + 1
#                del cleanImageArrays
#
#            if "cleanImageArrays" not in locals():
#                cleanImageArrays = np.array([])
#            np.append(cleanImageArrays, cleanImageArray, axis=0)

            # layer the current array on top of previous array, write to file. Slow.
            # Ideally, number of write processes should be minimized.
            f['clean_images'][numberIndex,:,:]=cleanImageArray
            numberIndex += 1

    #set attributes for image dataset
        clean_dataset.attrs['CLASS'] = 'IMAGE'
        clean_dataset.attrs['IMAGE_VERSION'] = '1.2'
        clean_dataset.attrs['IMAGE_SUBCLASS'] =  'IMAGE_GRAYSCALE'
        clean_dataset.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)
        clean_dataset.attrs['IMAGE_WHITE_IS_ZERO'] =  0




Transform_ECC_params = dict(warpMatrix = np.eye(2, 3, dtype=np.float32), # preparing unity matrix for x- and y- axis
                            motionType = cv.MOTION_TRANSLATION, # only motion in x- and y- axes
                            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1000,  1E-10)) # max iteration count and desired epsilon. Terminates when either is reached.
                            #gaussFiltSize = 5)


# Get total shift in x- and y- direction between two image frames / arrays
def calc_ECC_transform(prevFrame, curFrame):

    # Calculate the transform matrix which must be applied to prevFrame in order to match curFrame
    computedECC, ECCTransform = cv.findTransformECC(prevFrame, curFrame, **Transform_ECC_params)

    # Extract second element of first and second row to be shift in their respective directions
    pdx, pdy = ECCTransform[0,2],ECCTransform[1,2]

    # I think computedECC is the confidence that the transform matrix fits.
    print("\n\nECC confidence of transform: ", computedECC, "\npixel delta x-axis: ", pdx, "\npixel delta y-axis: ", pdy)

    return  pdx, pdy



# pick a random array from the current dataset to serve as a visualization image
def write_example_pictures(frame):
        Image.fromarray(frame.astype('uint8')).save(imagePath + '/excerpt_image.png')



# These two variables anchor motion estimates to real-world values
max_filament_speed = 140 # mm/s
pixels_per_mm = 611 # estimated by counting pixels between edges of known object

max_filament_speed = pixels_per_mm * max_filament_speed # pixels/second


# Instantiating stores for values
velocity_list_x = []
velocity_list_y = []
out_information = []
csv_field_names = ['mm/s X-axis', 'mm/s Y-axis', 'Timestamp [s]']
old_vx = 0
k = 0
tsList = [] # timestamps indexed per-frame
total_timestamp = 0
total_timestamp_list = [] # cumulative timestamps

denoise_hf5()

with h5py.File(imagePath + '/images.h5', 'r') as f:
    # load a slice containing n image arrays from clean dataset

    clean_slice = f['clean_images'][:] #[:chunksize]
    random_frame = random.choice(clean_slice)
    imageio.imsave(imagePath + '/excerpt_image.png', random_frame)

    print(clean_slice)
    # iterate over slice's first axis to make images from individual layers
    for z,x in zip(clean_slice, numberList):

        if k == 0:
            prevFrame = z
            k += 1
            continue # nothing to do with just the first image array
        else:
            # fetch specific line from cached file,an efficient method.
            line = linecache.getline(imagePath + "/tstamps.csv", int(x))
            timestamp = line.split(",")[0] # store whatever comes before comma in the specific line as timestamp. microsecond format
            timestamp_second = int(timestamp)/(10E+6) # convert from microsecond to second
            tsList.append(timestamp_second) # append to list of timestamps
            total_timestamp = total_timestamp + int(timestamp)
            total_timestamp_list.append(total_timestamp)


            pdx, pdy = calc_ECC_transform(prevFrame, z) # get pixel-relative motion between frames
            mm_dx, mm_dy = pdx / pixels_per_mm, pdy / pixels_per_mm # convert to millimeter-relative motion

            #converting from non-timebound relative motion to timebound (seconds) relative motion
            vx, vy = mm_dx / timestamp_second, mm_dy / timestamp_second

            xmax = max_filament_speed * timestamp_second # px/interval
            print("xmax = ", xmax, " pixels for this image interval")


            velocity_list_x.append(vx)
            velocity_list_y.append(vy)
            out_info = (vx, vy, timestamp_second)
            out_information.append(out_info)

            prevFrame = z # store current array as different variable to use next iteration
            k += 1


### write comma separated value file, for reuse in other software or analysis
with open(imagePath + '/velocity_estimates.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(csv_field_names)
    csvwriter.writerows(out_information)


# plot velocity along x-axis
plt.figure(figsize=(12,8))
plt.plot(total_timestamp_list, velocity_list_x, c = 'red', marker = 'o')
plt.xlabel('timestamp us', fontsize=12)
plt.ylabel('lateral motion', fontsize=12)
plt.show()


# plot velocity along y-axis
plt.figure(figsize=(12,8))
plt.plot(total_timestamp_list, velocity_list_y, c = 'green', marker = 'o')
plt.xlabel('timestamp us', fontsize=12)
plt.ylabel('perpendicular motion', fontsize=12)
plt.show()
