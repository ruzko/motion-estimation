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
from multiprocessing import Pool # to run multiple instances of time-consuming processes
import subprocess # to execute c-program "double"
import linecache # to read long files line by line efficiently
import random # to choose a random image from image range
import numpy as np # to manipulate images as arrays
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
    fileList = ["frame"+str(x) for x in list(numberList)]
    imagePath = str(srcDir)
    return fileList, numberList, imagePath, needsLineDoubling



fileList, numberList, imagePath, needsDoubling = retFileList()


denoiseNum = len(numberList)

def get_dims():
    while (1):
        cap = cv.VideoCapture(imagePath + "/video.mkv")
        # get vcap property
        width  = int(cap.get(3))   # float `width`
        height = int(cap.get(4))  # float `height`
        cap.release()
        break
    return width, height

width, height = get_dims()

hf5_params = dict(shape=(len(numberList), height, width),
                  maxshape=(len(numberList), height, width),
                  chunks = True,
                  dtype = 'uint8')




# read video, save frames as efficient hdf5 format file
def read_vid():
    with h5py.File(imagePath + '/images.h5', 'w') as f:
        # incoming data
        cap = cv.VideoCapture(imagePath + "/video.mkv")
        count = 0
        while cap.isOpened():
            frame_no = cap.set(cv.CAP_PROP_POS_FRAMES, count)
            ret, frame = cap.read()

            if ret != 1:
                break


            grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # first file; create the dummy dataset with no max shape
            if count == 0:
                    noisy_dataset = f.create_dataset("noisy_images", **hf5_params) # compression="lzf", shuffle=True)count

            # stack image array in 0-indexed position of third axis
            f['noisy_images'][count,:,:]=grayframe
            count += 1
            if count == denoiseNum:
                break


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
def enhance_contrast(bins=256): # https://gist.github.com/msameeruddin/8629aa0bf58521a22bb67ed0fea82fee
    numberIndex = 0
    with h5py.File(imagePath + '/images.h5', 'r+') as f:
         low_contrast_slice = f['noisy_images'][:]
         for z in low_contrast_slice:

             image_flattened = z.flatten()
             image_hist = np.zeros(bins)

             # frequency count of each pixel
             for pix in z:
                 image_hist[pix] += 1

              # cumulative sum
             cum_sum = np.cumsum(image_hist)
             norm = (cum_sum - cum_sum.min()) * 255
             #print(norm)

             # normalization of the pixel values
             n_ = cum_sum.max() - cum_sum.min()
             uniform_norm = norm / n_
             uniform_norm = uniform_norm.astype('int')
             print("uniform: ", uniform_norm)

             # flat histogram
             image_eq = uniform_norm[image_flattened]

             # reshaping the flattened matrix to its original shape
             image_eq = np.reshape(a=image_eq, newshape=z.shape)

             # make a dataset to hold hi-contrast images, for further processing later
             # first file; create the dummy dataset with no max shape
             if numberIndex == 0:
                 hi_contrast_dataset = f.create_dataset("hi_contrast_images", **hf5_params)

             # layer the current array on top of previous array, write to file. Slow.
             # Ideally, number of write processes should be minimized.
             f['hi_contrast_images'][numberIndex,:,:]=image_eq

             numberIndex += 1

         #set attributes for image dataset
         hi_contrast_dataset.attrs['CLASS'] = 'IMAGE'
         hi_contrast_dataset.attrs['IMAGE_VERSION'] = '1.2'
         hi_contrast_dataset.attrs['IMAGE_SUBCLASS'] =  'IMAGE_GRAYSCALE'
         hi_contrast_dataset.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)
         hi_contrast_dataset.attrs['IMAGE_WHITE_IS_ZERO'] =  0

    return


### Blur image to reduce noise. We don't really need sharp edges to estimate motion with findTransformECC()
def blurring(sharpImage):
    blurredImage = cv.GaussianBlur(sharpImage, (21, 21), sigmaX=0)

    return blurredImage



### Denoise image numberIndex by comparing with other images in num_frames_window
def denoising(arrays, numberIndex, num_frames_window):

    cleanImageArray = cv.fastNlMeansDenoisingMulti(srcImgs=arrays,
                        imgToDenoiseIndex=numberIndex, temporalWindowSize=num_frames_window,
                          h=20, templateWindowSize=19, searchWindowSize=41) # h is filter strength. h=10 is default
    return cleanImageArray




### Main function for denoising and contrast enhancing of image arrays
def denoise_hf5():
    numberIndex = 0
    with h5py.File(imagePath + '/images.h5', 'r+') as f:

        # load a slice containing n images from noisy dataset, not yet implemented
        noisy_slice = f['hi_contrast_images'][:]
        for z in noisy_slice:

#    with Pool() as p:
#        clean_arrays = p.map(denoise_hf5, noisy_slice)





        # denoise image
            if (numberIndex <= 1) or (numberIndex >= (denoiseNum - 2)):
                # denoise two first and last images individually
                cleanImageArray = cv.fastNlMeansDenoising(src=z,
                            h=41, templateWindowSize=21, searchWindowSize=45)

            elif (numberIndex <= 4) or (numberIndex >= (denoiseNum - 4)):
                # denoise using some neighbouring images as template
                cleanImageArray = denoising(noisy_slice, numberIndex, 5)

            else:  #(numberIndex <= 7) or (numberIndex >= (denoiseNum - 7)):
                # denoise using more neighbouring images as template
                cleanImageArray = denoising(noisy_slice, numberIndex, 9)

        #         else:
                # denoise using more neighbouring images as template
        #              cleanImageArray = denoising(noisy_slice, numberIndex, 13)

            blurredImageArray = blurring(cleanImageArray)
            if numberIndex == 0:
                # layer the current array on top of previous array, write to file. Slow.
                # Ideally, number of write processes should be minimized.
                clean_dataset = f.create_dataset("clean_images", **hf5_params) #, compression="lzf", shuffle=True)
                #set attributes for image dataset
                clean_dataset.attrs['CLASS'] = 'IMAGE'
                clean_dataset.attrs['IMAGE_VERSION'] = '1.2'
                clean_dataset.attrs['IMAGE_SUBCLASS'] =  'IMAGE_GRAYSCALE'
                clean_dataset.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)
                clean_dataset.attrs['IMAGE_WHITE_IS_ZERO'] =  0

            f['clean_images'][numberIndex,:,:]=blurredImageArray

            numberIndex += 1
    return


Transform_ECC_params = dict(warpMatrix = np.eye(2, 3, dtype=np.float32), # preparing unity matrix for x- and y- axis
                            motionType = cv.MOTION_TRANSLATION, # only motion in x- and y- axes
                            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1000,  1E-10)) # max iteration count and desired epsilon. Terminates when either is reached.
                            #gaussFiltSize = 5)



# Get total shift in x- and y- direction between two image frames / arrays
def calc_ECC_transform(prevFrame, curFrame):

    # Calculate the transform matrix which must be applied to prevFrame in order to match curFrame
    computedECC, ECCTransform = cv.findTransformECC(prevFrame, curFrame, **Transform_ECC_params)

    # Extract second element of first and second row to be shift in their respective directions
    pdx, pdy = ECCTransform[0,2], ECCTransform[1,2]

    # I think computedECC is the confidence that the transform matrix fits.
    print("\n\nECC confidence of transform: ", computedECC, "\npixel delta x-axis: ", pdx, "\npixel delta y-axis: ", pdy)

    return  pdx, pdy



# These two variables anchor motion estimates to real-world values
max_filament_speed = 140 # mm/min
max_filament_speed_sec = max_filament_speed / 60 # mm/s
pixels_per_mm = 611 # estimated by counting pixels between edges of known object

max_filament_speed = pixels_per_mm * max_filament_speed # pixels/second


# Instantiating stores for values
velocity_list_x = []
velocity_list_y = []
out_information = []
csv_field_names = ['mm/min X-axis', 'mm/min Y-axis', 'Timestamp [s]']
tsList = [] # timestamps indexed per-frame
total_timestamp_list = [] # cumulative timestamps



def end_process():
    k = 0
    old_vx = 0
    old_ts = 0
    total_timestamp = 0
    timestamp_gap = 0
    with h5py.File(imagePath + '/images.h5', 'r') as f:
        # load a slice containing n image arrays from clean dataset
        clean_slice = f['clean_images'][:] #[:chunksize]

        print(clean_slice)
        # iterate over slice's first axis to make images from individual layers
        for z,x in zip(clean_slice, numberList):

            if k == 0:
                prevFrame = z
                k += 1
                continue # nothing to do with just the first image array
            else:
                # fetch specific line from cached file,an efficient method.
                line = linecache.getline(imagePath + "/tstamps.txt", int(k+2))
                total_timestamp = line.split("\n")[0] # store whatever comes before comma in the specific line as timestamp. microsecond format

                timestamp_second = float(total_timestamp) / (1000) # convert from millisecond to second
                timestamp_gap_s = timestamp_second - old_ts
                timestamp_gap_m = timestamp_gap_s / 60 # convert from second to minute

                tsList.append(timestamp_second) # append to list of timestamps
                #total_timestamp = total_timestamp + int(timestamp)
                #total_timestamp_list.append(total_timestamp)
                old_ts = timestamp_second


                pdx, pdy = calc_ECC_transform(prevFrame, z) # get pixel-relative motion between frames
                mm_dx, mm_dy = pdx / pixels_per_mm, pdy / pixels_per_mm # convert to millimeter-relative motion

                #converting from non-timebound relative motion to timebound (seconds) relative motion
                vxs, vys = mm_dx / timestamp_gap_s, mm_dy / timestamp_gap_s
                vxm, vym = mm_dx / timestamp_gap_m, mm_dy / timestamp_gap_m

                xmax = max_filament_speed * timestamp_gap_m # px/interval
                print("xmax = ", xmax, " pixels for this image interval")


                velocity_list_x.append(vxm)
                velocity_list_y.append(vym)
                out_info = (vxm, vym, timestamp_second)
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
    plt.plot(tsList, velocity_list_x, c = 'red', marker = 'o')
    plt.xlabel('timestamp seconds', fontsize=12)
    plt.ylabel('lateral motion [mm/min]', fontsize=12)
    plt.show()


    # plot velocity along y-axis
    plt.figure(figsize=(12,8))
    plt.plot(tsList, velocity_list_y, c = 'green', marker = 'o')
    plt.xlabel('timestamp seconds', fontsize=12)
    plt.ylabel('perpendicular motion [mm/min]', fontsize=12)
    plt.show()




def main():
    # get input
    read_vid()

    # enhance contrast
    enhance_contrast()

    # denoise images
    denoise_hf5()

    # find velocity and present data
    end_process()

if __name__ == "__main__":
    main()



