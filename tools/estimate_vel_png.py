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
    numberList = []
    firstFrame = ''
    lastFrame = ''
    parser = argparse.ArgumentParser()
    parser.add_argument(type=int, nargs ='*', action='store', dest='fileIndex', \
          default='False', help='index of first image to be read. If full is passed, the whole video is used')

    parser.add_argument('-f', '--full', nargs ='?', action='store', dest='useWholeVideo', \
          default='False', const='True', help='index of last image to be read. If full is passed, the whole video is used')

    parser.add_argument('-p', '--path', nargs='?', type=Path, dest="srcDirectory", default='/dev/shm/', \
          help='which directory to read images from. Specify with "-p <path-to-folder>" or "--path <path-to-folder". Leave empty for /dev/shm/')

    parser.add_argument('-c', '--continue', nargs='?', action='store', dest='continuation', \
        default='False', const='True', help='continue analysis of video from previous attempt')

    args = parser.parse_args()
    srcDir = args.srcDirectory
    imagePath = str(srcDir)

    continuation = args.continuation

    if args.fileIndex != None and args.useWholeVideo != 'True':
        firstFrame, lastFrame = args.fileIndex
        r = range(firstFrame, lastFrame)
        numberList = list([*r])
        numberList.append(lastFrame)
    if args.useWholeVideo == "True":
        lastFrame = -1
    return firstFrame, lastFrame, numberList, imagePath, continuation



firstFrame, lastFrame, numberList, imagePath, continuation = retFileList()


def get_meta():
    while (1):
        cap = cv.VideoCapture(imagePath + "/video.mkv")
        # get vcap property
        width  = int(cap.get(3))   # float `width`
        height = int(cap.get(4))  # float `height`
        totalFrames = int(cap.get(7)) # cv.CAP_PROP_FRAME_COUNT

        cap.release()
        break
    return width, height, totalFrames

width, height, totalFrames = get_meta()



if lastFrame != -1:
    hf5_params = dict(maxshape=(len(numberList)+10, height, width),
                chunks = (10, height, width),
                dtype = 'uint8',
                compression="gzip",
                compression_opts=7,
                shuffle=True)


if lastFrame == -1:
    hf5_params = dict(maxshape=(totalFrames+10, height, width),
                chunks = (10, height, width),
                dtype = 'uint8',
                compression="gzip",
                compression_opts=7,
                shuffle=True)

    firstFrame = 0
    lastFrame = totalFrames
    r = range(firstFrame, lastFrame)
    numberList = list([*r])
    numberList.append(lastFrame)



# read relevant video frames
def read_vid():
    # incoming data
    cap = cv.VideoCapture(imagePath + "/video.mkv")
    k = 0
    while cap.isOpened():
        frame_no = cap.set(cv.CAP_PROP_POS_FRAMES, k)
        ret, frame = cap.read()
        if ret != 1:
            break
        grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        grayframe = grayframe[np.newaxis, ... ].astype(np.uint8)

        # first file; create the dummy dataset with no max shape
        if k == 0:
            noisy_arrs = np.asarray(grayframe)
        else:
            noisy_arrs = np.append(noisy_arrs, grayframe, axis=0)
        k += 1
        #print(noisy_arrs)
        if k == len(numberList):
            break
    return noisy_arrs


### Increase contrast by equalisizing histogram
def enhance_contrast(noisy_arrs, bins=256): # https://gist.github.com/msameeruddin/8629aa0bf58521a22bb67ed0fea82fee
    k = 0
    for z in noisy_arrs:

       # print(f'enhanceEQ z: {z}')
        image_flattened = z.flatten()
        image_hist = np.zeros(bins)

       # print(f'enhanceEQ noisyArrs: {noisy_arrs}')
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
        #print("uniform: ", uniform_norm)
        # flat histogram
        image_eq = uniform_norm[image_flattened]
        # reshaping the flattened matrix to its original shape
        image_eq = np.reshape(a=image_eq, newshape=z.shape)

        image_eq = image_eq[np.newaxis, ...].astype(np.uint8)
        print(image_eq)
        if k == 0:
            eq_arrs = np.asarray(image_eq)
            # layer the current array on top of previous array, write to file. Slow.
            # Ideally, number of write processes should be minimized.
        else:
            eq_arrs = np.append(eq_arrs, image_eq, axis=0)

        k += 1

    return eq_arrs



def adaptive_histogram_equalization(noisy_arrs):
    k = 0
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for z in noisy_arrs:
        equalized = clahe.apply(z)
        equalized = equalized[np.newaxis, ...].astype(np.uint8)
        print(equalized)
        if k == 0:
            eq_arrs = np.asarray(equalized)
            # layer the current array on top of previous array, write to file. Slow.
            # Ideally, number of write processes should be minimized.
        else:
            eq_arrs = np.append(eq_arrs, equalized, axis=0)

        k += 1

    return eq_arrs




### Blur image to reduce noise. We don't really need sharp edges to estimate motion with findTransformECC()
def blurring(sharpImage):
    blurredImage = cv.GaussianBlur(sharpImage, (5, 5), sigmaX=0)

    return blurredImage



### Denoise image numberIndex by comparing with other images in num_frames_window
def denoising(arrays, numberIndex, num_frames_window):

    cleanImageArray = cv.fastNlMeansDenoisingMulti(srcImgs=arrays,
                        imgToDenoiseIndex=numberIndex, temporalWindowSize=num_frames_window,
                          h=15, templateWindowSize=19, searchWindowSize=41) # h is filter strength. h=10 is default
    return cleanImageArray




### Main function for denoising and contrast enhancing of image arrays
def denoise_hf5(eq_arrs):
    k = 0
    with h5py.File(imagePath + '/images.h5', 'w') as f:

        # load a slice containing n images from noisy dataset, not yet implemented
        for z in eq_arrs:

            try: # Try, to enable error handling
                # denoise image
                if (k <= 1) or (k >= (len(numberList) - 3)):
                    # denoise two first and last images individually
                    cleanImageArray = cv.fastNlMeansDenoising(src=z,
                                h=20, templateWindowSize=21, searchWindowSize=45)

                elif (k <= 4) or (k >= (len(numberList) - 5)):
                    # denoise using some neighbouring images as template
                    cleanImageArray = denoising(eq_arrs, k, 5)

                else:  #(numberIndex <= 7) or (numberIndex >= (len(numberList) - 7)):
                    # denoise using more neighbouring images as template
                    cleanImageArray = denoising(eq_arrs, k, 9)
            except:
                print('something went wrong with denoising')

                break
            #         else:
                    # denoise using more neighbouring images as template
            #              cleanImageArray = denoising(noisy_slice, numberIndex, 13)

            blurredImageArray = blurring(cleanImageArray)
            blurredImageArray = blurredImageArray[np.newaxis, ...].astype(np.uint8)
            if k == 0:
                # layer the current array on top of previous array
                blurred_arrs = np.asarray(blurredImageArray)
                # Ideally, number of write processes should be minimized.

            if k != 0:
                blurred_arrs = np.append(blurred_arrs, blurredImageArray, axis=0)
                #breakpoint()
            k += 1
            print(f'Frame {k} of {len(numberList)} Denoised')
        clean_dataset = f.create_dataset("clean_images", shape=(blurred_arrs.shape), **hf5_params)
        #set attributes for image dataset
        clean_dataset.attrs['CLASS'] = 'IMAGE'
        clean_dataset.attrs['IMAGE_VERSION'] = '1.2'
        clean_dataset.attrs['IMAGE_SUBCLASS'] =  'IMAGE_GRAYSCALE'
        clean_dataset.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)
        clean_dataset.attrs['IMAGE_WHITE_IS_ZERO'] =  0

        f['clean_images'].write_direct(blurred_arrs) #write all arrays at once. fast.

    return


Transform_ECC_params = dict(motionType = cv.MOTION_TRANSLATION, # only motion in x- and y- axes
                            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10000,  0.0001)) # max iteration count and desired epsilon. Terminates when either is reached.
                            #gaussFiltSize = 5)



# Get total shift in x- and y- direction between two image frames / arrays
def calc_ECC_transform(prevFrame, curFrame):

    # Construct scale pyramid to speed up and improve accuracy of transform estimation
    nol = 4 # number of layers
    init_warp = np.eye(2, 3, dtype=np.float32)
    warp = init_warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)**(1-nol)
    prevFrame = [prevFrame]
    curFrame = [curFrame]
    for level in range(nol):
        prevFrame.insert(0, cv.resize(prevFrame[0], None, fx=1/2, fy=1/2,
                                   interpolation=cv.INTER_AREA))
        curFrame.insert(0, cv.resize(curFrame[0], None, fx=1/2, fy=1/2,
                                   interpolation=cv.INTER_AREA))

    # run pyramid ECC
    for level in range(nol):
        # Calculate the transform matrix which must be applied to prevFrame in order to match curFrame
        try:
            computedECC, ECCTransform = cv.findTransformECC(prevFrame[level], curFrame[level], warp, **Transform_ECC_params)
        except:
            print('ECCTransform could not be found, setting transform equal to identity matrix')
            ECCTransform = np.eye(2, 3, dtype=np.float32)
            computedECC = 0

        if level != nol-1:  # scale up for the next pyramid level
            warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)


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
        clean_slice = f['clean_images'][()] #[:chunksize]

        #print(clean_slice)
        # iterate over slice's first axis to make images from individual layers
        for z,x in zip(clean_slice, numberList):

            if k == 0:
                prevFrame = z

                k += 1
                continue # nothing to do with just the first image array
            else:
                # fetch specific line from cached file,an efficient method.
                line = linecache.getline(imagePath + "/tstamps.txt", (k+2)) # skip lines with metadata and first (0.0 sec) timestamp
                total_timestamp = line.split("\n")[0] # store the specific line as timestamp. microsecond format
                if total_timestamp == '':
                    total_timestamp = 1E-10
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
    fig1 = plt.figure(figsize=(100,40))
    plt.plot(tsList, velocity_list_x, c = 'red', marker = 'o', linewidth='4')
    plt.grid(color='green', linestyle='-')
    plt.xlabel('timestamp seconds', fontsize=32)
    plt.ylabel('lateral velocity [mm/min]', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    fig1.savefig(fname = (f'{imagePath}/lateral_velocity_frames_{firstFrame}-{lastFrame}.png'), dpi =100)
    plt.show()


    # plot velocity along y-axis
    fig2 = plt.figure(figsize=(100,20))
    plt.plot(tsList, velocity_list_y, c = 'green', marker = 'o', linewidth='4')
    plt.grid(color='r', linestyle='-')
    plt.xlabel('timestamp seconds', fontsize=32)
    plt.ylabel('perpendicular velocity [mm/min]', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    fig2.savefig(fname = (f'{imagePath}/perpendicular_velocity_frames_{firstFrame}-{lastFrame}.png'), dpi = 100)
    plt.show()




def main():
    if continuation != 'True':
        # get input
        noisy_arrs = read_vid()

        # enhance contrast
        eq_arrs = adaptive_histogram_equalization(noisy_arrs)

        # denoise images
        denoise_hf5(eq_arrs)

    # find velocity and present data
    end_process()

if __name__ == "__main__":
    main()



