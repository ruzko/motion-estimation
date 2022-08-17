#!/bin/env python3

# Copyright Jacob Dybvald Ludvigsen, 2022
# you may use this software for any purpose, as long as you include the above Copyright notice,
# and follow the conditions of the licence.
# This is free software, licenced under BSD-3-Clause


#install dependencies:
# python3 -m pip install numpy rawpy imageio matplotlib opencv-contrib-python h5py matplotlib

#import pandas as pd # to handle dataset correlation and resampling
from scipy import interpolate
import h264decoder # to directly open .h264 files
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

    parser.add_argument('-h264', nargs='?', action='store', dest='h264',
                        default='False', const='True', help='whether to read h264 file directly instead of mkv file.')

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


    return firstFrame, lastFrame, numberList, imagePath, continuation, args.h264



firstFrame, lastFrame, numberList, imagePath, continuation, use_h264 = retFileList()




def get_meta():
    for fileName in Path(imagePath).glob("*.mkv"):
        vid_file = str(fileName)
        break

    cap = cv.VideoCapture(vid_file)
    # get vcap property
    width  = int(cap.get(3))   # float `width`
    height = int(cap.get(4))  # float `height`
    totalFrames = int(cap.get(7)) # cv.CAP_PROP_FRAME_COUNT
    cap.release()

    return width, height, totalFrames


def get_meta_h264():
    #incoming data
    for fileName in Path(imagePath).glob("*.h264"):
        vid = open(fileName, 'rb')
        decoder = h264decoder.H264Decoder()
        while (1):
            data_in = vid.read(1024)
            if not data_in:
                break
            framedata, nread = decoder.decode_frame(data_in)
            data_in = data_in[nread:]
            (frame, width, height, lineSize) = framedata
            break
        break

    return width, height


if use_h264 != "True":
    width, height, totalFrames = get_meta()
else:
    width, height = get_meta_h264()


if lastFrame == -1:
    firstFrame = 0
    lastFrame = totalFrames
    r = range(firstFrame, lastFrame)
    numberList = list([*r])
    numberList.append(lastFrame)



def read_vid_h264():
    k = 0
    #incoming data
    vid = open(imagePath + "/*.h264", 'rb')
    decoder = h264decoder.H264Decoder()
    while (1):
        data_in = vid.read(1024)
        if not data_in:
            break
        framedata, nread = decoder.decode_frame(data_in)
        data_in = data_in[nread:]
        (frame, width, height, lineSize) = framedata
        if frame is not None:
            frame = np.frombuffer(frame, dtype=np.ubyte, count=len(frame))
            frame = frame.reshape((h, ls//3, 3))
            frame = frame[:,:w,:]

        grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grayframe = grayframe[np.newaxis, ... ].astype(np.uint8)

        if k == 0:
            noisy_arrs = np.asarray(grayframe)
        else:
            noisy_arrs = np.append(noisy_arrs, grayframe, axis=0)
        k += 1
        #print(noisy_arrs)
        if k == len(numberList):
            break
    return noisy_arrs



# read relevant video frames
def read_vid_mkv():
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




### Increase contrast by equalisizing histogram, without increasing noise
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



hf5_params = dict(maxshape=(len(numberList)+10, height, width),
            chunks = (10, height, width),
            dtype = 'uint8',
            compression="gzip",
            compression_opts=7,
            shuffle=True)


## Main function for denoising and contrast enhancing of image arrays
def denoise_hf5(eq_arrs):
    k = 0
    with h5py.File(imagePath + '/images.h5', 'w') as f:
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

            blurredImageArray = blurring(cleanImageArray) # blur to further reduce noise
            blurredImageArray = blurredImageArray[np.newaxis, ...].astype(np.uint8) # add axis to enable appending
            if k == 0:
                # make the first array
                blurred_arrs = np.asarray(blurredImageArray)

            if k != 0:
                # layer the current array on top of previous array
                blurred_arrs = np.append(blurred_arrs, blurredImageArray, axis=0)

            k += 1
            print(f'Frame {k} of {len(numberList)} Denoised')

            # create dataset with dimensions matching however many arrays were successfully processed. Avoids issues with broadcasting arrays to dataset
        clean_dataset = f.create_dataset("clean_images", shape=(blurred_arrs.shape), **hf5_params)
        #set attributes for image dataset
        clean_dataset.attrs['CLASS'] = 'IMAGE'
        clean_dataset.attrs['IMAGE_VERSION'] = '1.2'
        clean_dataset.attrs['IMAGE_SUBCLASS'] =  'IMAGE_GRAYSCALE'
        clean_dataset.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)
        clean_dataset.attrs['IMAGE_WHITE_IS_ZERO'] =  0

        f['clean_images'].write_direct(blurred_arrs) #write all arrays at once. fast.

    return blurred_arrs


Transform_ECC_params = dict(motionType = cv.MOTION_TRANSLATION, # only motion in x- and y- axes
                            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10000,  0.0001)) # max iteration count and desired epsilon. Terminates when either is reached.




# Get total shift in x- and y- direction between two image frames / arrays
def calc_ECC_transform(prevFrame, curFrame):

    # Construct scale pyramid to speed up and improve accuracy of transform estimation
    nol = 4 # number of layers
    init_warp = np.eye(2, 3, dtype=np.float32) # identity matrix
    warp = init_warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)**(1-nol) # adjust warp according to scale of array
    prevFrame = [prevFrame]
    curFrame = [curFrame]
    for level in range(nol): # add resized layers to original array, to get 3 dimensions.
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

        if level != nol-1:  # scale up for the next pyramid level, unless the next layer is the original image
            warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)


    # Extract second element of first and second row, which is translation in their respective directions
    pdx, pdy = ECCTransform[0,2], ECCTransform[1,2]

    # I think computedECC is the confidence that the transform matrix fits.
    print(f'\n\nECC confidence of transform: {computedECC}, \npixel delta x-axis: {pdx} \npixel delta y-axis:  {pdy}')

    return  pdx, pdy




# These two variables anchor motion estimates to real-world values
max_filament_speed = 140 # mm/min
max_filament_speed_sec = max_filament_speed / 60 # mm/s
pixels_per_mm = 611 # estimated by counting pixels between edges of known object.
max_filament_speed = pixels_per_mm * max_filament_speed # pixels/second


# Instantiating stores for values
encoder_out_info_list = []
enc_pos_list = []
enc_ts_list = []
velocity_list_x = []
velocity_list_y = []
motion_list_opt = []
out_information = []
csv_field_names = ['mm/min X-axis', 'mm/min Y-axis', 'Timestamp [s]']
tsList = [] # timestamps indexed per-frame
total_timestamp_list = [] # cumulative timestamps



def encoder_velocity():
    k = 2
    filament_motion = "0.00"
    total_timestamp = 0
    breakpoint()

    for fileName in Path(imagePath).glob("*encoder.csv"):
        encoder_log = str(fileName)
        break

    while filament_motion == "0.00":
        line = linecache.getline(encoder_log, k)
        filament_motion = line.split(",")[2]
        k += 1

        if filament_motion != "0.00":
            k -= 2
            break



    while (1):
        line = linecache.getline(encoder_log, k)
        if line == "":
            break
        encoder_timestamp = float(line.split(",")[0]) #
        filament_position = float(line.split(",")[2]) # mm

        if filament_position == 0.0:
            encoder_timestamp_second = encoder_timestamp / 1000
            old_ts = encoder_timestamp_second
            old_pos = filament_position
            k += 1
            continue

        encoder_timestamp_second = encoder_timestamp / 1000 # millisecond to second
        timestamp_gap = encoder_timestamp_second - old_ts
        total_timestamp = total_timestamp + timestamp_gap

        filament_motion = filament_position - old_pos
        velocity_encoder = (filament_motion / (timestamp_gap/60)) # mm/min

        print(f'\n\nencoder timestamp: {total_timestamp}, \nmotion: {filament_motion}, \nvelocity: {velocity_encoder}')

        encoder_out_info = (total_timestamp, filament_position, velocity_encoder)
        encoder_out_info_list.append(encoder_out_info)

        enc_pos_list.append(filament_position)
        enc_ts_list.append(total_timestamp)

        old_ts = encoder_timestamp_second
        old_pos = filament_position
        k += 1


    return enc_pos_list, enc_ts_list





def end_process():
    k = 0
    timestamp_k = firstFrame
    old_vx = 0
    old_ts = 0
    total_timestamp = 0
    timestamp_gap = 0
   # with h5py.File(imagePath + '/images.h5', 'r') as f:
        # load a slice containing n image arrays from clean dataset
    #    clean_slice = f['clean_images'][()]

        #print(clean_slice)
        # iterate over slice's first axis to make images from individual layers
    for z,x in zip(clean_slice, numberList):

        if k == 0:
            prevFrame = z


            k += 1
            continue # nothing to do with just the first image array

        else:
            # get timestamp file
            for fileName in Path(imagePath).glob("*tstamps.txt"):
                tstamp_fileName = str(fileName)

                break

            # fetch specific line from cached file,an efficient method.
            line = linecache.getline(tstamp_fileName, (firstFrame+2)) # skip lines with metadata and first (0.0 sec) timestamp
            total_timestamp = line.split("\n")[0] # store the specific portion of the line as timestamp. microsecond format
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
            motion_list_opt.append(mm_dx)

            out_info = (vxm, vym, timestamp_second)
            out_information.append(out_info)

            prevFrame = z # store current array as different variable to use next iteration
            k += 1
            timestamp_k += 1

    return out_information, velocity_list_x, velocity_list_y, tsList, motion_list_opt




def dataset_correlation(optical_pos, optical_ts, encoder_pos, encoder_ts):

    # interpolate data, hitting all original datapoints
    f_interpolated_opt_pos = interpolate.Akima1DInterpolator(optical_ts, optical_pos)
    f_interpolated_enc_pos = interpolate.Akima1DInterpolator(encoder_ts, encoder_pos)

    # make new timestamp list, with equally spaced intervals and equal number of points as camera frames
    tsList_new = np.linspace(0, optical_ts[-1], len(optical_ts))

    # resample datasets to new timestamp list
    interpolated_opt_pos = f_interpolated_opt_pos(tsList_new)
    interpolated_enc_pos = f_interpolated_enc_pos(tsList_new)




    return interpolated_opt_pos, interpolated_enc_pos, tsList_new





def savePlots(out_information, velocity_list_x, velocity_list_y, tsList):

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
    fig1.savefig(fname = (f'{imagePath}/lateral_velocity_frames{firstFrame}-{lastFrame}.png'), dpi =100)
    plt.show()


    # plot velocity along y-axis
    fig2 = plt.figure(figsize=(100,20))
    plt.plot(tsList, velocity_list_y, c = 'green', marker = 'o', linewidth='4')
    plt.grid(color='r', linestyle='-')
    plt.xlabel('timestamp seconds', fontsize=32)
    plt.ylabel('perpendicular velocity [mm/min]', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    fig2.savefig(fname = (f'{imagePath}/perpendicular_velocity_frames{firstFrame}-{lastFrame}.png'), dpi = 100)
    plt.show()




def main():
    if continuation != 'True':
        # get input
        if use_h264 == "True":
            noisy_arrs = read_vid_h264()
        else:
            noisy_arrs = read_vid_mkv()

        # enhance contrast
        eq_arrs = adaptive_histogram_equalization(noisy_arrs)

        # denoise images
        clean_arrs = denoise_hf5(eq_arrs)

    else:
        with h5py.File(imagePath + '/images.h5', 'r') as f:
            clean_arrs = f['clean_images'][()]

    # read encoder motion and timestamps from csv file
    enc_pos_list, enc_ts_list = encoder_velocity()

    # find velocity
    out_information, velocity_list_x, velocity_list_y, tsList, motion_list_opt = end_process(clean_arrs)

    dataset_correlation(motion_list_opt, tsList, enc_pos_list, enc_ts_list)

    # present data
    savePlots(out_information, velocity_list_x, velocity_list_y, tsList)


if __name__ == "__main__":
    main()



