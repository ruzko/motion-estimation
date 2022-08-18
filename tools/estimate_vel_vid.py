#!/bin/env python3

# Copyright Jacob Dybvald Ludvigsen, 2022
# you may use this software for any purpose, as long as you include the above Copyright notice,
# and follow the conditions of the licence.
# This is free software, licenced under BSD-3-Clause


#install dependencies:
# python3 -m pip install numpy rawpy imageio matplotlib opencv-contrib-python h5py matplotlib


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
    for fileName in Path(imagePath).glob("*.h264"):
        vid_file = str(fileName)
        break
    vid = open(vid_file, 'rb')
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
            frame = frame.reshape((height, lineSize//3, 3))
            frame = frame[:,:width,:]

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
    blurredImage = cv.GaussianBlur(sharpImage, (3, 3), sigmaX=0)

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
                                h=15, templateWindowSize=19, searchWindowSize=41)

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

            # blurredImageArray = blurring(cleanImageArray) # blur to further reduce noise
            blurredImageArray = cleanImageArray[np.newaxis, ...].astype(np.uint8) # add axis to enable appending
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
    ECCTransform = init_warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)**(1-nol) # adjust warp according to scale of array
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
            # breakpoint()
            computedECC, ECCTransform = cv.findTransformECC(prevFrame[level], curFrame[level], ECCTransform, **Transform_ECC_params)
            measurement_flag = 1
        except:
            print(f'\nECCTransform could not be found for layer {level+1} of {nol}')
            #ECCTransform =     # np.eye(2, 3, dtype=np.float32)
            #computedECC = 0
            measurement_flag = 0

        if level != nol-1:  # scale up for the next pyramid level, unless the next layer is the original image
            ECCTransform = ECCTransform * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)

        #if level == nol-1 and ECCTransform == None:



    # Extract second element of first and second row, which is translation in their respective directions
    try:
        pdx, pdy = ECCTransform[0,2], ECCTransform[1,2]
    except:
        print(f'no transform found for images')

        pdx = 0
        pdy = 0


    # I think computedECC is the confidence that the transform matrix fits.
    #print(f'\n\nECC confidence of transform: {computedECC}') #, \npixel delta x-axis: {pdx} \npixel delta y-axis:  {pdy}')

    return  pdx, pdy, measurement_flag




FB_opt_flow_params=dict(pyr_scale=.5,
                         levels=3,
                         winsize= 16,
                         iterations=15,
                         poly_n=5,
                         poly_sigma=1.2,
                         flags= 1)


def calculate_dense_flow(prevFrame, curFrame):

    flow = cv.calcOpticalFlowFarneback(prevFrame, curFrame, None, **FB_opt_flow_params)
    pdx, pdy = flow[..., 0], flow[..., 1]
    pdx_average, pdy_average = np.mean(pdx), np.mean(pdy)

    return pdx_average, pdy_average



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
csv_field_names = ['Timestamp [s]', 'mm/min X-axis optical', 'mm/min Y-axis optical']
tsList = [] # timestamps indexed per-frame
total_timestamp_list = [] # cumulative timestamps



def encoder_velocity():
    k = 2
    camera_triggered = 0
    filament_motion = 0
    total_timestamp_enc = 0
    firstRunFlag = 1

    # get camera timestamp file
    for fileName in Path(imagePath).glob("*tstamps.txt"):
        tstamp_fileName = str(fileName)
        break
    # fetch specific line from cached file,an efficient method.
    line = linecache.getline(tstamp_fileName, lastFrame)
    total_timestamp_opt = float(line.split("\n")[0]) / 1000



    for fileName in Path(imagePath).glob("*encoder.csv"):
        encoder_log = str(fileName)
        break

    while camera_triggered == 0:
        line = linecache.getline(encoder_log, k)
        camera_triggered = int(line.split(",")[8].split("\n")[0][0])

        k += 1

        if camera_triggered == 1:
            k -= 2

            break



    while (total_timestamp_enc <= total_timestamp_opt):

        line = linecache.getline(encoder_log, k)
        if line == "":
            break
        encoder_timestamp = float(line.split(",")[0]) #
        filament_position = float(line.split(",")[2]) # mm

        if firstRunFlag == 1:
            encoder_timestamp_second = encoder_timestamp / 1000
            old_ts = encoder_timestamp_second
            old_pos = filament_position
            k += 1
            firstRunFlag = 0
            continue

        encoder_timestamp_second = encoder_timestamp / 1000 # millisecond to second
        timestamp_gap_enc = encoder_timestamp_second - old_ts
        total_timestamp_enc = total_timestamp_enc + timestamp_gap_enc

        filament_motion = filament_position - old_pos
        velocity_encoder = (filament_motion / (timestamp_gap_enc/60)) # mm/min

        print(f'\n\nencoder timestamp: {total_timestamp_enc}, \nposition: {filament_position}, \nvelocity: {velocity_encoder}')

        encoder_out_info = (total_timestamp_enc, filament_position, velocity_encoder)
        encoder_out_info_list.append(encoder_out_info)

        enc_pos_list.append(filament_position)
        enc_ts_list.append(total_timestamp_enc)

        old_ts = encoder_timestamp_second
        old_pos = filament_position
        k += 1



    return enc_pos_list, enc_ts_list





def end_process(clean_arrs):
    k = 0
    failed_estimates = 0
    total_motion_optical_flow = 0
    opt_flow_list = []
    timestamp_k = firstFrame + 2 # skip lines with metadata and first (0.0 sec) timestamp
    old_vx = 0
    old_ts = 0
    total_timestamp = 0
    timestamp_gap = 0
    opt_lateral_position = 0

    # iterate over slice's first axis
    for z,x in zip(clean_arrs, numberList):

        if k == 0:
            prevFrame = z
            k += 1
            timestamp_k += 1
            continue # nothing to do with just the first image array

        else:
            # get timestamp file
            for fileName in Path(imagePath).glob("*tstamps.txt"):
                tstamp_fileName = str(fileName)

                break


            pdx, pdy, measurement_flag = calc_ECC_transform(prevFrame, z) # get pixel-relative motion between frames

            pdx_optical_flow, pdy_optical_flow = calculate_dense_flow(prevFrame, z)
            total_motion_optical_flow += pdx_optical_flow
            opt_flow_list.append(total_motion_optical_flow)

            #print(f'Optical flow pdx: {pdx_optical_flow} \nOptical flow pdy: {pdy_optical_flow}')


            if (pdx < 2 and measurement_flag == 1):
                k += 1
                timestamp_k += 1
                continue

            if measurement_flag == 0:
                failed_estimates += 1

            # fetch specific line from cached file,an efficient method.
            line = linecache.getline(tstamp_fileName, timestamp_k)
            total_timestamp = float(line.split("\n")[0]) # store the specific portion of the line as timestamp. microsecond format
            if total_timestamp == '':
                total_timestamp = 1E-10
            timestamp_second = total_timestamp / (1000) # convert from millisecond to second
            timestamp_gap_s = timestamp_second - old_ts
            timestamp_gap_m = timestamp_gap_s / 60 # convert from second to minute

            tsList.append(timestamp_second) # append to list of timestamps
            #total_timestamp = total_timestamp + int(timestamp)
            #total_timestamp_list.append(total_timestamp)
            old_ts = timestamp_second




            mm_dx, mm_dy = pdx / pixels_per_mm, pdy / pixels_per_mm # convert to millimeter-relative motion

            #converting from non-timebound relative motion to timebound (seconds) relative motion
            vxs, vys = mm_dx / timestamp_gap_s, mm_dy / timestamp_gap_s
            vxm, vym = mm_dx / timestamp_gap_m, mm_dy / timestamp_gap_m

            opt_lateral_position += mm_dx

            xmax = max_filament_speed * timestamp_gap_m # px/interval
            print(f'\n\nxmax = {xmax} pixels for this image interval. \npdx = {pdx} \npdy = {pdy}')


            velocity_list_x.append(vxm)
            velocity_list_y.append(vym)
            motion_list_opt.append(opt_lateral_position)

            out_info = (vxm, vym, timestamp_second)
            out_information.append(out_info)

            prevFrame = z # store current array as different variable to use next iteration
            k += 1
            timestamp_k += 1


    print(f'\nfailed motion estimates: {failed_estimates} of {len(numberList)}')
    print(f'final position according to findTransformECC: {opt_lateral_position}')
    print(f'final position according to Dense optical flow: {total_motion_optical_flow/pixels_per_mm}\n\n')
    return out_information, velocity_list_x, velocity_list_y, tsList, motion_list_opt




def dataset_correlation(optical_pos, optical_ts, encoder_pos, encoder_ts):
    #breakpoint()
    k = 0
    interpolated_vel_list_opt = []
    interpolated_vel_list_enc = []

    # interpolate data, hitting all original datapoints
    f_interpolated_opt_pos = interpolate.Akima1DInterpolator(optical_ts, optical_pos)
    f_interpolated_enc_pos = interpolate.Akima1DInterpolator(encoder_ts, encoder_pos)

    # make new timestamp list, with equally spaced intervals and equal number of points as camera frames
    tsList_new = np.linspace(0, optical_ts[-1], len(optical_ts))

    # resample datasets to new timestamp list
    interpolated_opt_pos = f_interpolated_opt_pos(tsList_new)
    interpolated_enc_pos = f_interpolated_enc_pos(tsList_new)


    for mm_dx_opt, mm_dx_enc, ts in zip(interpolated_opt_pos, interpolated_enc_pos, tsList_new):
        if k == 0:
            old_ts = ts
            k += 1
            continue
        timestamp_gap = ts - old_ts
        v_opt = (mm_dx_opt / timestamp_gap) * 60
        v_enc = (mm_dx_enc / timestamp_gap) * 60

        interpolated_vel_list_opt.append(v_opt)
        interpolated_vel_list_enc.append(v_enc)

        k += 1



    return interpolated_opt_pos, interpolated_enc_pos, tsList_new, interpolated_vel_list_enc, interpolated_vel_list_opt








correlated_position_csv_headers = ['timestamp [s]', 'position optical', 'position encoder']

def presentData(out_information, velocity_list_x, velocity_list_y, tsList, interpolated_opt_pos, interpolated_enc_pos, interpolated_tsList, interpolated_vel_list_enc, interpolated_vel_list_opt):

    ### write comma separated value file, for reuse in other software or analysis
    with open(imagePath + '/velocity_estimates.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_field_names)
        csvwriter.writerows(out_information)

    with open(imagePath + '/correlated_positions.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(correlated_position_csv_headers)
        for value in zip(interpolated_tsList, interpolated_opt_pos, interpolated_enc_pos):
            csvwriter.writerow(value)


#    # plot velocity along x-axis (lateral)
#    fig1 = plt.figure(figsize=(100,40))
#    plt.plot(tsList, velocity_list_x, c = 'red', marker = 'o', linewidth='4')
#    plt.grid(color='green', linestyle='-')
#    plt.xlabel('timestamp seconds', fontsize=32)
#    plt.ylabel('lateral velocity [mm/min]', fontsize=32)
#    plt.xticks(fontsize=24)
#    plt.yticks(fontsize=24)
#    fig1.savefig(fname = (f'{imagePath}/lateral_velocity_frames{firstFrame}-{lastFrame}.png'), dpi =100)
#    plt.show()
#
#
#    # plot velocity along y-axis (perpendicular)
#    fig2 = plt.figure(figsize=(100,20))
#    plt.plot(tsList, velocity_list_y, c = 'green', marker = 'o', linewidth='4')
#    plt.grid(color='r', linestyle='-')
#    plt.xlabel('timestamp seconds', fontsize=32)
#    plt.ylabel('perpendicular velocity [mm/min]', fontsize=32)
#    plt.xticks(fontsize=24)
#    plt.yticks(fontsize=24)
#    fig2.savefig(fname = (f'{imagePath}/perpendicular_velocity_frames{firstFrame}-{lastFrame}.png'), dpi = 100)
#    plt.show()


    fig1 = plt.figure(figsize=(100,40))
    plt.plot(interpolated_tsList, interpolated_enc_pos, c = 'red', marker = 'o', linewidth='4')
    plt.plot(interpolated_tsList, interpolated_opt_pos, c = 'blue', marker = 'o', linewidth='4')
    plt.grid(color='green', linestyle='-')
    plt.xlabel('timestamp seconds. Blue is optical, red is encoder', fontsize=32)
    plt.ylabel('lateral position [mm]', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    fig1.savefig(fname = (f'{imagePath}/enc+opt_lateral_position_frames{firstFrame}-{lastFrame}.png'), dpi =100)
    plt.show()



    fig1 = plt.figure(figsize=(100,40))
    plt.plot(interpolated_tsList[1:], interpolated_vel_list_enc, c = 'red', marker = 'o', linewidth='4')
    plt.plot(interpolated_tsList[1:], interpolated_vel_list_opt, c = 'blue', marker = 'o', linewidth='4')
    plt.grid(color='green', linestyle='-')
    plt.xlabel('timestamp seconds. Blue is optical, red is encoder', fontsize=32)
    plt.ylabel('perpendicular velocity [mm/min]', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    fig1.savefig(fname = (f'{imagePath}/enc+opt_vel_frames{firstFrame}-{lastFrame}.png'), dpi =100)
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

    interpolated_opt_pos, interpolated_enc_pos, interpolated_tsList, interpolated_vel_list_enc, interpolated_vel_list_opt = dataset_correlation(motion_list_opt, tsList, enc_pos_list, enc_ts_list)

    # present data
    presentData(out_information, velocity_list_x, velocity_list_y, tsList, interpolated_opt_pos, interpolated_enc_pos, interpolated_tsList, interpolated_vel_list_enc, interpolated_vel_list_opt)


if __name__ == "__main__":
    main()



