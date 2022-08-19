#!/bin/env python3

# script for auto-calibration of analog gain in raspberry pi image sensors using raspiraw.
# It checks multiple gain settings, and prints the gain with the best (most centered/widest) histogram distribution.


import rawpy
import numpy as np
import argparse
import subprocess
from PIL import ImageStat




# input is: exposure in microseconds, height and width

def getArgs():
    exposure = ''
    width = ''
    height = ''
    parser = argparse.ArgumentParser()
    parser.add_argument(type=int, nargs = 4, action='store', dest='arguments', \
          default=False, help='mode (integer), exposure in microseconds, height and width in pixels')
    args = parser.parse_args()
    exposure, width, height = args.arguments
    return mode, exposure, width, height


mode, exposure, width, height = getArgs()


def check_histogram_distribution(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cumulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255


    return cum_sum, norm

print norm

def signalToNoise(image, axis=0, ddof=0):
    a = np.asanyarray(image)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)




register_values = range(0,232, 5)

def capture_images(mode, exposure, width, height):
    hexed_gains = [hex(x) for x in register_values]
    for x,y in zip(hexed_gains, register_values):
        subprocess.Popen("raspiraw --mode ", mode,
                         "--expus ", exposure,
                         "--width ", width,
                         "--height ", height,
                         "--fps 1 --timeout 1000 --i2c 10 --regs ", x,
                         "--header --tstamps tstamps.csv --output gain_", y, ".raw")

    filenames = ["gain_" + y + ".raw" for y in register_values]
    return filenames



def raw_conversion(raw_images):
    for y in rawfiles:
        with rawpy.imread(y) as raw:
            rgb = raw.postprocess(
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full,
                no_auto_bright=False, output_bps=8)
                grayframe = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)





rawfiles = capture_images(mode, exposure, width, height)

