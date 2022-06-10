#!/bin/python3
# Copyright Jacob Dybvald Ludvigsen, crackwitz, 2022
# This is free software, licenced under GPLv3-or-later
#install dependencies: pip3 install numpy rawpy imageio matplotlib memory-profiler opencv-contrib-python

import numpy as np
import rawpy
import imageio
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
from memory_profiler import profile


# Take input, convert to list with leading zeroes and  return
@profile
def retFileList():
    fileList = []
    parser = argparse.ArgumentParser()
    parser.add_argument(type=int, nargs = 2, action='store', dest='fileList', default=False, help='numbes of first and last image files to be read')
    args = parser.parse_args()
    inputFileList = args.fileList
    fileList.extend(range(*inputFileList))
    fileList = map(str, fileList)
    numberList = [str(x).zfill(4) for x in list(fileList)]
    fileList = ["out."+str(x)+".raw" for x in list(numberList)]
    return fileList, numberList



# prepend headers to rawfiles if they don't already have a header
rawList, numberList = retFileList()
hf = open('/dev/shm/hd0.32k', 'rb')
header = hf.read()
hf.close()
for x in list(rawList):
    with open('/dev/shm/' + x, 'rb') as rawFile: partialRaw = rawFile.read(32)
    if header != partialRaw:
        with open('/dev/shm/' + x, 'rb') as original: data = original.read()
        with open('hd.' + x, 'wb') as modified: modified.write(header + data)

# list with modified filenames
headedList = ["hd."+str(x) for x in list(rawList)]

# Convert from raw to viewable format and save
for (x,y) in zip(headedList, numberList):
    with rawpy.imread(x) as raw:
        rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
    imageio.imsave('img.'+y+'.gif', rgb)


# list with viewable filenames
viewableList = ['img.'+str(x)+'.gif' for x in list(numberList)]

# store frames in numpy array and  print their shape and number
@profile
def imageArray():
    frames = []
    imgShape = []
    with list(viewableList) as List:
        for i in List:
        image = cv.imread(i)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        frames.append(image)
    frames = np.array(frames)
    imgShape = image.shape
    return frames, imgShape

frames, imgShape = imageArray()
nframes = len(frames)
height, width = imgShape


# counting pixels
toothx = 160 # pixels, ~4 teeth, width/4, 640/4

# calculate good points to track
@profile
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
    imshow(scores)
    canvas = cv.cvtColor(scores, cv.COLOR_GRAY2BGR)
    cv.circle(canvas, center=maxLoc, radius=3, thickness=cv.FILLED, color=(0,0,1))
#    imshow(canvas)


scale = 4
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



xmax = 1 * toothx
anchor_index = 0
anchor_pos = np.float32([0,0])


abs_shifts = []
@profile
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

k = 0
while k < nframes:
    anchor = scaled_frames[anchor_index]
    frame = scaled_frames[k]
    relshift = calculate_shift(anchor, frame, normalize=False) / scale
    (relx, rely) = relshift

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


