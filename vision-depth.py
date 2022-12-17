# %% [markdown]
# # PROJECT VISION - Rediscovering Mobility For Blind
# %%
import numpy as np
import cv2
import argparse
import sys
from  matplotlib import pyplot as plt
#import serial
import time
#ser = serial.Serial('COM6',9600)

# %% [markdown]
# # 1. Cam Calibration
def nothing(x):
    pass
# %%
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
def undistortRectify(frameR, frameL):

    # Undistort and rectify images
    undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


    return undistortedR, undistortedL
# %% [markdown]
# # 2. Finding the distance of each pixel of the image
cv2.createTrackbar('numDisparities','disp',1,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('preFilterType','disp',1,1,nothing)
cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',5,25,nothing)
# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()
# %%
def depth_map(imgL, imgR):
    """ 
    Depth map calculation. Works with SGBM and WLS. 
    Need rectified images, returns depth map ( left to right disparity ) 
    """
    # SGBM Parameters
    window_size = 7  
    # wsize 
    # default 3; 5; 
    # 7 for SGBM reduced size image; 
    # 15 for SGBM full size image (1300px and above); 
    # 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=1,
        numDisparities= 5 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize= window_size,
        P1=8 * 3 * window_size,
        P2= 32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    # wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR).astype(np.float32) / 16
    return displ

# %%
capL = cv2.VideoCapture(1, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# %%
while True:
    #rightFrame = cv2.imread("images/right.jpeg")
    #leftFrame = cv2.imread("images/left.jpeg", cv2.IMREAD_COLOR)
    if not (capL.grab() and capR.grab()):
        print("No more frames")
        break

    _, leftFrame = capL.read()
    _, rightFrame = capR.read()

    cv2.imshow("capL", leftFrame)
    cv2.imshow("capR", rightFrame)

    height, width, channel = leftFrame.shape  # We will use the shape for remap

    # print(height, width, channel)
    # print("Images from camera: ")
    # # plotting
    # f, ax = plt.subplots(1,2, figsize=(12, 3))
    # ax[0].imshow(cv2.cvtColor(leftFrame, cv2.COLOR_BGR2RGB))
    # ax[1].imshow(cv2.cvtColor(rightFrame, cv2.COLOR_BGR2RGB))
    # plt.show()

    """
    Undistortion and Rectification part! Undistorts and Rectifies the images using the Calibration codes
    """
    left_rectified, right_rectified = undistortRectify(leftFrame, rightFrame)

    # print("After rectification: ")
    # plotting
    # f, ax = plt.subplots(1,2, figsize=(12, 3))
    # ax[0].imshow(cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB))
    # ax[1].imshow(cv2.cvtColor(right_rectified, cv2.COLOR_BGR2RGB))
    # plt.show()

    # cv2.imshow("Rectified L", left_rectified)
    # cv2.imshow("Rectified R", right_rectified)

    L = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    R= cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType','disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv2.getTrackbarPos('minDisparity','disp')
        # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(L,R)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 time.time()s.

    # Converting to float32 
    disparity = disparity.astype(np.float32)/255
    cv2.imshow("BGM",disparity)
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities

    # Displaying the disparity map
    cv2.imshow("disp",disparity)

    # Close window using esc key
    if cv2.waitKey(1) == 27:
        break


    else:
        capL= cv2.VideoCapture(0)
        capR= cv2.VideoCapture(1)
