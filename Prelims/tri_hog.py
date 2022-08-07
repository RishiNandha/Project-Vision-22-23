import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os 

os.chdir("..")
os.chdir("images")
L=cv.imread("left.jpeg",0)
R=cv.imread("right.jpeg",0)

stereo = cv.StereoBM_create(numDisparities=128, blockSize=13)
img = stereo.compute(L,R)
plt.imshow(img,'gray')
plt.show()

winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#compute(img[, winStride[, padding[, locations]]]) -> descriptors
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)
hist = hog.compute(img,winStride,padding,locations)