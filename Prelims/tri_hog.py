import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os 

L=cv.imread("left.png",0)
R=cv.imread("right.png",0)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(L,R)
plt.imshow(disparity,'gray')
plt.show()

hog = cv.HOGDescriptor()
h = hog.compute(disparity)
print(h.shape)