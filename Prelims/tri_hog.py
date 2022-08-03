import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os 

os.chdir("..")
os.chdir("images")
L=cv.imread("left.jpeg",0)
R=cv.imread("right.jpeg",0)

stereo = cv.StereoBM_create(numDisparities=128, blockSize=13)
disparity = stereo.compute(L,R)
plt.imshow(disparity,'gray')
plt.show()

hog = cv.HOGDescriptor()
h = hog.compute(disparity)
print(h.shape)