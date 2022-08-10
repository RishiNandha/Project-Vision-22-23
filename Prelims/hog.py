from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

address_L = 'tsukuba_l.jpeg'
address_R = 'tsukuba_r.jpeg'

imgR = imread(address_R)

if imgR.ndim==2:
	imgR=imgR.reshape((imgR.shape[0],imgR.shape[1],1))

fd, hog_image = hog(imgR, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True) 
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 


ax1.imshow(imgR, cmap=plt.cm.gray) 
ax1.set_title('Input image') 

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
ax2.set_title('Histogram of Oriented Gradients')

imgR_g = cv.imread(address_R, 0)
imgL_g = cv.imread(address_L, 0)
stereo = cv.StereoBM_create(numDisparities=16, blockSize=13)
img = stereo.compute(imgL_g,imgR_g)
#	img = exposure.rescale_intensity(img, in_range=(0, 10)) 
ax3.imshow(img, cmap=plt.cm.gray)
ax3.set_title('Depth Map')
if img.ndim==2:
	img=img.reshape((img.shape[0],imgR.shape[1],1))
fd, hog2 = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
hog2 = exposure.rescale_intensity(hog2, in_range=(0,10)) 
ax4.imshow(hog2, cmap=plt.cm.gray)
ax4.set_title('HOG over Depth Map')

plt.show()