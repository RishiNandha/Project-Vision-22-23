import cv2 as cv
import numpy as np
import statistics as stats
import cv2


def pool(im, a=10, b=0.5):
	for x in range(250,0,-10):
		if ((x-a <= im) & (im <= x+a)).sum() > b * im.size:
			return(x)
	else:
		return x

def localize(im,a=10,b=0.5, pool_x=5, pool_y=5, H=1920, W=1080, label="Image"):
	w,h = im.shape
	newimg = np.array([pool(im[i:i+pool_x, j:j+pool_y], a, b) for i in range(0,w-w%pool_x,pool_x) for j in range(0,h - h%pool_y,pool_y)]).reshape(w//pool_x, h//pool_y)
	cv2.imshow(label+str(newimg.shape), cv2.resize(newimg.astype('uint8'),(H,W), interpolation = cv.INTER_NEAREST))
	return newimg

#img_matrix = np.array([shapenret(img[i:i+x,j:j+y]) for i in range(0,x*n,x) for j in range(0,y*n,y)]).reshape(n,n,x,y,3)

def test(path):
	img = cv.imread(path, cv2.IMREAD_GRAYSCALE)
	cv.imshow(path, img)
	x, y = img.shape
	while(img.size > 81):
		img = localize(img,10,0.3,5, 5, y, x, path)


test('depth_map_1.jpg')
#test('depth_map_2.jpg')
#test('depth_map_3.jpg')
#test('depth_map_4.jpg')
cv.waitKey(0)