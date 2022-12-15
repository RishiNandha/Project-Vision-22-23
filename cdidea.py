import cv2 as cv
import numpy as np

img = cv.imread('images/left.jpeg')

h,w,channels = img.shape
print(img.shape)
n = 3
y = w//n
x = h//n

def shapenret(n):
	print(n.shape)
	return n

img_matrix = np.array([shapenret(img[i:i+x,j:j+y]) for i in range(0,x*n,x) for j in range(0,y*n,y)]).reshape(n,n,x,y,3)

for L in range(n):
	for ratio in range(n):
		name = str(L) +":"+ str(ratio)
		cv.imshow(name,img_matrix[L][ratio])
cv.waitKey(0)