import cv2 as cv
import numpy as np
import statistics as stats
import cv2

img = cv.imread('depth_map_1.jpg')
cv.imshow("original", img)
h,w,channels = img.shape
n = 9
y = w//n
x = h//n

def shapenret(n):
	return n

def reduction(matrix):
    red_matrix = []
    for i in range(0,n,3):
        temp1 = []
        for j in range(0,n,3):
            temp2 = []
            for k in range(i,i+3):
                for l in range(j,j+3):
                    temp2.append(matrix[k][l])
            red = stats.mode(temp2)
            # print(temp2)
            # print(red)
            # print()
            temp1.append(red)

        # print(temp1)
        red_matrix.append(temp1)

    for i in matrix:
        for j in i:
            print(j," ",end ='')            
        print()
    
    if len(red_matrix) == 3:
        return red_matrix
    else:
        reduction(matrix)

img_matrix = np.array([shapenret(img[i:i+x,j:j+y]) for i in range(0,x*n,x) for j in range(0,y*n,y)]).reshape(n,n,x,y,3)

for L in range(n):
	for ratio in range(n):
		name = str(L) +":"+ str(ratio)
		#cv.imshow(name,img_matrix[L][ratio])
		
dom_color_matrix = []
for image_row in img_matrix:
    temp = []
    for image in image_row:
        data = np.reshape(image,(-1,3))
        data = np.float32(data)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv.KMEANS_RANDOM_CENTERS
        compactness,labels,centers = cv.kmeans(data,1,None,criteria,10,flags)

        temp.append(centers[0].astype(np.int32)[0])

    dom_color_matrix.append(temp)

print("Initial Color Matrix:")
reduced_matrix = reduction(dom_color_matrix)
redimg = np.array(reduced_matrix)
print(redimg.shape)
redimg = cv2.resize(redimg.astype(float),(x*n,y*n),interpolation= cv2.INTER_LINEAR)
cv2.imshow("Reduced Color Matrix", redimg)
trigger_matrix = []

print("\nReduced Color Matrix:")
for i in reduced_matrix:
    temp = []
    for j in i:
        if j > 110:
            temp.append(1)
        else:
            temp.append(0)
        print(j," ",end ='')
    
    trigger_matrix.append(temp)
        
    print()

print("\nTrigger Matrix:")

for i in trigger_matrix:
    for j in i:
        print(j," ",end = "")
    print()
triimg = np.array(trigger_matrix).resize(x*n,y*n)
cv2.imshow("Trigger Matrix", triimg)
cv.waitKey(0)