#!/usr/bin/env python
# coding: utf-8

# # PROJECT VISION - Rediscovering Mobility For Blind

# In[1]:


import numpy as np
import cv2
import argparse
import sys
from  matplotlib import pyplot as plt


# # 1. Cam Calibration

# In[2]:


def load_stereo_coefficients(path):
    """ 
    Loads stereo matrix coefficients. 
    """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]


# In[3]:


K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(
    "calibration/calibration_file.txt"
)  # Get cams params


# # 2. Finding the distance of each pixel of the image

# In[4]:


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


# In[5]:


rightFrame = cv2.imread("images/right.jpeg")
leftFrame = cv2.imread("images/left.jpeg", cv2.IMREAD_COLOR)

height, width, channel = leftFrame.shape  # We will use the shape for remap

print("Images from camera: ")
#plotting
f, ax = plt.subplots(1,2, figsize=(12, 3))
ax[0].imshow(cv2.cvtColor(leftFrame, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(rightFrame, cv2.COLOR_BGR2RGB))
plt.show()


# In[6]:


"""
Undistortion and Rectification part! Undistorts and Rectifies the images using the Calibration codes
"""
leftMapX, leftMapY = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1, (width, height), cv2.CV_32FC1
)
left_rectified = cv2.remap(
    leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2, (width, height), cv2.CV_32FC1
)
right_rectified = cv2.remap(
    rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
)

print("After rectification: ")
#plotting
f, ax = plt.subplots(1,2, figsize=(12, 3))
ax[0].imshow(cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(right_rectified, cv2.COLOR_BGR2RGB))
plt.show()


# In[7]:


"""
disp_matrix = depth_map(gray_left, gray_right)
"""
#We need grayscale for disparity map.

#gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
gray_left = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
#gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)

#disp_matrix = depth_map(right_rectified,left_rectified)  # Get the disparity map
disp_matrix = depth_map(rightFrame,leftFrame)  # Get the disparity map
#print(disp_matrix)

print("Depth map:")
plt.figure(figsize=(15, 4.5))
plt.imshow(cv2.cvtColor(disp_matrix, cv2.COLOR_BGR2RGB))
plt.show()


# In[8]:


"""
distance_matrix = (base offset x focal length)/disp_matrix
"""
distance_matrix = []
base =  0.07 # 1 / Q[3, 2] base offset (distance between the two cameras)
focal = Q[2, 3] # Focal Length of the cameras

infi = 10e15;count =0
for i in range(disp_matrix.shape[0]):
    for j in range(disp_matrix.shape[1]):
        if disp_matrix[i][j] == 0:
            disp_matrix[i][j] = (1/infi)

distance_matrix = (base*focal)/disp_matrix
print(distance_matrix)


# # 3. YOLO on the Right Image

# In[9]:


path_dir = 'yolo'
#extracting network from yolov3.weights 
net = cv2.dnn.readNet(f'{path_dir}/yolov3.weights' , f'{path_dir}/yolov3.cfg')

#extracting the name of objects
with open(f'{path_dir}/coco.names','r' ) as f:
    classes = f.read().splitlines()


# In[10]:


"""
VIDEO CAPTURE
"""
#cap = cv.VideoCapture(0)
#BGR image loaded

#img = right_rectified
img = rightFrame
#img_copy = right_rectified
img_copy = rightFrame

#while True:
#_,img = cap.read()
#height ,width and layersof the image
height,width,l = img.shape 

#resize to RGB (0-1 scale) 416 image for yolo
blob = cv2.dnn.blobFromImage(img, 1/255 , (416,416),(0,0,0) , swapRB= True , crop = False)


# In[11]:


"""
Model Input
"""
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames() #finiding the unconnected layers
layerOutputs = net.forward(output_layers_names) #returns an array of output layers


"""
Detecting the rectangle with max confidence
"""
boxes = [] #stores the top left corner index of the box and the h and w 
confidences = [] #storesthe max conf of the box 
class_ids = [] #stores the index of max conf 

for output in layerOutputs:
    for detection in output:
        scores = detection[5:] #first 5 are the dimensions of box and if object is present or not 
        class_id = np.argmax(scores)
        conf = scores[class_id]
        # Setting confidence threshold = 0.1
        if conf > 0.1:
            cx = int(detection[0]*width) #center x of the box
            cy = int(detection[1]*height) #center y of the box
            w= int(detection[2]*width)
            h=int(detection[3]*height)

            bx = int(cx-w/2) #left corner x
            by = int(cy-h/2) #left corner y

            boxes.append([bx,by,w,h])
            confidences.append((float(conf)))
            class_ids.append(class_id) 

#Non-max supression
indexes = cv2.dnn.NMSBoxes(boxes , confidences ,0.3 ,0.4)

#font and different colours
font = cv2.FONT_HERSHEY_PLAIN 
colors = np.random.uniform(0,255 , size=(len(boxes),3))


# # 4. Calculating Distance of the detected objects

# In[12]:


#Drawing Rectangles
for i in indexes.flatten():
    top_leftX,top_leftY,width,height = boxes[i] 
    label = str(classes[class_ids[i]]) #name of the object
    confidence = str(round(confidences[i],2)) #confidence of the object
    #print (i , " : " , label , " : " , confidence )
    print ("detected object: ", label)
    print ("confidence: " , confidence)
    print( "x-coordinate: " , top_leftX , "\t" , "y-coordinate:" , top_leftY)
    print("width: " , width , "\t" , "height: " , height)
    sum =0; count =0

    for x in range(top_leftY,top_leftY+height):
        for y in range(top_leftX,top_leftX+width):
            if distance_matrix[x][y]<15:
                sum += distance_matrix[x][y]
                count +=1
    dist = sum/count
    print(f"{label} distance: {round(dist,2)}m\n")
    color = colors[i] 
    cv2.rectangle(img_copy , (top_leftX,top_leftY) , (top_leftX+width , top_leftY+height) , color , 2)
    cv2.putText(img_copy,label+" - " + f"{round(dist,2)}m", (top_leftX , top_leftY+20) , font , 1 ,(255,255,255),2)

# cv2.imshow(img_copy) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.figure(figsize=(20, 6))
plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.show()


# ## Team Members

# 1. Vinayak Nishant Gudipaty, 2nd year Electrical, IIT Madras
# 2. Yogesh Agarwala, 2nd year Electrical, IIT Madras
# 3. Harshit Raj, 2nd year Mechanical, IIT Madras
# 4. Saroopa G, 2nd year Mechanical, IIT Madras
# 5. Anish Pophale, 1st year Chemical, IIT Madras
