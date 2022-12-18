
# # PROJECT VISION - Rediscovering Mobility For Blind

import numpy as np
import cv2
from  matplotlib import pyplot as plt
import time

import serial
arduino=serial.Serial(port='COM8',baudrate=9600)
# # 1. Cam Calibration
cv_file = cv2.FileStorage()
cv_file.open('Camera Calibration/stereoMap.xml', cv2.FileStorage_READ)

state_old='00'
state='00'

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
def undistortRectify(frameR, frameL):

    # Undistort and rectify images
    undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


    return undistortedR, undistortedL

# # 2. Finding the distance of each pixel of the image

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

path_dir = 'yolo'
#extracting network from yolov3.weights 
net = cv2.dnn.readNet(f'{path_dir}/yolov3.weights' , f'{path_dir}/yolov3.cfg')

#extracting the name of objects
with open(f'{path_dir}/coco.names','r' ) as f:
    classes = f.read().splitlines()


capL = cv2.VideoCapture(1, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(2, cv2.CAP_DSHOW)
while True:
    if not (capL.grab() and capR.grab()):
        print("No more frames")
        break

    _, leftFrame = capL.read()
    _, rightFrame = capR.read()



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

    """
    disp_matrix = depth_map(gray_left, gray_right)
    """
    #We need grayscale for disparity map.
    gray_left = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)

    disp_matrix = depth_map(rightFrame, leftFrame)  # Get the disparity map
    #print(disp_matrix)

    # print("Depth map:")
    # plt.figure(figsize=(15, 4.5))
    # cv2.imshow('Depth Map:',cv2.cvtColor(disp_matrix, cv2.COLOR_BGR2RGB))
    # plt.show()

    """
    distance_matrix = (base offset x focal length)/disp_matrix
    """
    distance_matrix = []
    base =  0.055 # 1 / Q[3, 2] base offset (distance between the two cameras)
    focal = 3 # Focal Length of the cameras

    infi = 10e15;count =0
    for i in range(disp_matrix.shape[0]):
        for j in range(disp_matrix.shape[1]): 
            if disp_matrix[i][j] == 0:
                disp_matrix[i][j] = (1/infi)

    distance_matrix = (base*focal)/disp_matrix
    # print(distance_matrix)

    """
    VIDEO CAPTURE
    """
    #cap = cv.VideoCapture(0)
    #BGR image loaded
    img = rightFrame
    img_copy = rightFrame

    #while True:
    #_,img = cap.read()
    #height ,width and layersof the image
    height,width,l = img.shape 

    #resize to RGB (0-1 scale) 416 image for yolo
    blob = cv2.dnn.blobFromImage(img, 1/255 , (224,224),(0,0,0) , swapRB= True , crop = False)



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
    # print(f'The type of indexes is {type(indexes)}')
    # print(len(indexes)==0)
    #font and different colours
    font = cv2.FONT_HERSHEY_PLAIN 
    colors = np.random.uniform(0,255 , size=(len(boxes),3))
    distance = []
    centre = []
    #Drawing Rectangles
    if(len(indexes) != 0):
        for i in indexes.flatten():
            if str(classes[class_ids[i]])=='tvmonitor':
                continue
            top_leftX,top_leftY,width,height = boxes[i] 
            label = str(classes[class_ids[i]]) #name of the object
            confidence = str(round(confidences[i],2)) #confidence of the object
            #print (i , " : " , label , " : " , confidence )
            # print ("detected object: ", label)
            #print ("confidence: " , confidence)
            #print( "x-coordinate: " , top_leftX , "\t" , "y-coordinate:indexes" , top_leftY)
            #print("width: " , width , "\t" , "height: " , height)
            sum =0; count=1; flag=1

            for x in range(top_leftY,min(top_leftY+height,480)):
                for y in range(top_leftX,min(top_leftX+width,640)):
                    if distance_matrix[x][y]<15:
                        sum += distance_matrix[x][y]
                        if (flag):
                            flag=0
                        else:
                            count+=1
            dist = sum/count
            # print(f"{label} distance: {round(dist,2)}m\n")
            mid = 320
            
            if(label == "person"):
                centreX = (width/2 + (top_leftX))
                centre.append(centreX)
                distance.append(dist)

            color = colors[i] 
            cv2.rectangle(img_copy , (top_leftX,top_leftY) , (top_leftX+width , top_leftY+height) , color , 2)
            cv2.putText(img_copy,label+'-'+f'{round(dist,2)*0.7}', (top_leftX , top_leftY+20) , font , 1 ,(255,255,255),2)
        
        if(len(distance) == 0):
            #arduino.write(bytes('00', 'utf-8'))
            state='00'
            print(f'Both Off')
        else:
            X = centre[distance.index(min(distance))]
            # print(f'X value of centre {X}')
            if(X >= mid):
                state='01'
                print(f'Right On')
                # arduino.write(bytes('01', 'utf-8'))
            else:
                state='10'
                # arduino.write(bytes('10', 'utf-8'))
                print(f'Left On')
    else:
        #ser.write(b'00\n')
        print(f'Both Off')
        state='00'
           
    if state!=state_old:
        arduino.write(bytes(state, 'utf-8'))
        state_old=state

    # cv2.imshow(img_copy) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.figure(figsize=(20, 6))
    # cv2.imshow('Final Image:',cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    cv2.imshow('Final Image:',img_copy)
    # plt.show()
    
    key = cv2.waitKey(1)
    if key == 27:
        break

# %%
# 

# %%



