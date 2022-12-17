import cv2 as cv
import numpy as np

#extracting network from yolov3.weights 
net = cv.dnn.readNet('yolov3.weights' , 'yolov3.cfg')

#extracting the name of objects
with open('coco.names','r' ) as f:
    classes = f.read().splitlines()

#VIDEO CAPTURE
cap = cv.VideoCapture('test.mp4')
# BGR image loaded
#img = cv.imread('data/kite.jpg')

while True:
    _,img = cap.read()
    #height ,width and layersof the image
    height , width , l = img.shape 

    #resize to RGB (0-1 scale) 416 image for yolo
    blob = cv.dnn.blobFromImage(img, 1/255 , (416,416),(0,0,0) , swapRB= True , crop = False)

    #Model Input
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames() #finiding the unconnected layers
    layerOutputs = net.forward(output_layers_names) #returns an array of output layers

    #Detecting the rectangle with max confidence 
    boxes = [] #stores the top left corner index of the box and the h and w 
    confidences = [] #storesthe max conf of the box 
    class_ids = [] #stores the index of max conf 

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:] #first 5 are the dimensions of box and if object is present or not 
            class_id = np.argmax(scores)
            conf = scores[class_id]
            # Setting confidence threshold = 0.5
            if conf > 0.5 :
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
    indexes = cv.dnn.NMSBoxes(boxes , confidences ,0.5 ,0.4)

    #font and different colours
    font = cv.FONT_HERSHEY_PLAIN 
    colors = np.random.uniform(0,255 , size=(len(boxes),3))

    #Drawing Rectangles
    for i in indexes.flatten():
        x,y,w,h = boxes[i] 
        label = str(classes[class_ids[i]]) #name of the object
        confidence = str(round(confidences[i],2)) #confidence of the object
        print (i , " : " , label , " : " , confidence )
        print( "x-coordinate :" , x , "\t" , "y-coordinate :" , y  )
        print("width : " , w , "\t" , "height : " , h , "\n")
        color = colors[i] 
        cv.rectangle(img , (x,y) , (x+w , y+h) , color , 2)
        cv.putText(img,label+" " + confidence , (x , y+20) , font , 2 ,(255,255,255),2)


    cv.imshow("Image" , img)    
    key = cv.waitKey(1)
    if key == 27 :
        break
    
cap.release()
cv.destroyAllWindows()
