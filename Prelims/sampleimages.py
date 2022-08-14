import numpy as np
import cv2 as cv
from  matplotlib import pyplot as plt
import os
import keyboard
os.chdir("sampleimages")

# Testing if Cameras are working
def list_ports():
    is_working = True
    dev_port = 0
    while is_working:
        camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
        dev_port +=1
list_ports()

camL = cv.VideoCapture(0)
camR = cv.VideoCapture(1)

# Preview for Focus Adjusting
while(True):
	# Capture frame-by-frame
	ret, frameL = camL.read()
	ret, frameR = camR.read()
	# Display the resulting frame
	cv.imshow('preview_L',frameL)
	cv.imshow('preview_R', frameR)
	cv.waitKey(1)
	# Waits for a user input to quit the application
	if keyboard.is_pressed('q'):
		cv.destroyAllWindows()
		break

index=int(input("Index : "))
while index>=0:
	temp, imgL = camL.read()
	temp, imgR = camR.read()
	# Checking if Right and Left are correct

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
	ax1.imshow(cv.cvtColor(imgL,cv.COLOR_BGR2RGB))
	ax2.imshow(cv.cvtColor(imgR,cv.COLOR_BGR2RGB))
	plt.show()

	okay = int(input("Works? (1=Yes, 2=Invert, -1=Nope) : "))

	# Writing

	if okay==1:
		cv.imwrite("L"+str(index)+".jpeg", imgL)
		cv.imwrite("R"+str(index)+".jpeg", imgR)
	elif okay==2:
		cv.imwrite("L"+str(index)+".jpeg", imgR)
		cv.imwrite("R"+str(index)+".jpeg", imgL)
	index=int(input("Index : "))
