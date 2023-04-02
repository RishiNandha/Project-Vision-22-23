import cv2 as cv
import keyboard


cam0 = cv.VideoCapture(0)
cam1 = cv.VideoCapture(1, cv.CAP_DSHOW)
cam2 = cv.VideoCapture(2)
cam3 = cv.VideoCapture(3)
while(True):
	# Capture frame-by-frame
	_, frame0 = cam0.read()
	_, frame1 = cam1.read()
	_, frame2 = cam2.read()
	_, frame3 = cam3.read()
	
	try:
		cv.imshow(f'preview_0',frame0)
		cv.imshow(f'preview_1',frame1)
		cv.imshow(f'preview_2',frame2)
		cv.imshow(f'preview_3',frame3)
	except:
		print(f"Some port didn't work")
		break
	
	cv.waitKey(1)
	# Waits for a user input to quit the application
	if keyboard.is_pressed('q'):
		cv.destroyAllWindows()
		break