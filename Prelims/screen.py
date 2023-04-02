import cv2 as cv
import keyboard


cam0 = cv.VideoCapture(0, cv.CAP_DSHOW)
cam1 = cv.VideoCapture(1, cv.CAP_DSHOW)

while(True):
	# Capture frame-by-frame
	_, frame0 = cam0.read()
	_, frame1 = cam1.read()
	
	try:
		cv.imshow(f'preview_0',frame0)
		cv.imshow(f'preview_1',frame1)
	except:
		print(f"Some port didn't work")
		break
	
	cv.waitKey(1)
	# Waits for a user input to quit the application
	if keyboard.is_pressed('q'):
		cv.destroyAllWindows()
		break