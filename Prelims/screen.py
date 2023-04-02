import cv2 as cv
import keyboard

while(True):
	p = int(input("Enter port number to preview: "))
	cam = cv.VideoCapture(p)
	while(True):
		# Capture frame-by-frame
		_, frame = cam.read()
		try:
			cv.imshow(f'preview_{p}',frame)
		except:
			print(f"Port {p} didn't work")
			break
		cv.waitKey(1)
		# Waits for a user input to quit the application
		if keyboard.is_pressed('q'):
			cv.destroyAllWindows()
			break