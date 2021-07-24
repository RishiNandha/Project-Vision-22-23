import numpy as np
import cv2
import glob
import argparse
import sys

# Set the values for your cameras
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(2)

# Use these if you need high resolution.
# capL.set(3, 1024)  # width
# capL.set(4, 768)  # height

# capR.set(3, 1024) # width
# capR.set(4, 768) # height
i = 1

while True:
    if not (capL.grab() and capR.grab()):
        print("No more frames")
        break

    _, leftFrame = capL.retrieve()
    _, rightFrame = capR.retrieve()

    # Use if you need high resolution. If you set the camera for high res, you can pass these.
    # cv2.namedWindow("capL", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("capL", 1024, 768)

    # cv2.namedWindow('capR', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('capR', 1024, 768)

    cv2.imshow("capL", leftFrame)
    cv2.imshow("capR", rightFrame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("c"):
        cv2.imwrite(
            "left/left" + str(i) + ".png",
            leftFrame,
        )
        cv2.imwrite(
            "right/right" + str(i) + ".png",
            rightFrame,
        )
        i += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()