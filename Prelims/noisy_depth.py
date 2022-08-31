import cv2
import numpy as np

imgL = cv2.imread('sampleimages/L0.jpeg')
imgR = cv2.imread('sampleimages/R0.jpeg')

height= 360
width = 640

def load_stereo_coefficients(path):

    # Loads stereo matrix coefficients. 
    
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

# %%
K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(
    "../calibration/calibration_file.txt"
)  # Get cams params

print(K1,D1,K2,D2,R, T, E, F, R1, R2, P1, P2, Q)
# %% [markdown]
# # 2. Finding the distance of each pixel of the image

# %%
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

leftMapX, leftMapY = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1, (width, height), cv2.CV_32FC1
)
left_rectified = cv2.remap(
    imgL, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2, (width, height), cv2.CV_32FC1
)
right_rectified = cv2.remap(
    imgR, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
)


import matplotlib.pyplot as plt 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True) 
ax1.imshow(imgL)

ax2.imshow(imgR)

dm = depth_map(imgL,imgR)
ax3.imshow(dm,cmap='gray')
cv2.imwrite('sampleimages/dm0.jpeg', dm)
plt.show()