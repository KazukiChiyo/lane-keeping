import cv2
import glob
import pickle
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read images
images = glob.glob('camera_cal/calibration*.jpg')

# Arrays to store object points and image points from all the test_images
objpoints = []
imgpoints = []

# Prepare object points, like (0,0,0),(1,0,0),...
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x,y coordinates

for fname in images:
    # Read in each image
    image = mpimg.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # Draw and display the corners
        cor_image = cv2.drawChessboardCorners(image, (9,6), corners, ret)
        scipy.misc.imsave('camera_cal/corners/'+fname.split('/')[1], cor_image)

# Save image points and object points to pickle file
points_dist = {}
points_dist['objpoints'] = objpoints
points_dist['imgpoints'] = imgpoints

with open('camera_cal/points_dist_pickle.p', 'wb') as handle:
    pickle.dump(points_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('camera_cal/points_dist_pickle.p', 'rb') as handle:
#     b = pickle.load(handle)
