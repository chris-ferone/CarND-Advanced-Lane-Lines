import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

#Camera Calibration: Correcting for Distortion
# function takes an image, object points, and image points
# performs the camera calibration, image distortion correction and
# returns the undistorted image


def findimagepoints(nx, ny):
    objpoints=[] #3d points in real world space; object points will be the same; z coordinate will be zero for every point
    #since board in on a falt image plane (i.e. (0,0,0), (0,1,0), (0,2,0), (0,3,0), (1,0,0), (1,1,0), etc.)
    imgpoints=[] #2d points in image plane

    #prepare object points
    objp=np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # read in and make a list of calibration images
    images = glob.glob('camera_cal\\calibration*.jpg')
    i=0
    for fname in images:
        # read in each image
        img = mpimg.imread(fname)

        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        #print("image name: ", fname)
        # if corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            #i=i+1
            #print("count: ", i)
            # draw chessboard corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)
            #temp_path=os.path.split(fname)
            #plt.savefig(temp_path[1])
            #plt.show()
    return objpoints, imgpoints


nx=9
ny=6
#calibrate camera
[objpoints, imgpoints] = findimagepoints(nx, ny)

pickle.dump( {"objpoints": objpoints, "imgpoints": imgpoints}, open( "cc.p", "wb" ) )