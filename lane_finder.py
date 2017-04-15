import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "cc.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

nx = 9
ny = 6

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def thresholding(img, s_thresh, sx_thresh):
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    binary=sxbinary | s_binary.astype(int)
    return binary

def perspectiveTransform(img):
    #offset = 100  # offset for dst points
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    w=img.shape[1]
    h = img.shape[0]
    # For source points I'm grabbing the outer four detected corners
    src = np.float32([[[200, h]], [[w/2-50, 450]], [[w/2+50, 450]], [[w-200, h]]])
    ovrlines1 = cv2.line(img.copy(), tuple(src[0][0]), tuple(src[1][0]), color=[255, 0, 0], thickness=2)
    ovrlines2 = cv2.line(img.copy(), tuple(src[1][0]), tuple(src[2][0]), color=[255, 0, 0], thickness=2)
    ovrlines3 = cv2.line(img.copy(), tuple(src[2][0]), tuple(src[3][0]), color=[255, 0, 0], thickness=2)
    newimg=cv2.addWeighted(img*100, 1, cv2.add(cv2.add(ovrlines1, ovrlines2), ovrlines3), .8, 0.)
    plt.imshow(newimg)
    #plt.show()

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = np.float32([[[300, h]], [[300, 0]], [[w-300, 0]], [[w-300, h]]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img.astype(float), M, img_size)
    return warped

def pipeline(img):
    # remove image distortion
    undistorted_img = cal_undistort(img, objpoints, imgpoints)

    # Color and Gradient Thresholding
    thrshd_img=thresholding(undistorted_img, s_thresh=(170, 255), sx_thresh=(20, 100))

    #Perspective Transform
    warped=perspectiveTransform(thrshd_img)
    #Finding the lines

    #result=undistorted_img
    return warped



image = mpimg.imread('test_images/test5.jpg')
#image = mpimg.imread('camera_cal/calibration1.jpg')

result = pipeline(image)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(result, cmap='gray')
ax2.set_title('Pipeline Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()