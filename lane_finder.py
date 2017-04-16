import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

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
    return warped, M

def findlanelines(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    #ax1.plot(histogram)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #print("leftx_base ", leftx_base)
    #print("rightx_base ", rightx_base)
    #print("binary warped type: ", binary_warped.dtype)
    #plt.imread(binary_warped)
    #plt.show()
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    #print("max: ", np.argmax(out_img))

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    cax = ax1.imshow(out_img)
    fig1.colorbar(cax)

    ax1.plot(left_fitx, ploty, color='yellow')
    ax1.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    [left_curverad, right_curverad] = rocCalc(ploty, lefty, leftx, righty, rightx)
    fitWithHistory(binary_warped, left_fit, right_fit)

    return histogram, left_fitx, right_fitx, ploty, left_curverad, right_curverad

def calcLaneOffset(left_fitx, right_fitx):
    #Calculate offset from lane center
    #left_lane = left_fit_cr[0] * y_eval ** 2 + left_fit_cr[1] * y_eval +left_fit_cr[0]
    #right_lane = right_fit_cr[0] * y_eval ** 2 + right_fit_cr[1] * y_eval + right_fit_cr[0]
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    midpoint = (left_fitx[-1]+right_fitx[-1])/2
    #print("left: ", left_fitx[-1])
    #print("right: ", right_fitx[-1])
    offset_p = 1280/2-midpoint #offset in pixels
    offset_m = offset_p * xm_per_pix  #offset in meters
    return offset_m

def rocCalc(ploty, lefty, leftx, righty, rightx):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad

def fitWithHistory(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    new_left_fit = np.polyfit(lefty, leftx, 2)
    new_right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = new_left_fit[0] * ploty ** 2 + new_left_fit[1] * ploty + new_left_fit[2]
    right_fitx = new_right_fit[0] * ploty ** 2 + new_right_fit[1] * ploty + new_right_fit[2]
    return new_left_fit, new_right_fit, ploty

    #Visualization Below

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(result)
    ax1.plot(left_fitx, ploty, color='yellow')
    ax1.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

def drawImageBackOnRoad(warped, left_fitx, right_fitx, ploty, M, undist, left_curverad, right_curverad, offset_m):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv=np.linalg.inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.putText(result, "left curv: %d (m)" % left_curverad, (20, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    result = cv2.putText(result, "right curv: %d (m)" % right_curverad, (900, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    result = cv2.putText(result, "pos: %.2f (m)" % offset_m, (500, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #print("offset: ", offset_m)
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(result)
    return result

def pipeline(img):
    # remove image distortion
    undistorted_img = cal_undistort(img, objpoints, imgpoints)

    # Color and Gradient Thresholding
    thrshd_img=thresholding(undistorted_img, s_thresh=(170, 255), sx_thresh=(20, 100))

    #Perspective Transform
    [warped, M] = perspectiveTransform(thrshd_img)

    #convert to warped image to binary warped
    binary_warped = np.zeros_like(warped, dtype=np.uint8)
    binary_warped[np.nonzero(warped)] = 1

    #Finding the lane lines
    fig=plt.figure()
    ax1 = fig.add_subplot(1,2,2)
    ax1.imshow(warped)
    ax2 = fig.add_subplot(1,2,1)
    ax2.imshow(binary_warped)
    [histogram, left_fitx, right_fitx, ploty, left_curverad, right_curverad] = findlanelines(binary_warped)

    #Calculate vehicle offset
    offset_m = calcLaneOffset(left_fitx, right_fitx)

    #draw image back on road
    result = drawImageBackOnRoad(warped, left_fitx, right_fitx, ploty, M, undistorted_img, left_curverad, right_curverad, offset_m)

    return result

UseStillImage = False

if UseStillImage:
    image = mpimg.imread('test_images/test5.jpg')
    # image = mpimg.imread('camera_cal/calibration1.jpg')

    result = pipeline(image)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result, cmap='gray')
    ax2.set_title('Pipeline Result', fontsize=40)
    #ax2.plot(result.shape[0]-histogram)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

else:
    input_clip = VideoFileClip('project_video.mp4')#.subclip(0,1)
    output_clip = input_clip.fl_image(pipeline)
    output_clip.write_videofile('output.mp4', audio=False)

