**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistort_road_output.png "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./examples/roc.png "roc eqn"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 14 through 51 of the file called `CameraCalibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The code for this next step is contained in lines 18 through 21 of the file called `lane_finder.py`. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 23 through 45 in `lane_finder.py`).  I converted the image RGB color space to HLS color space, and then seperated the L and S channels. First I take the derivative in the x direction of the L channel image. After scalling, I save all values with high and low thresholds. Then I threshold the S channel. Finally, I "OR" the output from both thresholding operations.  

Here's an example of my output for this step.  

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspectiveTransform(img)`, which appears in lines 47 through 74 in the file `lane_finder.py`.  The function takes as inputs an image (`img`). I chose to hardcode the source and destination points in the following manner:

```
    w=img.shape[1]
    h = img.shape[0]
src = np.float32(
	[[[200, h]], 
	[[w/2-50, 450]], 
	[[w/2+50, 450]], 
	[[w-200, h]]])
dst = np.float32(
	[[[300, h]], 
	[[300, 0]], 
	[[w-300, 0]], 
	[[w-300, h]]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 300, 720      | 
| 590, 450      | 300, 0        |
| 690, 450      | 980, 0        |
| 1080, 720     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I did this in lines 64 through 161 in my code in `lane_finder.py` First, I identified which pixels belonged to each lane line. I began by taking a histogram in the x direction. The peak of the left and right halves of the histogram gave me the starting point for the left and right lines. I then created two sliding windows and centered them about these starting points. Within each window, I searched for non-zero pixels. If I found a minimum number of pixels, I re-cented then next window around the mean of those pixels. I repeated this over a total of 9 sliding widows for each lane. I then extracted all of the non-zero pixels for each lane and fit a 2nd order polynomial to each lane. 

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 163 through 195 in my code in `lane_finder.py` To find the radius of curvature, I first coverted the set of non-zero(x,y) points found around each lane from pixels to meters. Then I fit a new polynomial to the world-space-based points. I then evaluated the following radius of curvature equation at the bottom of the image (i.e, y=719), which is the closest point to the vehicle.

![alt text][image7]

A and B are the coefficients that were found from the curve fit. 

The position offset of the vehicle was calculated by finding the difference between the center of the image (1280/2) and midpoint between the two curve fits (evaluated at the bottom of the image, y=719). The difference was then converted from pixel space to world space in meters.  

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 255 through 282 in my code in `lane_finder.py` in the function `drawImageBackOnRoad()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Certainly it was challenging to tune the saturation and gradient thresholds correctly. I also had a lot difficulty plotting the warped images correctly and overlaying the source and destination points, and therefore could not immediately verify that I was conducting the perspective transform correctly. 

The pipeline might fail under poor weather conditions and/or sharp turns. I have only tested it on a video clip showing sunny and clear weather conditions with gentle turns at highway speeds. 

Perhaps using an approach that dead reckons when lane position and curvature cannot be extracted from an image alone. Vehicle speed, steering angle, and previous path history could be used to estimate lane curvature when image-based calculations are unreliable and need to be discarded. This estimate could be combined with image-based lane "measurements" using a Kalman filter. 