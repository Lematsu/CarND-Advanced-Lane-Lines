## Self-Driving Car Enginner Nanodegree

### Advanced Lane Finding Project

---

**Overview**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

In order to achieve this, we have designed a softare pipeline described below:
0. Input: Image and/or Video
1. Calibrate Camera
2. Perspective Transform
3. Color threshold binarize
4. Lane line detection
5. Lane line calculation
6. Draw highlighted lane and display information
7. Combine and save images

I have written this pipeline described in "advanced_lane_lines/advanced_lane_lines.ipynb".

Please see the sample image and video I created in "./advanced_lane_lines/advanced_lane_lines.ipynb" below.

[//]:


[image1]: ./camera_cal/calibration1.jpg "Original chessboard"
[image2]: ./output_images/Undistort/undistort_chessboard_12.png "Undistorted chessboard"
[image3]: ./test_images/test1.jpg "Original image"
[image4]: ./test_images/undistort_lines_0.png "Undistorted image"
[image5]: ./output_images/Perspective/test_case_source.png "Original image"
[image6]: ./output_images/Perspective/test_case_warped.png "Birds-Eye image"
[image7]: ./test_images/test1.jpg "Original image"
[image8]: ./output_images/ColorBianry/color_lines_0.png "Binary image"
[image9]: ./output_images/ColorBinary/perspective_and_binarize_5.png
[image10]: ./output_images/LaneCurvature/test_curvature_withline.png
[image11]: ./test_images/test1.jpg
[image12]: ./output_images/LaneCurvature/curvature_0.png

[video0]: ./output_video/processed_project_video.mp4 "Video format"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this section is written in "./advanced_lane_lines/advanced_lane_lines.py" within the "CameraCalibration" class located at line 10. I decided to program in object-orientation because of the number of parameters and static variables I would frequently chnage to complete this project.

I first initialize the class objects with the following parameters:
1. filepath to calibration images
2. filepath to calibration data
3. number of x coordiantes of the chessboard corners
4. number of y coordinates of the chessboard corners
5. debug flag

When initializing, this class object first checks if there's already a pickle file stored. I decided to utilize a pickle file to improve efficiency when testing multiple times using the same calibration. The pickle file is created in the `calibrate` method described below. If a pickle file is not found, the initializer will run the `calibrate` method.

This class object has two methods:
1. `_calibrate`
2. `undistort`

I will first describe the `_calibrate` private method. I create `object_point` to represent (x,y,z=0) using the class parameters for x and y coordinates in float32 format. To calibrate each image, we read the image using OpenCV, perform grayscale on the image, and then find the chessboard corners using OpenCV.For each image I use for calibration, `object_point` is added to the class field `self.object_points` and corner coordinates are added to `self.image_points`. After all images have been iterated over, I perform the `calibrateCamera` with the parameters I found with `self.object_points` and `self.image_points` using OpenCV. After `calibrateCamera` returned it's variables, I create a dictionary called `calibrated_data` and store the data as a pickle file.


### Pipeline (single images)

#### Brief overview
The software pipeline is performed by the class object `LaneCurvature`. Within it's methods, the main public method called `self.process(img)` performs the software pipeline for single images, and video. Please see line 649 in "./advanced_lane_lines/advanced_lane_lines.py". For step by step demonstration, please check "./advanced_lane_lines/advanced_lane_lines.ipynb".

#### 1. Provide an example of a distortion-corrected image.

Next, I will illustrate the `undistort` public method in CameraCalibration. This method is called externally. It first checks if a valid pickle file has been created in the `_calibrate` method previously. If not, it raises an exception that the file does not exist. Next, in case of a valid process, it opens the pickle file, and calls `undistort` with the parameters from the pickle file using OpenCV. The `undistort` function transforms an image to compensate radial and tangetial lens distortion.

From the above calibration, the following images show the original distorted and processed undistorted image of a road.

Here is an example case for distortion correctness using chessboards.

This is the original distorted chessboard image.



![alt_text][image1]

This is the undistorted chessboard image.


![alt_text][image2]

This is the original distorted image. 



![alt_text][image3]

This is the processed undistorted image.



![alt_text][image4]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform is handled by the PerspectiveTransform class object in "./advanced_lane_lines/advanced_lane_lines.py". The paramters it takes for initialization are the original source points and the destination points where the trasnformation will occur. Other than the initializer, the class object contains two methods, `transform` and `transform_inverse`. The method `self.transform` is used to create a warped image for the original image to be bird-eye's viewed. I hard-coded the source points and destinations in "./advanced_lane_lines/advanced_lane_lines.ipynb" as follows below:
```python
left_bottom_corner = [250, 700]
left_top_corner = [580, 450]
right_top_corner = [700, 450]
right_bottom_corner = [1060, 690]

corners = np.float32([left_bottom_corner, left_top_corner,
                      right_top_corner, right_bottom_corner])

new_top_left = np.array([corners[0,0], 0])
new_top_right = np.array([corners[3,0], 0])
offset = [50, 0]

temp_img_size = (temp_img.shape[1], temp_img.shape[0])
src = np.float32([corners[0], corners[1], corners[2], corners[3]])
dst = np.float32([corners[0] + offset, new_top_left + offset,
                  new_top_right - offset, corners[3] - offset])
```

This resulted in the following source and destination points:

| Source         | Destination     | 
|:--------------:|:---------------:| 
| 250., 700.     | 300., 0.        | 
| 580., 450.     | 300., 700.      |
| 700., 450.     | 1010., 690.     |
| 1060., 690.    | 1010., 0.       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt_text][image5]


![alt_text][image6]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of three techniques below to generate a binary image:
1. Sobel Operation
2. Color thresholding in S channel of the HLS color space
3. Color thresholding in L channel of the HLS color space

The code for color thresholding was processed by the ColorThresholds class object in "./advanced_lane_lines/advanced_lane_lines.py".The parameter it needs would be the following below:
- gray_thresh  (gray scale threshold)
- s_thresh     (S color threshold in HLS)
- l_thresh     (L color threshold in HLS)
- thresh       (minimum number of neighbors when filtering noise)
- sobel_kernel (kernel size by sobel operation)

The class object contains the following methods, `__reduce_binary_noise` and `convertBinary`. `__reduce_binary_noise` would be a helper function for `convertBinary`.

The procedure for `self.convertBinary(image)` would be as follows.
1. Take a copy of the image.
2. Convert input RGB input to HLS color space
3. Apply sobel operation in X-direction and calculate scaled derivatives.
4. Generate a binary image based on grayscale thresh values provided during initialization
5. Generate a binary image using S channel of the HLS color scheme and provided S threshold
6. Generate a binary image using L channel of the HLS color scheme and provided L threshold
7. Combine images between sobel operation binary, L channel binary, and S channel binary iamges
8. Reduce noise by processing `self.__reduce_binary_noise`.

The `__reduce_binary_noise` function was be possibel thanks to OpenCV's `filter2D` function. This function applied an arbitrary linear filter to an image.

Here were example images on color thresholding described above.

![alt_text][image7]

![alt_text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify lane-line pixels, I processed the following steps.
1. Extract lane lines
2. Find peak points
3. Perform sliding windows to find lane-line pixels
4. Calculate X and Y coordinates
5. Fit a second order polynomial
6. Calculate X coordinate for each Y coordiante using polynomial

I have written the class object `LaneCurvature` in "./advanced_lane_lines/advanced_lane_lines.py". The object had the following parameters for initialization.
- calibration         (camera calibration object)
- colorThres          (color threshold object)
- perspective         (perspective transform object)
- max_buffer_size     (max buffer size to find average lane curvature)
- buffer_matrix_size  (buffer matrix size to find average lane curvature)
- nwindows            (variables needed for `self._lane_detect_init`)
- margin              (variables needed for `self._lane_detect_init`)
- min_num_pixels      (variables needed for `self._lane_detect_init`)
- txt_color           (variables to draw lane info)
- lane_txt_posit      (variables to draw lane info)
- deviat_txt_posit    (variables to draw lane info)

First, to extract lane lines, I set up two methods `self._lane_detect_init(img)` and `self._lane_detect_next(img)`. During the first iteration, `self._lane_detect_init(img)` would be proceeded first. For the next iterations, `self._lane_detect_next(img)` would be processed using information obtained from it's previous iteration. This check occurred in `self.process` line 674 in "./advanced_lane_lines/advanced_lane_lines.py".

Second `self._lane_detect_init(img)`, I first created a histogram to be able to find peak pixel points of lane lines. Then I find the midpoint of the histogram to be able to distinguish where the middle of the road would approximately be. Then, I get left and right halves of the histogram to identify potential lane lines. This grouped possible peak points within the lane lines.

Third, to perform sliding windows, I calculate the height of the windows I would be using. I further set up the format by storing margins for the size of the windows. Then, I extract all non-zero pixles in X and Y axis to `nonzero_y` and `nonzero_x`. To perform n numbers of windows, I set current iteration X coordinates for left and right lanes. For each iteration of `self.nwindows`, I calculated the low and high y-axis coordinates using the image shape given the which number of iteration of windows it would be at. This meant that as the loop continues to iterate, the y-axis for low and high would be increasing incrementally. Then, I set the left and right, low and high x-coordinates from the base x-axis and up with the `self.margin` value specified during initialization. This meant that as the loop continues to iterate, I would store new x-coordinate positions for the next iteration. Next, I would store all valid nonzero values into a list. I checked the validation by comparing `nonzero_y` and `nonzero_x` I stored in the beginning of the method and the current window values I've calucated in the beginning of the iteration of nwindows. Then, I store valid pixel indexes by appending left and right valid indexes to local variable `left_lane_ids` and `right_lane_ids` respectively. In order to maintain a high confidence interval to run `self._lane_detect_next` later on, I compared our result to `self.min_num_pixels` that stores the least amount of neighbors pixels the lane must have before the lane lines could be verified and stored. In case that it was verified, I added the values to `left_x_current` for left lane lines and `right_x_current` for right lane lines.  Please see the code snippet below for refernece of the loop.
```python
        # perform sliding windows to find valid indexes in nonzeros
        for window in range(self.nwindows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - (window) * window_height
            
            win_x_left_low = left_x_curr - self.margin
            win_x_left_high = left_x_curr + self.margin
            win_x_right_low = right_x_curr - self.margin
            win_x_right_high = right_x_curr + self.margin
            
            valid_left_ids = ((nonzero_y >= win_y_low) & 
                              (nonzero_y < win_y_high) &
                              (nonzero_x >= win_x_left_low) &
                              (nonzero_x < win_x_left_high)).nonzero()[0]
            
            valid_right_ids = ((nonzero_y >= win_y_low) &
                               (nonzero_y < win_y_high) &
                               (nonzero_x >= win_x_right_low) &
                               (nonzero_x < win_x_right_high)).nonzero()[0]
            
            # store valid pixel ids
            left_lane_ids.append(valid_left_ids)
            right_lane_ids.append(valid_right_ids)
            
            if len(valid_left_ids) > self.min_num_pixels:
                left_x_current = np.int(np.mean(nonzero_x[valid_left_ids]))
            if len(valid_right_ids) > self.min_num_pixels:
                right_x_current = np.int(np.mean(nonzero_x[valid_right_ids]))
```
Forth, once the nwindows loop was terminated, I extracted left and right pixel positions by using the valid indexes on `nonzero_x` and `nonzero_y` to identify lane lines. This was performed for both left and right lane lines. Please see the code snippet below.
```python
        # Concatenate indexes
        left_lane_ids = np.concatenate(left_lane_ids)
        right_lane_ids = np.concatenate(right_lane_ids)
        
        # Extract left and right pixel positions
        left_x = nonzero_x[left_lane_ids]
        left_y = nonzero_y[left_lane_ids]
        right_x = nonzero_x[right_lane_ids]
        right_y = nonzero_y[right_lane_ids]
```


Fifth, to calculate valid x and y coordinates, I created a fit second degree polynomial for left and right lane. This was possible by utilizing  the x and y cooridnates found using sliding windows. 

Sixth, to calculate X coordinates for each Y coordinates in the polynomial, I created a list of y cooridnates using image shape. Then, using the second degree polynomial, I calcuated the left and right x-coordinates respectively. Please see the code snippet below.
```python
        fit_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        fit_left_x = self.left_lane[0] * fit_y**2 + self.left_lane[1] * fit_y + \
                     self.left_lane[2]
        fit_right_x = self.right_lane[0] * fit_y ** 2 + self.right_lane[1] * fit_y + \
                      self.right_lane[2]
        
```

On a side note, within the method `self._lane_detect_init` was where `self.lane_init=True` occurs. This told the class object that the initialization process had been completed, and it could run `self._lane_detect_next`. 

Now, `self._lane_detect_next` would be very similar to `self._lane_detect_init`. The main difference would be that it would not have to perform the sliding windows algorithm because the valid lane indexes would already be stored in `self.left_lane` and `self.right_lane`. This would mean that it would not have to perform steps 1-3 mentioned above, and only need to run steps 4-6.

Here would be an example source image before the process.


![alt_text][image9]

This would be an example of what the process detects.


![alt_text][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the lane curavture and vehicle position in method `self._lane_info(img_size, left_x, right_x)` where img_size represented the x and y size of the image, and left_x and right_x represented the identified x-axis lane lines. Within this method, the following processes were performed.
1. Find left and right intercept
2. Convert to meters
3. fit new polynomials with new metric
4. calculate new radii
5. calcualte lane deviation

First, I found the left and right intercept by utilizing the second degree polynomial using left_lane and right_lane as variables, and img_size as constants. Please see the code snippet below.
```python
         # calculate intercept point at the bottom of the image
        left_intercept = self.left_lane[0] * img_size[0] ** 2 + \
                         self.left_lane[1] * img_size[0] + \
                         self.left_lane[2]
        right_intercept = self.right_lane[0] * img_size[0] ** 2 + \
                          self.right_lane[1] * img_size[0] + \
                          self.right_lane[2]
```

Second, I found all pixel width and height of the road. Next, I took the US highway regulation of highways, which would be 3.7m by 30m, and divided that with the road width in pixels. Thus, we found converts the x and y axis from pixel to meters. Please see the code snippet below.
```python
        # US highway regulator lengths in meters
        US_highway = (3.7, 30)
        
        # calculate road width in pixels
        road_width_pxls = right_intercept - left_intercept
        
        # convert length from pixel to meters
        meter_x = US_highway[0] / road_width_pxls
        meter_y = US_highway[1] / road_width_pxls
```

Third, I recalcualted the road curvature in meters for both x and y axis. This was accomplished by finding all possible y-values by utilizing the same buffer size as sliding windows and performing the second degree polynomial with all possible y-values with `meter_y` and `left_x` and `right_x`. The new polynomials were stored in `left_fit_m` and `right_fit_m`. Please see the following snippet.
```python
        # recalculate road curvature in x-y 
        plot_y = np.linspace(0, self.buffer_matrix_size - 1,
                            num=self.buffer_matrix_size)
        plot_y_max = np.max(plot_y)
        
        # fit new polynomials to x-y
        left_fit_m = np.polyfit(plot_y * meter_y, left_x * meter_x, 2)
        right_fit_m = np.polyfit(plot_y * meter_y, right_x * meter_x, 2)
```

Forth, I calculated the radii with the following formula below.
```python
        # calculate new radii
        left_radii = ((1 + (2 * left_fit_m[0] * plot_y_max * meter_y + \
                      left_fit_m[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_m[0])
        
        right_radii = ((1 + (2 * right_fit_m[0] * plot_y_max * meter_y + \
                      right_fit_m[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_m[0])
```

Fifth, I calcualted lane deviation by finding the lane center, and performing the following formula below.
```python
        # calculate lane center
        lane_center = (left_intercept + right_intercept) / 2.0
        
        # calculate lane deviation
        lane_deviation = (lane_center - img_size[1] / 2.0) * meter_x
```

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step within the method `self.process(img)` in class object `LaneCurvature` within "./advanced_lane_lines/advanced_lane_lines.py" from lines 685 to 706. Somethings that I would like to highlight would be the following methods below.
- `_draw_lane_section` in line 702.
- `_combine_img` in line 704.

In `_draw_lane_section` that took parameters image, left_x (left lane x coordinates), and right_x (right lane x coordiantes), I transposed both left and right lane to find the points to highlight. Please see the snippet below.
```python
        # create valid pixel y-coordinates
        fit_y = np.linspace(0, copy_img.shape[0] - 1, copy_img.shape[0])
        
        # transpose left lane
        left_pts = np.array([np.transpose(np.vstack([left_x, fit_y]))])
        # transpose right lane; flip array for right points
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_x, fit_y])))])
```
A note I would like to make was for the right lane, I had to flip the right points in order to correctly draw the highlight sections. Finally, to draw the polynomial, I called the `fillPoly` function provided by OpenCV.

In `_combine_img` that took parameters new image and source image, I copied each image and performed inverse perspective transform on the new image to correctly combine the new and source image. Please see the snippet below.
```python
        # perform inverse perspective transform on new image
        copy_new_perspective = self.perspective.transform_inverse(copy_new_img)
        # combine new and source image
        combine = cv2.addWeighted(copy_src_img, 1, copy_new_perspective, 0.3, 0)
```

Please see the sample source image below:


![alt_text][image11]

Please see the processed image below:


![alt_text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The biggest problem with this software pipeline would be that the degree of accuracy in which the lane lines were detected mainly relied on hard-coded color thresholding values. This meant that when the condition of the roads changed according to lighting, the processed output image would not have accurate highlights of the lane. Therefore, increased amounts of shadow on the road would cause the pipeline to break. This was relevant in the challenge project video would at the 40% processing mark, the pipeline would fail due to the inability to find lanes.

Another severe problem would be that I assumed the conditions of the road would be similar to the previous frame on the road. For example, the way I identify lane lines would be that I would first run `self.lane_detect_init` that would utilize sliding windows to accurately find lane lines based on the number of windows I configured when creating the object. Then, for the rest of all frames within the video, if we were processing videos, I would use a similar format that was found in the first frame of the video. On a highway road, where there would be less changes in the conditions of the road, my software pipeline would work fine, but on non-highway roads where there could be sharp turns that would deviate from the configured cooridnates, the software pipeline would break. This was also evident in the clannege project video.

The most difficulty I had working on this project was dealing with the format necessary to call OpenCV and numpy functions. For instance, we were using numerous color spaces in color thresholding. Then, we would be using OpenCV and numpy functions to filter, format, tranpose, transform, store, read, and more. For each of these functions, there were specified formats to where you must have your inputs to be set before use. Because I was unaware that each function would output a possible different format, I at times struggled to keep consistency within data formats. To able to keep consistency, I wrote test functions that would minimize the amounts of repetitive I/O I had to write and possible make mistakes.

Another minor difficulty I faced was the amount of hard-code that was needed to complete this project. Although I believe there were some improvements to go from static to dynamic variables, a portion of my code was dependent on static variables. For example, for color thresholding and perspective transform was solely dependent on the hard-coded values I found when tweeking around what the output image held. Because of this, there were times where some values would work better for some images, and then for other. Thus, it was hard to determine what values was the best values for a certain situation. I decided to use values where the software pipeline did not fail when running the basic project video.

As mentioned above, some improvements that I would like to make would be to be able to handle more situations, for instance, when the road was dark in the challenge video. I believe that my main problem would be the amount of hard-coded values that I have and the single initialization of the lane line detection system I implemented. The software pipeline was optimized to perform for a certain situation, in my example would be on a sunny day on the highway, where if there were other obstructions within the frame of the image that would cause the situation to change drastically, my software pipeline would crash.

Something I could do to make it robust would be to be able to detect more than one situation in a frame. Currently, it's set to minimum amount of shadow on a sunny day. But, I could add more settings such as "a lot of shadow on a less sunny day with trees" where the color threshold would be able to handle different situations. I believe this could be detected by tweeking the color space and finding out the average depth in a certain color space. If this value of the average color space would be above a certain value, I could deem this frame situation to be "sunny with little amounts of shadow". In the other hand, I could deem another frame as "cloudy with a lot of shadow". Therefore, the software pipeline would be able to handle situations that occur in the challenge video.
