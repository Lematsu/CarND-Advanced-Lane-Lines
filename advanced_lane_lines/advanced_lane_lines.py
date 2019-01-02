# necessary imports
import glob               # open files
import os                 # manage filepaths
import pickle             # use pickle data
import cv2                # read, write, and manipulate png files
import numpy as np        # manage arrays
from pathlib import Path  # manage files


class CameraCalibration:
    def __init__(self, fp_calibration_images, fp_calibration_data, x, y, debug_flag=False):
        """
        This class handles camera calibration.

        :param fp_calibration_images:
            Filepath to camera images used for camera calibration.
            It would read from fp_calibration_images if fp_calibration_data does not exist.
        :param fp_calibration_data:
            Filepath to camera calibration data stored in pickle format.
            It would read from fp_calibration_data if file exists.
        :param x:
            A number of corners in the x-axis.
        :param y:
            A number of corners in the y-axis.
        :param debug_flag:
            Debug flag
        """
        # camera calibration object
        self.__fp_calibration_images = fp_calibration_images
        # camera calibration data
        self.__fp_calibration_data = fp_calibration_data
        # calibration x axis points
        self.__x = x
        # calibration y axis points
        self.__y = y
        # debug flag
        self.__debug_flag = debug_flag
        # object points
        self.__object_points = []
        # image points
        self.__image_points = []
        
        # TODO: check if file is also a picke file
        if not Path(self.__fp_calibration_data).is_file():
            if debug_flag:
                print("/advanced_lane_lines.py/CameraCalibration/__init__/: using " + 
                      self.__fp_calibration_images)
            self.__calibrate()
        else:
            if self.__debug_flag:
                print("/advanced_lane_lines.py/CameraCalibration/__init__/: using " + 
                      self.__fp_calibration_data)
       
    def __calibrate(self):
        """
        
        This method performs camera calibration.
        
        :stores:
            Camera calibration coefficients as a python dictionary in a pickle file.
        """
        # configure object points
        object_point = np.zeros((self.__x * self.__y, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.__x, 0:self.__y].T.reshape(-1,2)
        
        # iterate over calibration images
        for idx, fname in enumerate(glob.glob(self.__fp_calibration_images)):
            # read image
            img = cv2.imread(fname)
            if self.__debug_flag:
                print("/advanced_lane_lines.py/CameraCalibration/__calibrate/: " +
                      "read image %d" % idx)
            # grayscale image
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # find chessboard corners from given image
            ret, corners = cv2.findChessboardCorners(gray_img, (self.__x, self.__y), None)
            # if found, append current object and image points
            if ret:
                self.__object_points.append(object_point)
                self.__image_points.append(corners)
                if self.__debug_flag:
                    print("/advanced_lane_lines.py/CameraCalibration/__calibrate/: " +
                          "found corners. Appended object and image points.")
        # calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.__object_points,
                                                           self.__image_points,
                                                           (img.shape[1], img.shape[0]),
                                                           None, None)
        # store data in dictionary
        calibration_data = {'mtx': mtx, 'dist': dist}
        
        # store dictionary in pickle flie
        with open(self.__fp_calibration_data, 'wb') as f:
            pickle.dump(calibration_data, file=f)
            if self.__debug_flag:
                print("/advanced_lane_lines.py/CameraCalibration/__calibrate/: " +
                      "dumped to pickle file " + self.__fp_calibration_data)
    
    def undistort(self, img):
        """
        
        This method takes an undistorted image as input and returns an undistroted image.
        
        :param img:
            takes orignal distorted image and undistorts image.
        """
        # try opening pickle file
        try:
            with open(self.__fp_calibration_data, 'rb') as f:
                calibration_data = pickle.load(file=f)
                if self.__debug_flag:
                    print("/advanced_lane_lines.py/CameraCalibration/undistort/: " +
                          "loaded " + self.__fp_calibration_data)
        # raise error if pickle file not found
        except IOError:
            print("/advanced_lane_lines.py/Cameracalibration/undistort/ ERROR: " +
                  self.__fp_calibration_data + " does not exist.")

        # undistort source image
        undistort= cv2.undistort(img, calibration_data['mtx'], calibration_data['dist'],
                                 None, calibration_data['mtx'])
        if self.__debug_flag:
            print("/advanced_lane_lines.py/CameraCalibration/undistort/: " +
                  "created undistort image.")
        
        return undistort


class ColorThresholds:
    def __init__(self, gray_thresh, s_thresh, l_thresh, thresh, sobel_kernel):
        """
        
        This class applies color channel thresholds to an image.
        
        :param gray_thresh:
            Tuple with minimum and maximum gray color thresholds.
        
        :param s_thresh:
            Tuple with minimum and maximum S color threshold in HLS color scheme.
        
        :param l_thresh:
            Tuple with minimum and maximum L color threshol in HLS color scheme.
        
        :param thresh:
            Minimum number of neighbors with value.
        
        :param sobel_kernel:
            Size of the kernel size by Sobel operation.
        """
        # gray scale threshold
        self.gray_thresh = gray_thresh
        # S color threshold in HLS
        self.s_thresh = s_thresh
        # L color threshold in HLS
        self.l_thresh = l_thresh
        # minimum number of neighbors when filtering noise
        self.thresh = thresh
        # kernel size by sobel operation
        self.sobel_kernel = sobel_kernel

        
    def __reduce_binary_noise(self, img):
        """
        
        This method reduce binary image noise.
        
        :param image:
            binary image (0 or 1)
        
        :param threshold:
            minimum number of neighbors with value
        
        :return:
            binary image with reduced noise
        """
        # format kernel array to filter
        k = np.array([[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]])
        # filter noise
        nb_neighbors = cv2.filter2D(img, ddepth=-1, kernel=k)
        img[nb_neighbors < self.thresh] = 0
        
        return img

 
    def convertBinary(self, img):
        """
        
        This method creates a binary image where lines are marked in white color and 
        rest of the image is marked in black color.
        
        :param image:
            Source image
        
        :return:
            The binary image where lines are marked in white color and 
            rest of the iamge is marked in black color.
        """
        #take a copy of image.
        image_copy = np.copy(img)

        #convert RBG image to HLS color space.
        hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]

        #apply sobel opration in X-direction and calculate scaled derivatives.
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        #generate a binary image based on gray_thresh values.
        thresh_min = self.gray_thresh[0]
        thresh_max = self.gray_thresh[1]
        sobel_x_binary = np.zeros_like(scaled_sobel)
        sobel_x_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        #generate a binary image using S channel of the HLS color scheme and
        #provided S threshold
        s_binary = np.zeros_like(s_channel)
        s_thresh_min = self.s_thresh[0]
        s_thresh_max = self.s_thresh[1]
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        #generate a binary image using L channel of the HLS color scheme and
        #provided L threshold
        l_binary = np.zeros_like(l_channel)
        l_thresh_min = self.l_thresh[0]
        l_thresh_max = self.l_thresh[1]
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        #combine images
        binary = np.zeros_like(sobel_x_binary)
        binary[((l_binary == 1) & (s_binary == 1) | (sobel_x_binary == 1))] = 1
        binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')

        return self.__reduce_binary_noise(binary)

class PerspectiveTransform:
    def __init__(self, src_p, dest_p):
        """
        
        This class performs perspective transformation on images from source points
        to destination points.
        
        :param src_p:
            original source points of image
        :param dest_p:
            destination points of image to transform
        """
        # original source point
        self.src_p = src_p
        # destination points
        self.dest_p = dest_p
        
        # perspective transform matrix
        self.matrix = cv2.getPerspectiveTransform(self.src_p, self.dest_p)
        # inverse perspective transform matrix
        self.matrix_inverse = cv2.getPerspectiveTransform(self.dest_p, self.src_p)

    def transform(self, img):
        """
        
        This method performs matrix transformation.
        
        :param img:
            takes orignal image and performs matrix transformation
        :return:
        """
        size = (img.shape[1], img.shape[0])
        # warp image
        return cv2.warpPerspective(img, self.matrix, size, flags=cv2.INTER_LINEAR)
    
    def transform_inverse(self, img):
        """
        
        This method performs inverse matrix transformation.
        
        :param img:
            takes orignal image and perform inverse matrix transformation
        :return:
        """
        size = (img.shape[1], img.shape[0])
        # warp image
        return cv2.warpPerspective(img, self.matrix_inverse, size, flags=cv2.INTER_LINEAR)

class LaneCurvature:
    def __init__(self, calibration, colorThres, perspective,
                 max_buffer_size, buffer_matrix_size,
                 nwindows, margin, min_num_pixels,
                 txt_color, lane_txt_posit, deviat_txt_posit):
        """
        This class detects lane curvature in given picture by calling the process method.
        
        :param calibration:
            camera calibration object
        
        :param perspective:
            perspective transformation object
        
        """
        # perspective transform object
        self.perspective = perspective
        # camera calibration object 
        self.calibration = calibration
        # color threshold object
        self.colorThres = colorThres
        
        # variables needed for _lane_detect_init
        self.nwindows = nwindows
        self.margin = margin
        self.min_num_pixels = min_num_pixels
        
        # variables to draw lane info
        self.txt_color = txt_color
        self.lane_txt_posit = lane_txt_posit
        self.deviat_txt_posit = deviat_txt_posit
        
        # determine either to run lane_detect_init or lane_detect_next
        self.lane_init = False
        
        # cache of left and right lane pixel coordinates from previous image
        self.left_lane = None
        self.right_lane = None
        
        # max buffer size to find avereage
        self.MAX_BUFFER_SIZE = max_buffer_size
        # buffer matrix size to find average
        self.buffer_matrix_size = buffer_matrix_size
        # buffer index counter
        self.buffer_index = 0
        # increment counter
        self.iter_counter = 0
        # buffer left matrix
        self.buffer_left = np.zeros((self.MAX_BUFFER_SIZE,
                                     self.buffer_matrix_size))
        # buffer right matrix
        self.buffer_right = np.zeros((self.MAX_BUFFER_SIZE,
                                      self.buffer_matrix_size))
        

    def _lane_avg(self, fit_left_x, fit_right_x):
        """
        
        This method takes left and right lane pixels and finds the average
        
        :param fit_left_x:
            best fit left lane pixel x-coordinates
        
        :param fit_right_x:
            best fit right lane pixel x-coordiantes
        
        :return:
            average left and right lane pixels
        """
        # store left and right lane pixel x-coordinates
        self.buffer_left[self.buffer_index] = fit_left_x
        self.buffer_right[self.buffer_index] = fit_right_x
        
        # increment buffer index
        self.buffer_index += 1
        self.buffer_index %= self.MAX_BUFFER_SIZE
        
        # find average when increments are less than the max buffer size
        if self.iter_counter < self.MAX_BUFFER_SIZE:
            self.iter_counter += 1
            avg_left = np.sum(self.buffer_left, axis=0) / self.iter_counter
            avg_right = np.sum(self.buffer_right, axis=0) / self.iter_counter
        # find average when increments are max buffer size
        else:
            avg_left = np.average(self.buffer_left, axis=0)
            avg_right = np.average(self.buffer_right, axis=0)
        
        return avg_left, avg_right

    
    def _lane_info(self, img_size, left_x, right_x):
        """
        
        This method calculates left and right curvature and lane deviation.
        
        :param img_size:
            size of image
        
        :param left_x:
            Left lane pixels' x-coordiantes
        
        :param right_x:
            Right lane pixels' x-coordinates
        
        :return:
            Left and right curvature and lane deviation.
        """
        # US highway regulator lengths in meters
        US_highway = (3.7, 30)
        
         # calculate intercept point at the bottom of the image
        left_intercept = self.left_lane[0] * img_size[0] ** 2 + \
                         self.left_lane[1] * img_size[0] + \
                         self.left_lane[2]
        right_intercept = self.right_lane[0] * img_size[0] ** 2 + \
                          self.right_lane[1] * img_size[0] + \
                          self.right_lane[2]
        
        # calculate road width in pixels
        road_width_pxls = right_intercept - left_intercept
        
        # raise error when lane deviation is negative
        # assert road_width_pxls > 0, 'Error: Road width (pixels) cannot be negative'
        
        # convert length from pixel to meters
        meter_x = US_highway[0] / road_width_pxls
        meter_y = US_highway[1] / road_width_pxls
        
        # recalculate road curvature in x-y 
        plot_y = np.linspace(0, self.buffer_matrix_size - 1,
                            num=self.buffer_matrix_size)
        plot_y_max = np.max(plot_y)
        
        # fit new polynomials to x-y
        left_fit_m = np.polyfit(plot_y * meter_y, left_x * meter_x, 2)
        right_fit_m = np.polyfit(plot_y * meter_y, right_x * meter_x, 2)
        
        # calculate new radii
        left_radii = ((1 + (2 * left_fit_m[0] * plot_y_max * meter_y + \
                      left_fit_m[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_m[0])
        
        right_radii = ((1 + (2 * right_fit_m[0] * plot_y_max * meter_y + \
                      right_fit_m[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_m[0])
        
        # calculate lane center
        lane_center = (left_intercept + right_intercept) / 2.0
        
        # calculate lane deviation
        lane_deviation = (lane_center - img_size[1] / 2.0) * meter_x
        
        return left_radii, right_radii, lane_deviation
        
    def _lane_detect_init(self, img):
        """
        
        This method performs the following actions:
        * takes a binary image and produces X coordinates of left and right lane lines.
        * uses sliding window to identify lane lines from the binary image and then 
          uses a second order polynomial estimation technique to calculate road curvature.
        
        
        :param img:
            input image to perform the above actions.
        """
        # create histogram
        histogram = np.sum(img[img.shape[0] // 2:, :, 0], axis=0)
        
        # find midpoint of histogram
        midpoint = np.int(histogram.shape[0] / 2)
        
        # get left and right halves of the histogram
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # calculate the height of the window
        window_height = np.int(img.shape[0] / self.nwindows)
        
        # extract all non-zero pixels in x and y axis
        nonzero = img.nonzero()
        nonzero_x = np.array(nonzero[1])
        nonzero_y = np.array(nonzero[0])
        
        # set current x coordinated for left and right
        left_x_curr = left_x_base
        right_x_curr = right_x_base
        
        # list for valid pixel ids
        left_lane_ids = []
        right_lane_ids = []
        
        # perform sliding windows
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
        
        # Concatenate indexes
        left_lane_ids = np.concatenate(left_lane_ids)
        right_lane_ids = np.concatenate(right_lane_ids)
        
        # Extract left and right pixel positions
        left_x = nonzero_x[left_lane_ids]
        left_y = nonzero_y[left_lane_ids]
        right_x = nonzero_x[right_lane_ids]
        right_y = nonzero_y[right_lane_ids]
        
        # Fit polynomial for left and right lane
        self.left_lane = np.polyfit(left_y, left_x, 2)
        self.right_lane = np.polyfit(right_y, right_x, 2)
        
        fit_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        fit_left_x = self.left_lane[0] * fit_y**2 + self.left_lane[1] * fit_y + \
                     self.left_lane[2]
        fit_right_x = self.right_lane[0] * fit_y ** 2 + self.right_lane[1] * fit_y + \
                      self.right_lane[2]
        
        self.lane_init = True
        
        return fit_left_x, fit_right_x

    
    def _lane_detect_next(self, img):
        """
        
        This image performs lane detection after lane detection initialization.
        It utilizies left and right lanes stored during initializaiton.
        
        :param img:
            source image
        
        :return:
            fit left and right lane pixel x coordinates
        """
        # find nonzero pixel coordinates
        nonzero = img.nonzero()
        # store nonzero x-coordinates
        nonzero_y = np.array(nonzero[0])
        # store nonzero y-coordinates
        nonzero_x = np.array(nonzero[1])
        
        # store left lane indexes
        left_lane_ids = (
            (nonzero_x > (self.left_lane[0] * (nonzero_y ** 2) + \
                          self.left_lane[1] * nonzero_y + \
                          self.left_lane[2] - self.margin)) \
            & (nonzero_x < (self.left_lane[0] * (nonzero_y ** 2) + \
                          self.left_lane[1] * nonzero_y + \
                          self.left_lane[2] + self.margin)))
        # store right lane indexes
        right_lane_ids = (
            (nonzero_x > (self.right_lane[0] * (nonzero_y ** 2) + \
                         self.right_lane[1] * nonzero_y + \
                         self.right_lane[2] - self.margin)) \
            & (nonzero_x < (self.right_lane[0] * (nonzero_y ** 2) + \
                         self.right_lane[1] * nonzero_y + \
                         self.right_lane[2] + self.margin)))
        
        # find left and right line pixel positions
        left_x = nonzero_x[left_lane_ids]
        left_y = nonzero_y[left_lane_ids]
        
        right_x = nonzero_x[right_lane_ids]
        right_y = nonzero_y[right_lane_ids]
        
        # find second order polynomial
        self.left_lane = np.polyfit(left_y, left_x, 2)
        self.right_lane = np.polyfit(right_y, right_x, 2)
        
        # Find left and right lane fits
        fit_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fit_x = self.left_lane[0] * fit_y ** 2 + \
                     self.left_lane[1] * fit_y + \
                     self.left_lane[2]
        right_fit_x = self.right_lane[0] * fit_y ** 2 +\
                      self.right_lane[1] * fit_y + \
                      self.right_lane[2]
        
        return left_fit_x, right_fit_x
        
    def _draw_lane_section(self, img, left_x, right_x):
        """
        
        This method draws the lane section
        
        :param img:
            input image to draw lane section
        
        :param left_x:
            Left lane pixels' x-coordiantes
        
        :param right_x:]
            Right lane pixels' x-coordinates
        
        :return:
            image with lane section drawn
        """
        # copy image
        copy_img = np.zeros_like(img)
        # create valid pixel y-coordinates
        fit_y = np.linspace(0, copy_img.shape[0] - 1, copy_img.shape[0])
        
        # transpose left lane
        left_pts = np.array([np.transpose(np.vstack([left_x, fit_y]))])
        # transpose right lane; flip array for right points
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_x, fit_y])))])
        
        # format points
        pts = np.hstack((left_pts, right_pts))
        
        # fill polynomial on image
        cv2.fillPoly(copy_img, np.int_([pts]), (0, 255, 0))
                     
        return copy_img
        
    def _combine_img(self, new_img, src_img):
        """
        
        This method combines two images
        
        :param new_img:
            bianry image with lane section drawn.
        :param src_img:
            origianl image
        
        :return:
            return the result of combining new_img and src_img
        """
        # copy new image
        copy_new_img = np.copy(new_img)
        # copy source image
        copy_src_img = np.copy(src_img)
        # perform inverse perspective transform on new image
        copy_new_perspective = self.perspective.transform_inverse(copy_new_img)
        # combine new and source image
        combine = cv2.addWeighted(copy_src_img, 1, copy_new_perspective, 0.3, 0)
        
        return combine
       
    def process(self, img):
        """
        
        This method performs the following actions:
        * takes an image and highlights lane lines
        * display left and right lane curvatures
        * vehicle offset from the center of the line
        
        :param img:
            input image to perform the above actions.
        
        :return:
            image with highlighted line lines, display of left and right
            lane curvature, and vehicle offset from center.
        """
        # copy image
        img = np.copy(img)
        # undistort image
        undistorted_img = self.calibration.undistort(img)
        # perspective transform image
        birdsview_img = self.perspective.transform(undistorted_img)
        # color threshold binarize image
        binary_img = self.colorThres.convertBinary(birdsview_img)
        
        # check if image was previously processed
        if self.lane_init:
            left_lane_x, right_lane_x = self._lane_detect_next(binary_img)
        # use self.lane_left and self.lane_right if it was already processed
        else:
            left_lane_x, right_lane_x = self._lane_detect_init(binary_img)
            

        # find lane average for display
        avg_left, avg_right = self._lane_avg(left_lane_x, right_lane_x)
        
        # find left, right, and deviation for display
        left_curve, right_curve, lane_deviation = self._lane_info(img.shape,
                                                                  avg_left, avg_right)

        # format lane curvature display
        curve_txt = 'Left Curvature: {:.2f} m Right Curvature: {:.2f} m'.format(left_curve, 
                                                                               right_curve)
        
        
        # format deviation display
        deviation_txt = 'Lane Deviation: {:.3f} m'.format(lane_deviation)
        # style font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # display lane curvature
        cv2.putText(img, curve_txt, self.lane_txt_posit, font, 1, self.txt_color, 2)
        # display vehicle disposition
        cv2.putText(img, deviation_txt, self.deviat_txt_posit, font, 1, self.txt_color, 2)
        # create highlight image
        img_drawn = self._draw_lane_section(binary_img, avg_left, avg_right)
        # combine image        
        combine_img = self._combine_img(img_drawn, img)
        
        return combine_img
        