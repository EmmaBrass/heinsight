"""
The liquid level class uses computer vision to analyze an image of a container 
with a liquid in it, and be able to determine the position of the liquid levels 
in it; one or more, depending on the liquid. Broadly speaking, the determination 
of the liquid level is based on [insert algo description]

The user can also specify the number of liquid levels to look for, 
or not specify, and so the number of liquid levels in an image can be 
determined dynamically. A camera/webcam is required to use the liquid level
class. Images taken by the camera (raw) and images with liquid level lines drawn 
on them (drawn) can be accessed to be saved and stored.

This class also has complementary classes - the TrackLiquidToleranceLevels 
subclasses. These tracking classes can be used to track any number of user 
defined reference levels and either one or two user defined tolerance levels, 
and can determine whether a liquid level identified in an image falls 
above/below a single tolerance level or within two tolerance levels, or not. 
It can also determine the distance (within the image) between the reference line 
and the current liquid level.

This liquid level class can be leveraged to be used as control code for a larger 
system by providing visual feedback on the liquid level(s) in a container, 
especially when also using a liquid level tracking class.

General background:
If you call image.shape, you get back (height, width, channels), and if you want 
to access a pixel in the image, you need to search it by 
image[height][row][channel]. If there is only one channel in the image there 
will not be a channel specified. Height is equivalent to the row in an image, 
and width is equivalent to a column in an image. For an image the top-left 
corner of the image is (0,0), and the width of the image increases as you travel
towards the top-right corner of the image (0, width), and the height of the 
image increases as you travel towards the bottom-left of the image (height, 0).

The user can:
Select a region of interest within to search for the liquid level.
Adjust parameters of how defined a line must be for it to be identified 
as a liquid level.
Select the number of liquid levels to look for.

"""

import os
import sys
import logging
import json
import cv2
import imutils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter, find_peaks
from numpy import diff
import pandas as pd
from datetime import datetime
from skimage.color import rgb2lab, lab2rgb
from heinsight.liquidlevel.track_tolerance_levels import \
    TrackLiquidToleranceLevels, TrackTwoLiquidToleranceLevels

module_logger = logging.getLogger('liquid_level')


def set_up_module_logger():
    module_logger.setLevel(logging.DEBUG)

    log_folder = os.path.join(os.path.abspath(os.path.curdir), 'logs')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    (f'log folder: {log_folder}')

    fh = logging.FileHandler(filename=os.path.join(log_folder, 
        f'test_logging.txt'))
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    #  create formatter and add it to the handlers
    file_formatter = logging.Formatter("%(asctime)s ; %(levelname)s ; \
        %(module)s ; %(threadName)s ; %(message)s")
    fh.setFormatter(file_formatter)
    console_formatter = logging.Formatter("%(asctime)s ; %(module)s ; \
        %(message)s")
    ch.setFormatter(console_formatter)

    module_logger.addHandler(fh)
    module_logger.addHandler(ch)


if __name__ == '__main__':
    set_up_module_logger()


class NoMeniscusFound(Exception):
    def __init__(self, image, contour_image):
        self.error_image = image
        self.contour_image = contour_image


class LiquidLevel:
    def __init__(self,
                 camera=None,
                 track_liquid_tolerance_levels: TrackLiquidToleranceLevels=None,
                 use_tolerance = False,
                 use_reference = False,
                 number_of_liquid_levels_to_find: int = 1,
                 volumes_list = [],
                 rows_to_count: int = 2,
                 width: int = None,
                 find_meniscus_minimum: float = 0,
                 no_error: bool = False,
                 liquid_level_data_save_folder: str = None,
                 ):
        """
        Class to find a/many liquid levels in

        :param Camera, camera: camera object from heinsight
        :param TrackOneLiquidToleranceLevel, TrackTwoLiquidToleranceLevels, track_liquid_tolerance_levels: tracker
            for the tolerance levels, or None. In the case of None then you can't set tolerance levels
        :param int, number_of_liquid_levels_to_find: int, number of menisci to keep track of in a single image. if it
        is 0,
            then it will be changed in here to be 999. this allows for LiquidLevel to find all the menisci so the
            user doesn't have to specify a single number; should be used in conjunction with using a non-zero value for
            find_meniscus_minimum
        :param None, int, width: int: width to resize all images to, if None, then don't resize the images
        :param float, find_meniscus_minimum: float between 0 and 1, is the minimum fraction of pixels that must be
            white in a contour image within a single slice of looking for the meniscus. A slice of looking for a
            meniscus is the width of the region to look for the meniscus * the number of rows of pixels to count to
            look for the meniscus.
        :param bool, no_error: False if you want an error to be thrown when a meniscus can't be found, True if you
            want the error when a meniscus wasn't found to not be thrown
        :param str, liquid_level_data_save_folder: string, the path to the folder of where to save a JSON file of every
            measurement that gets made using an instance of the Liquid Level class. In this JSON, store the important
            user set parameters for the Liquid Level instance, and also save timestamped values of the liquid level
            location found ein this JSON
            If the value is None, then don't create this file
        """
        # logging set up
        self.logger = logging.getLogger('liquid_level.LiquidLevel')
        # set log level
        self.logger.setLevel(logging.DEBUG)
        # define file handler and set formatter
        file_handler = logging.FileHandler('logfile.log')
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : \
            %(name)s : %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter

        # add file handler to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        # how to format datetime objects into strings
        self.datetime_format = '%Y_%m_%d_%H_%M_%S'

        self.camera = camera

        # do we want to use tolerance and/or reference?
        self.use_tolerance = use_tolerance
        self.use_reference = use_reference

        self.volumes_list = volumes_list # list of volumes to track using the reference levels.

        # attributes for drawing the current liquid level on an image
        bgr_green = (0, 255, 0)
        self.current_level_colour = bgr_green
        self.current_level_text_position = (0, 30)

        self.width = width  # width to resize an image to when trying to find the contours in the image
        self.track_liquid_tolerance_levels = track_liquid_tolerance_levels

        self.loaded_image = None  # the image that was most recently loaded
        self.loaded_bw_image = None  # the b&w image of the image that was most recently loaded

        self.row = 0  # keeping track of current (not ref) meniscus level/height. float; this height value is
        # relative to the height of the image

        self.list_of_frame_points = []  # just for use in select frame frame points instead of lists
        self.mask_to_search_inside = None  # mask, inside of which to look for the liquid level - the region of
        # interest to search inside for the liquid level
        # self.current_frame_point = None

        self.number_of_liquid_levels_to_find = number_of_liquid_levels_to_find  # if menisci to check is 0, then later
        # when finding the number of menisci it will be converted to 999 to look for all the menisci in the image
        self.liquid_level_array = []
        self.rows_to_count = rows_to_count  # int, number of rows to use to rank horizontal lines; assume that the higher
                                           # ranked a contour is, the more likely it is to be a menisci

        self.all_images_with_lines = []  # list of list of timestamp, img: [[timestamp, img], [timestamp, img]...]
                                        # image has lines
        self.all_images_no_lines = []  # list of list of timestamp, img: [[timestamp, img], [timestamp, img]...]
                                        # images dont have lines
        self.all_images_bw = []  # # list of list of timestamp, img: [[timestamp, img], [timestamp, img]...]
                                        # images are the b&w images

        self.pump_to_pixel_ratio = {}  # this will only be used if the liquid level object is used for a pump,
        # and you want to do some self-corrective things based on images that are taken of the liquid level. This is
        # a dictionary, so this is applicable whether using a peristaltic pump or a syringe pump. Regardless of the
        # kind of pump being used, pixels must be a key, and is the number of pixels difference after a specific pump.
        # Then if using a peristaltic pump, need rpm and time as keys, corresponding to the rpm and time used to move
        #  the meniscus that was detected. if using a syringe pump, then volume must be a key, that corresponds to
        # the volume that was pumped to move the meniscus that was detected.
        self.pixel_to_mm_ratio = None  # ratio of how many pixels is in a mm (px/mm)

        self.find_meniscus_minimum = find_meniscus_minimum  # float, minimum area of the slide in which to look for a
        #  meniscus, that needs to be white pixels in the contour image in order to validate that a meniscus was found
        self.no_error = no_error  # bool, if true, then no error will be thrown if a meniscus was not found in the
        # entire region of interest; what is displayed on the image instead is a line at the top of the image in the
        # colour of what the current liquid level line should be

        # initial parameters to be used by reset()
        self.initial_arguments = {
            'camera': self.camera,
            'track_liquid_tolerance_levels': self.track_liquid_tolerance_levels,
            'width': self.width,
            'number_of_liquid_levels_to_find': self.number_of_liquid_levels_to_find,
            'rows_to_count': self.rows_to_count,
            'find_meniscus_minimum': self.find_meniscus_minimum,
            'no_error': self.no_error,
        }

        self.liquid_level_data = {}  # dictionary, where keys are timestamps, and values are liquid level locations
        self.liquid_level_data_save_folder = liquid_level_data_save_folder
        self.liquid_level_data_save_file_path = None

        if self.liquid_level_data_save_folder is not None:
            self.set_up_liquid_level_data_save_file()

    def set_up_liquid_level_data_save_file(self):
        json_file_name = 'liquid_level_data.json'
        self.liquid_level_data_save_file_path = os.path.join(self.liquid_level_data_save_folder, json_file_name)

        path_to_json_file = self.liquid_level_data_save_file_path

        set_up_data = self.get_set_liquid_level_data_as_dictionary()

        with open(path_to_json_file, 'w') as file:
            json.dump(set_up_data, file)

    def get_set_liquid_level_data_as_dictionary(self):
        set_up_data = {'number_of_liquid_levels_to_find': self.number_of_liquid_levels_to_find,
                       'rows_to_count': self.rows_to_count,
                       'find_meniscus_minimum': self.find_meniscus_minimum,
                       'reference_level_relative': None,
                       'tolerance_level_relative': None,
                       'liquid_level_data': {},
                       }
        return set_up_data

    def update_json_file_with_new_data_values(self,):
        """

        :return:
        """
        json_file = open(self.liquid_level_data_save_file_path, "r")  # Open the JSON file for reading
        data = json.load(json_file)  # Read the JSON into the buffer
        json_file.close()  # Close the JSON file

        # Working with buffered content
        dictionary_of_time_stamp_and_liquid_level_location = self.liquid_level_data

        liquid_level_data_buffer = data["liquid_level_data"]
        # add new values of time stamp and liquid level location
        for timestamp_in_dictionary in dictionary_of_time_stamp_and_liquid_level_location:
            self.update_json_file_with_single_data_value(liquid_level_data_buffer=liquid_level_data_buffer,
                                                         timestamp_in_dictionary=timestamp_in_dictionary,
                                                         dictionary_of_time_stamp_and_liquid_level_location=dictionary_of_time_stamp_and_liquid_level_location,
                                                         )

        # Save changes to JSON file
        json_file = open(self.liquid_level_data_save_file_path, "w+")
        json_file.write(json.dumps(data))
        json_file.close()

    def update_json_file_with_single_data_value(self,
                                                liquid_level_data_buffer,
                                                timestamp_in_dictionary,
                                                dictionary_of_time_stamp_and_liquid_level_location,
                                                ):
        """

        :param liquid_level_data_buffer:
        :param timestamp_in_dictionary: a key in the dictionary_of_time_stamp_and_liquid_level_location
        :param dict, dictionary_of_time_stamp_and_liquid_level_location: a dictionary of the liquid level data
        :return:
        """
        liquid_level_location = dictionary_of_time_stamp_and_liquid_level_location[timestamp_in_dictionary]
        liquid_level_data_buffer[timestamp_in_dictionary] = liquid_level_location
        return liquid_level_data_buffer

    def reset(self):
        """
        Function to reset everything so it is like starting with a new version 
        of liquid level; so no reference line, tolerance goes back to the 
        initial value, there are no images saved in memory. 
        Also call reset() for the camera so if there were any images were saved 
        to memory they will be deleted.
        """
        self.logger.debug('reset function called')
        
        if self.camera is not None:
            self.camera.reset()
        if self.track_liquid_tolerance_levels is not None:
            self.track_liquid_tolerance_levels.reset()
        self.width = self.initial_arguments['width']
        self.loaded_image = None
        self.loaded_bw_image = None

        self.reset_region_of_interest()

        self.number_of_liquid_levels_to_find = self.initial_arguments['number_of_liquid_levels_to_find']

        self.reset_row()

        self.rows_to_count = self.initial_arguments['rows_to_count']
        self.all_images_with_lines = []
        self.all_images_no_lines = []
        self.all_images_edge = []

        self.pump_to_pixel_ratio = {}
        self.pixel_to_mm_ratio = None
        self.find_meniscus_minimum = self.initial_arguments['find_meniscus_minimum']
        self.no_error = self.initial_arguments['no_error']

        self.logger.debug('reset function done')

    def reset_row(self):
        """
        Resets the latest liquid level row that was found
        :return:
        """
        self.logger.debug('reset row function called')
        self.row = 0
        self.liquid_level_array = []

    def reset_region_of_interest(self):
        """
        Resets the region of interest selection
        :return:
        """
        self.logger.info('reset_region_of_interest function called')
        self.list_of_frame_points = []
        self.mask_to_search_inside = None

        # when resetting the region of interest, then should also reset the tracker
        if self.track_liquid_tolerance_levels is not None:
            self.track_liquid_tolerance_levels.reset()

        # and reset the current liquid level row

        self.logger.info('reset_region_of_interest done')

    def set_number_of_liquid_levels_to_find(self, int):
        self.logger.debug(f'set set_number_of_liquid_levels_to_find from {self.set_number_of_liquid_levels_to_find}'
                          f' to {int}')
        self.number_of_liquid_levels_to_find = int
        return

    def set_width(self, int):
        self.width = int
        return

    def set_rows_to_count(self, int):
        self.rows_to_count = int
        return

    def load_image_and_select_and_set_parameters(self,
                                                 img,
                                                 select_region_of_interest=True,
                                                 set_reference=True,
                                                 volumes_list=[],
                                                 select_tolerance=True,
                                                 ):
        """
        load an image and set the reference variables: the row with the most white pixels in the frame of the closed
        image after finding a frame (frame made by cropping), and an array of rows with the most white pixels in them;
        the number is dependent on the self.number_of_liquid_levels_to_find value

        :param img: 'str' or numpy.ndarray: image to load
        :param bool, select_region_of_interest: If True, then allow user to select the area to look for the meniscus
        :param bool, set_reference: If True, then find where the most prominent liquid level is and set the reference
            level
        :param bool, select_tolerance: If True, then allow user to select the area that will be the tolerance -
            really only applies for when you have applications where you want to keep the meniscus within a range
            using computer vision
        :return: black and white contour image that was passed through
        """
        self.logger.info('load_image_and_select_and_set_parameters function called')
        image = self.load_img(img)
        bw_image = self.convert_bw(image)
        self.loaded_bw_image = bw_image

        if select_region_of_interest:
            self.select_region_of_interest()
            # draw the region of interest box on the image to get passed onto everything else
            cv2.rectangle(image, self.list_of_frame_points[-2], 
                    self.list_of_frame_points[-1], (0, 255, 0), 2)

        # next line can throw NoMeniscusFound exception
        if set_reference:
            self.set_reference(image=image, volumes_list=volumes_list)
        if select_tolerance:
            self.set_tolerance(image=image)
        self.logger.info('load_image_and_select_and_set_parameters function done')
        return bw_image

    def set_reference(self, image, volumes_list):
        if self.track_liquid_tolerance_levels is None:
            raise AttributeError('No tracker - cannot set reference lines')
        for vol in volumes_list:
            self.track_liquid_tolerance_levels.select_reference_row(image=image, vol=vol)

        # add the reference level to the json file
        if self.liquid_level_data_save_folder is not None:
            json_file = open(self.liquid_level_data_save_file_path, "r")  # Open the JSON file for reading
            data = json.load(json_file)  # Read the JSON into the buffer
            json_file.close()  # Close the JSON file

            # Working with buffered content
            reference_level_relative = self.track_liquid_tolerance_levels.get_relative_reference_heights()
            data["reference_level_relative"] = reference_level_relative

            # Save changes to JSON file
            json_file = open(self.liquid_level_data_save_file_path, "w+")
            json_file.write(json.dumps(data))
            json_file.close()

    def set_tolerance(self, image):
        if self.track_liquid_tolerance_levels is None:
            raise AttributeError('No tracker - cannot set tolerance')
        self.track_liquid_tolerance_levels.select_tolerance(image=image)

        # add the tolerance level to the json file
        if self.liquid_level_data_save_folder is not None:
            # add the reference level to the json file
            json_file = open(self.liquid_level_data_save_file_path, "r")  # Open the JSON file for reading
            data = json.load(json_file)  # Read the JSON into the buffer
            json_file.close()  # Close the JSON file

            # Working with buffered content
            tolerance_level_relative = self.track_liquid_tolerance_levels.get_relative_tolerance_height()
            data["tolerance_level_relative"] = tolerance_level_relative

            # Save changes to JSON file
            json_file = open(self.liquid_level_data_save_file_path, "w+")
            json_file.write(json.dumps(data))
            json_file.close()

    def load_and_convert_bw(self, img):
        self.logger.debug('load_and_find_contour function called')
        fill = self.load_img(img)
        bw_image = self.convert_bw(fill)
        self.loaded_bw_image = bw_image
        self.logger.debug('load_and_find_contour function done')
        return bw_image

    def load_and_find_level(self, img):
        """
        loads an image, finds its contour, gets a frame (frame made by cropping) of the closed image, and then
        finds the row with the most white pixels in the frame, and an array of rows with the most white pixels in them;
        the number is dependent on the self.number_of_liquid_levels_to_find value
        :param img: 'str' or numpy.ndarray: image to load
        :return:
        """
        self.logger.debug('load_and_find_level function called')
        fill = self.load_img(img)
        cv2.imshow("fill", fill)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        bw_image = self.convert_bw(fill)
        row = self.find_liquid_level(fill)
        self.logger.debug('load_and_find_level function done')
        return bw_image, row

    def load_img(self, img):
        """
        Load and resize an image to a particular width; height will automatically adjust
        :param str, img, img: 'str' or numpy.ndarray: image to load
        :return: img: resized image as numpy.ndarray
        """
        self.logger.debug('load_img function called')
        # loads and resizes image
        if type(img) is str:
            img = cv2.imread(img)

        if self.width is not None:
            img = imutils.resize(img, width=self.width)

        self.loaded_image = img

        self.logger.debug('load_img function done')
        return img

    def convert_bw(self,
                     fill,
                     ):
        """
        Given an image converts it to a black and white image.

        :param fill: numpy.ndarray: image to convert
        :return: closed: black and white image
        """
        self.logger.debug('convert_bw function called')
        # get these values from the object's attributes

        return_image = fill

        return_image = cv2.cvtColor(return_image, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite("b&w_image.jpg", return_image)
        cv2.imshow("b&w image", return_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.logger.debug('convert_bw function done')
        return return_image

    def find_parameters_for_canny_edge(self, image, sigma=0.33):
        self.logger.debug('find_parameteres_for_canny_edge function called')
        # compute the median of the single channel pixel intensities
        median = np.median(image)
        # find bounds for Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        self.logger.debug('find_parameteres_for_canny_edge function done')
        return lower, upper

    def find_frame(self):
        """
        Finds a frame to look for menisci within a closed image. This will be based off the mask that was created
        when the user specified the area to look for the meniscus, and get the left, right, top, and bottom row or
        column number that represents a rectangle that encompasses the non zero values in the entire mask (aka the
        area of the mask of where to look for a meniscus, as that area has non-zero values)

        :return: left, right, top, bottom, are all int, that represent the row or column that together define the frame
        """
        left, right, top, bottom = self.find_maximum_edges_of_mask()

        return left, right, top, bottom

    def select_region_of_interest(self):
        """
        Allows user to see an image of both the latest loaded image and the contour version of that image, and to draw
        a polygon on the image and use that as the selected frame of where to look for a meniscus. After drawing,
        you can press 'r' to clear it to reselect a different box or press 'c' to choose the box that will be
        the frame.

        This sets self.mask_to_search_inside and list_of_frame_points
        in order to create the mask for the region of interest
        """
        self.logger.info('select_region_of_interest function called')
        # resource:  https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
        # have it so that it opens a window so the user can click and drag to define the frame for the image, and then
        # that is the frame that should be set, so it also needs to  set crop left, right, top, bottom

        # reset any selections made before
        self.mask_to_search_inside = None
        self.list_of_frame_points = [] # a tuple of the frame points
        # also need to add stuff to make a mask

        def make_selection(event, x, y, flags, param):

            # if left mouse button clicked, record the starting(x, y) coordinates
            # press the mouse button down at the top left and bottom right corners of your desired ROI
            if event is cv2.EVENT_LBUTTONDOWN:
                self.list_of_frame_points.append((x, y))
                self.logger.info("list of points: %s", self.list_of_frame_points)

        # clone image and set up cv2 window. the cloned images will be used if the user wants to reset the points
        # that have been selected on the image - use the most recently loaded image for this method
        image = self.loaded_image.copy()
        clone = self.loaded_image.copy()
        closed_image = self.loaded_bw_image.copy()
        closed_clone = self.loaded_bw_image.copy()

        cv2.namedWindow('Select frame')
        cv2.namedWindow('Select frame - closed')
        cv2.setMouseCallback('Select frame', make_selection)
        cv2.setMouseCallback('Select frame - closed', make_selection)

        # keep looping until 'q' is pressed - q to quit
        while True:
            # during this time the user can left click on the image to select the region of interest

            # display image, wait for a keypress
            cv2.imshow('Select frame', image)
            #cv2.imshow('Select frame - closed', closed_image) # TODO why do we need to see the closed image too?
            key = cv2.waitKey(1) & 0xFF

            if len(self.list_of_frame_points) >= 2:
                image = self.loaded_image.copy()
                # if there are a total of two or more points saved, draw a rectangle
                # to visualise the region that has been selected
                cv2.rectangle(image, self.list_of_frame_points[-2], 
                    self.list_of_frame_points[-1], (0, 255, 0), 2)

            # if 'r' key is pressed, reset the cropping region
            if key == ord('r'):
                image = clone.copy()
                closed_image = closed_clone.copy()
                self.list_of_frame_points = []

            # if 'c' key pressed break from while True loop
            elif key == ord('c'):
                # make list into np array because this is the format it is needed to make a mask
                self.list_of_frame_points = np.array(self.list_of_frame_points)
                break

        # create mask inside of which the liquid level will be searched for
        # use this for help:
        # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python

        # create initial blank (all black) mask. For the mask, when it is applied to an image, pixels that coincide
        #         # with black pixels on the mask will not be shown
        self.mask_to_search_inside = np.zeros(shape=clone.shape[0:2], dtype=np.uint8)
        cv2.imshow("mask_to_search_inside_initial", self.mask_to_search_inside)
        # make the mask; draw white rectangle for the region that the user has selected
        cv2.rectangle(self.mask_to_search_inside, self.list_of_frame_points[-2], 
                        self.list_of_frame_points[-1], 255, -1)

        # do bit wise operation, this gives the original image back but with only selected region showing,
        # and everything else is black (aka apply a mask, where only the things inside the mask,
        # which is self.mask_to_search_inside, will be visible and the rest of the image blacked out). This is just
        # to show that creating the self.mask_to_search_inside worked.
        dst = cv2.bitwise_and(clone, clone, mask=self.mask_to_search_inside)
        cv2.imshow("image+mask", dst)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.logger.info('select_region_of_interest function done')

    def find_maximum_edges_of_mask(self):
        # from the mask to search for a liquid level, find the row and 
        # column values that together would make a rectangle surrounding the 
        # area that encompasses all non zero values of the mask. 
        # top, would be the top row, right would be the right column, and so on
        left = 1000*1000
        right = -1
        top = 1000*1000
        bottom = -1
        for idx, pixel_point_value in enumerate(self.list_of_frame_points):
            # loop through all the points that the user selected to create the 
            # outline of the mask inner list is a list of the x and y 
            # coordinates of that point
            if pixel_point_value[0] < left:
                left = pixel_point_value[0]
            if pixel_point_value[0] > right:
                right = pixel_point_value[0]
            if pixel_point_value[1] < top:
                top = pixel_point_value[1]
            if pixel_point_value[1] > bottom:
                bottom = pixel_point_value[1]
        return left, right, top, bottom

    def find_max_change_location(self, rows, aspect, aspect_name: str):
        """
        Find the row location of the most prominent change of the aspect value.

        :param rows: the array of the row values
        :param aspect: the array of the color aspect (e.g. hue)
        :param aspect_name: the name of the color aspect being analysed
        """

        # implement FIR/IIR forwards and backwards filter
        # reference: https://scipy.github.io/old-wiki/pages/Cookbook/FiltFilt.html#:~:text=filtfilt%2C%20a%20linear%20filter%20that,stable%20response%20(via%20lfilter_zi).

         # Create an order 3 lowpass butterworth filter.
        b, a = butter(9, 0.09)
         
        # Apply the filter to xn.  Use lfilter_zi to choose the initial 
        # condition of the filter.
        zi = lfilter_zi(b, a)
        z, _ = lfilter(b, a, aspect, zi=zi*aspect[0])
         
        # Apply the filter again, to have a result filtered at an order
        # the same as filtfilt.
        z2, _ = lfilter(b, a, z, zi=zi*z[0])
         
        # Use filtfilt to apply the filter.
        aspect_smoothed = filtfilt(b, a, aspect)

        plt.plot(rows, aspect, c='b')
        plt.plot(rows, aspect_smoothed, c='r')  # smoothed by filter
        plt.title(aspect_name)
        plt.show()

        # Here we start the differentiation steps.

        dh = 1

        aspect_smoothed_1deriv = diff(aspect_smoothed)/dh
        aspect_smoothed_2deriv = diff(aspect_smoothed_1deriv)/dh
        aspect_smoothed_3deriv = diff(aspect_smoothed_2deriv)/dh

        # plt.plot(rows, aspect_smoothed, c='r')  # Smoothed by filter
        plt.plot(rows[1:], aspect_smoothed_1deriv, c='b') # 1st derivative
        plt.plot(rows[2:], aspect_smoothed_2deriv, c='g') # 2nd derivative
        plt.plot(rows[3:], aspect_smoothed_3deriv, c='pink') # 3rd derivative
        plt.title(aspect_name)
        plt.show()

        aspect_smoothed_2deriv_abs = []
        aspect_smoothed_3deriv_abs = []

        # The liquid level will be at the position of the greatest |d3I/dh3| 
        # values between the two greatest |d2I/dh2| values.

        # Get the absolute values.
        for i in aspect_smoothed_2deriv:
            a_i = abs(i)
            aspect_smoothed_2deriv_abs.append(a_i)

        for i in aspect_smoothed_3deriv:
            a_i = abs(i)
            aspect_smoothed_3deriv_abs.append(a_i)

        # Find the two highest peaks in the plot of 
        # aspect_smoothed_2deriv_abs vs rows.

        # Find the peaks in absolute 2nd derivative of aspect
        peaks = find_peaks(aspect_smoothed_2deriv_abs, height=0)
        i_peaks = []
        for p in peaks[0]:
            i_peaks.append(p)
        
        self.logger.debug("aspect peak indicies: %s", i_peaks)

        h_peaks = []
        for p in i_peaks:
            h_peaks.append(p+2)
        self.logger.debug("rows peak indicies: %s", h_peaks)
        
        plt.plot(rows[2:], aspect_smoothed_2deriv_abs, c='g') # 2nd derivative
        for i, h in zip(i_peaks, h_peaks):
            plt.plot(rows[h], aspect_smoothed_2deriv_abs[i], 
                marker="o", ls="", ms=3, color='r' )
        plt.title(aspect_name)
        plt.show()

        # From these list of indices, we want to return the two indicies that 
        # correspond to the greatest absolute 2nd derivative values.
        peak_heights = np.array(list(peaks[1].values())).flatten()
        self.logger.debug("peak heights: %s", peak_heights)

        # Get indices of the 2 greatest absolute 2nd derivative values.
        n = 2
        highest_2_peaks_index = [np.where(peak_heights==i) for i in \
            sorted(peak_heights, reverse=True)][:n]

        highest_2_peaks_index = [highest_2_peaks_index[0][0][0], 
            highest_2_peaks_index[1][0][0]]    

        self.logger.debug("highest_2_peaks_index 0: %s", highest_2_peaks_index)

        largest_2ndderiv_index = []
        largest_2ndderiv_index.append(i_peaks[highest_2_peaks_index[0]])
        largest_2ndderiv_index.append(i_peaks[highest_2_peaks_index[1]])

        largest_2ndderiv_index.sort()

        self.logger.debug("largest_2ndderiv_index: %s", largest_2ndderiv_index)

        max_i = 0
        max_i_idx = 0
        for idx, i in enumerate(aspect_smoothed_3deriv_abs):
            if (idx >= largest_2ndderiv_index[0] and 
                idx <= largest_2ndderiv_index[1]):
                if i > max_i:
                    max_i = i
                    max_i_idx = idx

        self.logger.debug("max_i_idx: %s", max_i_idx)

        # max_i_idx gives the index of the greatest value of 
        # the abs of the 3rd derivative.
        # The corresponding value from the rows array will be the row where
        # the liquid level is located.

        prominent_row = rows[max_i_idx+3]  # location of most prominent line/where liquid level is most likely
            #  to be

        self.logger.info("The most prominent row using aspect %s is %s", aspect_name, prominent_row)

        return prominent_row

    def find_liquid_level(self, image):
        """
        Finds a/multiple menisci within a frame of a b&w image. 
        Updates self.row and self.liquid_level_array. 
        self.row is the single row where the only/strongest horizontal 
        line/meniscus is, and self.liquid_level_array is an array of the rows 
        with lines/menisci ordered by rank (higher means stronger 
        horizontal line)
        :param bw_image: black and white image
        :return:
        """

        bw_image = self.convert_bw(image)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        self.logger.debug('find_liquid_level function called')
        liquid_level_data_frame = pd.DataFrame(columns=('row', 
            'fraction_of_pixels'))  # create a pandas dataframe, with 2 rows
        img_height, img_width = bw_image.shape  # size of the edge image
        left_row, right_row, top_row, bottom_row = \
            self.find_maximum_edges_of_mask()
        rows = range(top_row, bottom_row)  # the rows to consider for iteration
        cols = range(left_row, right_row)  # columns to consider
        num_cols = len(cols)

        hue_pixels = [] # array of hue pixel values created from the average of the row
        saturation_pixels = [] # array of saturation pixel values created from the average of the row
        brightness_pixels = [] # array of brightness pixel values created from the average of the row

        for row in rows:  # iterate through each row
            hue_pixel_total = 0
            saturation_pixel_total = 0
            brightness_pixel_total = 0
            for col in cols: # iterate through each coloumn
                # For the hue.
                hue_pixel_total += hsv_image[row][col][0]
                # For the saturation.
                saturation_pixel_total += hsv_image[row][col][1]
                # For the brightness.
                brightness_pixel_total += hsv_image[row][col][2]
            hue_pixel_value = hue_pixel_total/num_cols
            hue_pixels.append(hue_pixel_value)
            saturation_pixel_value = saturation_pixel_total/num_cols
            saturation_pixels.append(saturation_pixel_value)
            brightness_pixel_value = brightness_pixel_total/num_cols
            brightness_pixels.append(brightness_pixel_value)

        height = np.array(rows)
        hue = np.array(hue_pixels)
        saturation = np.array(saturation_pixels)
        brightness = np.array(brightness_pixels)

        # location of most prominent line for these modes
        hue_row = self.find_max_change_location(height, hue, "hue")
        saturation_row = self.find_max_change_location(height, saturation, "saturation")
        brightness_row = self.find_max_change_location(height, brightness, "brightness")

        self.logger.info("hue row: %s", hue_row)
        self.logger.info("saturation row: %s", saturation_row)
        self.logger.info("brightness row: %s", brightness_row)

        # Find average saturation value above and below the saturation_row
        rows_above = range(rows[0], saturation_row)
        num_rows_above = len(rows_above)
        rows_below = range(saturation_row, rows[-1])
        num_rows_below = len(rows_below)

        saturation_above_total = 0
        for row in rows_above:  # iterate through each row
            for col in cols: # iterate through each coloumn
                # For the saturation.
                saturation_above_total += hsv_image[row][col][1]
        saturation_above_av = saturation_above_total/(num_cols*num_rows_above)

        # Here we also find the average brightness value below the saturation_row
        saturation_below_total = 0
        brightness_below_total = 0
        for row in rows_below:  # iterate through each row
            for col in cols: # iterate through each coloumn
                # For the saturation.
                saturation_below_total += hsv_image[row][col][1]
                brightness_below_total += hsv_image[row][col][2]
        saturation_below_av = saturation_below_total/(num_cols*num_rows_below)
        brightness_below_av = brightness_below_total/(num_cols*num_rows_below)

        self.logger.info("saturation_above_av, %s", saturation_above_av)
        self.logger.info("saturation_below_av, %s", saturation_below_av)
        self.logger.info("brightness_below_av, %s", brightness_below_av)

        # Liquid level location is stored as a fraction of image height.
        # If liquid is brightly colored, saturation_row is used.
        # Otherwise, brightness_row is used.
        if (brightness_below_av > 50) and \
            (saturation_below_av >= (saturation_above_av + 50)):
            self.logger.info("Most prominent line is saturation row.")
            liquid_level_location = saturation_row/img_height
        else:
            self.logger.info("Most prominent line is brightness row.")
            liquid_level_location = brightness_row/img_height

        self.row = liquid_level_location  

        if self.number_of_liquid_levels_to_find == 0:
            number_of_number_of_liquid_levels_to_find = 999
        else:
            number_of_number_of_liquid_levels_to_find = self.number_of_liquid_levels_to_find

        self.liquid_level_array.append(liquid_level_location)  # store the values for the
        # rows for the number of menisci that the user wanted to check aka if user wanted to look for 2 menisci,
        # then save the row values for the top 2 'found menisci'. if the self.number_of_liquid_levels_to_find was set
        #  to 0 because user doesn't want to set a specific number to check, this will still work because 999 will
        # actually be passed to the row_array call to find 999 menisci; it will go to the maximum number it can go to

        self.logger.debug(f'liquid level found at {liquid_level_location}')
        self.logger.debug('find_liquid_level function done')
        return liquid_level_location

    def number_of_levels_last_found(self):
        # return number of liquid levels last found when find_liquid_level was run
        return len(self.liquid_level_array)

    def in_tolerance(self,
                     row: float = None):
        """
        Checks if a just measured liquid level is within the tolerance of the reference liquid level. if no row (
        float of row relative to image height) is passed, then use the latest row found by the liquid level algorithm
        and that self.row was set to

        :return: bool: whether you are in the tolerance (True) or not (False). float, percent_diff: the fraction away
            from the tolerance level the current meniscus is. The float value is a percentage relative to the entire
            height of the image
        """
        self.logger.debug('in_tolerance function called')
        if self.track_liquid_tolerance_levels is None:
            return None

        list_of_tolerance_levels = self.track_liquid_tolerance_levels.get_absolute_tolerance_height()
        if len(list_of_tolerance_levels) == 0:
            raise AttributeError('No tolerance levels have been set yet')

        if row is None:
            current_row = self.row
        else:
            current_row = row
        row_within_tolerance = self.track_liquid_tolerance_levels.in_tolerance(height=current_row)

        self.logger.debug(f'liquid level at height {row} in row: {row_within_tolerance}')
        self.logger.debug('in_tolerance function done')
        return row_within_tolerance

    def distance_from_reference(self):
        """
        Return the distance away from the reference line, if one has been selected, that the current liquid level
        line; the distance is normalized by the height of the image. If the value is a positive number then the
        current liquid level is above the reference line.

        :return:
        """
        self.logger.debug('distance_from_reference function called')
        if self.track_liquid_tolerance_levels is None:
            return None

        if self.track_liquid_tolerance_levels.reference_rows is None:
            raise AttributeError('No reference has been set yet')

        current_row = self.row
        difference_from_reference = self.track_liquid_tolerance_levels.distance_from_reference(height=current_row)

        percent_diff = difference_from_reference

        self.logger.debug(f'liquid level row percent diff from reference level: {percent_diff}')
        self.logger.debug('distance_from_reference function done')
        return percent_diff

    def draw_menisci(self, img):
        """
        Draws all menisci lines on an image at where the menisci were calculated
        :param img: image to draw line on
        :return: line: the image with all the menisci drawn on it
        """

        for row in self.liquid_level_array:
            self.row = row
            img = self.draw_level_line(img)

        return img

    def draw_lines(self, img=None):
        """
        Draws all the lines (reference, actual level, and tolerance) on an image - draw on the original image if no
        image is passed

        :param img, the image you want to draw on
        :return image: image with lines drawn on it
        """
        if img is None:
            image = self.loaded_image
        else:
            image = img
        image = self.draw_menisci(image)
        if self.track_liquid_tolerance_levels is not None:
            image = self.draw_ref_line(image)
            image = self.draw_tolerance_lines(image)

        return image

    def draw_ref_on_loaded_image(self):
        """
        Draw the reference line on the original image
        :return: image: image with line drawn on
        """
        image = self.loaded_image

        # if no reference row has been selected yet just return the original image
        if self.track_liquid_tolerance_levels is None:
            return image
        if self.track_liquid_tolerance_levels.reference_rows is None:
            return image

        image = self.draw_ref_line(image)

        return image

    def find_image_height_width(self, image):
        """
        Helper method to find the height and width of an image, whether it is a grey scale image or not

        :param image: an image, as a numpy array
        :return:
        """
        if len(image.shape) == 3:
            image_height, image_width, _ = image.shape
        elif len(image.shape) == 2:
            image_height, image_width = image.shape
        else:
            raise ValueError('Image must be passed as a numpy array and have either 3 or 2 channels')

        return image_height, image_width

    def draw_level_line(self, img):
        """
        Draw the current meniscus on the image
        :param img: image to draw line on
        :return:
        """

        img_height, img_width = self.find_image_height_width(image=img)

        absolute_current_level = int(img_height * self.row)
        current_level_left_point = (0, absolute_current_level)
        current_level_right_point = (img_width, absolute_current_level)

        # draw green line for the current level
        colour = self.current_level_colour
        text_position = self.current_level_text_position
        image = self.draw_line_on_image(image=img,
                                        left_point=current_level_left_point,
                                        right_point=current_level_right_point,
                                        colour=colour,
                                        text='liquid level',
                                        text_position=text_position
                                        )

        return image

    def draw_line_on_image(self,
                           image,
                           left_point,
                           right_point,
                           colour,
                           text,
                           text_position):
        """
        Helper function to draw a single line on an image

        :param image: image to draw the line on
        :param (int, int), left_point: the left point of the line, as (width, height) or equivalently (
        column, row)
        :param (int, int), right_point: the right point of the line, as (width, height) or equivalently (
        column, row)
        :param (int, int, int), colour: colour of the line in (b, g, r)
        :param str, text: text to put on the image
        :param (int, int), text_position: the point in the image to place the text, , as (width, height) or
            equivalently (column, row)
        :return: image with line and text drawn on the image
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        image = cv2.line(image,
                         left_point,
                         right_point,
                         colour,
                         thickness = 1)
        cv2.putText(image, text, text_position, font, font_scale, colour)

        return image

    def draw_ref_line(self, img):
        """
        Draw the reference line on the image
        :param img: image to draw line on
        :return:
        """

        image = self.track_liquid_tolerance_levels.draw_reference_level(image=img)

        return image

    def draw_tolerance_lines(self, image):
        """
        Draw lines of the tolerance boundaries for the reference meniscus on the image
        :param image: image to draw line on
        :return:
        """
        image = self.track_liquid_tolerance_levels.draw_tolerance_levels(image=image)
        return image

    def start(self,
              image=None,
              select_region_of_interest=True,
              set_reference=False,
              volumes_list=[],
              select_tolerance=False):
        """
        First thing that should be run after making the liquid level instance. This will cause the camera to take a
        picture and allow the user to choose if they want to select a region of interest to look for the meniscus and
        if they want to select a tolerance level in the image - tolerance mainly used if you want to also use computer
        vision to self correct for meniscus drift

        :param image: image to analyze, or if left none, then a photo will be taken by the camera
        :param bool, select_region_of_interest: If true, then allow user to select the area to look for the meniscus
        :param bool, set_reference: If True, then find where the most prominent liquid level is and set the reference
            level
        :param bool, select_tolerance: If true, then allow user to select the area that will be the tolerance -
            really only applies for when you have applications where you want to keep the meniscus within a range
            using computer vision
        :return:
        """
        self.logger.info('start function called')
        if image is None:
            image = self.camera.take_picture()
        # next line can throw NoMeniscusFound exception
        bw_image = self.load_image_and_select_and_set_parameters(img=image,
                                                                   select_region_of_interest=select_region_of_interest,
                                                                   set_reference=set_reference,
                                                                   volumes_list=volumes_list,
                                                                   select_tolerance=select_tolerance
                                                                   )
        self.volumes_list=volumes_list
        self.logger.info("Volumes list: %s", self.volumes_list)    

        time = datetime.now()
        if self.track_liquid_tolerance_levels is not None:
            if self.track_liquid_tolerance_levels.reference_rows is not None:
                self.all_images_with_lines.append([time.strftime(self.datetime_format), self.draw_ref_on_loaded_image()])
        self.all_images_no_lines.append([time.strftime(self.datetime_format), image])
        self.all_images_bw.append([time.strftime(self.datetime_format), bw_image])

        # while the experiment is still running
        # call self.run(cam) after every cycle of liquid transfer
        self.logger.info('start function done')
        return

    def take_photo(self):
        self.logger.debug('take_photo function called')
        img = self.camera.take_picture()
        time = datetime.now()
        self.logger.debug('take_photo function done')
        return img, time

    def take_photo_add_to_memory(self):
        img, time = self.take_photo()
        self.add_image_to_memory(img=img,
                                 img_name=time.strftime(self.datetime_format),
                                 array_to_save_to=self.all_images_no_lines,
                                 )
        return img, time
        # self.all_images_no_lines.append([time.strftime(self.datetime_format), img])

    def add_image_to_memory(self, img, img_name, array_to_save_to):
        # add images to one of the arrays used to store images taken in the experiment, organized as
        # [[timestamp, img], [timestamp, img]...]
        array_to_save_to.append([img_name, img])
        # to prevent using too much memory, delete all except the latest image taken after more than n images have be
        #  taken
        n = 50
        if len(array_to_save_to) >= n:
            array_to_save_to = array_to_save_to[-1:]

    def take_photo_find_levels_add_to_memory(self):
        img, time = self.take_photo()
        bw_image, _ = self.load_and_find_level(img=img)
        self.add_image_to_memory(img=self.loaded_image,
            img_name=time.strftime(self.datetime_format),
            array_to_save_to=self.all_images_no_lines,
            )
        self.add_image_to_memory(img=bw_image,
            img_name=time.strftime(self.datetime_format),
            array_to_save_to=self.all_images_bw,
            )
        self.add_image_to_memory(img=self.draw_lines(),
            img_name=time.strftime(self.datetime_format),
            array_to_save_to=self.all_images_with_lines,
            )
        return img, bw_image, time

    def save_drawn_image(self):
        """
        Write the last image with drawn to the computer

        :return:
        """
        # todo need to separate this and have liquid level have its own separate folder
        date_time_with_line, line_img = self.all_images_with_lines[-1]
        save_image_name = f'drawn_{date_time_with_line}'
        self.camera.save_folder.save_image_to_folder(image_name=save_image_name,
            image=line_img,
            file_format='jpg',
            )

    def test_run(self,
                input_image=None
                ):

        self.logger.debug('test run function called')
        if input_image is None:
            image = self.camera.take_picture()
        else:
            # clone the image so no issues with it being edited and then re-used, causing errors
            image = input_image.copy()

        bw_image, row = self.load_and_find_level(image)

        self.logger.info("self.row: %s", self.row)

        image_with_lines = self.draw_lines(img=image)
        cv2.imshow("image with lines", image_with_lines)
        cv2.setWindowProperty("image with lines", cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.logger.debug('test run function complete')

    def run(self,
            input_image=None,
            volume='0'
            ):
        """
        Given an image (or take a photo), load the image and find the current liquid level or the set number to look
        for, and set self.row and self.liquid_level_array. Check whether the liquid level is within any tolerance
        levels, if they were set, and the relative distance from the reference line, if it was set.

        :param image: An image, or if none, then the camera will take a photo to be used as the image.
        :param volume: The volumes to return the percent diff for.  This volume must have been specified in volumes_list in start method.
        :return: bool, tolerance_bool, whether the liquid level was within tolerance or not. float, percent_diff,
            the relative distance of the current liquid level from a set reference level
        """
        self.logger.debug('run function called')
        if input_image is None:
            image = self.camera.take_picture()
        else:
            # clone the image so no issues with it being edited and then re-used, causing errors
            image = input_image.copy()
        # next line can throw NoMeniscusFound exception
        bw_image, _ = self.load_and_find_level(image)
        if self.use_tolerance == True:
            tolerance_bool = self.in_tolerance()
        else:
            tolerance_bool = None
        if self.use_reference == True:
            percent_diff_list = self.distance_from_reference()
            # find the array value that matches the index for the vol in self.volumes_list
            self.logger.info("volumes list is: %s", self.volumes_list)
            idx = self.volumes_list.index(volume)
            percent_diff = percent_diff_list[idx]
            self.logger.info("percent diff distance from volume %s is %s", volume, percent_diff)
        else:
            percent_diff = None
        time = datetime.now()
        time_formatted = time.strftime(self.datetime_format)
        image_with_lines = self.draw_lines(img=image)
        cv2.imshow("image with lines", image_with_lines)
        cv2.setWindowProperty("image with lines", cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.all_images_with_lines.append([time_formatted, image_with_lines])
        self.all_images_no_lines.append([time_formatted, image])
        self.all_images_bw.append([time_formatted, bw_image])

        # to prevent using too much memory, delete all except the latest image taken after more than n images have be
        #  taken
        n = 5
        if len(self.all_images_with_lines) >= n:
            self.all_images_with_lines = self.all_images_with_lines[-1:]
        if len(self.all_images_no_lines) >= n:
            self.all_images_no_lines = self.all_images_no_lines[-1:]
        if len(self.all_images_bw) >= n:
            self.all_images_bw = self.all_images_bw[-1:]

        # if user wants to create a file of all the time stamped values with the liquid level, this is the code block
        #  that will fill in the array of liquid_level_data with a dicitonary items of timestamp: liquid_level_location,
        # and then check if the array is a certain size to update the json file with new values

        self.liquid_level_data[time_formatted] = self.row

        # to prevent using too much memory, after there are n data points in the dictionary, update the json file
        # with this data then reset the dictionary again
        n = 5
        if self.liquid_level_data_save_file_path is not None:
            if len(self.liquid_level_data) >= n:
                self.logger.debug('add more data values to liquid level data JSON file')
                self.update_json_file_with_new_data_values()
                self.liquid_level_data = {}

        self.logger.debug('run function done')
        return tolerance_bool, percent_diff

    def take_picture_find_levels(self):
        # take a picture, find the liquid levels, and return the number of levels that were found
        self.take_photo_find_levels_add_to_memory()
        number_of_levels_found = self.number_of_levels_last_found()

        return number_of_levels_found


# live view test of the tracking class for the liquid level class
def video_run(cam=1):
    camera = None

    # only hae one tracker instance at a time
    # tracker = TrackOneLiquidToleranceLevel(above_or_below='above',
    #                                        )
    tracker = TrackTwoLiquidToleranceLevels()

    l = LiquidLevel(camera=camera,
                    track_liquid_tolerance_levels=tracker,
                    rows_to_count=10,
                    number_of_liquid_levels_to_find=1,
                    find_meniscus_minimum=0.1,
                    no_error=True,
                    liquid_level_data_save_folder=os.path.join(os.path.abspath(os.path.curdir), 'logs')
                    )

    l.start(select_region_of_interest=True,
            select_tolerance=True,
            set_reference=True)

    while True:
        tolerance_bool, percent_diff = l.run()

        edge_image = l.all_images_edge[-1][1]
        cv2.imshow('Edge image', edge_image)

        line = l.draw_lines()

        cv2.imshow('Video for liquid_level', line)

        # if press the q button exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_run(1)
