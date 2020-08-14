"""
Image analysis to contain methods used to do simple image analysis on images, and to prepare images for analysis.
Example of preparing an image for analysis is letting the user select an area in an image, and returning that area so
other analyses can be done for just that area
"""

import os
import cv2
import numpy as np
import pandas as pd
import imutils

class ImageAnalysis:
    def __init__(self):
        # things to do with selecting a rectangular area of an image
        self.frame_points = []  # list used for selecting a frame; inputs are lists: [x, y]
        self.list_of_frame_points_for_multiple_selection = []
        self.crop_left = None  # float, how much to crop from the left to get to the region of interest as a fraction
        #  of the entire image
        self.crop_right = None
        self.crop_top = None
        self.crop_bottom = None

        # things to do with selecting a polygonal area of an image
        self.mask_to_search_inside = None  # mask, inside of which to look for the liquid level - the region of
        # interest to search inside for the liquid level
        self.frame_points_tuple = []  # list; same as self.frame_points except the inputs are tuples: (x, y)
        self.list_of_frame_points_tuple_for_multiple_selection = []
        self.list_of_masks_to_search_inside = []

    def reset(self):
        self.frame_points = []  # used for selecting a frame
        self.crop_left = None  # float, how much to crop from the left to get to the region of interest as a fraction
        #  of the entire image
        self.crop_right = None
        self.crop_top = None
        self.crop_bottom = None

    def display_image(self,
                      image_name: str,
                      image):
        """
        Display a cv2 image. User needs to press any key before anything else will happen. Image will stop being
        displayed when user exits out of the image window

        :param image:
        :return:
        """
        cv2.imshow(image_name, image)
        cv2.waitKeyEx(0)

    def load_image(self,
                 image,
                 width: int = None,
                 height: int = None):
        """
        Load. If either width or height are not None, then the image will be resized using the imutils package so
        only width or height needs to be specified, and

        :param str, image, image: 'str' or numpy.ndarray: image to load
        :param int, width: image width you want the image to be resized to. height will be automatically adjusted to
            maintain image
        :param int, height: image height you want the image to be resized to. width will be automatically adjusted to
            maintain image
        :return: image: resized image as numpy.ndarray
        """
        if type(image) is str:
            image = cv2.imread(image)

        if width is not None or height is not None:
            image = imutils.resize(image=image,
                                   width=width,
                                   height=height)

        return image

    def find_image_height_width(self, image):
        """
        Helper method to find the height and width of an image, whether it is a grey scale image or not

        :param image: an image, as a numpy array
        :return: int, int: the height and width of an image
        """
        if len(image.shape) is 3:
            image_height, image_width, _ = image.shape
        elif len(image.shape) is 2:
            image_height, image_width = image.shape
        else:
            raise ValueError('Image must be passed as a numpy array and have either 3 or 2 channels')

        return image_height, image_width

    def select_rectangular_area(self, image):
        """
        Allows user to see an image image, and to draw
        a box on the image and use that as the selected frame of where to look for something. After drawing a box
        you can press 'r' to clear it to reselect a different box or press 'c' to choose the box that will be
        the frame.

        :return: float, the fraction relative to the image size for each side that you would have to crop from to get
            the rectangular region of interest.
        """
        # helpful resource https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
        # have it so that it opens a window so the user can click and drag to define the frame for the image, and then
        # that is the frame that should be set, so it also needs to set crop left, right, top, bottom

        image_height, image_width, _ = image.shape

        def make_selection(event, x, y, flags, param):
            # if left mouse button clicked, record the starting(x, y) coordinates
            if event is cv2.EVENT_LBUTTONDOWN:
                self.frame_points = [(x, y)]
            # check if left mouse button was released
            elif event is cv2.EVENT_LBUTTONUP:
                self.frame_points.append((x, y))

                # draw rectangle around selected region
                cv2.rectangle(image, self.frame_points[0], self.frame_points[1], (0, 255, 0), 2)
                cv2.imshow('Select frame', image)

        # clone image and set up cv2 window
        image = image.copy()
        clone = image.copy()

        cv2.namedWindow('Select frame')
        cv2.setMouseCallback('Select frame', make_selection)

        # keep looping until 'q' is pressed
        while True:
            # display image, wait for a keypress
            cv2.imshow('Select frame', image)
            key = cv2.waitKey(1) & 0xFF

            # if 'r' key is pressed, reset the cropping region
            if key == ord('r'):
                image = clone.copy()

            # if 'c' key pressed break from while True loop
            elif key == ord('c'):
                break

        # if there are 2 reference points then select the region of interest from the image to display
        if len(self.frame_points) == 2:
            cv2.namedWindow('Selected frame - roi')
            roi = clone[self.frame_points[0][1]:self.frame_points[1][1],
                  self.frame_points[0][0]:self.frame_points[1][0]]

            cv2.imshow('Selected frame - roi', roi)
            cv2.waitKey(0)

            # set the self.crop right, left, top, bottom from the selected frame
            self.crop_left = (self.frame_points[0][0] / image_width)
            self.crop_right = ((image_width - self.frame_points[1][0]) / image_width)
            self.crop_top = (self.frame_points[0][1] / image_height)
            self.crop_bottom = ((image_height - self.frame_points[1][1]) / image_height)

        cv2.destroyAllWindows()

        return self.crop_left, self.crop_right, self.crop_top, self.crop_bottom

    def find_rectangular_area_rows_and_columns(self, image, show=False, crop_left=None, crop_right=None, crop_top=None,
                                crop_bottom=None):
        """
        Finds a frame to look for something within a closed image. Also finds it on the image, and has the option
        for user to view the image. What this returns are the row and column values for the region of interest that
        the user selected in self.select_rectangular_area; since select_rectangular_area returns a float to give the
        fraction you need to
        move inwards from the 4 edges of the image in order to get the frame, and so this function will give the row
        and column values that correspond to those float values for the image

        Example of using this method, which requires use of self.select_rectangular_area first to be run:
        crop_left, crop_right, crop_top, crop_bottom = self.select_rectangular_area(image)  # select rectangle on an image,
            gives the fraction to move inside from the edges of the image to get the selected frame (rectangle)
        left, right, top, bottom = self.find_rectangular_area_rows_and_columns(closed, crop_left=crop_left,
        crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom)  # gives row and column values for the
        image (absolute, not relative as is given by find_rectangular_area())

        Then you can do something like:
        row = self.find_something(image, left, right, top, bottom)

        Where the find_something() code iterates through the image according to an algorithm to find something
        or to do something, and the search can be constrained only to the  Example of this is find_liquid_level in
        LiquidLevel.

        Example of how left, right, top, and bottom can be used in find_something():
        def find_something():
            rows = range(top, bottom)
            cols = range(left, right)
            ...

        or the output of left, right, top, bottom can also be used to as parameters to crop an image to only get the
        selected rectangular area

        :param image: image
        :param float, crop_left: fraction, relative to image length, how much to go in from the left to get to the
            region on interest
        :param float, crop_right: fraction, relative to image length, how much to go in from the right to get to the
            region on interest
        :param float, crop_top: fraction, relative to image height, how much to go in from the top to get to the
            region on interest
        :param float, crop_bottom: fraction, relative to image height, how much to go in from the bottom to get to the
            region on interest
        :param bool, show: True to view the drawn frame on the closed image
        :return: left, right, top, bottom, are all int, that represent the row or column that together define the frame
            that is the region of interest
        """
        if crop_left is None:
            crop_left = self.crop_left
        if crop_right is None:
            crop_right = self.crop_right
        if crop_top is None:
            crop_top = self.crop_top
        if crop_bottom is None:
            crop_bottom = self.crop_bottom

        if len(image.shape) is 3:
            image_height, image_width, channels = image.shape
        else:
            image_height, image_width = image.shape

        left = int(image_width * crop_left)
        right = image_width - int(image_width * crop_right)
        top = int(image_height * crop_top)
        bottom = image_height - int(image_height * crop_bottom)

        if show is True:
            image = image.copy()
            image = cv2.line(image, (left, top), (right, top), (0, 255, 0))
            image = cv2.line(image, (left, top), (left, bottom), (0, 255, 0))
            image = cv2.line(image, (left, bottom), (right, bottom), (0, 255, 0))
            image = cv2.line(image, (right, bottom), (right, top), (0, 255, 0))

            cv2.imshow('Selected frame', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return left, right, top, bottom

    def select_multiple_polygonal_areas(self, image, number_of_areas_to_select: int):
        image_clone = image.copy()  # cloned image to have all selected drawn lines show when selecting more than
        # one polygonal area
        print(f'select multiple polygonal area. press "c" to choose an area, press "r" to restart a single selection')

        self.list_of_frame_points_for_multiple_selection = []
        self.list_of_frame_points_tuple_for_multiple_selection = []
        self.list_of_masks_to_search_inside = []

        for i in range(number_of_areas_to_select):
            self.select_polygonal_area(image=image_clone)
            selected_frame_points = self.frame_points
            selected_frame_points_tuple = self.frame_points_tuple
            self.list_of_frame_points_for_multiple_selection.append(selected_frame_points)
            self.list_of_frame_points_tuple_for_multiple_selection.append(selected_frame_points_tuple)
            self.list_of_masks_to_search_inside.append(self.mask_to_search_inside)

        image_with_drawn_areas = self.draw_multiple_selected_polygonal_areas_on_image(image=image)
        masked_image = self.draw_mask_on_image(image=image)

        cv2.imshow("masked image", masked_image)
        cv2.imshow("image with selected area", image_with_drawn_areas)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def select_polygonal_area(self, image):
        # right now this is copy and paste from liquid_level.py with slight tweaks to variable names
        """
        Allows user to see an image of both the latest loaded image and the contour version of that image, and to draw
        a polygon on the image and use that as the selected frame of where to look for a meniscus. After drawing,
        you can press 'r' to clear it to reselect a different box or press 'c' to choose the box that will be
        the frame.


        This sets self.mask_to_search_inside, list_of_frame_points_frame_points_tuple,
        and list_of_frame_points_frame_points_list in order to create the mask for the region of interest
        """
        # have it so that it opens a window so the user can click and drag to define the frame for the image, and then
        # that is the frame that should be set, so it also needs to  set crop left, right, top, bottom

        # reset any selections made before
        self.mask_to_search_inside = None
        self.frame_points = []
        self.frame_points_tuple = []

        def make_selection(event, x, y, flags, param):
            # if left mouse button clicked, record the starting(x, y) coordinates
            if event is cv2.EVENT_LBUTTONDOWN:
                self.frame_points_tuple.append((x, y))
                self.frame_points.append([x, y])

            if event is cv2.EVENT_LBUTTONUP:
                if len(self.frame_points_tuple) >= 2:
                    # if there are a total of two or more clicks made on the image, draw a line to connect the dots
                    # to easily visualize the region of interest that has been created so far
                    cv2.line(image, self.frame_points_tuple[-2],
                             self.frame_points_tuple[-1], (0, 255, 0), 2)

            if len(self.frame_points_tuple) > 0:
                # I think this means that even if only a single click has occurred so far, to still draw a single line
                #  that starts and ends at the position (aka a dot) to allow visualization of where the user just
                # clicked
                cv2.line(image, self.frame_points_tuple[-1],
                         self.frame_points_tuple[-1],
                         (0, 255, 0), 2)
                cv2.imshow('Select frame', image)

        # clone image and set up cv2 window
        image = image.copy()
        clone = image.copy()

        cv2.namedWindow('Select frame')
        cv2.setMouseCallback('Select frame', make_selection)

        # keep looping until 'q' is pressed - q to quit
        while True:
            # during this time the user can left click on the image to select the region of interest

            # display image, wait for a keypress
            cv2.imshow('Select frame', image)
            key = cv2.waitKey(1) & 0xFF

            # if 'r' key is pressed, reset the cropping region
            if key == ord('r'):
                image = clone.copy()
                self.frame_points = []
                self.frame_points_tuple = []

            # if 'c' key pressed break from while True loop
            elif key == ord('c'):
                # make into np array because this is the format it is needed to make a mask
                self.frame_points = np.array(self.frame_points)
                break

        # create mask inside of which the liquid level will be searched for
        # use this for help:
        # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python

        # create initial blank (all black) mask. For the mask, when it is applied to an image, pixels that coincide
        # with black pixels on the mask will not be shown
        self.mask_to_search_inside = np.zeros(shape=clone.shape[0:2], dtype=np.uint8)

        # make the mask; connect the points that the user selected to create the mask area, and make that area white
        # pixels. the mask automatically connects the first and last points that were made, to create an enclosed
        # area for the mask
        cv2.drawContours(self.mask_to_search_inside, [self.frame_points], -1,
                         (255, 255, 255), -1, cv2.LINE_AA)

        # do bit wise operation, this gives the original image back but with only selected region showing,
        # and everything else is black (aka apply a mask, where only the things inside the mask,
        # which is self.mask_to_search_inside, will be visible and the rest of the image blacked out). This is just
        # to show that creating the self.mask_to_search_inside worked.
        clone = self.draw_selected_polygonal_area_on_image(image=clone)
        masked_image = cv2.bitwise_and(clone, clone, mask=self.mask_to_search_inside)
        cv2.imshow("image with outlined area", clone)
        cv2.imshow("masked image", masked_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_multiple_selected_polygonal_areas_on_image(self, image):
        for i in range(len(self.list_of_frame_points_for_multiple_selection)):
            image = self.draw_selected_polygonal_area_on_image(image=image,
                                                               list_of_polygon_points=self.list_of_frame_points_for_multiple_selection[i])

        return image

    def draw_selected_polygonal_area_on_image(self, image, list_of_polygon_points=None):
        """
        Display the one selected polygonal area on an image. This is for showing a single selected polygonal area

        :param list_of_polygon_points: list of points for a single polygonal area; so for a single selected polygonal
            area this would be self.frame_points, which is also the default value if nothing is passed in
        :return:
        """
        # draw mask outline on the  image, then return the image
        if image.shape is 2:  # if image is black and white, not rgb
            line_colour = (255, 255, 255)  # make line colour white
        else:
            # cv image is rgb
            line_colour = (0, 255, 0)  # make line colour green

        if list_of_polygon_points is None:
            list_of_polygon_points = [self.frame_points]
        else:
            list_of_polygon_points = [list_of_polygon_points]

        image = cv2.drawContours(image,
                                 list_of_polygon_points,
                                 -1,
                                 line_colour,
                                 1,
                                 cv2.LINE_AA)
        return image

    def draw_mask_on_image(self, image):
        # draw mask on the  image, then return the image
        line_colour = (255, 255, 255)  # make line colour white

        # draw outline of shape on the image to better see edges of the mask
        if self.list_of_frame_points_for_multiple_selection == []:
            image = self.draw_selected_polygonal_area_on_image(image=image)
            # make the mask; connect the points that the user selected to create the mask area, and make that area white
            # pixels. the mask automatically connects the first and last points that were made, to create an enclosed
            # area for the mask
            cv2.drawContours(self.mask_to_search_inside,
                             [self.frame_points],
                             -1,
                             line_colour,
                             -1,
                             cv2.LINE_AA)
            # put mask on image, and black out anything not in the area you want to search in
            # do bit wise operation, this gives the original image back but with only selected region showing,
            # and everything else is black (aka apply a mask, where only the things inside the mask,
            # which is self.mask_to_search_inside, will be visible and the rest of the image blacked out). This is just
            # to show that creating the self.mask_to_search_inside worked.
            image = cv2.bitwise_and(image, image, mask=self.mask_to_search_inside)

        else:
            image = self.draw_multiple_selected_polygonal_areas_on_image(image=image)
            all_masks = np.zeros(shape=image.shape[0:2], dtype=np.uint8)
            for i in range(len(self.list_of_masks_to_search_inside)):
                all_masks = all_masks + self.list_of_masks_to_search_inside[i]

            # if when putting all the masks together there is overlap, change the value in the mask to 1
            image_height, image_width = self.find_image_height_width(image)
            for row in range(image_height):
                for column in range(image_width):
                    if all_masks[row][column] > 1:
                        all_masks[row][column] = 1

            self.mask_to_search_inside = []
            self.frame_points = []
            self.frame_points_tuple = []

            # do bit wise operation, this gives the original image back but with only selected region showing,
            # and everything else is black (aka apply a mask, where only the things inside the mask,
            # which is self.mask_to_search_inside, will be visible and the rest of the image blacked out). This is just
            # to show that creating the self.mask_to_search_inside worked.
            image = cv2.bitwise_and(image, image, mask=all_masks)

        return image

    def find_maximum_edges_of_mask(self):
        """
        :return: left, right, top, bottom, are all int, that represent the row or column that together define
        the edges of the mask. So left would be the left column, and top would be the top row, and so on
        """
        # from the mask to search inside of, find the row and column values that together would make a
        # rectangle surrounding the area that encompasses all non zero values of the mask. top, would be the top row,
        # right would be the right column, and so on
        left = 1000*1000
        right = 0
        top = 1000*1000
        bottom = 0
        for idx, pixel_point_value in enumerate(self.frame_points):
            # loop through all the points that the user selected to create the outline of the mask inner list is
            # a list of the x and y coordinates of that point
            if pixel_point_value[0] < left:
                left = pixel_point_value[0]
            if pixel_point_value[0] > right:
                right = pixel_point_value[0]
            if pixel_point_value[1] < top:
                top = pixel_point_value[1]
            if pixel_point_value[1] > bottom:
                bottom = pixel_point_value[1]
        return left, right, top, bottom

    def crop_horizontal(self, image, crop_left, crop_right):
        """
        Crops image in horizontal direction by the the crop_left fraction on the left and the crop_right fraction on
        the right.

        :param numpy.ndarray, image: image
        :param float, crop_left: value between 0 and 1, is the fraction to crop out from the left part of the image
        :param float, crop_right: value between 0 and 1, is the fraction to crop out from the right part of the image
        :return: numpy.ndarray, cropped, the image but with a fraction of the left and right parts of the image removed
        """
        if len(image.shape) is 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape

        new_left = int(width * crop_left)
        crop_right = 1 - crop_right
        new_right = int(width * crop_right)

        cropped = image[0:height, new_left:new_right]

        return cropped

    def crop_vertical(self, image, crop_top, crop_bottom):
        """
        Crops image in horizontal direction by the the crop_left fraction on the left and the crop_right fraction on
        the right.

        :param numpy.ndarray, image: image
        :param float, crop_top: value between 0 and 1, is the fraction to crop out from the top part of the image
        :param float, crop_bottom: value between 0 and 1, is the fraction to crop out from the bottom part of the image
        :return: numpy.ndarray, cropped, the image but with a fraction of the top and bottom parts of the image removed
        """
        if len(image.shape) is 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape

        new_top = int(height * crop_top)
        crop_bottom = 1 - crop_bottom
        new_bottom = int(height * crop_bottom)

        cropped = image[new_top:new_bottom, 0:width]

        return cropped

    def draw_line(self, image, left_point, right_point, colour, text, text_position):
        """
        Helper function to draw a single line on an image

        :param image: image to draw the line on
        :param (int, int), left_point: the left point of the line, as (width, height) or equivalently (column, row)
        :param (int, int), right_point: the right point of the line, as (width, height) or equivalently (column, row)
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
                         colour)
        cv2.putText(image, text, text_position, font, font_scale, colour)

        return image


from PyQt5.QtWidgets import QApplication

def folder_of_images_to_video(folder_path,
                              output_video_file_location=None,
                              output_video_name=None,
                              fps=30,
                              display_image_name=False,
                              progress_bar=None):
    """
    For this, assuming that the images names are the time stamp of when the image was taken, since the images will be
    have text placed on it, based on the file name. take a folder of images, and turn it into a video with file name
    written on each frame in the video

    can either give the output video file location OR output video name

    :param str, folder_path: path to folder of images to turn into a video
    :param str, output_video_file_location: path to save the video file to
    :param str, output_video_name: name of the video that will be created; it needs to include the file type in it too
    :param int, fps: frames per second; the frames per second for the output video
    :param bool, display_image_name: if True, then overlap the name of the image on each frame in the video
    :param progress_bar: only used by the gui for being able to update the progress bar, user shouldn't ever need to
        use this
    :return:
    """
    ia = ImageAnalysis()

    folder = folder_path
    if output_video_file_location is None:
        if output_video_name is None:
            output_video_name = 'output_video.mp4'
        output_video_file_location = os.path.join(folder_path, output_video_name)

    if output_video_name is None:
        output_video_name = output_video_file_location.split('/')[-1]

    # use the first image in the folder to get the width and height of all the images
    only_once = 0
    for filename in os.listdir(folder):
        if only_once > 0:
            break
        path_to_first_image = os.path.join(folder, filename)
        first_image = cv2.imread(path_to_first_image)
        image_height, image_width = ia.find_image_height_width(image=first_image)
        only_once += 1

    # Define the codec and create VideoWriter object
    output_file_type = output_video_name.split('.')[-1]
    if output_file_type == 'mp4':
        fourcc = 0x00000021
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_file_location,
                                   fourcc,
                                   fps,
                                   (image_width, image_height))

    green = (0, 255, 0)
    colour = green
    bottom_of_the_image_row = image_height
    bottom_of_the_image_row -= 30
    text_position = (0, bottom_of_the_image_row)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.70

    for index, filename_with_extension in enumerate(os.listdir(folder)):
        image = cv2.imread(os.path.join(folder, filename_with_extension))
        if display_image_name is True:
            split_up_filename_path = filename_with_extension.split('.')
            filename_without_file_type = split_up_filename_path[0]
            text = filename_without_file_type
            cv2.putText(image, text, text_position, font, font_scale, colour)
        output_video.write(image)
        # update gui progress bar if it has been passed into this method
        if progress_bar is not None:
            progress_bar.setValue(index)
            QApplication.processEvents()

    output_video.release()
    cv2.destroyAllWindows()

