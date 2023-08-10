"""
The tracking classes are used to record user selections of a reference liquid level and either one or two tolerance
levels for liquid level applications. It is complementary to the liquid level class. The usefulness of this class is
the ability for the user to set a reference and boundaries for a system that has a dynamically changing liquid
level(s) and needs the ability to respond to changes in liquid level, especially once the liquid level(s) need to be
controlled or there needs to be a response to liquid level movement/changes

The user can:
Interactively select a reference liquid level
Depending on the implementation used, either select one or two tolerance levels. if one tolerance level is set,
the user must define whether a liquid level that is above the tolerance level is considered 'within tolerance' or not

"""

import cv2


class TrackLiquidToleranceLevels:
    def __init__(self,):
        """
        Class to keep track of tolerance levels. A tolerance level(s) is a level relative to the height of an image
        that the user can select, and this class will keep track of that. This class also is used to set a reference
        level in an image.

        The initial/main purpose of this class is to be able to keep track of the tolerance level for applications
        that require the use of the LiquidLevel class in ada_peripherals, for when an application needs to make
        decisions based on a liquid level based on its proximity/distance from/between one/two tolerance level(s).

        """
        self.reference_rows = []  # the user can select reference rows for the 
        # image, and this value (float) is the relative height of that row in 
        # the image (relative to the image height)
        self.reference_image = None  # copy of the reference image
        self.reference_image_height = 0
        self.reference_image_width = 0

        self.tolerance_levels = []  # list of tolerance levels (height in an 
        # image) where values are between 0 and 1; tolerance levels are relative 
        # to the height of the image

        # bgr colours
        bgr_red = (0, 0, 255)
        bgr_blue = (255, 0, 0)

        # setting colours for the different lines
        self.reference_level_colour = bgr_red
        self.tolerance_level_colour = bgr_blue

        # setting text positions for the different texts to appear on images when drawing levels on the images
        self.reference_level_text_position = (0, 15)
        self.tolerance_level_text_position = (0, 45)
        

    def reset(self):
        """
        Reset all the initial attributes
        """
        self.reset_reference()
        self.tolerance_levels = []

    def reset_reference(self):
        self.reference_rows = []
        self.reference_image = None
        self.reference_image_height = 0
        self.reference_image_width = 0

    def get_relative_reference_heights(self):
        """
        Return the heights of the reference rows relative to the height of the 
        image.

        :return: float
        """
        if len(self.reference_rows) == 0:
            raise AttributeError('No reference row has been set')
        else:
            return self.reference_rows

    def get_absolute_reference_heights(self):
        """
        Return the absolute heights of the reference rows in the reference image.

        :return: int, absolute height of the reference row the user selected
        """
        if len(self.reference_rows) == 0:
            return self.reference_rows

        absolute_reference_heights = []
        for row in self.reference_rows:
            absolute_reference_heights.append(int(self.reference_image_height 
                * row))
        return absolute_reference_heights

    def get_relative_tolerance_height(self):
        """
        Return the height of the tolerance level(s) relative to the height of 
        the image.

        :return: float
        """
        if self.tolerance_levels is None:
            raise AttributeError('Tolerance level(s) have not been set')
        else:
            return self.tolerance_levels

    def get_absolute_tolerance_height(self):
        """
        Return a list of the tolerance level(s) the user set.

        :return: list, a list of the absolute heights (row number) in an image 
        of the tolerance level(s) the user selected.
        """
        list_of_absolute_tolerance_heights = []
        for idx, tolerance_level in enumerate(self.tolerance_levels):
            absolute_tolerance_height = int(self.reference_image_height * 
                tolerance_level)
            list_of_absolute_tolerance_heights.append(absolute_tolerance_height)

        return list_of_absolute_tolerance_heights

    def find_image_height_width(self, image):
        """
        Helper method to find the height and width of an image, whether it is a 
        grey scale image or not.

        :param image: an image, as a numpy array
        :return: int, int: the height and width of an image
        """
        if len(image.shape) == 3:
            image_height, image_width, _ = image.shape
        elif len(image.shape) == 2:
            image_height, image_width = image.shape
        else:
            raise ValueError('Image must be passed as a numpy array and have \
                either 3 or 2 channels')

        return image_height, image_width

    def select_reference_row(self, image, vol):
        """
        Allows user to select a horizontal reference line in the image that will 
        be tracked as self.reference_rows.
        The set reference row will be relative to the height of the image

        :param image: numpy array - an image
        :return:
        """
        print(f'Select a reference line for {vol} volumes.')

        self.reference_image = image

        image_height, image_width = self.find_image_height_width(image=image)
        self.reference_image_height = image_height
        self.reference_image_width = image_width

        line_height = self.select_one_line(image=image, description=vol)
        relative_line_height = line_height/image_height

        self.set_reference_level(height=relative_line_height)

    def set_reference_level(self,
                            height: float,
                            ):
        """
        The set reference row will be relative to the height of the image
        :param height:
        :return:
        """
        self.reference_rows.append(height)
        return

    def distance_from_reference(self, height: float):
        """
        Determine the distance between a given height value (relative to image height) of an image, to the reference
        height (also relative to image height), and return that to the user

        :param float, height: relative height of a horizontal row in an image to compare to the reference row height
        :return: float, percent_diff: the fraction away from the references level the given height is, relative to
            the entire height of the image. If the percent_diff value is negative this means that the height of the
            line given is below the reference line, and if it is >= 0 then the liquid level is above the reference
        """
        # if difference from reference is a positive number, then the height of the liquid level is above the
        # reference row
        difference_from_reference = []
        for row in self.reference_rows:
            difference_from_reference.append(row - height)

        return difference_from_reference

    def above_tolerance_level(self, height: float):
        """
        Determine if a given height value (relative to image height) is above or below a single user defined
        tolerance level, or above both of the tolerance levels if the user selected 2. Return true if the height
        is above the tolerance level

        :param float, height: relative height of a horizontal row in an image
        :return: bool, whether the given height is above the single tolerance level or not
        """
        # must be implemented in the sub classes
        raise NotImplementedError('Function only implemented in the TrackOneLiquidToleranceLevel class')

    def select_tolerance(self, image):
        """
        Allow the user to select 1 or 2 tolerance level(s) interactively - The relative heights (relative to image
        height) of the selected levels will be used to set the tolerance levels. The tolerance levels are stored in
        the self.tolerance_levels attribute as a list of the selected tolerance level(s)

        :param image: Image that will be displayed for user to be able to interactively set a height line for
        :return:
        """
        # must be implemented in the sub classes
        raise NotImplementedError

    def set_tolerance_levels(self,
                             height_1: float,
                             height_2: float = None
                             ):
        """
        set tolerance level(s) to be height_1 (and height_2), depending on if one or 2 tolerance level class is used.
        :param height_1: height relative to image height to set tolerance level to
        :return:
        """
        # must be implemented in the sub classes
        raise NotImplementedError

    def select_one_line(self, image, description):
        """
        Display an image, and allow the user to click anywhere on the image to create a horizontal line. The height
        of this horizontal line, not normalized by the height of the image, is returned. This only lets the user
        select a single horizontal line in the image. Press 'r' to reset line selection, and press 'c' to finalize
        selection of the line, press 'q' to exit.

        :param image: Image that will be displayed for user to be able to interactively set a height line for
        :return: int, line_height: the absolute height of the point/horizontal line that the user selected from the
            image.
        """
        # make an initial copy of the image first
        image = image.copy()
        points_from_selection = []  # list of points from when the user left clicks on the displayed image

        image_height, image_width = self.find_image_height_width(image=image)

        def make_selection(event, x, y, flags, param):

            # if left mouse button clicked, record the starting(x, y) coordinates
            if event is cv2.EVENT_LBUTTONDOWN:
                points_from_selection.append((x, y))

                # draw horizontal line from the point selected
                line_left = (0, y)
                line_right = (image_width, y)
                cv2.line(image, line_left, line_right, (0, 255, 0), 2)
                cv2.imshow(f'Select line for {description}', image)

        # clone image and set up cv2 window
        clone = image.copy()

        cv2.namedWindow(f'Select line for {description}')
        cv2.setMouseCallback(f'Select line for {description}', make_selection)

        # keep looping until 'q' is pressed
        while True:
            # display image, wait for a keypress
            cv2.imshow(f'Select line for {description}', image)
            key = cv2.waitKey(1) & 0xFF

            # if 'r' key is pressed, reset the cropping region
            if key == ord('r'):
                points_from_selection = []
                image = clone.copy()

            # if 'c' key pressed break from while True loop
            elif key == ord('c'):
                # if there was a single selection made then return the absolute height of the selected point from the
                # image
                if len(points_from_selection) == 1:
                    line_height = points_from_selection[0][1]
                    cv2.destroyAllWindows()
                    return line_height
                else:
                    print('You must only select a single level. Press "r" to reset line selection')

    def in_tolerance(self, height: float):
        """
        Determine whether a height is within the tolerance level(s) or not; height here is the height of a row of
        pixels, normalized by the image's height. For a single tolerance level, within tolerance depends on if the
        user define within tolerance to be above or below the tolerance line. For two tolerance levels,
        within tolerance is defined as the space between the two tolerance levels.

        :param float, height: relative height of a horizontal row in an image to check and see if it is within the
            tolerance level(s) or not
        :return: bool, whether the current height is within the tolerance level(s) the user set
        """
        # must be implemented in the sub classes
        raise NotImplementedError

    def draw_reference_level(self, image):
        """
        Draw all the reference lines on the image
        :param image: image to draw line on
        :return: image with lines
        """

        _, img_width = self.find_image_height_width(image=image)

        absolute_reference_height = self.get_absolute_reference_heights()

        if len(absolute_reference_height) == 0:
            # if user hasnt yet selected a reference level
            return image

        for height in absolute_reference_height:
            ref_top_left = (0, height)
            ref_lower_right = (img_width, height)

            colour = self.reference_level_colour
            text_position = self.reference_level_text_position
            image = self.draw_line_on_image(image=image,
                                            left_point=ref_top_left,
                                            right_point=ref_lower_right,
                                            colour=colour,
                                            text='reference',
                                            text_position=text_position
                                            )

        return image

    def draw_tolerance_levels(self, image):
        """
        Draw lines of the tolerance level(s) for the reference meniscus on the image.
        :param image: image to draw line on
        :return:
        """
        _, img_width = self.find_image_height_width(image=image)

        list_of_absolute_tolerance_levels = self.get_absolute_tolerance_height()

        if len(list_of_absolute_tolerance_levels) == 0:
            return image

        for absolute_tolerance_level in list_of_absolute_tolerance_levels:
            tolerance_left_point = (0, absolute_tolerance_level)
            tolerance_right_point = (img_width, absolute_tolerance_level)

            colour = self.tolerance_level_colour
            text_position = self.tolerance_level_text_position
            image = self.draw_line_on_image(image=image,
                                            left_point=tolerance_left_point,
                                            right_point=tolerance_right_point,
                                            colour=colour,
                                            text='tolerance',
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
                         thickness=1)
        cv2.putText(image, text, text_position, font, font_scale, colour)

        return image


class TrackOneLiquidToleranceLevel(TrackLiquidToleranceLevels):
    def __init__(self,
                 above_or_below: str,
                 ):
        """
        Class specifically for setting and tracking a single tolerance level in an image.

        :param str, above_or_below: either "above" or "below". "above" for if a liquid level that is above the
            tolerance level means it is within bounds, or "below" if a liquid level that is below the tolerance level
            means it is within bounds
        """
        super().__init__()
        if above_or_below == "above":
            self.above_in_tolerance = True  # true if a liquid level above the tolerance line is considered within
            # bounds
        elif above_or_below == "below":
            self.above_in_tolerance = False
        else:
            raise AttributeError('above_or_below value must be either "above" or "below"')

    def above_tolerance_level(self, height: float):
        """
        Return true if the given height is above the tolerance level

        :param float, height: relative height of a horizontal row in an image
        :return: bool, whether the given height is above the single tolerance level or not
        """

        if self.tolerance_levels[0] is None:
            return ValueError('No tolerance levels have been set yet')
        else:
            if self.tolerance_levels[0] > height:
                return True
            else:
                return False

    def select_tolerance(self, image):
        """
        See description of superclass.
        :return:
        """
        print('Select one tolerance level')

        image_height, image_width = super().find_image_height_width(image=image)
        self.reference_image_height = image_height
        self.reference_image_width = image_width

        # if there is already a reference level drawn, draw it on the image
        clone = image.copy()
        clone = self.draw_reference_level(clone)

        line_height = super().select_one_line(image=clone)
        relative_line_height = line_height/image_height

        self.set_tolerance_levels(height_1=relative_line_height)

    def set_tolerance_levels(self,
                            height_1: float,
                            height_2: float = None):

        self.tolerance_levels = [height_1]
        return

    def in_tolerance(self, height: float):
        """
        Checks if a just measured liquid level is within the tolerance, based if the lines is above or below the
        tolerance. If self.above_in_tolerance is true, then a liquid level is defined to be within tolerance if it is
        above the tolerance level. If self.above_in_tolerance is false, then the liquid level is defined to be out of
        tolerance if it is above the tolerance level. All heights should be normalized by image height, and a smaller
        height means higher up in the image.

        :param float, height: relative height of a horizontal row in an image to check and see if it is within the
            tolerance level or not.
        :return: bool: whether you are in the tolerance (True) or not (False)
        """

        if self.above_in_tolerance is True:
            # if anything above the tolerance level is considered in tolerance
            if height < self.tolerance_levels[0]:
                return True
        else:
            # if anything below the tolerance level is considered within tolerance
            if height > self.tolerance_levels[0]:
                return True

        # if the height is not within tolerance
        return False


class TrackTwoLiquidToleranceLevels(TrackLiquidToleranceLevels):
    def __init__(self):
        """
        Class specifically for setting and tracking two tolerance levels in an image.

        """
        super().__init__()

    def select_tolerance(self, image):
        """
        Let user interactively select 2 lines on an image to be used as the tolerance lines. See description of
        superclass.
        :return:
        """
        print('Select two tolerance levels')

        image_height, image_width = super().find_image_height_width(image=image)
        self.reference_image_height = image_height
        self.reference_image_width = image_width

        # if there is already a reference level selected, draw it on the image
        image = self.draw_reference_level(image=image)

        # make an initial copy of the image first
        image = image.copy()
        points_from_selection = []  # list of points from when the user left clicks on the displayed image

        image_height, image_width = self.find_image_height_width(image=image)

        def make_selection(event, x, y, flags, param):

            # if left mouse button clicked, record the starting(x, y) coordinates
            if event is cv2.EVENT_LBUTTONDOWN:
                points_from_selection.append((x, y))

                # draw line from the point selected
                line_left = (0, y)
                line_right = (image_width, y)
                cv2.line(image, line_left, line_right, (0, 255, 0), 2)
                cv2.imshow('Select line', image)

        # clone image and set up cv2 window
        clone = image.copy()

        cv2.namedWindow('Select line')
        cv2.setMouseCallback('Select line', make_selection)

        # keep looping until 'q' is pressed
        while True:
            # display image, wait for a keypress
            cv2.imshow('Select line', image)
            key = cv2.waitKey(1) & 0xFF

            # if 'r' key is pressed, reset the cropping region
            if key == ord('r'):
                points_from_selection = []
                image = clone.copy()

            # if 'c' key pressed break from while True loop
            elif key == ord('c'):
                # if there were two selections made then return the absolute height of the selected points from the
                # image
                if len(points_from_selection) == 2:
                    line_one_height = points_from_selection[0][1]
                    line_two_height = points_from_selection[1][1]
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    relative_line_one_height = line_one_height / image_height
                    relative_line_two_height = line_two_height / image_height

                    self.set_tolerance_levels(height_1=relative_line_one_height,
                                              height_2=relative_line_two_height
                                              )

                    print(f'Selected tolerance levels: {self.tolerance_levels}')
                    break
                else:
                    print('You must only select two levels. Press "r" to reset line selection')

    def set_tolerance_levels(self,
                            height_1: float,
                            height_2: float = None):

        self.tolerance_levels = [height_1, height_2]
        return

    def in_tolerance(self, height: float):
        """
        Checks if a just measured liquid level is within the two set tolerance levels.

        if it is above the tolerance level. All heights should be relative to the image height, and a smaller height
        means higher up in the image.

        :param float, height: relative height of a horizontal row in an image to check and see if it is within the
            tolerance level or not.
        :return: bool: whether you are in the tolerance (True) or not (False)
        """

        if self.tolerance_levels is []:
            raise AttributeError('No tolerance levels were selected')

        tolerance_one = self.tolerance_levels[0]
        tolerance_two = self.tolerance_levels[1]
        upper_tolerance_level = min(tolerance_one, tolerance_two)
        lower_tolerance_level = max(tolerance_one, tolerance_two)

        if (height > upper_tolerance_level) and (height < lower_tolerance_level):
            return True
        else:
            return False

    def above_tolerance_level(self, height: float):
        """
        Determine if the given height is above both the tolerance levels that the user selected.

        :param float, height: relative height of a horizontal row in an image
        :return: bool, whether the given height is above both the tolerance levels or not
        """
        if self.tolerance_levels[0] is None:
            return ValueError('No tolerance levels have been set yet')
        else:
            tolerance_one = self.tolerance_levels[0]
            tolerance_two = self.tolerance_levels[1]
            upper_tolerance_level = min(tolerance_one, tolerance_two)
            if height < upper_tolerance_level:
                return True
            else:
                return False


