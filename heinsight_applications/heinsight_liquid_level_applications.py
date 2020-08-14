"""
Main script to create a superclass for an application that uses a camera, the liquid level class, and a liquid level
tracker class. This generic application will also be used to collect functions that will be useful for these kinds of
liquid level monitoring applications.

When making the peristaltic pump control instance, the direction set, is the direction to pump liquid out of the vial
that is being watched by the webcam

"""

import os
import cv2
import time
import json
import numpy as np
from datetime import datetime
from heinsight.liquidlevel.liquid_level import LiquidLevel, NoMeniscusFound
from heinsight.liquidlevel.time_manager import TimeManager
from heinsight.liquidlevel.try_tracker import TryTracker
from heinsight.liquidlevel.track_tolerance_levels import TrackTwoLiquidToleranceLevels, TrackOneLiquidToleranceLevel
from heinsight.vision.image_analysis import ImageAnalysis
from heinsight.files import HeinsightFolder
from hein_utilities.slack_integration.bots import RTMSlackBot


class LiquidLevelMonitor:
    """
    Generic class for liquid level monitoring applications
    """
    def __init__(self,
                 liquid_level: LiquidLevel,
                 try_tracker: TryTracker,
                 time_manager: TimeManager,
                 number_of_monitor_liquid_level_replicate_measurements: int = 3,
                 slack_bot: RTMSlackBot = None,
                 show: bool = False,
                 save_folder_bool: bool = False,
                 save_folder_name: str = None,
                 save_folder_location: str = None,
                 ):
        """
        Generic liquid level monitor class
        :param LiquidLevel, liquid_level: instance of LiquidLevel
        :param TryTracker, try_tracker:
        :param TimeManager, time_manager:
        :param int, number_of_monitor_liquid_level_replicate_measurements: must be an odd number, is the number of times to try
            to find a liquid level when monitoring the liquid level, the results of which will be averaged to a
            single measurements for where the liquid level is and if it in tolerance or not
        :param RTMSlackBot, slack_bot:
        :param bool, show: True to display the last image that was taken from python using cv2 on the screen
        :param bool, save_folder_bool: True if at the end of the application you want to save all the images
            that were taken and used throughout the application run to use
        :param str, save_folder_name: Name of the save folder - generally it would be the experiment name
        :param str, save_folder_location: location to create the folder to save everything
        """
        self.liquid_level = liquid_level
        self.application_liquid_level_data = {}  #dictionary, where keys are timestamps, and values are liquid level locations
        self.try_tracker = try_tracker
        self.number_of_monitor_liquid_level_replicate_measurements = number_of_monitor_liquid_level_replicate_measurements
        self.slack_bot = slack_bot
        self.time_manager = time_manager
        self.show = show
        self.save_folder_bool = save_folder_bool

        self.run_experiment = True
        self.pause_experiment = False  # used pause the experiment

        # create initial main folder to save all images to
        if save_folder_name == None:
            save_folder_name = 'temp'

        if save_folder_location == None:
            save_folder_location = os.path.join(os.path.abspath(os.path.curdir), save_folder_name)

        self.save_folder = HeinsightFolder(
            folder_name=save_folder_name,
            folder_path=save_folder_location,
        )
        self.save_folder_name = self.save_folder.get_name()
        self.slack_images_folder: HeinsightFolder = None  # folder for all images that will be sent through slack
        self.all_drawn_images_folder: HeinsightFolder = None  # folder of all images from liquid level with lines
        # drawn on to
        # indicate where the liquid levels and tolerance levels and reference level are
        self.raw_images_folder: HeinsightFolder = None  # folder to same all raw non drawn on images
        self.application_liquid_level_data_save_file_path = None  # path to json file to save liquid level data for
        # the application; only averaged values used for decision making included here

        # how to format datetime objects into strings
        self.datetime_format = '%Y_%m_%d_%H_%M_%S'

        self.create_folder_hierarchy()

    def pre_run(self,
                image=None,
                select: bool = True,
                ):
        """
        Typically the first step for the run of an experiment. This has been singled out because there may be
        additional pre-run steps or optimizations that must be done before the application can be run, and so this
        pre-run step has been separated from the normal run step for ease of re-running an application and continuing
        to run an experiment without having to restart the whole script again.

        :param image, an image. This will be the image that will be used to select the reference and tolerance
            levels. If none, the camera in the liquid level attribute will be used to take a photo.
        :param bool, select: Whether you need to do any additional steps before the first step of the run or not.
            typically involves deciding to select a region of interest, a reference row, or tolerance levels, or not
        :return:
        """
        raise NotImplementedError

    def run(self,
            image=None,
            do_pre_run: bool = True,
            select=True,
            ):
        """
        The main function that puts together the methods to run liquid level monitoring of an image (that either must be
        passed through or else take a picture with the camera and use that), and then advances the system based on
        analysis of the image. What decision needs to be made based on the analysis needs to be separately
        implemented in each of the subclasses.

        The default action for the superclass run is only to delete the folder of images that were saved to the disk
        if the user doesn't want to save the images

        :param bool, do_pre_run: Whether to do the pre_run step or not
        :param image, an image. This will be the image that will be used to monitor the liquid level and go through
            the run. If none, the camera in the liquid level attribute will be used to take a photo.
        :param bool, select:
        :return:
        """

        try:
            if do_pre_run is True:  # if you do want to run the pre_run step
                # next line can throw NoMeniscusFound exception
                try:
                    self.pre_run(image=None, select=select)
                except NoMeniscusFound as error:
                    self.check_if_should_try_again(no_meniscus_error=error)

            self.set_up_applications_liquid_level_data_save_file()

            # the overall loop
            while self.run_experiment is True:
                if self.pause_experiment is not True:
                    # send slack message every interval of time specified in init to remind user that CPC is still running
                    curr_time = datetime.now()
                    time_since_started = self.time_manager.time_since_started(time=curr_time)

                    has_a_time_interval_elapsed = self.time_manager.has_a_time_interval_elapsed(time=curr_time)
                    if has_a_time_interval_elapsed:
                        self.do_if_a_time_interval_has_passed(time=curr_time)

                    # check if experiment has run for the amount of time user specified or not, if it has or has gone
                    # over; if gone over then end the script
                    after_end_time = self.time_manager.is_after_end_time(time=curr_time)
                    if after_end_time:
                        self.end_sequence(time_since_started=time_since_started)
                        return

                    if self.show is True:
                        # even though in the monitor_liquid_level function multiple images may be taken and the average
                        # result for tolerance_bool and percent_diff is taken, to not cause much lag and save screen
                        # space only the latest image the liquid level algorithm was run on will be displayed
                        _, last_image = self.liquid_level.all_images_with_lines[-1]
                        cv2.imshow('Last image that was taken', last_image)
                        # show the image, and if user presses the 'q' button, exit out of the run
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    try:
                        tolerance_bool, percent_diff = self.monitor_liquid_level()
                        if self.try_tracker.get_try_counter() is not 0:
                            self.post_slack_message('Liquid level found. Reset try counter to 0')
                            self.try_tracker.reset_try_counter()

                            date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
                            line_image_path = self.slack_images_folder.save_image_to_folder(
                                image_name=f'reset_try_counter_{date_time_with_line}',
                                image=line_img,
                            )
                            # self.post_slack_file(
                            #     file_path=line_image_path,
                            #     message='The last image taken')

                    except NoMeniscusFound as error:
                        self.check_if_should_try_again(error)
                        continue

                    if tolerance_bool:  # if the most recent measured meniscus level found is within the tolerance bounds
                        # advance normally next line can throw NoMeniscusFound exception
                        try:
                            self.advance()
                        except NoMeniscusFound as error:  # if algorithm couldnt find a meniscus
                            self.check_if_should_try_again(error)
                            continue
                    else:  # most recent measured meniscus was not within tolerance bounds
                        self.not_in_tolerance(percent_diff=percent_diff)
                else:  # self.pause_experiment is True
                    time.sleep(30)
            else:  # self.run_experiment is not True
                self.post_slack_message('experiment was manually ended')
                print('experiment was manually ended')
                return
                # raise KeyboardInterrupt

        except KeyboardInterrupt as error:
            print('Stopped application script using Keyboard Interrupt')
            self.post_slack_message('Stopped application script using Keyboard Interrupt')
            print(f'Error: {error}')

        except Exception as error:
            print('Run has failed')
            print(f'Error: {error}')
            # write the last seen image before the error image, and send that to user through slack
            self.post_slack_message(f'Something went wrong with the run. Error encountered: {error} :cry:')

            date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
            date_time_for_edge, edge_img = self.liquid_level.all_images_edge[-1]

            line_image_path = self.slack_images_folder.save_image_to_folder(
                image_name=f'failed_exit_run_image_{date_time_with_line}',
                image=line_img
            )
            edge_image_path = self.slack_images_folder.save_image_to_folder(
                image_name=f'failed_exit_run_image_edge_{date_time_for_edge}',
                image=edge_img
            )
            self.post_slack_file(line_image_path,
                                 'Last image before error')
            self.post_slack_file(edge_image_path,
                                 'Last image before error')

            raise error

        finally:
            if self.application_liquid_level_data_save_file_path is not None:
                self.update_json_file_with_new_liquid_level_data_values()

            if self.save_folder_bool is False:
                self.save_folder.delete_from_disk()

    def advance(self):
        """
        The main thing the application needs to do, either before or after it monitors the liquid level, depending on
        how the run method is implemented nd the order of this function and the monitor_liquid_level function inside
        of run().

        :return:
        """
        # this should take the place of cycle in the peristaltic one pump loop, and should be the action/whatever
        # happens in between when you need to use the camera again to monitor the liquid level
        raise NotImplementedError

    def end_sequence(self,
                     time_since_started: float):
        """
        do this if the run has ended by going to the end time based on the time manager
        :param: float, time_since_started: number of hours that have elapsed since start time
        :return:
        """
        raise NotImplementedError

    def monitor_liquid_level(self, list_of_percent_diff=None):
        """
        Method used to monitor the liquid level
        :param list_of_percent_diff: list to keep track of the difference between the current liquid level height
            with the reference height, in terms of percentage of the entire image height
        :return:
        """
        average_percent_diff = 0  # average difference between the identified current liquid level and the reference
        # liquid level, ignoring values that are outliers

        #----------------------------------------------------------------------------------

        # can throw a NoMeniscusFound exception
        # implement the variance method of doing this
        if list_of_percent_diff is None:
            list_of_percent_diff = self.take_multiple_photos_and_run_liquid_level()

        # use the reject_current_outliers method to remove outliers from measuring the percent_diff values
        list_of_percent_diff = self.reject_outliers_modified_z(list_of_percent_diff)

        # find the average percent diff from the remaining values
        for percent_diff_value in list_of_percent_diff:
            average_percent_diff += percent_diff_value
        average_percent_diff = average_percent_diff/len(list_of_percent_diff)

        # then need to convert the average percent diff back into relative to the image height value
        absolute_diff = average_percent_diff * self.liquid_level.track_liquid_tolerance_levels.reference_image_height
        average_liquid_level_height = self.liquid_level.track_liquid_tolerance_levels.get_absolute_reference_height() - absolute_diff
        average_liquid_level_percent_height = average_liquid_level_height / self.liquid_level.track_liquid_tolerance_levels.reference_image_height

        # on average, was the liquid level within tolerance or not, taking into account getting rid of outlier values
        average_tolerance_bool = self.liquid_level.in_tolerance(average_liquid_level_percent_height)
        rounded_tolerance_bool = average_tolerance_bool

        # save new average liquid level percent height to memory, and after n values are in memory, save it to the
        # application liquid level data json file
        time = datetime.now()
        time_formatted = time.strftime(self.datetime_format)

        self.application_liquid_level_data[time_formatted] = average_liquid_level_percent_height
        # to prevent using too much memory, after there are n data points in the dictionary, update the json file
        # with this data then reset the dictionary again
        n = 5
        if self.application_liquid_level_data_save_file_path is not None:
            if len(self.application_liquid_level_data) >= n:
                self.update_json_file_with_new_liquid_level_data_values()
                self.application_liquid_level_data = {}
        # # ----------------------------------------------------------------------------------
        # the old way of how things were done
        # list_of_tolerance_bool = []  # list to keep track of the images that indicate if the liquid level is within
        # # the set tolerance boundaries
        # average_tolerance_bool = 0  # average of the values in list_of_tolerance_bool, will be between 0 and 1 and
        # # then rounded to 0 or 1. the way this will be used is that False will be treated as 0 and True as 1 when
        # #  checking whether the liquid level line is within the tolerance bounds or not
        #
        # for i in range(number_of_times_to_try):
        #     # loop through the number of times to try to take pictures to try to find the meniscus. every time,
        #     # the tolerance_bool, of whether the meniscus is within the tolerance bounds or not, and percent_diff is
        #     # how far away the meniscus is from the the reference liquid level line
        #     tolerance_bool, percent_diff, _ = self.take_picture_and_run_liquid_level_algorithm()
        #     list_of_tolerance_bool.append(tolerance_bool)
        #     list_of_percent_diff.append(percent_diff)
        #     # add one to the average_tolerance_bool if the liquid level was found to be within the tolerance bounds
        #     if tolerance_bool is True:
        #         average_tolerance_bool = average_tolerance_bool + 1
        # # find the average value by dividing by number_of_times_to_try
        # average_tolerance_bool = average_tolerance_bool / number_of_times_to_try
        # # round the value so it is either 0 or 1, where 0 would mean the majority of the measured liquid level
        # # readings were outside of the tolerance bounds, and 1 means that the majority were inside the tolerance bounds
        # rounded_tolerance_bool = round(average_tolerance_bool)
        #
        # # majority of the tries to find the meniscus found it outside of the tolerance bounds
        # if rounded_tolerance_bool is 0:
        #     # loop through the list_of_tolerance_bool, and if the value is False, then add the value at the
        #     # corresponding index in the list_of_percent_diff to average_percent_diff
        #     for i in range(number_of_times_to_try):
        #         if list_of_tolerance_bool[i] is False:
        #             average_percent_diff = average_percent_diff + list_of_percent_diff[i]
        # else:
        #     # else then the rounded tolerance is 1 and so the majority of the tries to find the meniscus found it inside
        #     #  of the tolerance bounds
        #     # loop through the list_of_tolerance_bool, and if the value is True, then add the value at the
        #     # corresponding index in the list_of_percent_diff to average_percent_diff
        #     for i in range(number_of_times_to_try):
        #         if list_of_tolerance_bool[i] is True:
        #             average_percent_diff = average_percent_diff + list_of_percent_diff[i]
        #
        # # then divide by the number_of_times_to_try to get the average value
        # average_percent_diff = average_percent_diff / number_of_times_to_try
        # #----------------------------------------------------------------------------------

        # then return the rounded tolerance bool and average percentage diff
        return rounded_tolerance_bool, average_percent_diff

    def take_multiple_photos_and_run_liquid_level(self):
        # instead of just taking one picture, take 3 and find the average of the tolerance_bool values and
        # round the value. Once rounded, then only take the corresponding percent_diff values that have the same
        # tolerance_bool value, where 0 is False and 1 is True, and average them to get the position of the liquid
        # level relative to how far from the reference level it is. this way, if the analysis of one image incorrectly
        # identifies the location of the liquid level compared to the other images, its value will be ignored
        list_of_percent_diff = []
        number_of_times_to_try = self.number_of_monitor_liquid_level_replicate_measurements  # must be an odd number
        while number_of_times_to_try > 0:
            # loop through the number of times to try to take pictures to try to find the liquid level. percent_diff is
            # how far away the meniscus is from the the reference liquid level line. collect these all into
            # list_of_percent_diff
            try:
                _, percent_diff, _ = self.take_picture_and_run_liquid_level_algorithm()
                self.try_tracker.reset_try_counter()
                list_of_percent_diff.append(percent_diff)
                number_of_times_to_try -= 1
            except NoMeniscusFound as error:
                self.check_if_should_try_again(no_meniscus_error=error)
        return list_of_percent_diff

    def reject_outliers_modified_z(self, df, threshold=1):
        # given an array of numbers, will return set that rejects any numbers outside modified_z_score threshold and
        # return back the original array with only the values beneath the threshold to keep
        median = np.median(df)
        median_absolute_deviation = np.median([np.abs(i - median) for i in df])
        if median_absolute_deviation == 0:
            median_absolute_deviation = 0.001
        modified_z_scores = [0.6745 * (i - median) / median_absolute_deviation for i in df]
        # creates array that gives indeces to keep
        keep = [df[index] for index, value in enumerate(modified_z_scores) if abs(value) < threshold]
        return keep

    def take_picture_and_run_liquid_level_algorithm(self):
        """
        Convenience method to take a photo and add that to the folder of raw saved images, and run the liquid level
        finding algorithm on that image and save an image of the image with the drawn levels on it. Then return the
        tolerance_bool (if the liquid level was within tolerance) and the percent_diff (percentage of full image
        height difference between the identified current liquid level and user set reference level). also return the
        path to the drawn image that was saved

        :return: tolerance_bool, bool: True if the liquid level is within tolerance. float, percent_diff: the
            relative distance the identified liquid level is away from the user set reference level.
            str, path_to_save_image
        """
        curr_time = (datetime.now()).strftime('%Y_%m_%d_%H_%M_%S')
        image = self.liquid_level.camera.take_picture()
        self.raw_images_folder.save_image_to_folder(image_name=curr_time,
                                                    image=image)
        tolerance_bool, percent_diff = self.liquid_level.run(image=image)

        path_to_drawn_save_image = self.save_last_drawn_image()
        return tolerance_bool, percent_diff, path_to_drawn_save_image

    def not_in_tolerance(self,
                         percent_diff: float,
                         ):
        """
        Do this if the liquid level was found not to be within tolerance
        :param: float, percent_diff, the relative distance of the current liquid level from a set reference level
        :return:
        """
        """
        What to do if the liquid level was not found to be within tolerance

        :param float, percent_diff: the relative distance of the current liquid level from a set reference level
        :return: 
        """
        date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]

        line_image_path = self.slack_images_folder.save_image_to_folder(
            image_name=f'out_of_tolerance_{date_time_with_line}',
            image=line_img
        )

        if percent_diff > 0:
            above_or_below = 'above'
        else:  # if less than tolerance then pump into the vial being watched
            above_or_below = 'below'

        print(f'liquid level {above_or_below} tolerance')
        print(f'current liquid level difference from the reference liquid level by {percent_diff}')
        self.post_slack_message(f'Liquid level moved {above_or_below} the tolerance boundaries. '
                                f'Current liquid level differs from the reference liquid level by {percent_diff}')

        # self.post_slack_file(line_image_path,
        #                      'Liquid level out of tolerance bounds image')

        # next line can throw NoMeniscusFound exception
        self.self_correct(percent_diff=percent_diff)

        # next line can throw NoMeniscusFound exception
        try:
            # immediately analyze a photo after self correction step and send message and picture to user so they can
            # tell immediately if self correction worked or not
            tolerance_bool, percent_diff, path_to_drawn_save_image = self.take_picture_and_run_liquid_level_algorithm()
            self.post_slack_message(f'After self correction - liquid level within tolerance bounds: {tolerance_bool}, '
                                    f'liquid level differs from the reference liquid level by {percent_diff}')
            # self.post_slack_file(path_to_drawn_save_image,
            #                      'Liquid level image after self correction')
            print(f'After self correction - liquid level within tolerance bounds: {tolerance_bool}, '
                  f'liquid level differs from the reference liquid level by {percent_diff}')
        except NoMeniscusFound as error:
            self.check_if_should_try_again(no_meniscus_error=error)

    def self_correct(self,
                     percent_diff: float,
                     ):
        """
        Simple self correction step to do.

        :return:
        """
        raise NotImplementedError

    def calculate_how_much_to_self_correct(self,
                                           percent_diff: float,
                                           ):
        """
        Calculate how much to self correct by

        For the subclass implementations, a parameter can be passed through for the function.
        :param: float, percent_diff, the relative distance of the current liquid level from a set reference level
        :return:
        """
        raise NotImplementedError

    def pump_self_correct(self,
                          time_to_pump,
                          direction,
                          ):
        raise NotImplementedError

    def check_if_should_try_again(self,
                                  no_meniscus_error: NoMeniscusFound,
                                  ):
        """
        If run into a NoMeniscusFound error, then need to run this to check if the application run should continue or
        end.

        More specifically: Check, in the try tracker, if the try counter is less than the maximum number of tries; the
        tries are tries to find the liquid level again after unsuccessfully finding it, if the try  counter is not
        within the maximum number of tries the stop the application run.

        For the subclass implementations, maybe also do something like record time the the maximum number of tries
        was hit and send a slack message or save the last image that caused the error.

        :param NoMeniscusFound, no_meniscus_error: The error thrown; should be a NoMeniscusError
        :return:
        """
        try_tracker_not_reached_maximum_number_of_tries = self.try_tracker.not_reached_maximum_number_of_tries()

        if try_tracker_not_reached_maximum_number_of_tries:
            self.try_again(no_meniscus_error=no_meniscus_error)
        else:
            self.do_not_try_again(no_meniscus_error=no_meniscus_error)

    def try_again(self,
                  no_meniscus_error: NoMeniscusFound,
                  ):
        """
        Do this if the try limit has not been reached yet. A more specific implementation may be implemented in
        the subclasses

        :param NoMeniscusFound, no_meniscus_error:
        :return:
        """
        current_try = self.try_tracker.get_try_counter()
        maximum_number_of_tries = self.try_tracker.get_max_number_of_tries()
        print(f'Could not find a liquid level. Maximum number of tries has not been reached yet - will try again. On '
              f'try {current_try} out of {maximum_number_of_tries}')
        self.post_slack_message(f'Could not find a liquid level. Maximum number of tries has not been reached yet - '
                                f'will try again. On try {current_try} out of {maximum_number_of_tries}')

        self.try_tracker.increment_try_counter()
        # writes the image where the error occurred to the main folder of images
        time = (datetime.now()).strftime('%Y_%m_%d_%H_%M_%S')
        error_image = no_meniscus_error.error_image
        error_contour_image = no_meniscus_error.contour_image

        error_image_path = self.slack_images_folder.save_image_to_folder(
            image_name=f'{time}_no_meniscus_found',
            image=error_image
        )
        error_contour_image_path = self.slack_images_folder.save_image_to_folder(
            image_name=f'{time}_no_meniscus_found_contour',
            image=error_contour_image
        )

        # self.post_slack_file(error_image_path, 'Error image')
        # self.post_slack_file(error_contour_image_path, 'Error closed image')

    def do_not_try_again(self,
                         no_meniscus_error: NoMeniscusFound,
                         ):
        """
        Do this if the try limit has been reached . A more specific implementation may be implemented in the
        subclasses. Raise the error in order to end the application run

        :param NoMeniscusFound no_meniscus_error:
        :return:
        """
        maximum_number_of_tries = self.try_tracker.get_max_number_of_tries()
        print(f'Could not find a liquid level. Maximum number of {maximum_number_of_tries} tries has been reached - '
              f'application run will end')
        self.post_slack_message(f'Could not find a liquid level. Maximum number of {maximum_number_of_tries} tries '
                                f'has been reached - application run '
                                f'will end')

        time = (datetime.now()).strftime('%Y_%m_%d_%H_%M_%S')
        error_image = no_meniscus_error.error_image
        error_contour_image = no_meniscus_error.contour_image

        error_image_path = self.slack_images_folder.save_image_to_folder(
            image_name=f'{time}_no_meniscus_found',
            image=error_image
        )
        error_contour_image_path = self.slack_images_folder.save_image_to_folder(
            image_name=f'{time}_no_meniscus_found_contour',
            image=error_contour_image
        )

        self.post_slack_file(error_image_path, 'Error image')
        self.post_slack_file(error_contour_image_path, 'Error closed image')

        raise no_meniscus_error

    def do_if_a_time_interval_has_passed(self,
                                         time: datetime):
        hours_elapsed = self.time_manager.hours_elapsed(time=time)
        interval = self.time_manager.calculate_interval(hours_elapsed=hours_elapsed)
        self.time_manager.append_interval(interval=interval)

        time_since_started = self.time_manager.time_since_started(time=time)
        self.post_slack_message(message=f'application has been running for {round(time_since_started, 2)} hours and is '
                                        f'still running')

        if self.time_manager.more_than_one_interval_has_elapsed():  # if the experiment has been running
            # and its not the start of the run, then send an image to the user of what the last image
            # with lines looks like
            date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
            line_image_path = self.slack_images_folder.save_image_to_folder(
                image_name=f'interval_{interval}_at_{date_time_with_line}',
                image=line_img
            )
            self.post_slack_file(
                file_path=line_image_path,
                message='The last image taken')

    def save_last_drawn_image(self):
        """
        Save the last image with drawn lines of liquid levels

        :return:
        """
        date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
        path_to_save_image = self.all_drawn_images_folder.save_image_to_folder(
            image_name=f'{date_time_with_line}',
            image=line_img
        )
        return path_to_save_image


    def post_slack_message(self,
                           message: str):
        """
        Convenience function to send a slack message with the associated RTMSlackBot. if no RTMSlackBot, do nothing

        :param message:
        :return:
        """
        if self.slack_bot is None:
            return
        self.slack_bot.post_slack_message(message)

    def post_slack_file(self,
                        file_path: str,
                        message: str,
                        ):
        """
        Convenience function to post a slack file with the associated RTMSlackBot. if no RTMSlackBot, do nothing
        :param file_path: location of file to send
        :param message:
        :return:
        """
        if self.slack_bot is None:
            return
        self.slack_bot.post_slack_file(filepath=file_path,
                                       title=message,
                                       comment=message,
                                       )

    def create_folder_hierarchy(self):
        """
        create the folders required for the specific application
        :return:
        """
        self.slack_images_folder = self.save_folder.make_and_add_sub_folder(sub_folder_name='slack_images')
        self.all_drawn_images_folder = self.save_folder.make_and_add_sub_folder(sub_folder_name='all_drawn_images')
        self.raw_images_folder = self.save_folder.make_and_add_sub_folder('raw_images')

    def set_up_applications_liquid_level_data_save_file(self):
        json_file_name = 'application_liquid_level_data.json'
        self.application_liquid_level_data_save_file_path = os.path.join(self.save_folder.get_path(), json_file_name)

        path_to_json_file = self.application_liquid_level_data_save_file_path

        set_up_data = self.get_set_up_application_liquid_level_data_as_dictionary()

        with open(path_to_json_file, 'w') as file:
            json.dump(set_up_data, file)

    def get_set_up_application_liquid_level_data_as_dictionary(self):
        reference_level_relative = self.liquid_level.track_liquid_tolerance_levels.get_relative_reference_height()
        tolerance_level_relative = self.liquid_level.track_liquid_tolerance_levels.get_relative_tolerance_height()

        set_up_data = {'number_of_liquid_levels_to_find': self.liquid_level.number_of_liquid_levels_to_find,
                       'rows_to_count': self.liquid_level.rows_to_count,
                       'find_meniscus_minimum': self.liquid_level.find_meniscus_minimum,
                       'reference_level_relative': reference_level_relative,
                       'tolerance_level_relative': tolerance_level_relative,
                       'liquid_level_data': {},
                       'liquid_level_correction_data': [],
                       }
        return set_up_data

    def update_json_file_with_new_liquid_level_data_values(self, ):
        """

        :return:
        """
        json_file = open(self.application_liquid_level_data_save_file_path, "r")  # Open the JSON file for reading
        data = json.load(json_file)  # Read the JSON into the buffer
        json_file.close()  # Close the JSON file

        # Working with buffered content
        dictionary_of_time_stamp_and_liquid_level_location = self.application_liquid_level_data

        liquid_level_data_buffer = data["liquid_level_data"]
        # add new values of time stamp and liquid level location
        for timestamp_in_dictionary in dictionary_of_time_stamp_and_liquid_level_location:
            liquid_level_data_buffer = self.update_json_file_with_single_liquid_level_data_value(
                liquid_level_data_buffer=liquid_level_data_buffer,
                timestamp_in_dictionary=timestamp_in_dictionary,
                dictionary_of_time_stamp_and_liquid_level_location=dictionary_of_time_stamp_and_liquid_level_location,
            )

        # Save changes to JSON file
        json_file = open(self.application_liquid_level_data_save_file_path, "w+")
        json_file.write(json.dumps(data))
        json_file.close()
        return

    def update_json_file_with_single_liquid_level_data_value(self,
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

    def update_json_file_with_new_liquid_level_correction_data_values(self,):
        """

        :return:
        """
        json_file = open(self.application_liquid_level_data_save_file_path, "r")  # Open the JSON file for reading
        data = json.load(json_file)  # Read the JSON into the buffer
        json_file.close()  # Close the JSON file

        # Working with buffered content
        liquid_level_correction_data_buffer = data["liquid_level_correction_data"]
        # add new values of time stamp and liquid level location
        time = (datetime.now()).strftime(self.datetime_format)
        liquid_level_correction_data_buffer.append(time)

        # Save changes to JSON file
        json_file = open(self.application_liquid_level_data_save_file_path, "w+")
        json_file.write(json.dumps(data))
        json_file.close()
        return

    def pause_the_experiment(self):
        self.pause_experiment = True

    def resume_experiment(self):
        self.pause_experiment = False

    def end_experiment(self):
        self.run_experiment = False


class AutomatedCPC(LiquidLevelMonitor):
    def __init__(self,
                 liquid_level: LiquidLevel,
                 try_tracker: TryTracker,
                 time_manager: TimeManager,
                 pump,
                 initial_pump_rate: float = None,
                 self_correction_pump_rate: float = None,
                 number_of_monitor_liquid_level_replicate_measurements: int = 3,
                 slack_bot: RTMSlackBot = None,
                 show: bool = False,
                 save_folder_bool: bool = True,
                 save_folder_name: str = None,
                 save_folder_location: str = None,
                 ):
        """

        :param liquid_level:
        :param try_tracker:
        :param time_manager:
        :param pump:
        :param int, initial_pump_rate: rate for the pump to use for anything other than self correction, if there is
            an advance step that requires the pump to run. default should be to use whatever rate the pump is
            currently set at
        :param int, self_correction_pump_rate: rate for the pump to use for self correction. default is to use the
            rate the pump is currently set at
        :param number_of_monitor_liquid_level_replicate_measurements:
        :param slack_bot:
        :param show:
        :param save_folder_bool:
        :param save_folder_name:
        :param save_folder_location:
        """
        super().__init__(liquid_level=liquid_level,
                         try_tracker=try_tracker,
                         time_manager=time_manager,
                         number_of_monitor_liquid_level_replicate_measurements
                         =number_of_monitor_liquid_level_replicate_measurements,
                         slack_bot=slack_bot,
                         show=show,
                         save_folder_bool=save_folder_bool,
                         save_folder_name=save_folder_name,
                         save_folder_location=save_folder_location
                         )
        self.pump = pump
        self.initial_pump_rate = initial_pump_rate
        if self.initial_pump_rate is None:
            self.initial_pump_rate = pump.get_rate()
        self.self_correction_pump_rate = self_correction_pump_rate
        if self_correction_pump_rate is None:
            self.self_correction_pump_rate = self.initial_pump_rate

    # todo maybe move some of this up to superclass but not all
    def pre_run(self,
                image=None,
                select: bool = True,
                ):
        """
        see superclass

        :param image:
        :param select:
        :return:
        """
        # todo consider moving this up to liquid level monitor class
        # next line can throw NoMeniscusFound exception
        if image is None:
            image = self.liquid_level.camera.take_picture()

        curr_time = (datetime.now()).strftime('%Y_%m_%d_%H_%M_%S')
        self.raw_images_folder.save_image_to_folder(image_name=curr_time,
                                                    image=image)

        if select is True:
            self.liquid_level.track_liquid_tolerance_levels.reference_row = 0
            self.liquid_level.start(
                image=image,
                select_region_of_interest=True,
                set_reference=True,
                select_tolerance=True,
            )
        else:
            self.liquid_level.start(
                image=image,
                select_region_of_interest=False,
                set_reference=False,
                select_tolerance=False,
            )

        self.pump.set_rate(rate=self.initial_pump_rate)

        print(f'pre_run complete')

    def run(self,
            image=None,
            do_pre_run: bool = True,
            select: bool = True,
            ):
        super(AutomatedCPC, self).run(image=image,
                                      do_pre_run=do_pre_run,
                                      select=select)

    def advance(self):
        """
        Does one cycle of pumping liquid pump out of the main vial and then pump back into the main vial.

        :return:
        """
        raise NotImplementedError

    # todo maybe move some of this up to superclass but not all
    def end_sequence(self,
                     time_since_started: float):
        """
        ending sequence of things to do. post slack messages to let user know the end time has been reached

        :param: float, time_since_started: number of hours that have elapsed since start time
        :return:
        """
        self.post_slack_message(f'CPC has run for {time_since_started} hours, the end time for CPC was'
                                f'{self.time_manager.end_time} hours after starting at {self.time_manager.start_time}. '
                                f'The run has completed :tada:')

        date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
        line_image_path = self.slack_images_folder.save_image_to_folder(
            image_name=f'last_image_for_run',
            image=line_img
        )
        self.post_slack_file(
            line_image_path,
            'The last image taken for experiment')
        print('run completed')

    def monitor_liquid_level(self, list_of_percent_diff=None):
        return super().monitor_liquid_level(list_of_percent_diff)

    # todo maybe move some of this to the superclass
    def not_in_tolerance(self, percent_diff):
        return super().not_in_tolerance(percent_diff=percent_diff)

    def self_correct(self,
                     percent_diff: float,
                     ):
        """
        Self-correct according to if there is too little/too much liquid in the vial relative to the reference line
        by pumping

        :param float, percent_diff: percentage relative to height of the image of the distance between the most
           recently measured meniscus line and the reference meniscus line
        :return:
        """
        raise NotImplementedError

    def calculate_how_much_to_self_correct(self,
                                           percent_diff: float,
                                           ):
        raise NotImplementedError

    def pump_self_correct(self,
                          time_to_pump,
                          direction,
                          ):
        raise NotImplementedError

    def check_if_should_try_again(self,
                                  no_meniscus_error: NoMeniscusFound,
                                  ):
        super().check_if_should_try_again(no_meniscus_error=no_meniscus_error)

    def try_again(self,
                  no_meniscus_error: NoMeniscusFound,
                  ):
        """
        Do what the superclass does but also save images of what the error looks like to disk
        :param no_meniscus_error:
        :return:
        """
        super().try_again(no_meniscus_error=no_meniscus_error)

    def do_not_try_again(self,
                         no_meniscus_error: NoMeniscusFound,
                         ):
        """
        Do what the superclass does but also save images of what the error looks like to disk and send slack messages
        of the error images.

        :param no_meniscus_error:
        :return:
        """
        super().do_not_try_again(no_meniscus_error=no_meniscus_error)

    # todo maybe move some of this up to superclass but not all
    def do_if_a_time_interval_has_passed(self,
                                         time: datetime):
        """
        Do this if a time interval has passed based on time passed through. add the new time interval to the time
        manager, and save the latest image analyzed to the folder for slack images to be saved in. then post that
        image to slack if there is a RTMSlackBot

        :param datetime, time:
        :return:
        """
        super(AutomatedCPC, self).do_if_a_time_interval_has_passed(time=time)

    def save_last_drawn_image(self):
        """
        Save the last image with drawn lines of where the liquid levels are to the computer

        :return: str, path_to_save_image: path to the image that was saved
        """
        return super().save_last_drawn_image()

    def post_slack_message(self,
                           message: str):
        super().post_slack_message(message=message)

    def post_slack_file(self,
                        file_path: str,
                        message: str,
                        ):
        super().post_slack_file(file_path=file_path,
                                message=message)

    def create_folder_hierarchy(self):
        super().create_folder_hierarchy()


class AutomatedCPCDualPumps(AutomatedCPC):

    def __init__(self,
                 liquid_level: LiquidLevel,
                 try_tracker: TryTracker,
                 time_manager: TimeManager,
                 withdraw_pump,  # pump that is set to withdraw from the vessel being watched
                 dispense_pump,  # pump that is set to dispense into the vessel being watched
                 initial_pump_rate: float,  # rate for the pump that runs constantly, in ml/min
                 time_to_self_correct: int = 10,  # number of seconds to allow only 1 pump to run for liquid level
                 # self correction
                 self_correction_pump_rate: float = None,
                 number_of_monitor_liquid_level_replicate_measurements: int = 3,
                 advance_time: int = 30,
                 slack_bot: RTMSlackBot = None,
                 show: bool = False,
                 save_folder_bool: bool = True,
                 save_folder_name: str = None,
                 save_folder_location: str = None,
                 ):
        super().__init__(liquid_level=liquid_level,
                         try_tracker=try_tracker,
                         time_manager=time_manager,
                         # really matter here, only one way to do things
                         initial_pump_rate=initial_pump_rate,
                         self_correction_pump_rate=self_correction_pump_rate,
                         number_of_monitor_liquid_level_replicate_measurements
                         =number_of_monitor_liquid_level_replicate_measurements,
                         pump=withdraw_pump,
                         slack_bot=slack_bot,
                         show=show,
                         save_folder_bool=save_folder_bool,
                         save_folder_name=save_folder_name,
                         save_folder_location=save_folder_location
                         )
        self.withdraw_pump = self.pump
        self.dispense_pump = dispense_pump
        self.initial_pump_rate = initial_pump_rate
        self.advance_time = advance_time
        self.track_liquid_tolerance_levels_fail_safe = TrackTwoLiquidToleranceLevels()

        self.time_to_self_correct = time_to_self_correct

        self.direction_to_withdraw = 'withdraw'
        self.direction_to_dispense = 'dispense'

        self.ia = ImageAnalysis()
        self.pumps_are_running = False  # for tracking if pumps are running or not, used to trigger the pumps when
        # un-pausing the experiment

    def pause_the_experiment(self):
        self.pause_experiment = True
        self.pumps_are_running = False
        self.stop_both_pumps()

    def pre_run(self,
                image=None,
                select: bool = True,
                ):
        """
        run the pre-run step to allow the user to select the reference and tolerance levels. but in a addition,
        also let the user select the fail-safe tolerance levels, where if the liquid level goes outside of the
        fail-safe tolerance levels, the the pumps are stopped and the run ends
        :param image:
        :param select:
        :return:
        """
        super().pre_run(image=image,
                        select=select)

        if image is None:
            image = self.liquid_level.camera.take_picture()

        curr_time = (datetime.now()).strftime('%Y_%m_%d_%H_%M_%S')
        self.raw_images_folder.save_image_to_folder(image_name=curr_time,
                                                    image=image)

        if select is True:
            self.track_liquid_tolerance_levels_fail_safe.select_tolerance(image=image)

    def run(self,
            image=None,
            do_pre_run: bool = True,
            select: bool = True,
            ):
        """
        The main function that puts together the methods to run liquid level monitoring of an image (that
        either must be passed through or else take a picture with the camera and use that), and then advances the
        system based on analysis of the image. What decision needs to be made based on the analysis needs to be
        separately implemented in each of the subclasses.

        The default action for the superclass run is only to delete the folder of images that were saved to
        the disk if the user doesn't want to save the images

        For this version specifically, two pumps are controlled, one will constantly run at a single rate,
        and the second will also constantly run, but it's rate may be adjusted to keep the liquid level within the
        user selected tolerance levels. If the liquid level goes outside of the fail-safe tolernace levels then both
        the pumps are stopped and the experiment ends.

        :param bool, do_pre_run: Whether to do the pre_run step or not
        :param image, an image. This will be the image that will be used to monitor the liquid level and go
        through
            the run. If none, the camera in the liquid level attribute will be used to take a photo.
        :param bool, select:
        :return:
        """

        try:
            if do_pre_run is True:  # if you do want to run the pre_run step
                # next line can throw NoMeniscusFound exception
                try:
                    self.pre_run(image=None, select=select)
                except NoMeniscusFound as error:
                    self.check_if_should_try_again(no_meniscus_error=error)

            self.set_up_applications_liquid_level_data_save_file()

            # the overall loop
            # initially start both the pumps
            self.start_both_pumps_at_initial_rate()
            self.pumps_are_running = True

            while self.run_experiment is True:
                if self.pause_experiment is not True:
                    # send slack message every interval of time specified in init to remind user that CPC is still
                    # running
                    curr_time = datetime.now()
                    time_since_started = self.time_manager.time_since_started(time=curr_time)

                    has_a_time_interval_elapsed = self.time_manager.has_a_time_interval_elapsed(time=curr_time)
                    if has_a_time_interval_elapsed:
                        self.do_if_a_time_interval_has_passed(time=curr_time)

                    # check if experiment has run for the amount of time user specified or not, if it has or has gone
                    # over; if gone over then end the script
                    after_end_time = self.time_manager.is_after_end_time(time=curr_time)
                    if after_end_time:
                        self.end_sequence(time_since_started=time_since_started)
                        return

                    if self.show is True:
                        # even though in the monitor_liquid_level function multiple images may be taken and the average
                        # result for tolerance_bool and percent_diff is taken, to not cause much lag and save screen
                        # space only the latest image the liquid level algorithm was run on will be displayed
                        _, last_image = self.liquid_level.all_images_with_lines[-1]
                        cv2.imshow('Last image that was taken', last_image)
                        # show the image, and if user presses the 'q' button, exit out of the run
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    try:
                        fail_safe_tolerance_bool, tolerance_bool, percent_diff = self.monitor_liquid_level()
                        if self.try_tracker.get_try_counter() is not 0:
                            self.post_slack_message('Liquid level found. Reset try counter to 0')
                            self.try_tracker.reset_try_counter()

                            date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
                            line_img = self.draw_fail_safe_tolerance_lines(image=line_img)
                            line_image_path = self.slack_images_folder.save_image_to_folder(
                                image_name=f'reset_try_counter_{date_time_with_line}',
                                image=line_img,
                            )
                            # self.post_slack_file(
                            #     file_path=line_image_path,
                            #     message='The last image taken')

                    except NoMeniscusFound as error:
                        self.check_if_should_try_again(error)
                        continue

                    if fail_safe_tolerance_bool is False:  # if the liquid level was found to be outside of the fail
                        # safe tolerance levels, then stop the pumps and stop the run
                        message = 'Stopped application because liquid level went outside fail safe tolerance bounds'
                        print(message)
                        self.post_slack_message(f'{message}')

                        if self.pumps_are_running is True:
                            self.stop_both_pumps()
                        return

                    if tolerance_bool:  # if the most recent measured meniscus level found is within the tolerance
                        # bounds advance normally next line can throw NoMeniscusFound exception
                        # START THE PUMPS AGAIN IF THEY had BEEN stopped previously aka if the liquid level went out of
                        # tolerance this causes the pumps to be stopped
                        if self.pumps_are_running is False:
                            self.start_both_pumps_at_initial_rate()
                            self.pumps_are_running = True
                        try:
                            self.advance()
                        except NoMeniscusFound as error:  # if algorithm couldnt find a meniscus
                            self.check_if_should_try_again(error)
                            continue
                    else:  # most recent measured meniscus was not within tolerance bounds
                        self.not_in_tolerance(percent_diff=percent_diff)
                else:  # self.pause_experiment is True
                    time.sleep(30)
            else:  # self.run_experiment is not True
                self.post_slack_message('experiment was manually ended')
                print('experiment was manually ended')
                raise KeyboardInterrupt

        except KeyboardInterrupt as error:
            print('Stopped application script using Keyboard Interrupt')
            self.post_slack_message('Stopped application script using Keyboard Interrupt')

        except Exception as error:
            print('Run has failed')
            # write the last seen image before the error image, and send that to user through slack
            self.post_slack_message(f'Something went wrong with the run. Error encountered: {error} :cry:')

            date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
            line_img = self.draw_fail_safe_tolerance_lines(image=line_img)
            date_time_for_edge, edge_img = self.liquid_level.all_images_edge[-1]

            line_image_path = self.slack_images_folder.save_image_to_folder(
                image_name=f'failed_exit_run_image_{date_time_with_line}',
                image=line_img
            )
            edge_image_path = self.slack_images_folder.save_image_to_folder(
                image_name=f'failed_exit_run_image_edge_{date_time_for_edge}',
                image=edge_img
            )
            self.post_slack_file(line_image_path,
                                 'Last image before error')
            self.post_slack_file(edge_image_path,
                                 'Last image before error')

            _, _, _, path_to_drawn_save_image = self.take_picture_and_run_liquid_level_algorithm()
            self.post_slack_file(path_to_drawn_save_image,
                                 'current image')

            raise error

        finally:
            if self.application_liquid_level_data_save_file_path is not None:
                self.update_json_file_with_new_liquid_level_data_values()

            try:
                self.dispense_pump.stop()
            except:
                pass
            try:
                self.withdraw_pump.stop()
            except:
                pass

            if self.save_folder_bool is False:
                self.save_folder.delete_from_disk()

    def save_last_drawn_image(self):
        """
        Save the last image with drawn lines of liquid levels

        :return:
        """
        date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
        line_img = self.draw_fail_safe_tolerance_lines(image=line_img)
        path_to_save_image = self.all_drawn_images_folder.save_image_to_folder(
            image_name=f'{date_time_with_line}',
            image=line_img
        )
        return path_to_save_image

    def start_both_pumps_at_initial_rate(self):
        self.dispense_pump.set_rate(rate=self.initial_pump_rate)
        self.withdraw_pump.set_rate(rate=self.initial_pump_rate)
        self.withdraw_pump.set_direction('withdraw')
        self.dispense_pump.set_direction('dispense')
        self.dispense_pump.run()
        self.withdraw_pump.run()

    def stop_both_pumps(self):
        self.dispense_pump.stop()
        self.withdraw_pump.stop()

    def advance(self):
        time.sleep(self.advance_time)

    def monitor_liquid_level(self, list_of_percent_diff=None):
        """
        Method used to monitor the liquid level. mostly similar to the superclass version. the difference here is
        that the fail safe tolerance level tracker is also used to check to see if the liquid level is within the
        tolerance levels of the fail safe tolerance tracker or not

        :return:
        """
        # todo new way
        # instead of just taking one picture, take 3 and find the average of the tolerance_bool values and
        # round the value. Once rounded, then only take the corresponding percent_diff values that have the same
        # tolerance_bool value, where 0 is False and 1 is True, and average them to get the position of the liquid
        # level relative to how far from the reference level it is. this way, if the analysis of one image incorrectly
        # identifies the location of the liquid level compared to the other images, its value will be ignored
        number_of_times_to_try = self.number_of_monitor_liquid_level_replicate_measurements  # must be an odd number

        list_of_percent_diff = []  # list to keep track of the difference between the current liquid level height
        # with the reference height, in terms of percentage of the entire image height

        average_percent_diff = 0  # average difference between the identified current liquid level and the reference
        # liquid level, ignoring values that are outliers

        # given an array of numbers, will return set that rejects any numbers outside modified_z_score threshold and
        # return back the original array with only the values beneath the threshold to keep
        def reject_outliers_modified_z(df, threshold=50):
            median = np.median(df)
            median_absolute_deviation = np.median([np.abs(i - median) for i in df])
            if median_absolute_deviation == 0:
                median_absolute_deviation = 0.001
            modified_z_scores = [0.6745 * (i - median) / median_absolute_deviation for i in df]
            # creates array that gives indeces to keep
            keep = [df[index] for index, value in enumerate(modified_z_scores) if value < threshold]
            return keep

        # can throw a NoMeniscusFound exception
        # implement the variance method of doing this
        for i in range(number_of_times_to_try):
            # loop through the number of times to try to take pictures to try to find the liquid level. percent_diff is
            # how far away the meniscus is from the the reference liquid level line. collect these all into
            # list_of_percent_diff
            try:
                _, _, percent_diff, _ = self.take_picture_and_run_liquid_level_algorithm()
                self.try_tracker.reset_try_counter()
                list_of_percent_diff.append(percent_diff)
            except NoMeniscusFound as error:
                self.check_if_should_try_again(no_meniscus_error=error)
                number_of_times_to_try += 1

        # use the reject_current_outliers method to remove outliers from measuring the percent_diff values
        list_of_percent_diff = reject_outliers_modified_z(list_of_percent_diff)

        # find the average percent diff from the remaining values
        for percent_diff_value in list_of_percent_diff:
            average_percent_diff += percent_diff_value
        average_percent_diff = average_percent_diff / len(list_of_percent_diff)

        # then need to convert the average percent diff back into relative to the image height value
        absolute_diff = average_percent_diff * self.liquid_level.track_liquid_tolerance_levels.reference_image_height
        average_liquid_level_height = self.liquid_level.track_liquid_tolerance_levels.get_absolute_reference_height() - absolute_diff
        average_liquid_level_percent_height = average_liquid_level_height/self.liquid_level.track_liquid_tolerance_levels.reference_image_height

        # on average, was the liquid level within tolerance or not, taking into account getting rid of outlier values
        average_tolerance_bool = self.liquid_level.in_tolerance(average_liquid_level_percent_height)
        rounded_tolerance_bool = average_tolerance_bool

        average_fail_safe_tolerance_bool = self.track_liquid_tolerance_levels_fail_safe.in_tolerance(average_liquid_level_percent_height)
        rounded_fail_safe_tolerance_bool = average_fail_safe_tolerance_bool

        # save new average liquid level percent height to memory, and after n values are in memory, save it to the
        # application liquid level data json file
        time = datetime.now()
        time_formatted = time.strftime(self.datetime_format)

        self.application_liquid_level_data[time_formatted] = average_liquid_level_percent_height
        # to prevent using too much memory, after there are n data points in the dictionary, update the json file
        # with this data then reset the dictionary again
        n = 5
        if self.application_liquid_level_data_save_file_path is not None:
            if len(self.application_liquid_level_data) >= n:
                self.update_json_file_with_new_liquid_level_data_values()
                self.application_liquid_level_data = {}

        return rounded_fail_safe_tolerance_bool, rounded_tolerance_bool, average_percent_diff
    #Old way--------------------------------------------------------------------------------------------------
        # # instead of just taking one picture, take 3 and find the average of the tolerance_bool values and
        # # round the value. Once rounded, then only take the corresponding percent_diff values that have the same
        # # tolerance_bool value, where 0 is False and 1 is True, and average them to get the position of the liquid
        # # level relative to how far from the reference level it is. this way, if the analysis of one image incorrectly
        # # identifies the location of the liquid level compared to the other images, its value will be ignored
        # number_of_times_to_try = self.number_of_monitor_liquid_level_replicate_measurements  # must be an odd number
        # list_of_tolerance_bool = []  # list to keep track of the images that indicate if the liquid level is within
        # # the set tolerance boundaries
        # lis_of_fail_safe_tolerance_bool = []
        # list_of_percent_diff = []  # list to keep track of the difference between the current liquid level height
        # # with the reference height, in terms of percentage of the entire image height
        # average_tolerance_bool = 0  # average of the values in list_of_tolerance_bool, will be between 0 and 1 and
        # # then rounded to 0 or 1. the way this will be used is that False will be treated as 0 and True as 1 when
        # #  checking whether the liquid level line is within the tolerance bounds or not
        # average_fail_safe_tolerance_bool = 0
        # average_percent_diff = 0  # average difference between the identified current liquid level and the reference
        # # liquid level, ignoring values that are outliers
        #
        # # can throw a NoMeniscusFound exception
        # for i in range(number_of_times_to_try):
        #     # loop through the number of times to try to take pictures to try to find the meniscus. every time,
        #     # the tolerance_bool, of whether the meniscus is within the tolerance bounds or not, and percent_diff is
        #     # how far away the meniscus is from the the reference liquid level line
        #     fail_safe_tolerance_bool, tolerance_bool, percent_diff, _ = \
        #         self.take_picture_and_run_liquid_level_algorithm()
        #     list_of_tolerance_bool.append(tolerance_bool)
        #     lis_of_fail_safe_tolerance_bool.append(fail_safe_tolerance_bool)
        #     list_of_percent_diff.append(percent_diff)
        #     # add one to the average_tolerance_bool if the liquid level was found to be within the tolerance bounds
        #     if tolerance_bool is True:
        #         average_tolerance_bool = average_tolerance_bool + 1
        #     if fail_safe_tolerance_bool is True:
        #         average_fail_safe_tolerance_bool = average_fail_safe_tolerance_bool + 1
        # # find the average value by dividing by number_of_times_to_try
        # average_tolerance_bool = average_tolerance_bool / number_of_times_to_try
        # average_fail_safe_tolerance_bool = average_fail_safe_tolerance_bool / number_of_times_to_try
        # # round the value so it is either 0 or 1, where 0 would mean the majority of the measured liquid level
        # # readings were outside of the tolerance bounds, and 1 means that the majority were inside the tolerance bounds
        # rounded_tolerance_bool = round(average_tolerance_bool)
        # rounded_fail_safe_tolerance_bool = round(average_fail_safe_tolerance_bool)
        #
        # # majority of the tries to find the meniscus found it outside of the tolerance bounds
        # if rounded_tolerance_bool is 0:
        #     # loop through the list_of_tolerance_bool, and if the value is False, then add the value at the
        #     # corresponding index in the list_of_percent_diff to average_percent_diff
        #     for i in range(number_of_times_to_try):
        #         if list_of_tolerance_bool[i] is False:
        #             average_percent_diff = average_percent_diff + list_of_percent_diff[i]
        # else:
        #     # else then the rounded tolerance is 1 and so the majority of the tries to find the meniscus found it inside
        #     #  of the tolerance bounds
        #     # loop through the list_of_tolerance_bool, and if the value is True, then add the value at the
        #     # corresponding index in the list_of_percent_diff to average_percent_diff
        #     for i in range(number_of_times_to_try):
        #         if list_of_tolerance_bool[i] is True:
        #             average_percent_diff = average_percent_diff + list_of_percent_diff[i]
        #
        # # then divide by the number_of_times_to_try to get the average value
        # average_percent_diff = average_percent_diff / number_of_times_to_try
        # # then return the rounded tolerance bool and average percentage diff
        # return rounded_fail_safe_tolerance_bool, rounded_tolerance_bool, average_percent_diff

    #Old way--------------------------------------------------------------------------------------------------

    def take_picture_and_run_liquid_level_algorithm(self):
        """
        mostly like superclass version, except also check whether the liquid level is within the fail safe tolerance
        liquid level or not

        :return: tolerance_bool, bool: True if the liquid level is within tolerance. float, percent_diff: the
            relative distance the identified liquid level is away from the user set reference level.
            str, path_to_save_image. fail_safe_tolerance_bool, True if the liquid level is within tolerance of the
            fail safe tolerance bounds
        """
        curr_time = (datetime.now()).strftime('%Y_%m_%d_%H_%M_%S')
        image = self.liquid_level.camera.take_picture()
        self.raw_images_folder.save_image_to_folder(image_name=curr_time,
                                                    image=image)
        tolerance_bool, percent_diff = self.liquid_level.run(image=image)
        liquid_level_current_row = self.liquid_level.row
        fail_safe_tolerance_bool = self.track_liquid_tolerance_levels_fail_safe.in_tolerance(height=liquid_level_current_row)

        path_to_drawn_save_image = self.save_last_drawn_image()
        return fail_safe_tolerance_bool, tolerance_bool, percent_diff, path_to_drawn_save_image

    def not_in_tolerance(self, percent_diff):
        """
        Do this if the liquid level was found not to be within tolerance to adjust the liquid level to be within
        tolerance

        :param: float, percent_diff, the relative distance of the current liquid level from a set reference level
        :return:
        """
        if self.pumps_are_running is True:
            self.stop_both_pumps()
            self.pumps_are_running = False

        date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
        line_img = self.draw_fail_safe_tolerance_lines(image=line_img)

        line_image_path = self.slack_images_folder.save_image_to_folder(
            image_name=f'out_of_tolerance_{date_time_with_line}',
            image=line_img
        )
        if percent_diff > 0:
            above_or_below = 'above'
        else:  # if less than tolerance then pump into the vial being watched
            above_or_below = 'below'

        print(f'liquid level {above_or_below} tolerance')
        print(f'current liquid level difference from the reference liquid level by {percent_diff}')
        self.post_slack_message(f'Liquid level moved {above_or_below} the tolerance boundaries. '
                                f'Current liquid level differs from the reference liquid level by {percent_diff}')

        # self.post_slack_file(line_image_path,
        #                      'Liquid level out of tolerance bounds image')

        # next line can throw NoMeniscusFound exception
        self.self_correct(percent_diff=percent_diff)

        # next line can throw NoMeniscusFound exception
        try:
            # immediately analyze a photo after self correction step and send message and picture to user so they can
            # tell immediately if self correction worked or not
            _, tolerance_bool, percent_diff, path_to_drawn_save_image = \
                self.take_picture_and_run_liquid_level_algorithm()
            self.post_slack_message(f'After self correction - liquid level within tolerance bounds: {tolerance_bool}, '
                                    f'liquid level differs from the reference liquid level by {percent_diff}')
            # self.post_slack_file(path_to_drawn_save_image,
            #                      'Liquid level image after self correction')
            print(f'After self correction - liquid level within tolerance bounds: {tolerance_bool}, '
                  f'liquid level differs from the reference liquid level by {percent_diff}')
        except NoMeniscusFound as error:
            self.check_if_should_try_again(no_meniscus_error=error)

    def self_correct(self,
                     percent_diff: float,
                     ):
        """
        stop one of the pumps so that self correction of the liquid level can occur; done with a single user set self
        correction rate and time that is not determined by the distance between the currently liquid level and the
        reference liquid level

        :param percent_diff:
        :return:
        """
        if percent_diff > 0:  # if more
            # than tolerance ( aka if meniscus travelled upwards)
            self_correct_direction = 'withdraw'
        else:  # if less than tolerance (aka meniscus travelled downwards)
            self_correct_direction = 'dispense'

        message = f'to self correct need to {self_correct_direction}. percent diff is: {percent_diff} for ' \
                  f'{self.time_to_self_correct}'
        print(message)
        self.post_slack_message(message=message)

        self.pump_self_correct(time_to_pump=self.time_to_self_correct,
                               direction=self_correct_direction
                               )
        if self.application_liquid_level_data_save_file_path is not None:
            self.update_json_file_with_new_liquid_level_correction_data_values()

    def pump_self_correct(self,
                          time_to_pump,
                          direction,
                          ):
        if direction == 'withdraw':
            self_correction_pump = self.withdraw_pump
        else:  # is self_correct_direction is 'dispense'
            self_correction_pump = self.dispense_pump

        self_correction_pump.set_rate(rate=self.self_correction_pump_rate)
        self_correction_pump.start()

        time.sleep(time_to_pump)

        self_correction_pump.stop()

    def draw_fail_safe_tolerance_lines(self, image):
        """
        Draw two lines at the location of the two user selected fail safe tolerance lines. this is nearly identical
        to the draw_tolerance_lines() method in the liquid_level class
        :param image:
        :return:
        """
        _, img_width = self.ia.find_image_height_width(image=image)

        list_of_absolute_tolerance_levels = self.track_liquid_tolerance_levels_fail_safe.get_absolute_tolerance_height()

        if len(list_of_absolute_tolerance_levels) is 0:
            return image

        for absolute_tolerance_level in list_of_absolute_tolerance_levels:
            tolerance_left_point = (0, absolute_tolerance_level)
            tolerance_right_point = (img_width, absolute_tolerance_level)

            # draw blue line for tolerance
            pink = (88, 100, 300)
            text_position = (0, 60)
            image = self.ia.draw_line(image=image,
                                      left_point=tolerance_left_point,
                                      right_point=tolerance_right_point,
                                      colour=pink,
                                      text='fail-safe tolerance',
                                      text_position=text_position
                                      )
        return image


class AutomatedCPCPeristaltic(AutomatedCPC):
    def __init__(self,
                 liquid_level: LiquidLevel,
                 try_tracker: TryTracker,
                 time_manager: TimeManager,
                 pump,
                 initial_pump_rate: float = None,
                 self_correction_pump_rate: float = None,
                 number_of_monitor_liquid_level_replicate_measurements: int = 3,
                 slack_bot: RTMSlackBot = None,
                 show: bool = False,
                 save_folder_bool: bool = True,
                 save_folder_location: str = None,
                 save_folder_name: str = None,
                 wait_time: int = 1,
                 ):
        """
        Automated CPC

        :param liquid_level:
        :param try_tracker:
        :param time_manager:
        :param PeristalticPumpControl, pump: For the pump, the set direction of the pump should be so that liquid
            will be pumped out of the vial that is being watched by the camera. the pump class is located in the
            gronckle repository in north_robotics_hardware.peristaltic_pump_control.py
        :param slack_bot:
        :param show:
        :param save_folder_bool:
        :param save_folder_location:
        :param save_folder_name:
        :param wait_time: how long to wait after a pump action gets completed before the next action is taken
        """
        super().__init__(liquid_level=liquid_level,
                         try_tracker=try_tracker,
                         time_manager=time_manager,
                         pump=pump,
                         initial_pump_rate=initial_pump_rate,
                         self_correction_pump_rate=self_correction_pump_rate,
                         number_of_monitor_liquid_level_replicate_measurements
                         =number_of_monitor_liquid_level_replicate_measurements,
                         slack_bot=slack_bot,
                         show=show,
                         save_folder_bool=save_folder_bool,
                         save_folder_location=save_folder_location,
                         save_folder_name=save_folder_name,
                         )
        self.wait_time = wait_time  # how long to wait between consecutive pumps from the same pump

    def pre_run(self,
                image=None,
                select: bool = True,
                ):
        """
        see superclass

        :param select:
        :return:
        """
        super().pre_run(image=image, select=select)

    def run(self,
            image=None,
            do_pre_run: bool = True,
            select: bool = True,
            ):
        super().run(image=image,
                    do_pre_run=do_pre_run,
                    select=select)

    def advance(self):
        """
        Does one cycle of pumping liquid pump out of the main vial and then pump back into the main vial.

        :return:
        """
        raise NotImplementedError

    def self_correct(self,
                     percent_diff: float,
                     ):
        raise NotImplementedError

    def calculate_how_much_to_self_correct(self,
                                           percent_diff: float,
                                           ):
        raise NotImplementedError

    def pump_self_correct(self,
                          time_to_pump,
                          direction,
                          ):
        raise NotImplementedError


class AutomatedCPCNewEraPeristalticPump(AutomatedCPCPeristaltic):
    def __init__(self,
                 liquid_level: LiquidLevel,
                 try_tracker: TryTracker,
                 time_manager: TimeManager,
                 pump,
                 initial_pump_rate: float = None,
                 self_correction_pump_rate: float = None,
                 time_to_self_correct: int = 10,
                 number_of_monitor_liquid_level_replicate_measurements: int = 3,
                 advance_time: int = 15,
                 slack_bot: RTMSlackBot = None,
                 show: bool = False,
                 save_folder_bool: bool = True,
                 save_folder_location: str = None,
                 save_folder_name: str = None,
                 wait_time: int = 1,
                 ):
        """
      Uses a New Era peristaltic pump

        :param liquid_level:
        :param try_tracker:
        :param time_manager:
        :param pump:
        :param slack_bot:
        :param show:
        :param save_folder_bool:
        :param save_folder_location:
        :param save_folder_name:
        :param wait_time:
        """
        super().__init__(liquid_level=liquid_level,
                         try_tracker=try_tracker,
                         time_manager=time_manager,
                         pump=pump,
                         initial_pump_rate=initial_pump_rate,
                         self_correction_pump_rate=self_correction_pump_rate,
                         number_of_monitor_liquid_level_replicate_measurements
                         =number_of_monitor_liquid_level_replicate_measurements,
                         slack_bot=slack_bot,
                         show=show,
                         save_folder_bool=save_folder_bool,
                         save_folder_location=save_folder_location,
                         save_folder_name=save_folder_name,
                         wait_time=wait_time,
                         )
        self.advance_time = advance_time
        self.time_to_pump_correction = time_to_self_correct  # hard coded
        self.direction_to_withdraw = 'withdraw'
        self.direction_to_dispense = 'dispense'

    def pre_run(self,
                image=None,
                select: bool = True,
                ):
        super().pre_run(image=image,
                        select=select,
                        )

    def run(self,
            image=None,
            do_pre_run: bool = True,
            select: bool = True,
            ):
        super().run(image=image,
                    do_pre_run=do_pre_run,
                    select=select)

    def advance(self):
        print(f'start a cycle')

        self.pump.pump(
            pump_time=self.advance_time,
            direction=self.direction_to_dispense,
            wait_time=self.wait_time
        )

        time.sleep(2)

        self.pump.pump(
            pump_time=self.advance_time,
            direction=self.direction_to_withdraw,
            wait_time=self.wait_time
        )

        time.sleep(2)

        print(f'finished a cycle')

    def self_correct(self,
                     percent_diff: float,
                     ):
        self_correct_time = self.calculate_how_much_to_self_correct(percent_diff=percent_diff)

        if percent_diff > 0:  # if more than tolerance then pump out of the vial being watched
            self_correct_direction = self.direction_to_withdraw
        else:  # if less than tolerance then pump into the vial being watched
            self_correct_direction = self.direction_to_dispense

        print(f'direction to pump: {self_correct_direction}, '
              f'time to pump to self correct: f{self_correct_time}')
        self.post_slack_message(f'direction to pump: {self_correct_direction}, '
                                f'time to pump to self correct: {self.time_to_pump_correction}')

        self.pump_self_correct(time_to_pump=self_correct_time,
                               direction=self_correct_direction,
                               )

        if self.application_liquid_level_data_save_file_path is not None:
            self.update_json_file_with_new_liquid_level_correction_data_values()

    def calculate_how_much_to_self_correct(self,
                                           percent_diff: float,
                                           ):
        return self.time_to_pump_correction

    def pump_self_correct(self,
                          time_to_pump,
                          direction,
                          ):
        self.pump.set_rate(rate=self.self_correction_pump_rate)
        self.pump.pump(pump_time=time_to_pump,
                       direction=direction,
                       wait_time=self.wait_time,
                       )
        self.pump.set_rate(rate=self.self_correction_pump_rate)


class AutomatedSlurryFiltration(LiquidLevelMonitor):
    def __init__(self,
                 liquid_level: LiquidLevel,
                 try_tracker: TryTracker,
                 time_manager: TimeManager,
                 pump,
                 number_of_monitor_liquid_level_replicate_measurements: int = 3,
                 advance_time: int = 15,
                 slack_bot: RTMSlackBot = None,
                 show: bool = False,
                 save_folder_bool: bool = True,
                 save_folder_name: str = None,
                 save_folder_location: str = None,
                 ):
        """
        Generic liquid level monitor class
        :param LiquidLevel, liquid_level: instance of LiquidLevel, should have a TrackOneLiquidToleranceLevel tracker
        :param TryTracker, try_tracker:
        :param TimeManager, time_manager:
        :param int, advance_time: the number of seconds to allow filtration to occur before checking the liquid level
        :param RTMSlackBot, slack_bot:
        :param bool, show: True to display the last image that was taken from python using cv2 on the screen
        :param bool, save_folder_bool: True if at the end of the application you want to save all the images
            that were taken and used throughout the application run to use
        :param str, save_folder_name: Name of the save folder - generally it would be the experiment name
        :param str, save_folder_location: location to create the folder to save everything
        """
        super().__init__(liquid_level=liquid_level,
                         try_tracker=try_tracker,
                         time_manager=time_manager,
                         number_of_monitor_liquid_level_replicate_measurements
                         =number_of_monitor_liquid_level_replicate_measurements,
                         slack_bot=slack_bot,
                         show=show,
                         save_folder_bool=save_folder_bool,
                         save_folder_name=save_folder_name,
                         save_folder_location=save_folder_location
                         )
        self.pump = pump
        self.advance_time = advance_time

    def pre_run(self,
                image=None,
                select: bool = True,
                ):
        """
        Allow user to select the tolerance level for when self correction needs to be done, and set the liquid level
        to self correct to

        :param image:
        :param select:
        :return:
        """

        if image is None:
            image = self.liquid_level.camera.take_picture()

        curr_time = (datetime.now()).strftime('%Y_%m_%d_%H_%M_%S')
        self.raw_images_folder.save_image_to_folder(image_name=curr_time,
                                                    image=image)

        if select is True:
            self.liquid_level.track_liquid_tolerance_levels.reference_row=0
            self.liquid_level.start(
                image=image,
                select_region_of_interest=True,
                set_reference=False,
                select_tolerance=True,
            )
        else:
            self.liquid_level.start(
                image=image,
                select_region_of_interest=False,
                set_reference=False,
                select_tolerance=False,
            )
        print(f'pre_run complete')

    def run(self,
            image=None,
            do_pre_run: bool = True,
            select=True,
            ):
        super(AutomatedSlurryFiltration, self).run(image=image,
                                                   do_pre_run=do_pre_run,
                                                   select=select)

    def set_advance_time(self,
                         advance_time: int):
        self.advance_time = advance_time

    def advance(self):
        time.sleep(self.advance_time)

    # todo very similar to cpc version - only difference in message that gets posted to slack
    def end_sequence(self,
                     time_since_started: float):
        """
        ending sequence of things to do. post slack messages to let user know the end time has been reached

        :param: float, time_since_started: number of hours that have elapsed since start time
        :return:
        """
        'this is the same as in automated cpc'

        self.post_slack_message(f'Automated slurry filtration has run for {round(time_since_started, 2)} hours, '
                                f'the end time was {self.time_manager.end_time} hours after starting at '
                                f'{self.time_manager.start_time}. The run has completed :tada:')

        date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
        line_image_path = self.slack_images_folder.save_image_to_folder(
            image_name=f'last_image_for_run',
            image=line_img
        )
        self.post_slack_file(
            line_image_path,
            'The last image taken for experiment')
        print('run completed')

    def monitor_liquid_level(self, list_of_percent_diff=None):
        return super(AutomatedSlurryFiltration, self).monitor_liquid_level(list_of_percent_diff)

    def not_in_tolerance(self, percent_diff):
        """
        What to do if the liquid level was not found to be within tolerance

        :param float, percent_diff: the relative distance of the current liquid level from a set reference level
        :return:
        """
        return super().not_in_tolerance(percent_diff=percent_diff)

    def self_correct(self,
                     percent_diff: float,
                     ):
        return NotImplementedError

    def check_if_should_try_again(self,
                                  no_meniscus_error: NoMeniscusFound,
                                  ):
        super().check_if_should_try_again(no_meniscus_error=no_meniscus_error)

    def try_again(self,
                  no_meniscus_error: NoMeniscusFound,
                  ):
        """
        Do what the superclass does but also save images of what the error looks like to disk
        :param no_meniscus_error:
        :return:
        """
        super().try_again(no_meniscus_error=no_meniscus_error)


class AutomatedSlurryFiltrationPeristaltic(AutomatedSlurryFiltration):
    def __init__(self,
                 liquid_level: LiquidLevel,
                 try_tracker: TryTracker,
                 time_manager: TimeManager,
                 pump,
                 # pump: NewEraPeristalticPumpInterface,
                 number_of_monitor_liquid_level_replicate_measurements: int = 3,
                 advance_time: int = 15,
                 slack_bot: RTMSlackBot = None,
                 show: bool = False,
                 save_folder_bool: bool = True,
                 save_folder_name: str = None,
                 save_folder_location: str = None,
                 time_to_self_correct: int = 5,
                 ):
        """
        Uses a new era peristaltic pump

        :param liquid_level:
        :param try_tracker:
        :param time_manager:
        :param pump:
        :param slack_bot:
        :param show:
        :param save_folder_bool:
        :param save_folder_name:
        :param save_folder_location:
        :param int, time_to_self_correct: default time to self make the pump run for to self correct
        """
        super().__init__(liquid_level=liquid_level,
                         try_tracker=try_tracker,
                         time_manager=time_manager,
                         pump=pump,
                         number_of_monitor_liquid_level_replicate_measurements
                         =number_of_monitor_liquid_level_replicate_measurements,
                         advance_time=advance_time,
                         slack_bot=slack_bot,
                         show=show,
                         save_folder_bool=save_folder_bool,
                         save_folder_name=save_folder_name,
                         save_folder_location=save_folder_location
                         )
        self.time_to_self_correct = time_to_self_correct

    def set_rate(self,
                 rate: float,
                 unit=None):
        """
        Set the rate of the pump. see method in NewEraPeristalticPumpInterface

        :param rate:
        :param unit:
        :return:
        """
        self.pump.set_rate(rate=rate,
                           unit=unit

                           )

    def set_direction(self,
                      direction):
        """
        Set the direction of the pump. see method in NewEraPeristalticPumpInterface

        :param str, direction: etiher 'dispense' or 'withdray'
        :return:
        """
        self.pump.set_direction(direction=direction)

    def pre_run(self,
                image=None,
                select: bool = True,
                ):
        """
        see superclass

        :param select:
        :return:
        """
        # todo check this is all it should do
        super().pre_run(image=image, select=select)

    def run(self,
            image=None,
            do_pre_run: bool = True,
            select: bool = True,
            ):
        super().run(image=image,
                    do_pre_run=do_pre_run,
                    select=select)

    def self_correct(self,
                     percent_diff: float,
                     ):
        self.pump.pump(pump_time=self.time_to_self_correct,
                       direction='dispense',
                       wait_time=1,
                       )

        if self.application_liquid_level_data_save_file_path is not None:
            self.update_json_file_with_new_liquid_level_correction_data_values()


class AutomatedContinuousDistillation(LiquidLevelMonitor):
    def __init__(self,
                 liquid_level: LiquidLevel,
                 try_tracker: TryTracker,
                 time_manager: TimeManager,
                 pump,
                 number_of_monitor_liquid_level_replicate_measurements: int = 3,
                 advance_time: int = 15,
                 slack_bot: RTMSlackBot = None,
                 show: bool = False,
                 save_folder_bool: bool = True,
                 save_folder_name: str = None,
                 save_folder_location: str = None,
                 wait_time: int = 1,
                 ):
        """
        :param liquid_level:
        :param try_tracker:
        :param time_manager:
        :param pump:
        :param int, advance_time: the number of seconds to allow filtration to occur before checking the liquid level
        :param slack_bot:
        :param show:
        :param save_folder_bool:
        :param save_folder_name:
        :param save_folder_location:
        :param wait_time:
        """
        super().__init__(liquid_level=liquid_level,
                         try_tracker=try_tracker,
                         time_manager=time_manager,
                         number_of_monitor_liquid_level_replicate_measurements
                         =number_of_monitor_liquid_level_replicate_measurements,
                         slack_bot=slack_bot,
                         show=show,
                         save_folder_bool=save_folder_bool,
                         save_folder_name=save_folder_name,
                         save_folder_location=save_folder_location
                         )
        self.wait_time = wait_time
        self.pump = pump
        self.advance_time = advance_time
        self.direction_to_withdraw = 'withdraw'
        self.direction_to_dispense = 'dispense'

    def pre_run(self,
                image=None,
                select: bool = True,
                ):
        """
        see superclass

        :param image:
        :param select:
        :return:
        """
        # next line can throw NoMeniscusFound exception
        if image is None:
            image = self.liquid_level.camera.take_picture()

        curr_time = (datetime.now()).strftime('%Y_%m_%d_%H_%M_%S')
        self.raw_images_folder.save_image_to_folder(image_name=curr_time,
                                                    image=image)

        if select is True:
            self.liquid_level.track_liquid_tolerance_levels.reference_row = 0
            self.liquid_level.start(
                image=image,
                select_region_of_interest=True,
                set_reference=False,
                select_tolerance=True,
            )
        else:
            self.liquid_level.start(
                image=image,
                select_region_of_interest=False,
                set_reference=False,
                select_tolerance=False,
            )
        print(f'pre_run complete')

    def run(self,
            image=None,
            do_pre_run: bool = True,
            select: bool = True,
            ):
        super(AutomatedContinuousDistillation, self).run(image=image,
                                                         do_pre_run=do_pre_run,
                                                         select=select)

    def set_advance_time(self,
                         advance_time: int):
        self.advance_time = advance_time

    def advance(self):
        """
        wait for a certain period of time before monitoring the liquid level to know if more liquid needs to be added
        in to maintain the level
        :return:
        """
        time.sleep(self.advance_time)

    def end_sequence(self,
                     time_since_started: float):
        """
        ending sequence of things to do. post slack messages to let user know the end time has been reached

        :param: float, time_since_started: number of hours that have elapsed since start time
        :return:
        """
        self.post_slack_message(f'Continuous distillation has run for {time_since_started} hours, the end time was'
                                f'{self.time_manager.end_time} hours after starting at {self.time_manager.start_time}. '
                                f'The run has completed :tada:')

        date_time_with_line, line_img = self.liquid_level.all_images_with_lines[-1]
        line_image_path = self.slack_images_folder.save_image_to_folder(
            image_name=f'last_image_for_run',
            image=line_img
        )
        self.post_slack_file(
            line_image_path,
            'The last image taken for experiment')
        print('run completed')

    def monitor_liquid_level(self, list_of_percent_diff=None):
        return super().monitor_liquid_level(list_of_percent_diff)

    # todo maybe move some of this to the superclass
    def not_in_tolerance(self, percent_diff):
        return super().not_in_tolerance(percent_diff=percent_diff)

    def self_correct(self,
                     percent_diff: float,
                     ):
        """
        Self-correct according to if there is too little liquid in the vial relative to the reference line by adding
        more in

        :param float, percent_diff: percentage relative to height of the image of the distance between the most
           recently measured meniscus line and the reference meniscus line
        :return:
        """
        raise NotImplementedError

    def calculate_how_much_to_self_correct(self,
                                           percent_diff: float,
                                           ):
        raise NotImplementedError

    def pump_self_correct(self,
                          time_to_pump,
                          direction,
                          ):
        raise NotImplementedError

    def check_if_should_try_again(self,
                                  no_meniscus_error: NoMeniscusFound,
                                  ):
        super().check_if_should_try_again(no_meniscus_error=no_meniscus_error)

    def try_again(self,
                  no_meniscus_error: NoMeniscusFound,
                  ):
        """
        Do what the superclass does but also save images of what the error looks like to disk
        :param no_meniscus_error:
        :return:
        """
        super().try_again(no_meniscus_error=no_meniscus_error)

    def do_not_try_again(self,
                         no_meniscus_error: NoMeniscusFound,
                         ):
        """
        Do what the superclass does but also save images of what the error looks like to disk and send slack messages
        of the error images.

        :param no_meniscus_error:
        :return:
        """
        super().do_not_try_again(no_meniscus_error=no_meniscus_error)

    def do_if_a_time_interval_has_passed(self,
                                         time: datetime):
        """
        Do this if a time interval has passed based on time passed through. add the new time interval to the time
        manager, and save the latest image analyzed to the folder for slack images to be saved in. then post that
        image to slack if there is a RTMSlackBot

        :param datetime, time:
        :return:
        """
        super(AutomatedContinuousDistillation, self).do_if_a_time_interval_has_passed(time=time)

    def save_last_drawn_image(self):
        """
        Save the last image with drawn lines of where the liquid levels are to the computer

        :return: str, path_to_save_image: path to the image that was saved
        """
        return super().save_last_drawn_image()

    def post_slack_message(self,
                           message: str):
        super().post_slack_message(message=message)

    def post_slack_file(self,
                        file_path: str,
                        message: str,
                        ):
        super().post_slack_file(file_path=file_path,
                                message=message)

    def create_folder_hierarchy(self):
        super().create_folder_hierarchy()


class AutomatedContinuousDistillationPeristalticPump(AutomatedContinuousDistillation):
    def __init__(self,
                 liquid_level: LiquidLevel,
                 try_tracker: TryTracker,
                 time_manager: TimeManager,
                 pump,
                 time_to_self_correct: int = 10,  # number of seconds to allow only 1 pump to run for liquid level
                 # self correction
                 number_of_monitor_liquid_level_replicate_measurements: int = 3,
                 advance_time: int = 15,
                 slack_bot: RTMSlackBot = None,
                 show: bool = False,
                 save_folder_bool: bool = True,
                 save_folder_location: str = None,
                 save_folder_name: str = None,
                 wait_time: int = 1,
                 ):
        """
        Uses a New Era peristaltic pump

        :param liquid_level:
        :param try_tracker:
        :param time_manager:
        :param pump:
        :param slack_bot:
        :param show:
        :param save_folder_bool:
        :param save_folder_location:
        :param save_folder_name:
        :param wait_time:
        """
        super().__init__(liquid_level=liquid_level,
                         try_tracker=try_tracker,
                         time_manager=time_manager,
                         pump=pump,
                         number_of_monitor_liquid_level_replicate_measurements
                         =number_of_monitor_liquid_level_replicate_measurements,
                         advance_time=advance_time,
                         slack_bot=slack_bot,
                         show=show,
                         save_folder_bool=save_folder_bool,
                         save_folder_location=save_folder_location,
                         save_folder_name=save_folder_name,
                         wait_time=wait_time,
                         )
        self.time_to_pump_correction = time_to_self_correct  # hard coded
        self.direction_to_withdraw = 'withdraw'
        self.direction_to_dispense = 'dispense'

    def pre_run(self,
                image=None,
                select: bool = True,
                ):
        super().pre_run(image=image,
                        select=select,
                        )

    def run(self,
            image=None,
            do_pre_run: bool = True,
            select: bool = True,
            ):
        super().run(image=image,
                    do_pre_run=do_pre_run,
                    select=select)

    def advance(self):
        super().advance()

    def self_correct(self,
                     percent_diff: float,
                     ):
        self_correct_time = self.calculate_how_much_to_self_correct(percent_diff=percent_diff)

        # since there will be 1 tolerance line, under which the liquid level is said to be out of bounds, this means
        # that the only direction you would need to self correct is to add more liquid in aka dispense
        self_correct_direction = self.direction_to_dispense

        print(f'direction to pump: {self_correct_direction}, '
              f'time to pump to self correct: f{self_correct_time}')
        self.post_slack_message(f'direction to pump: {self_correct_direction}, '
                                f'time to pump to self correct: {self.time_to_pump_correction} seconds')

        self.pump.pump(pump_time=self_correct_time,
                       direction=self_correct_direction,
                       wait_time=self.wait_time,
                       )

        if self.application_liquid_level_data_save_file_path is not None:
            self.update_json_file_with_new_liquid_level_correction_data_values()



    def pump_self_correct(self,
                          time_to_pump,
                          direction,
                          ):
        self.pump.pump(pump_time=time_to_pump,
                       direction=direction,
                       wait_time=self.wait_time,
                       )
