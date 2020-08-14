import os
import cv2
import imutils
import numpy as np
import sys
import json
from datetime import datetime
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from heinsight import files
from heinsight.liquidlevel import liquid_level_data_analysis_gui_design
from heinsight.liquidlevel.liquid_level import LiquidLevel


"""
run in terminal from heinsight project directory to convert gui.ui into gui.py file
venv\Scripts\pyuic5.exe heinsight/liquidlevel/liquid_level_data_analysis_gui_design.ui -o heinsight/liquidlevel/liquid_level_data_analysis_gui_design.py

"""
_unit_seconds = 'seconds'
_unit_minutes = 'minutes'
_unit_hours = 'hours'

_time_units = {0: _unit_seconds,
               1: _unit_minutes,
               2: _unit_hours,
               }

_reverse_time_units = dict((v, k) for k, v in _time_units.items())


def plot_time_course_graph(graph_save_location,
                           x_axis_label,
                           y_axis_label,
                           x_value_array=None,
                           y_value_array=None,
                           array_of_x_y_values=None,
                           relative_tolerance_levels=None,
                           graph_title='figure',
                           plot_format_string='k-',
                           reference_image=None,
                           ):
    """
    make a line graph with a dot for each point of the graph. either pass in both the x_value_array and the y_value
    array, OR only pass in an array_of_x_y_values

    :param str, graph_save_location: location to save graph with the file extension to save the graph
    :param str, x_axis_label:
    :param str, y_axis_label:
    :param array, x_value_array: array of datetime objects or floats
    :param array, y_value_array: array of y axis values to plot
    :param array, array_of_x_y_values: 2D array of x and y values to be plotted. an array, where the values in the
        array are arrays themselves, which are [x_value, y_value]. so it looks like [[x1, y1], [x2, y2], ... [xn, yn]].
        The x values are either all datetime objects of floats that represent time elapsed. the y values can be
        either ints and floats
    :param str, graph_title: title of the graph
    :param str, plot_format_string: format string used by matplotlib to format what the graph points and trendline (
        if you want a trendline) will look like. check the matplotlib documentation for the plot function,
        and look for the fmt parameter. https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
    :param reference_image: an image, if passed, that will be plotted next to the graph with the axes as the height
        and width of the image
    :return:
    """
    if x_value_array is not None and y_value_array is None:
        print('must pass both x_value_array and y_value_array')
        return
    if x_value_array is None and y_value_array is not None:
        print('must pass both x_value_array and y_value_array')
        return
    if (x_value_array is not None or y_value_array is not None) and array_of_x_y_values is not None:
        print('cannot pass through both an x_value_array and an array_of_x_y_values OR a y_value_array and an '
              'array_of_x_y_values')
        return

    if array_of_x_y_values is not None:
        x_value_array = [x_value for [x_value, y_value] in array_of_x_y_values]
        y_value_array = [y_value for [x_value, y_value] in array_of_x_y_values]

    # plot and customize graph
    if reference_image.any() != None:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax0 = ax[0]
        ax1 = ax[1]
    else:
        fig, ax0 = plt.subplots(nrows=1, ncols=1)

    ax0.set_title(graph_title)
    ax0.plot(x_value_array, y_value_array, plot_format_string)
    ax0.set_xlabel(x_axis_label)
    ax0.set_ylabel(y_axis_label)
    ax0.set_ylim([0, 1])
    fig.autofmt_xdate()

    ax_0_min_y_value = min(y_value_array)
    ax_0_max_y_value = max(y_value_array)

    if relative_tolerance_levels is not None:
        ''' draw blue horizontal lines on the graph to indicate the relative heights of the tolerance levels'''
        min_x_value = min(x_value_array)
        max_x_value = max(x_value_array)
        x_values = [min_x_value, max_x_value]
        for relative_tolerance_level in relative_tolerance_levels:
            y_values = [relative_tolerance_level, relative_tolerance_level]
            ax0.plot(x_values, y_values, 'b-')  # plot blue horizontal line

    if reference_image.any() != None:
        '''plots an image (bgr image from cv2) on a matplotlub axis (ax), and inverting the y-axis and rotating the image
        by 180 degree and flipping horizontally so a higher y value corresponds to the top of the image'''
        ax1.set_title('Reference image')
        reference_image_rotated_180 = imutils.rotate(image=reference_image,
                                                     angle=180)
        reference_image_rotated_180_mirrored_horizontally = np.fliplr(reference_image_rotated_180)
        plot_image_on_graph(image=reference_image_rotated_180_mirrored_horizontally, ax=ax1)
        ax1.invert_yaxis()
        # ax1.set_ylim([ax_0_min_y_value, ax_0_max_y_value])

    plt.tight_layout()
    # plt.show()

    # save graph to disk
    fig.savefig(graph_save_location)

    plt.close()


def plot_liquid_level_data_over_time_course_graph(data_file_path,
                                                  graph_save_location,
                                                  x_axis_label,
                                                  y_axis_label,
                                                  relative_x_axis=False,
                                                  relative_x_axis_units=None,
                                                  relative_tolerance_levels=None,
                                                  graph_title='figure',
                                                  datetime_format='%Y_%m_%d_%H_%M_%S',
                                                  reference_image=None,
                                                  ):
    """

    :param str, data_file_path: path to json file. Inside the json file there is a key called 'liquid_level_data',
        and the value is a dictionary, where the keys of the dictionary are timestamps with a datetime format,
        and the values are the location of the liquid level relative to the image height at that time point
    :param str, graph_save_location: location to save graph with the file extension to save the graph
    :param x_axis_label:
    :param y_axis_label:
    :param bool, relative_x_axis: if true, then make the time points be relative to each other so the first
        point is at 0 time and all subsequent points are relative to that point. if false then use the actual times
        in the time stamp for the x axis
    :param relative_tolerance_levels: list of the tolerance levels to be drawn as horizontal lines on the graph
    :param relative_x_axis_units: one of the values in _time_units at the top of the script, to specify the units if
        the x axis values are relative instead of absolute values with respect to the first value
    :param str, graph_title:
    :param str, datetime_format: how the datetime object appears as a string in the data file. this is needed to
        know how to convert the datetime as strings back into datetime
    :param reference_image: an image, if passed, that will be plotted next to the graph with the axes as the height
        and width of the image
    :return:
    """

    with open(data_file_path) as file:
        data_file_json = json.load(file)

    liquid_level_data = data_file_json['liquid_level_data']

    try:
        relative_tolerance_levels = data_file_json['tolerance_level_relative']
        # but then the value needs to be flipped because the axes on the graph is inverted with respect to the axes for
        # an image in python
        relative_tolerance_levels = [abs(1-value) for value in relative_tolerance_levels]
    except:
        pass

    # make x value array
    x_value_array = list(liquid_level_data.keys())
    # then make datetime objects from every value in there, based on the date time format
    array_of_datetime_objects = []
    for value in x_value_array:
        datetime_object = datetime.strptime(value, datetime_format)
        array_of_datetime_objects.append(datetime_object)

    if relative_x_axis is True:
        # make an array of timedelta objects where each value is the difference between the actual time relative to
        # the first time point
        array_of_datetime_timedelta = [datetime_value - array_of_datetime_objects[0] for datetime_value in
                                       array_of_datetime_objects]

        # convert the relative timedeltas to floats, where the float number is the number of seconds since the first
        # time point
        array_of_relative_x_in_seconds = [array_of_datetime_timedelta[index].total_seconds() for index
                                          in range(len(array_of_datetime_timedelta))]

        if relative_x_axis_units == _unit_seconds:
            array_of_datetime_objects = array_of_relative_x_in_seconds
        elif relative_x_axis_units == _unit_minutes:
            array_of_relative_x_in_minutes = [array_of_relative_x_in_seconds[index] / 60.0 for index in
                                              range(len(array_of_relative_x_in_seconds))]
            array_of_datetime_objects = array_of_relative_x_in_minutes
        elif relative_x_axis_units == _unit_hours:
            array_of_relative_x_in_hours = [array_of_relative_x_in_seconds[index] / 3600.0 for index in
                                            range(len(array_of_relative_x_in_seconds))]
            array_of_datetime_objects = array_of_relative_x_in_hours

    x_value_array = array_of_datetime_objects

    # array of levels the liquid level was found at
    y_value_array = list(liquid_level_data.values())
    # since it is based on height, and a higher relative height level is actually lower in the image (due to the way
    # rows and image height is counted in an array), need to essentially invert the numbers on the y axis on a scale
    # from 0 to 1
    y_value_array = [abs(1-value) for value in y_value_array]

    plot_time_course_graph(graph_save_location=graph_save_location,
                           x_axis_label=x_axis_label,
                           y_axis_label=y_axis_label,
                           x_value_array=x_value_array,
                           y_value_array=y_value_array,
                           relative_tolerance_levels=relative_tolerance_levels,
                           graph_title=graph_title,
                           plot_format_string='k-',
                           reference_image=reference_image)


def plot_constantly_increasing_y_value_time_course_graph(data_file_path,
                                                         graph_save_location,
                                                         x_axis_label,
                                                         y_axis_label,
                                                         y_axis_increment=1,
                                                         relative_x_axis=False,
                                                         relative_x_axis_units=None,
                                                         reference_image=None,
                                                         graph_title='figure',
                                                         datetime_format='%Y_%m_%d_%H_%M_%S',
                                                         ):
    """
    from a folder of files, where there is a datetime object that has been converted into a string as a part of the
    names of the files in the folder, be able to create a graph where the x axis is the datetime of a point of
    interest. the file names can be filtered by a string so only the files with the included string will be used as a
    part of array of x values. for this there must be a known y axis increment for every x (datetime) value; the y
    axis value cannot be parsed from the file name

    :param str, data_file_path:
    :param str, graph_save_location: location to save graph with the file extension to save the graph
    :param str, x_axis_label:
    :param str, y_axis_label:
    :param float, y_axis_increment: how much the y axis should be incremented for every datetime value. so this might
        be the self correction volume, if the y axis is total volume dispensed, or time to self correct,
        if the y axis is the total time spent self correcting
    :param bool, relative_x_axis: if true, then make the time points be relative to each other so the first
        point is at 0 time and all subsequent points are relative to that point. if false then use the actual times
        in the time stamp for the x axis
    :param relative_x_axis_units: one of the values in _time_units at the top of the script, to specify the units if
        the x axis values are relative instead of absolute values with respect to the first value
    :param str, graph_title:
    :param str, datetime_format: how the datetime object appears as a string in the file names. this is needed to
        know how to convert the datetime as strings back into datetime
    :return:
    """

    with open(data_file_path) as file:
        data_file_json = json.load(file)

    liquid_level_data = data_file_json['liquid_level_data']

    # make x value array
    # make x value array
    x_value_array = list(liquid_level_data.keys())
    # then make datetime objects from every value in there, based on the date time format
    array_of_datetime_objects = []
    for value in x_value_array:
        datetime_object = datetime.strptime(value, datetime_format)
        array_of_datetime_objects.append(datetime_object)

    if relative_x_axis is True:
        # make an array of timedelta objects where each value is the difference between the actual time relative to
        # the first time point
        array_of_datetime_timedelta = [datetime_value - array_of_datetime_objects[0] for datetime_value in
                                       array_of_datetime_objects]

        # convert the relative timedeltas to floats, where the float number is the number of seconds since the first
        # time point
        array_of_relative_x_in_seconds = [array_of_datetime_timedelta[index].total_seconds() for index
                                          in range(len(array_of_datetime_timedelta))]

        if relative_x_axis_units == _unit_seconds:
            array_of_datetime_objects = array_of_relative_x_in_seconds
        elif relative_x_axis_units == _unit_minutes:
            array_of_relative_x_in_minutes = [array_of_relative_x_in_seconds[index] / 60.0 for index in
                                              range(len(array_of_relative_x_in_seconds))]
            array_of_datetime_objects = array_of_relative_x_in_minutes
        elif relative_x_axis_units == _unit_hours:
            array_of_relative_x_in_hours = [array_of_relative_x_in_seconds[index] / 3600.0 for index in
                                            range(len(array_of_relative_x_in_seconds))]
            array_of_datetime_objects = array_of_relative_x_in_hours

    x_value_array = array_of_datetime_objects

    # use list comprehension to make y value array using the y axis increment and length of x value array
    # need to do index+1 because range in python goes from 0 to len-1
    y_value_array = [y_axis_increment*(index+1) for index in range(len(x_value_array))]

    plot_time_course_graph(graph_save_location=graph_save_location,
                           x_axis_label=x_axis_label,
                           y_axis_label=y_axis_label,
                           x_value_array=x_value_array,
                           y_value_array=y_value_array,
                           graph_title=graph_title,
                           reference_image=reference_image,
                           plot_format_string='k-')



def load_date_time_from_folder_of_images(folder_path,
                                         relative_x_axis=False,
                                         relative_x_axis_units=None,
                                         datetime_format='%Y_%m_%d_%H_%M_%S',
                                         file_name_filter=None
                                         ):
    """
    from a folder with files with datetime string in the file name, create an array of the datetime objects that were
    in the filenames. if needed, only files with a certain name can be filtered for use, by specifying what else
    needs to be in the file name for the datetime to be included in the array

    maybe later should have something to be able to specify how the datetime string should be

    :param str, folder_path:
    :param bool, relative_x_axis: if true, then make the time points be relative to each other so the first
        point is at 0 time and all subsequent points are relative to that point. if false then use the actual times
        in the time stamp for the x axis
    :param relative_x_axis_units: one of the values in _time_units at the top of the script, to specify the units if
        the x axis values are relative instead of absolute values with respect to the first value
    :param str, datetime_format: how the datetime object appears as a string in the file names. this is needed to
        know how to convert the datetime as strings back into datetime
    :param str, file_name_filter: if you want to filter the files in the folder to must include a string in the name
        in order to use that file name and to get the datetime string to be returned as part of the return array. the
        file_name_filter must be everything except the datetime string in the filename, as this part will be removed
        from the filename and the rest of the filename must be the datetime string, to be added into the array that
        gets returned
    :return: an array of datetime objects if relative x axis is false, or an array of floats (?) if the relative x axis
        is True
    """
    array_of_datetime_objects = []

    # loop through the files in the folder
    for filename_with_extension in os.listdir(folder_path):
        split_up_filename_with_extension = filename_with_extension.split('.')
        filename_without_file_type = split_up_filename_with_extension[0]
        filename = filename_without_file_type

        # if a file name filter was specified
        if file_name_filter is not None:
            if file_name_filter in filename:
                # if the filename includes what you want to the filename to contain, then strip the filter from the
                # filename, and what remains should just be the datetime string
                filename = filename.strip(file_name_filter)
            else:
                continue

        # now only the filename should only be the datetime string
        # from the datetime string, create a datetime object
        datetime_object = datetime.strptime(filename, datetime_format)
        array_of_datetime_objects.append(datetime_object)

    if relative_x_axis is True:
        # make an array of timedelta objects where each value is the difference between the actual time relative to
        # the first time point
        array_of_datetime_timedelta = [datetime_value - array_of_datetime_objects[0] for datetime_value in
                                       array_of_datetime_objects]

        # convert the relative timedeltas to floats, where the float number is the number of seconds since the first
        # time point
        array_of_relative_x_in_seconds = [array_of_datetime_timedelta[index].total_seconds() for index
                                          in range(len(array_of_datetime_timedelta))]

        if relative_x_axis_units == _unit_seconds:
            array_of_datetime_objects = array_of_relative_x_in_seconds
        elif relative_x_axis_units == _unit_minutes:
            array_of_relative_x_in_minutes = [array_of_relative_x_in_seconds[index] / 60.0 for index in
                                              range(len(array_of_relative_x_in_seconds))]
            array_of_datetime_objects = array_of_relative_x_in_minutes
        elif relative_x_axis_units == _unit_hours:
            array_of_relative_x_in_hours = [array_of_relative_x_in_seconds[index] / 3600.0 for index in
                                            range(len(array_of_relative_x_in_seconds))]
            array_of_datetime_objects = array_of_relative_x_in_hours

    return array_of_datetime_objects


def plot_image_on_graph(image, ax):
    '''plots an image (bgr image from cv2) on a matplotlub axis (ax)'''
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb_image, aspect='auto')

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


class LiquidLevelDataAnalysisGUI(QtWidgets.QMainWindow,
                                 liquid_level_data_analysis_gui_design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(LiquidLevelDataAnalysisGUI, self).__init__(parent)
        self.setupUi(self)

        # the valid file formats
        self.png_format = 'png'
        # dictionary of valid file formats with the index they are at in the combo box in the gui
        self.output_file_formats = {0: self.png_format,
                                    }
        # reversed dictionary of valid file formats
        self.reverse_output_file_formats = dict((v, k) for k, v in self.output_file_formats.items())

        # valid time units
        self.unit_seconds = _unit_seconds
        self.unit_minutes = _unit_minutes
        self.unit_hours = _unit_hours
        # dictionary of time unit types with the index they are at in the combo box in the gui
        self.time_units = _time_units
        # reversed dictionary of experiment types
        self.reverse_time_units = _reverse_time_units

        # valid y axis type
        self.constantly_increasing_y_value_type = 'Constantly increasing Y value'
        self.liquid_level_type = 'Liquid level'
        # dictionary of y axis types with the index they are at in the combo box in the gui
        self.y_axis_types = {0: self.constantly_increasing_y_value_type,
                             1: self.liquid_level_type,
                             }
        # reversed dictionary of y axis types
        self.reverse_y_axis_types = dict((v, k) for k, v in self.y_axis_types.items())

        # default values in the gui
        self.default_output_file_format = self.png_format
        self.default_graph_title = ''
        self.default_x_axis_label = ''
        self.default_relative_x_axis = False
        self.default_relative_x_axis_units = self.unit_seconds
        self.default_y_axis_label = ''
        self.default_y_axis_type = self.constantly_increasing_y_value_type
        self.default_y_axis_increment = 10.0
        self.default_datetime_format = '%Y_%m_%d_%H_%M_%S'
        self.default_reference_image_path = ''

        # parameters to be set from the gui
        self.liquid_level_data = None
        self.output_file_name = None
        self.output_file_location = None
        self.output_file_format = self.default_output_file_format
        self.graph_title = self.default_graph_title
        self.x_axis_label = self.default_x_axis_label
        self.relative_x_axis = self.default_relative_x_axis
        self.relative_x_axis_units = self.default_relative_x_axis_units
        self.y_axis_label = self.default_y_axis_label
        self.y_axis_type = self.default_y_axis_type
        self.y_axis_increment = self.default_y_axis_increment
        self.reference_image_path = self.default_reference_image_path
        self.datetime_format = self.default_datetime_format

        # path to load and save a json file gui set up
        self.path_to_set_up_data_json_file = None

        # useful for the gui attributes
        self.number_of_files_in_folder = 0

    def setupUi(self, MainWindow):
        super().setupUi(self)
        self.graph_title_lineEdit.textChanged.connect(self.set_graph_title)
        self.x_axis_label_lineEdit.textChanged.connect(self.set_x_axis_label)
        self.relative_x_axis_checkBox.toggled.connect(self.set_relative_x_axis)
        self.relative_x_axis_units_comboBox.activated.connect(self.set_relative_x_axis_units)
        self.y_axis_label_lineEdit.textChanged.connect(self.set_y_axis_label)
        self.y_axis_type_comboBox.currentIndexChanged.connect(self.set_y_axis_type)
        self.y_axis_increment_doubleSpinBox.valueChanged.connect(self.set_y_axis_increment)
        self.datetime_format_lineEdit.textChanged.connect(self.set_datetime_format)
        self.liquid_level_data_browse_button.clicked.connect(self.set_liquid_level_data)
        self.reference_image_browse_button.clicked.connect(self.set_reference_image)
        self.output_file_location_browse_button.clicked.connect(self.set_output_file_location)
        self.output_video_file_format_comboBox.activated.connect(self.set_output_file_format)
        self.generate_file_pushButton.clicked.connect(self.generate_file)

        self.actionOpen_2.triggered.connect(self.open_action)
        self.actionSave_2.triggered.connect(self.save_action)
        self.actionSave_As.triggered.connect(self.save_as_action)

    def create_pop_up_message_box(self,
                                  message: str,
                                  window_title: str):
        '''convenience method for creating a pop up message box, with a message and a window title'''
        print(f'{message}')
        message_box = QtWidgets.QMessageBox()
        message_box.setIcon(QtWidgets.QMessageBox.Information)
        message_box.setText(f"{message}")
        message_box.setWindowTitle(f"{window_title}")
        message_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        message_box.exec()

    def open_action(self):
        try:
            set_up_json_tuple = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      "Select set up json file to open",
                                                                      filter="JSON (*.json)"
                                                                      )
            path_to_set_up_json_file = set_up_json_tuple[0]
            self.path_to_set_up_data_json_file = path_to_set_up_json_file

            with open(path_to_set_up_json_file) as file:
                set_up_data_dictionary = json.load(file)

            # set attributes
            self.liquid_level_data = set_up_data_dictionary['liquid_level_data']
            self.output_file_name = set_up_data_dictionary['output_file_name']
            self.output_file_location = set_up_data_dictionary['output_file_location']
            self.output_file_format = set_up_data_dictionary['output_file_format']
            self.graph_title = set_up_data_dictionary['graph_title']
            self.x_axis_label = set_up_data_dictionary['x_axis_label']
            self.relative_x_axis = set_up_data_dictionary['relative_x_axis']
            self.relative_x_axis_units = set_up_data_dictionary['relative_x_axis_units']
            self.y_axis_label = set_up_data_dictionary['y_axis_label']
            self.y_axis_type = set_up_data_dictionary['y_axis_type']
            self.y_axis_increment = set_up_data_dictionary['y_axis_increment']
            self.datetime_format = set_up_data_dictionary['datetime_format']
            self.reference_image_path = set_up_data_dictionary['reference_image_path']

            # put values on gui so user can visually see the values loaded
            self.update_gui_set_up_tab()
        except:
            return

    def update_gui_set_up_tab(self,):
        """update all of what the values for the gui that the user can see looks like based on the gui attributes"""

        self.update_gui_output_format()
        self.update_gui_relative_axis_units()
        self.update_gui_y_axis_type()
        self.graph_title_lineEdit.setText(self.graph_title)
        self.x_axis_label_lineEdit.setText(self.x_axis_label)
        self.relative_x_axis_checkBox.setChecked(self.relative_x_axis)
        self.y_axis_label_lineEdit.setText(self.y_axis_label)
        self.y_axis_increment_doubleSpinBox.setValue(self.y_axis_increment)
        self.datetime_format_lineEdit.setText(self.datetime_format)
        self.liquid_level_data_LineEdit.setText(self.liquid_level_data)
        self.reference_image_lineEdit.setText(self.reference_image_path)
        self.output_file_location_LineEdit.setText(self.output_file_location)

    def update_gui_output_format(self):
        index_of_current_output_file_format = self.reverse_output_file_formats[self.output_file_format]
        self.output_video_file_format_comboBox.setCurrentIndex(index_of_current_output_file_format)

    def update_gui_relative_axis_units(self):
        index_of_current_relative_x_axis_units = self.reverse_time_units[self.relative_x_axis_units]
        self.relative_x_axis_units_comboBox.setCurrentIndex(index_of_current_relative_x_axis_units)

    def update_gui_y_axis_type(self):
        index_of_current_y_axis_type = self.reverse_y_axis_types[self.y_axis_type]
        self.y_axis_type_comboBox.setCurrentIndex(index_of_current_y_axis_type)

    def save_as_action(self):
        """
        prompt user to choose a location and file name to save a JSON file with the current set up from the gui
        :return:
        """
        try:
            set_up_data = self.get_set_up_data_as_dictionary()
            set_up_json_tuple = QtWidgets.QFileDialog.getSaveFileName(self,
                                                                      "Select where to save set up parameters JSON"
                                                                      "file",
                                                                      filter="JSON (*.json)"
                                                                      )
            path_to_set_up_json_file = set_up_json_tuple[0]
            with open(path_to_set_up_json_file, 'w') as file:
                json.dump(set_up_data, file)

            self.path_to_set_up_data_json_file = path_to_set_up_json_file
            print(f'set up json file save path: {path_to_set_up_json_file}')

        except:
            return

    def save_action(self):
        """
        save the current set up parameters to a json file that the current set up is either loaded from,
        or redirect to the action from the "save as" button
        :return:
        """
        if self.path_to_set_up_data_json_file is None:
            self.save_as_action()
        else:
            set_up_data = self.get_set_up_data_as_dictionary()

            path_to_set_up_json_file = self.path_to_set_up_data_json_file
            with open(path_to_set_up_json_file, 'w') as file:
                json.dump(set_up_data, file)

            print(f'set up json file save path: {path_to_set_up_json_file}')

    def get_set_up_data_as_dictionary(self):
        set_up_data = {'liquid_level_data': self.liquid_level_data,
                       'output_file_name': self.output_file_name,
                       'output_file_location': self.output_file_location,
                       'output_file_format': self.output_file_format,
                       'graph_title': self.graph_title,
                       'x_axis_label': self.x_axis_label,
                       'relative_x_axis': self.relative_x_axis,
                       'relative_x_axis_units': self.relative_x_axis_units,
                       'y_axis_label': self.y_axis_label,
                       'y_axis_type': self.y_axis_type,
                       'y_axis_increment': self.y_axis_increment,
                       'datetime_format': self.datetime_format,
                       'reference_image_path': self.reference_image_path,
                       }
        return set_up_data

    def set_graph_title(self,):
        self.graph_title = self.graph_title_lineEdit.text()
        print(f'graph title: {self.graph_title}')

    def set_x_axis_label(self, ):
        self.x_axis_label = self.x_axis_label_lineEdit.text()
        print(f'x axis label: {self.x_axis_label}')

    def set_relative_x_axis(self):
        self.relative_x_axis = self.relative_x_axis_checkBox.isChecked()
        if self.relative_x_axis is True:
            self.relative_x_axis_units_comboBox.setEnabled(True)
        else:
            self.relative_x_axis_units_comboBox.setEnabled(False)
        print(f'relative x axis: {self.relative_x_axis}')

    def set_relative_x_axis_units(self, index):
        """
        :return:
        """
        self.relative_x_axis_units = self.time_units[index]
        print(f'relative x axis units: {self.relative_x_axis_units}')

    def set_y_axis_label(self, ):
        self.y_axis_label = self.y_axis_label_lineEdit.text()
        print(f'y axis label: {self.y_axis_label}')

    def set_y_axis_type(self, index):
        """
        :return:
        """
        self.y_axis_type = self.y_axis_types[index]

        if self.y_axis_type == self.constantly_increasing_y_value_type:
            self.set_constantly_increasing_y_value_type()
        elif self.y_axis_type == self.liquid_level_type:
            self.set_liquid_level_type_y_value_type()

        print(f'y axis type: {self.y_axis_type}')

    def set_constantly_increasing_y_value_type(self,):
        self.y_axis_increment_doubleSpinBox.setEnabled(True)
        # self.include_reference_image_checkBox.setChecked(False)
        # self.include_reference_image_checkBox.setEnabled(False)

    def set_liquid_level_type_y_value_type(self,):
        self.y_axis_increment_doubleSpinBox.setEnabled(False)
        # self.include_reference_image_checkBox.setEnabled(True)

    # def set_include_reference_image(self):
    #     self.include_reference_image = self.include_reference_image_checkBox.isChecked()
    #     print(f'include reference image: {self.include_reference_image}')

    def set_y_axis_increment(self):
        spinbox_value = self.y_axis_increment_doubleSpinBox.value()
        self.y_axis_increment = spinbox_value
        print(f'y axis increment: {spinbox_value}')

    def set_datetime_format(self):
        self.datetime_format = self.datetime_format_lineEdit.text()
        print(f'datetime format: {self.datetime_format}')

    def set_liquid_level_data(self, ):
        try:
            json_file_tuple = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                    "Select liquid level data JSON file",
                                                                    filter="JSON (*.json)")
            path_to_liquid_level_data_json_file = json_file_tuple[0]

            # then display the folder path on the selected folder to save to label
            self.liquid_level_data_LineEdit.setText(f'{path_to_liquid_level_data_json_file}')
            self.liquid_level_data = path_to_liquid_level_data_json_file

        except:  # if user clicks cancel there wouldn't be a directory to save or use
            return

    def set_reference_image(self, ):
        try:
            reference_image_file_tuple = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                    "Select reference image file",
                                                                    # filter="JSON (*.json)",
                                                                    )
            path_to_reference_image_file = reference_image_file_tuple[0]

            # then display the folder path on the selected folder to save to label
            self.reference_image_lineEdit.setText(f'{path_to_reference_image_file}')
            self.reference_image_path = path_to_reference_image_file

        except:  # if user clicks cancel there wouldn't be a directory to save or use
            return

    def set_output_file_location(self):
        """
        prompt user to choose a location and file name to save the graph as a file
        :return:
        """
        try:
            output_file_location_tuple = QtWidgets.QFileDialog.getSaveFileName(self,
                                                                               "Select where to save graph file"
                                                                               "file",
                                                                               )
            path_to_output_file_location = output_file_location_tuple[0]
            split_path_tuple = os.path.split(path_to_output_file_location)
            output_file_name = split_path_tuple[-1]

            self.output_file_location = path_to_output_file_location
            self.output_file_name = output_file_name

            self.output_file_location_LineEdit.setText(f'{path_to_output_file_location}')

            print(f'file save path: {path_to_output_file_location}')
            print(f'file name: {output_file_name}')
        except:
            return

    def set_output_file_format(self, index):
        """
        :return:
        """
        self.output_file_format = self.output_file_formats[index]
        print(f'output file format: {self.output_file_format}')

    # def set_up_completed(self):
    #     liquid_level_data = self.liquid_level_data
    #     output_video_name = self.output_video_name
    #
    #     if liquid_level_data is None or output_video_name is None:
    #         return False
    #     else:
    #         return True

    def generate_file(self):
        # if self.set_up_completed() is False:
        #     return
        # else:

        print('generate graph')

        output_file_path = self.output_file_location + '.' + self.output_file_format

        reference_image = None
        if self.reference_image_path != self.default_reference_image_path:
            reference_image = cv2.imread(self.reference_image_path)

        if self.y_axis_type == self.constantly_increasing_y_value_type:
            plot_constantly_increasing_y_value_time_course_graph(data_file_path=self.liquid_level_data,
                                                                 graph_save_location=output_file_path,
                                                                 x_axis_label=self.x_axis_label,
                                                                 relative_x_axis=self.relative_x_axis,
                                                                 relative_x_axis_units=self.relative_x_axis_units,
                                                                 y_axis_label=self.y_axis_label,
                                                                 y_axis_increment=self.y_axis_increment,
                                                                 graph_title=self.graph_title,
                                                                 datetime_format=self.datetime_format,
                                                                 )
        elif self.y_axis_type == self.liquid_level_type:
            plot_liquid_level_data_over_time_course_graph(data_file_path=self.liquid_level_data,
                                                          graph_save_location=output_file_path,
                                                          x_axis_label=self.x_axis_label,
                                                          y_axis_label=self.y_axis_label,
                                                          relative_x_axis=self.relative_x_axis,
                                                          relative_x_axis_units=self.relative_x_axis_units,
                                                          graph_title=self.graph_title,
                                                          reference_image=reference_image,
                                                          datetime_format=self.datetime_format,
                                                          )

        self.create_pop_up_message_box(message='File generated!',
                                       window_title='Generate progress')


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    form = LiquidLevelDataAnalysisGUI()

    form.show()
    app.exec_()


if __name__ == '__main__':
    main()


