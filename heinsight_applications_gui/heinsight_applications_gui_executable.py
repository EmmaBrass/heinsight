import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import os
import sys
import json
import cv2
import threading
from datetime import datetime
import slack
from heinsight_applications_gui.old_liquid_level_gui import LiquidLevelGUI
from heinsight.liquidlevel.liquid_level_data_analysis import plot_liquid_level_data_over_time_course_graph, \
    _unit_seconds, _unit_minutes, _unit_hours
from hein_utilities.slack_integration.bots import RTMSlackBot

from heinsight_applications_gui import heinsight_applications_gui_design
from heinsight_applications.heinsight_liquid_level_applications import AutomatedCPCNewEraPeristalticPump, \
    AutomatedSlurryFiltrationPeristaltic, AutomatedContinuousDistillationPeristalticPump, AutomatedCPCDualPumps, \
    LiquidLevelMonitor
from heinsight.liquidlevel.liquid_level import LiquidLevel
from heinsight.liquidlevel.track_tolerance_levels import TrackOneLiquidToleranceLevel, TrackTwoLiquidToleranceLevels
from heinsight.liquidlevel.time_manager import TimeManager
from heinsight.liquidlevel.try_tracker import TryTracker
from heinsight.vision.camera import Camera, ImageAnalysis
from newera.new_era import NewEraPeristalticPumpInterface

# global variables so we have a way to access objects to have slack integration
LIQUID_LEVEL_APPLICATION: LiquidLevelMonitor = None
SLACK_BOT = None
TIME_TO_SELF_CORRECT = None
SELF_CORRECTION_RATE = None

# decide for this script run if should attempt to allow slack integration or not
DO_SLACK_INTEGRATION = True

"""
create a .py from the gui design.ui file
pyuic5.exe heinsight_applications_gui_design.ui -o heinsight_applications_gui_design.py

"""

'''The liquid level gui, which is subclassed from the .py file that was created by the .ui file in designer; that 
part creates the gui itself and how it looks, but in this class, the methods and associations of how the user 
interacts with the gui is specified'''


class LiquidLevelGui(QtWidgets.QMainWindow, heinsight_applications_gui_design.Ui_MainWindow):
    LIQUID_LEVEL_APPLICATION = None

    def __init__(self, parent=None):
        super(LiquidLevelGui, self).__init__(parent)
        self.setupUi(self)

        # the valid experiment types
        self.single_cpc_system = 'single cpc'
        self.dual_cpc_system = 'dual cpc'
        self.filtration_system = 'filtration'
        self.continuous_distillation_system = 'continuous distillation'
        # dictionary of experiment types with the index they are at in the combo box in the gui
        self.experiment_types = {0: self.single_cpc_system,
                                 1: self.dual_cpc_system,
                                 2: self.continuous_distillation_system,
                                 3: self.filtration_system}
        # reversed dictionary of experiment types
        self.reverse_experiment_types = dict((v, k) for k, v in self.experiment_types.items())

        # default values in the gui
        self.gui_default_pump_one_port_label = 'Pump 1 port'
        self.gui_default_pump_two_port_label = 'Pump 2 port'
        self.gui_default_camera_port = 0
        self.gui_default_pump_one_port = None
        self.gui_default_pump_two_port = None
        self.gui_default_initial_pump_rate = 5.0
        self.gui_default_self_correction_pump_rate = 5.0
        self.gui_default_time_to_self_correct = 10
        self.gui_default_try_tracker_max_number_of_tries = 5
        self.gui_default_number_of_monitor_liquid_level_replicate_measurements = 5
        self.gui_default_advance_time = 5
        self.gui_default_experiment_type = self.single_cpc_system
        self.set_experiment_type_single_cpc()
        self.gui_default_time_to_pump_to_calculate_pump_to_pixel_ratio = 10

        # parameters to be set from the gui (includes default values from the gui) - they get updated as the user
        # interacts with the gui
        self.experiment_name = None
        self.folder_to_save_to = None
        self.camera_port = self.gui_default_camera_port
        self.experiment_type = self.single_cpc_system
        self.number_of_liquid_levels_to_find = 1
        self.rows_to_count_for_liquid_level = 4  # number of rows to use to find the liquid level in the liquid level
        # detection algorithm
        self.find_meniscus_minimum = 0.06  # see liquid level object for what this is
        self.time_to_self_correct = self.gui_default_time_to_self_correct
        self.try_tracker_max_number_of_tries = self.gui_default_try_tracker_max_number_of_tries
        self.number_of_monitor_liquid_level_replicate_measurements = \
            self.gui_default_number_of_monitor_liquid_level_replicate_measurements
        self.advance_time = self.gui_default_advance_time
        self.pump_one_port = None
        self.pump_two_port = None
        # updated to have place to input pump port
        self.initial_pump_rate = self.gui_default_initial_pump_rate
        self.self_correction_pump_rate = self.gui_default_self_correction_pump_rate
        self.slack_integration_json_file_path = None

        self.time_to_pump_to_calculate_pump_to_pixel_ratio = \
            self.gui_default_time_to_pump_to_calculate_pump_to_pixel_ratio

        self.slack_integration_arguments = None  # dictionary of arguments from json file for slack integration

        # python objects for controlling the experiment
        self.ne_pump_one = None
        self.ne_pump_two = None
        self.camera = None
        self.liquid_level = None
        self.liquid_level_application = None
        self.slack_bot = None

        # path to load and save a json file for the set up part of the experiment
        self.path_to_set_up_data_json_file = None

    def setupUi(self, MainWindow):
        super().setupUi(self)
        self.directory_to_save_experiment_to_browse_button.clicked.connect(self.set_image_folder_to_save_to)
        self.experiment_type_comboBox.currentIndexChanged.connect(self.set_experiment_type)
        self.ExpName_entry.textChanged.connect(self.set_experiment_name)
        self.CamPort_SpinBox.valueChanged.connect(self.set_camera_port)
        self.pump_one_port_lineEdit.textChanged.connect(self.set_pump_one_port)
        self.pump_two_port_lineEdit.textChanged.connect(self.set_pump_two_port)
        self.pump_rate_SpinBox.valueChanged.connect(self.set_initial_pump_rate)
        self.slack_integration_json_browse_button.clicked.connect(self.set_slack_integration_json_file)
        self.slack_integration_json_lineEdit.textChanged.connect(self.set_slack_integration_json_file_from_line_edit)
        self.time_to_self_correct_spinBox.valueChanged.connect(self.set_time_to_self_correct)
        self.self_correction_pump_rate_spinBox.valueChanged.connect(self.set_self_correction_pump_rate)
        self.try_tracker_max_number_of_tries_label_spinBox.valueChanged.connect(self.set_try_tracker_max_number_of_tries)
        self.number_of_monitor_liquid_level_replicate_measurements_spinBox.valueChanged.connect(self.set_number_of_monitor_liquid_level_replicate_measurements)
        self.advance_time_spinBox.valueChanged.connect(self.set_advance_time)

        self.tabWidget.currentChanged.connect(self.tab_changed)
        self.initialize_experiment_button.clicked.connect(self.initialize_experiment)
        self.webcam_stream_pushButton.clicked.connect(self.open_webcam_stream)
        self.pump_pushButton.clicked.connect(self.pump)
        self.start_experiment_Button.clicked.connect(self.start_experiment)

        self.actionOpen.triggered.connect(self.open_action)
        self.actionSave.triggered.connect(self.save_action)
        self.actionSave_as.triggered.connect(self.save_as_action)

    def set_image_folder_to_save_to(self,):
        try:
            folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select image directory"))
            self.folder_to_save_to = folder
            print(f'folder to save to: {folder}')

            # then display the folder path on the selected folder to save to label
            self.directory_to_save_experiment_to_LineEdit.setText(f'{folder}')
        except:  # if user clicks cancel there wouldn't be a directory to save or use
            return

    def set_experiment_name(self, string):
        self.experiment_name = string
        print(f'experiment name: {self.experiment_name}')

    def set_camera_port(self):
        self.camera_port = self.CamPort_SpinBox.value()
        print(f'camera port: {self.camera_port}')

    def set_pump_one_port(self):
        self.pump_one_port = self.pump_one_port_lineEdit.text()
        print(f'pump one port: {self.pump_one_port}')

    def set_pump_two_port(self):
        self.pump_two_port = self.pump_two_port_lineEdit.text()
        print(f'pump two port: {self.pump_two_port}')

    def set_experiment_type(self, index):
        """
        :return:
        """
        self.experiment_type = self.experiment_types[index]
        print(f'experiment type: {self.experiment_type}')

        if self.experiment_type == self.single_cpc_system:
            self.set_experiment_type_single_cpc()
        elif self.experiment_type == self.dual_cpc_system:
            self.set_experiment_type_dual_cpc()
        elif self.experiment_type == self.continuous_distillation_system:
            self.set_experiment_type_continuous_distillation()
        elif self.experiment_type == self.filtration_system:
            self.set_experiment_type_filtration()

    def set_experiment_type_single_cpc(self):
        self.pump_two_port_lineEdit.setText(self.gui_default_pump_two_port)
        self.pump_two_port_lineEdit.setEnabled(False)
        self.advance_time_spinBox.setValue(self.gui_default_advance_time)
        self.advance_time_spinBox.setEnabled(False)
        self.self_correction_pump_rate_spinBox.setEnabled(True)
        self.pump_one_port_label.setText(f'{self.gui_default_pump_one_port_label}')
        self.pump_two_port_label.setText(f'{self.gui_default_pump_two_port_label}')

    def set_experiment_type_dual_cpc(self):
        self.pump_two_port_lineEdit.setText(self.gui_default_pump_two_port)
        self.pump_two_port_lineEdit.setEnabled(True)
        self.advance_time_spinBox.setEnabled(True)
        self.self_correction_pump_rate_spinBox.setEnabled(True)
        self.pump_one_port_label.setText(f'{self.gui_default_pump_one_port_label} - Dispense')
        self.pump_two_port_label.setText(f'{self.gui_default_pump_two_port_label} - Withdraw')

    def set_experiment_type_continuous_distillation(self):
        self.pump_two_port_lineEdit.setText(self.gui_default_pump_two_port)
        self.pump_two_port_lineEdit.setEnabled(False)
        self.advance_time_spinBox.setEnabled(True)
        self.self_correction_pump_rate_spinBox.setEnabled(False)
        self.pump_one_port_label.setText(f'{self.gui_default_pump_one_port_label}')
        self.pump_two_port_label.setText(f'{self.gui_default_pump_two_port_label}')

    def set_experiment_type_filtration(self):
        self.pump_two_port_lineEdit.setText(self.gui_default_pump_two_port)
        self.pump_two_port_lineEdit.setEnabled(False)
        self.advance_time_spinBox.setEnabled(True)
        self.self_correction_pump_rate_spinBox.setEnabled(False)
        self.pump_one_port_label.setText(f'{self.gui_default_pump_one_port_label}')
        self.pump_two_port_label.setText(f'{self.gui_default_pump_two_port_label}')

    def set_time_to_self_correct(self):
        self.time_to_self_correct = self.time_to_self_correct_spinBox.value()
        print(f'time to self correct: {self.time_to_self_correct}')

    def set_try_tracker_max_number_of_tries(self):
        """
        number of times that the liquid level algorithm can be run consecutively and not find a liquid level
        before the program exits (run stops).
        :return:
        """
        spin_box_value = self.try_tracker_max_number_of_tries_label_spinBox.value()
        self.try_tracker_max_number_of_tries = spin_box_value

        print(f'try tracker max no. of tries: {self.try_tracker_max_number_of_tries}')

    def set_number_of_monitor_liquid_level_replicate_measurements(self):
        number_of_monitor_liquid_level_replicate_measurements = \
            self.number_of_monitor_liquid_level_replicate_measurements_spinBox.value()
        self.number_of_monitor_liquid_level_replicate_measurements = \
            number_of_monitor_liquid_level_replicate_measurements

        # value must be odd
        even_if_zero = self.number_of_monitor_liquid_level_replicate_measurements % 2

        if even_if_zero == 0:
            odd_spin_box_value = number_of_monitor_liquid_level_replicate_measurements + 1
            self.number_of_monitor_liquid_level_replicate_measurements_spinBox.setValue(odd_spin_box_value)
        else:
            print(f'number of liquid level replicate measurements to make: '
                  f'{self.number_of_monitor_liquid_level_replicate_measurements}')

    def set_advance_time(self):
        self.advance_time = self.advance_time_spinBox.value()
        print(f'advance time: {self.advance_time}')

    def set_initial_pump_rate(self):
        self.initial_pump_rate = self.pump_rate_SpinBox.value()
        print(f'initial pump rate: {self.initial_pump_rate}')

    def set_self_correction_pump_rate(self):
        self.self_correction_pump_rate = self.self_correction_pump_rate_spinBox.value()
        print(f'self correction pump rate: {self.self_correction_pump_rate}')

    def set_slack_integration_json_file(self):
        """
        load a JSON file with arguments to create a slack bot; let the user select the file path to the JSON file to
        use

        JSON file needs to be organized as so:
        {
        "BOTNAME": "botname",
        "SLACKCHANNEL": "#slackchannel",
        "BOTTOKEN": "bot-token",
        "SLACKUSERTOKEN": "slackusertoken",
        "SLACKUSERNAME": "username"
        }
        where BOTNAME is the name of the bot, SLACKCHANNEL is the channel for the bot to send messages to,
        BOTTOKEN is the token for the bot, which must be found from Slack for the specific bot, SLACKUSERTOKEN is the
        token for the user that will interact with the bot, which must be found from Slack for the specific user,
        SLACKUSERNAME is the name of the Slack user.
        :return:
        """
        try:
            slack_json_file_tuple = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                          "Select slack integration JSON file",
                                                                          filter="JSON (*.json)")
            path_to_slack_json_file = slack_json_file_tuple[0]
            with open(path_to_slack_json_file) as file:
                self.slack_integration_arguments = json.load(file)
            print(f'file to slack integration json: {path_to_slack_json_file}')

            # then display the folder path on the selected folder to save to label
            self.slack_integration_json_lineEdit.setText(f'{path_to_slack_json_file}')
            self.slack_integration_json_file_path = path_to_slack_json_file
        except:  # if user clicks cancel there wouldn't be a file to use
            return

    def set_slack_integration_json_file_from_line_edit(self):
        """
        load a JSON file with arguments to create a slack bot. file path to the JSON file is determined already,
        the user does not have to manually select a file to use

        JSON file needs to be organized
        as so:
        {
        "BOTNAME": "botname",
        "SLACKCHANNEL": "#slackchannel",
        "BOTTOKEN": "bot-token",
        "SLACKUSERTOKEN": "slackusertoken",
        "SLACKUSERNAME": "username"
        }
        where BOTNAME is the name of the bot, SLACKCHANNEL is the channel for the bot to send messages to,
        BOTTOKEN is the token for the bot, which must be found from Slack for the specific bot, SLACKUSERTOKEN is the
        token for the user that will interact with the bot, which must be found from Slack for the specific user,
        SLACKUSERNAME is the name of the Slack user.
        :return:
        """
        try:
            path_to_slack_json_file = self.slack_integration_json_file_path
            with open(path_to_slack_json_file) as file:
                self.slack_integration_arguments = json.load(file)
            print(f'file to slack integration json: {path_to_slack_json_file}')

            self.slack_integration_json_file_path = path_to_slack_json_file
        except:  # to catch the case where the line change isn't a valid filepath
            return

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

    def create_slack_bot(self):
        self.slack_bot = CPCSlackBot(user_name=self.slack_integration_arguments["SLACKUSERNAME"],
                                     user_member_id=self.slack_integration_arguments["SLACKUSERTOKEN"],
                                     bot_name=self.slack_integration_arguments["BOTNAME"],
                                     token=self.slack_integration_arguments["BOTTOKEN"],
                                     channel_name=self.slack_integration_arguments["SLACKCHANNEL"],
                                     )
        global SLACK_BOT
        SLACK_BOT = self.slack_bot

    def tab_changed(self, tab_index):
        current_tab_index = tab_index
        current_tab = self.tabWidget.tabText(current_tab_index)
        print(f'tab selected: {current_tab}')

        if current_tab == 'Set-Up':
            self.reset_experiment_objects()
        elif current_tab == 'Run':
            self.disable_all_except_initialize_button_in_run_tab()

    def initialize_experiment(self):
        self.reset_experiment_objects()

        self.initialize_experiment_objects()

        if self.experiment_type == self.single_cpc_system:
            self.initialize_single_cpc_experiment()
        if self.experiment_type == self.dual_cpc_system:
            self.initialize_dual_cpc_experiment()
        if self.experiment_type == self.continuous_distillation_system:
            self.initialize_continuous_distillation_experiment()
        if self.experiment_type == self.filtration_system:
            self.initialize_filtration_experiment()

        self.pump_pushButton.setEnabled(True)
        self.webcam_stream_pushButton.setEnabled(True)

    def initialize_dual_cpc_experiment(self):
        self.start_experiment_Button.setEnabled(True)

    def initialize_continuous_distillation_experiment(self):
        self.start_experiment_Button.setEnabled(True)

    def initialize_filtration_experiment(self):
        self.start_experiment_Button.setEnabled(True)

    def reset_experiment_objects(self, ):
        """
        Reset the python objects for controlling the experiment
        :return:
        """
        if self.ne_pump_one is not None:
            self.ne_pump_one.disconnect()
        self.ne_pump_one = None
        if self.ne_pump_two is not None:
            self.ne_pump_two.disconnect()
        self.ne_pump_two = None
        self.camera = None
        self.liquid_level = None
        self.liquid_level_application = None
        self.slack_bot = None
        print(f'experiment objects reset')

    def set_up_completed(self):
        experiment_name = self.experiment_name
        folder_to_save_to = self.folder_to_save_to
        pump_one_port = self.pump_one_port
        pump_two_port = self.pump_two_port
        experiment_type = self.experiment_type

        if experiment_name is None or folder_to_save_to is None or pump_one_port is None:
            return False

        if experiment_type == self.dual_cpc_system and pump_two_port is None:
            return False

        return True

    def initialize_experiment_objects(self):
        """then create all the python objects for the experiment"""
        self.disable_all_except_initialize_button_in_run_tab()

        if self.set_up_completed() is False:
            message = "Experiment was not fully set up. Go back to the set up tab. Ensure experiment name, " \
                      "folder to save to, camera port, and one or both pump ports have been specified"
            window_title = "Set up incomplete"
            self.create_pop_up_message_box(message=message, window_title=window_title)

            return

        self.initialize_camera()
        self.initialize_pump_one()
        if self.experiment_type == self.dual_cpc_system:
            self.initialize_pump_two()

        if self.experiment_type == self.filtration_system or self.experiment_type == self.continuous_distillation_system:
            track_liquid_tolerance_level = TrackOneLiquidToleranceLevel(above_or_below='above')
            liquid_level = LiquidLevel(camera=self.camera,
                                       track_liquid_tolerance_levels=track_liquid_tolerance_level,
                                       number_of_liquid_levels_to_find=self.number_of_liquid_levels_to_find,
                                       rows_to_count=self.rows_to_count_for_liquid_level,
                                       find_meniscus_minimum=self.find_meniscus_minimum,
                                       )
            self.liquid_level = liquid_level

        else:
            track_two_liquid_tolerance_levels = TrackTwoLiquidToleranceLevels()
            liquid_level = LiquidLevel(camera=self.camera,
                                       track_liquid_tolerance_levels=track_two_liquid_tolerance_levels,
                                       number_of_liquid_levels_to_find=self.number_of_liquid_levels_to_find,
                                       rows_to_count=self.rows_to_count_for_liquid_level,
                                       find_meniscus_minimum=self.find_meniscus_minimum,
                                       )
            self.liquid_level = liquid_level

            global LIQUID_LEVEL_APPLICATION
            LIQUID_LEVEL_APPLICATION = self.liquid_level_application

    def initialize_camera(self):
        try:
            self.camera = Camera(cam=self.camera_port)
        except:
            message = 'Could not connect to camera. Make sure selected port is correct and the camera is available'
            window_title = "Set up incomplete"
            self.create_pop_up_message_box(message=message, window_title=window_title)

            self.disable_all_except_initialize_button_in_run_tab()
            return

    def initialize_pump_one(self):
        try:
            self.ne_pump_one = NewEraPeristalticPumpInterface(port=self.pump_one_port)
            self.ne_pump_one.set_rate(rate=self.initial_pump_rate)
        except:
            message = 'Could not connect to pump 1. Make sure selected port is correct and the pump is available'
            window_title = "Set up incomplete"
            self.create_pop_up_message_box(message=message, window_title=window_title)

            self.disable_all_except_initialize_button_in_run_tab()
            return

    def initialize_pump_two(self):
        try:
            self.ne_pump_two = NewEraPeristalticPumpInterface(port=self.pump_two_port)
            self.ne_pump_two.set_rate(rate=self.initial_pump_rate)
        except:
            message = 'Could not connect to pump 2. Make sure selected port is correct and the pump is available'
            window_title = "Set up incomplete"
            self.create_pop_up_message_box(message=message, window_title=window_title)

            self.disable_all_except_initialize_button_in_run_tab()
            return

    def disable_all_except_initialize_button_in_run_tab(self):
        # disable all buttons if set up hasn't been completed yet
        self.start_experiment_Button.setEnabled(False)
        self.pump_pushButton.setEnabled(False)
        self.webcam_stream_pushButton.setEnabled(False)

    def find_dead_volume_time_for_pump(self):
        '''for single pump cpc, need to find the time for the dead volume to be moved between the twp vials,
        so need to run this method, which then sets the advance time to be the time for tge dead volume to be
        transferred'''
        print(f'find dead volume time')
        self.ne_pump_one.find_dead_volume_time()
        self.advance_time = self.ne_pump_one.dead_volume_time_in_seconds
        self.advance_time_spinBox.setValue(self.advance_time)

        self.start_experiment_Button.setEnabled(True)

    def open_webcam_stream(self):
        rows_to_count = self.rows_to_count_for_liquid_level
        track_liquid_tolerance_levels = TrackTwoLiquidToleranceLevels()

        # create the LiquidLevel object
        liquid_level = LiquidLevel(self.camera,
                                   track_liquid_tolerance_levels=track_liquid_tolerance_levels,
                                   rows_to_count=rows_to_count,
                                   number_of_liquid_levels_to_find=1,
                                   find_meniscus_minimum=0.1,
                                   no_error=True,
                                   )

        # create slack bot instance
        slack_bot = None

        my_gui = LiquidLevelGUI(liquid_level=liquid_level,
                                slack_bot=slack_bot,
                                )
        my_gui.root.mainloop()

    def pump(self):
        pump_time = self.pump_time_spinBox.value()
        if self.dispense_radioButton.isChecked():
            pump_direction = 'dispense'
        else:
            pump_direction = 'withdraw'

        pump_rate = self.self_correction_pump_rate

        print(f'{pump_direction} for {pump_time} seconds at rate {pump_rate}')

        self.ne_pump_one.pump(pump_time=pump_time,
                              direction=pump_direction,
                              wait_time=1,
                              rate=pump_rate,
                              )

        self.ne_pump_one.set_rate(rate=self.initial_pump_rate)

        print(f'pumping complete')

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
            self.experiment_name = set_up_data_dictionary['experiment_name']
            self.folder_to_save_to = set_up_data_dictionary['folder_to_save_to']
            self.camera_port = set_up_data_dictionary['camera_port']
            self.experiment_type = set_up_data_dictionary['experiment_type']
            self.number_of_liquid_levels_to_find = set_up_data_dictionary['number_of_liquid_levels_to_find']
            self.rows_to_count_for_liquid_level = set_up_data_dictionary['rows_to_count_for_liquid_level']
            self.find_meniscus_minimum = set_up_data_dictionary['find_meniscus_minimum']
            self.time_to_self_correct = set_up_data_dictionary['time_to_self_correct']
            self.try_tracker_max_number_of_tries = set_up_data_dictionary['try_tracker_max_number_of_tries']
            self.number_of_monitor_liquid_level_replicate_measurements = \
                set_up_data_dictionary['number_of_monitor_liquid_level_replicate_measurements']
            self.advance_time = set_up_data_dictionary['advance_time']
            self.pump_one_port = set_up_data_dictionary['pump_one_port']
            self.pump_two_port = set_up_data_dictionary['pump_two_port']
            self.initial_pump_rate = set_up_data_dictionary['initial_pump_rate']
            self.self_correction_pump_rate = set_up_data_dictionary['self_correction_pump_rate']
            self.slack_integration_json_file_path = set_up_data_dictionary['slack_integration_json_file_path']
            self.time_to_pump_to_calculate_pump_to_pixel_ratio = set_up_data_dictionary['time_to_pump_to_calculate_pump_to_pixel_ratio']

            # put values on gui so user can visually see the values loaded
            self.update_gui_set_up_tab()
        except:
            return

    def update_gui_set_up_tab(self,):
        """update all of what the values for the gui that the user can see looks like based on the gui attributes"""

        self.update_gui_set_experiment_type()
        self.directory_to_save_experiment_to_LineEdit.setText(self.folder_to_save_to)
        self.ExpName_entry.setText(self.experiment_name)
        self.CamPort_SpinBox.setValue(self.camera_port)
        self.pump_one_port_lineEdit.setText(self.pump_one_port)
        self.pump_two_port_lineEdit.setText(self.pump_two_port)
        self.pump_rate_SpinBox.setValue(self.initial_pump_rate)
        self.self_correction_pump_rate_spinBox.setValue(self.self_correction_pump_rate)
        self.slack_integration_json_lineEdit.setText(self.slack_integration_json_file_path)
        self.time_to_self_correct_spinBox.setValue(self.time_to_self_correct)
        self.try_tracker_max_number_of_tries_label_spinBox.setValue(self.try_tracker_max_number_of_tries)
        self.number_of_monitor_liquid_level_replicate_measurements_spinBox.setValue(
            self.number_of_monitor_liquid_level_replicate_measurements)
        self.advance_time_spinBox.setValue(self.advance_time)

    def update_gui_set_experiment_type(self):
        index_of_current_experiment_type = self.reverse_experiment_types[self.experiment_type]
        self.experiment_type_comboBox.setCurrentIndex(index_of_current_experiment_type)

    def save_as_action(self):
        """
        prompt user to choose a location and file name to save a JSON file with the current set up from the set up
        tab of the gui
        :return:
        """
        try:
            set_up_data = self.get_set_up_data_as_dictionary()
            set_up_json_tuple = QtWidgets.QFileDialog.getSaveFileName(self,
                                                                      "Select where to save set up parameters JSON "
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
        set_up_data = {'experiment_name': self.experiment_name,
                       'folder_to_save_to': self.folder_to_save_to,
                       'camera_port': self.camera_port,
                       'experiment_type': self.experiment_type,
                       'number_of_liquid_levels_to_find': self.number_of_liquid_levels_to_find,
                       'rows_to_count_for_liquid_level': self.rows_to_count_for_liquid_level,
                       'find_meniscus_minimum': self.find_meniscus_minimum,
                       'time_to_self_correct': self.time_to_self_correct,
                       'try_tracker_max_number_of_tries': self.try_tracker_max_number_of_tries,
                       'number_of_monitor_liquid_level_replicate_measurements':
                           self.number_of_monitor_liquid_level_replicate_measurements,
                       'advance_time': self.advance_time,
                       'pump_one_port': self.pump_one_port,
                       'pump_two_port': self.pump_two_port,
                       'initial_pump_rate': self.initial_pump_rate,
                       'self_correction_pump_rate': self.self_correction_pump_rate,
                       'slack_integration_json_file_path': self.slack_integration_json_file_path,
                       'time_to_pump_to_calculate_pump_to_pixel_ratio':
                           self.time_to_pump_to_calculate_pump_to_pixel_ratio,
                       }
        return set_up_data

    def start_experiment(self):

        try_tracker = TryTracker(
            max_number_of_tries=self.try_tracker_max_number_of_tries  # maximum number of times to try to search for a
            # liquid level, and if one hasn't been found by the maximum number of tries then the application will
            # error out
        )
        time_manager = TimeManager(
            end_time=999,
            time_interval=0.5,  # fraction of an hour to get updates from slack on how the experiment is going
        )

        experiment_name = self.experiment_name
        folder_to_save_to = self.folder_to_save_to
        folder_to_save_to = os.path.join(folder_to_save_to, experiment_name)

        liquid_level_data_save_folder = folder_to_save_to

        liquid_level = self.liquid_level
        advance_time = self.advance_time
        time_to_self_correct = self.time_to_self_correct
        number_of_monitor_liquid_level_replicate_measurements = self.number_of_monitor_liquid_level_replicate_measurements

        if self.experiment_type == self.filtration_system:
            if self.slack_integration_arguments is not None:
                self.create_slack_bot()
            self.liquid_level_application = AutomatedSlurryFiltrationPeristaltic(liquid_level=liquid_level,
                                                                                 try_tracker=try_tracker,
                                                                                 time_manager=time_manager,
                                                                                 pump=self.ne_pump_one,
                                                                                 number_of_monitor_liquid_level_replicate_measurements=number_of_monitor_liquid_level_replicate_measurements,
                                                                                 advance_time=advance_time,
                                                                                 show=False,
                                                                                 slack_bot=self.slack_bot,
                                                                                 save_folder_bool=True,
                                                                                 save_folder_name=experiment_name,
                                                                                 save_folder_location=folder_to_save_to,
                                                                                 time_to_self_correct=time_to_self_correct,
                                                                                 )

        elif self.experiment_type == self.single_cpc_system:
            if self.slack_integration_arguments is not None:
                self.create_slack_bot()
            self.liquid_level_application = AutomatedCPCNewEraPeristalticPump(liquid_level=liquid_level,
                                                                              try_tracker=try_tracker,
                                                                              time_manager=time_manager,
                                                                              pump=self.ne_pump_one,
                                                                              number_of_monitor_liquid_level_replicate_measurements=number_of_monitor_liquid_level_replicate_measurements,
                                                                              advance_time=advance_time,
                                                                              initial_pump_rate=self.initial_pump_rate,
                                                                              self_correction_pump_rate=self.self_correction_pump_rate,
                                                                              time_to_self_correct=time_to_self_correct,
                                                                              show=False,
                                                                              slack_bot=self.slack_bot,
                                                                              save_folder_bool=True,
                                                                              save_folder_name=experiment_name,
                                                                              save_folder_location=folder_to_save_to,
                                                                              )

        elif self.experiment_type == self.dual_cpc_system:
            if self.slack_integration_arguments is not None:
                self.create_slack_bot()
            self.liquid_level_application = AutomatedCPCDualPumps(liquid_level=liquid_level,
                                                                  try_tracker=try_tracker,
                                                                  time_manager=time_manager,
                                                                  dispense_pump=self.ne_pump_one,
                                                                  withdraw_pump=self.ne_pump_two,
                                                                  initial_pump_rate=self.initial_pump_rate,
                                                                  self_correction_pump_rate=self.self_correction_pump_rate,
                                                                  time_to_self_correct=time_to_self_correct,
                                                                  number_of_monitor_liquid_level_replicate_measurements=number_of_monitor_liquid_level_replicate_measurements,
                                                                  advance_time=advance_time,
                                                                  slack_bot=self.slack_bot,
                                                                  show=False,
                                                                  save_folder_bool=True,
                                                                  save_folder_name=experiment_name,
                                                                  save_folder_location=folder_to_save_to,
                                                                  )

        elif self.experiment_type == self.continuous_distillation_system:
            if self.slack_integration_arguments is not None:
                self.create_slack_bot()
            self.liquid_level_application = AutomatedContinuousDistillationPeristalticPump(liquid_level=liquid_level,
                                                                                           try_tracker=try_tracker,
                                                                                           time_manager=time_manager,
                                                                                           pump=self.ne_pump_one,
                                                                                           number_of_monitor_liquid_level_replicate_measurements=number_of_monitor_liquid_level_replicate_measurements,
                                                                                           time_to_self_correct=time_to_self_correct,
                                                                                           advance_time=advance_time,
                                                                                           slack_bot=self.slack_bot,
                                                                                           show=False,
                                                                                           save_folder_bool=True,
                                                                                           save_folder_name=experiment_name,
                                                                                           save_folder_location=folder_to_save_to,
                                                                                           wait_time=1,
                                                                                           )
        # create the liquid level instance and set up a json file for saving all the time stamped liquid level
        # location data i the folder where everything will be saved to
        liquid_level.liquid_level_data_save_folder = self.liquid_level_application.save_folder.get_path()
        liquid_level.set_up_liquid_level_data_save_file()

        print('start experiment')

        global LIQUID_LEVEL_APPLICATION
        LIQUID_LEVEL_APPLICATION = self.liquid_level_application
        self.close()

        threading.Thread(target=LIQUID_LEVEL_APPLICATION.run).start()

        if self.slack_bot is not None:
            self.slack_bot.post_slack_message(f'Experiment {self.experiment_name}')
            self.slack_bot.introduction_message()
            if DO_SLACK_INTEGRATION is True:
                threading.Thread(target=self.slack_bot.start_rtm_client).start()


#####################


SLACK_COMMANDS = ['pause experiment',  # pause a running experiment
                  'resume experiment',  # resume a paused experiment
                  'end experiment',  # set end experiment attribute to end the experiment
                  'current image',  # return current image from the webcam to slack
                  'dispense',  # allows user to also put an number in the command to pump for that amount of time
                  'withdraw',  # allows user to also put an number in the command to pump for that amount of time
                  'liquid level graph [sec || min || hour])',  # send a graph of liquid level over time for the
                  # application so far
                  'ABORT'  # end an experiment; similar to end experiment, but this is a different way of doing it in
                  #  case end experiment doesnt work
                  ]


class CPCSlackBot(RTMSlackBot):
    def __init__(self,
                 user_name: str = None,
                 user_member_id: str = None,
                 token: str = None,
                 bot_name: str = None,
                 channel_name: str = None,
                 ):
        super().__init__(user_member_id=user_member_id,
                         token=token,
                         bot_name=bot_name,
                         channel_name=channel_name,
                         )
        self.user_name = user_name

    def introduction_message(self):
        message = f'You can control this experiment though Slack!'
        self.post_slack_message(msg=message)
        self.send_command_list()

    def send_command_list(self):
        slack_commands_list = ''

        how_to_send_commands = f'To send commands through Slack, send one of the recognized commands to this channel ' \
                               f'and include my name in the message :smile:'

        self.post_slack_message(msg=how_to_send_commands)

        for command in SLACK_COMMANDS:
            slack_commands_list = slack_commands_list + f'*{command}*, '

        message = f'Command list for CPC bot: \n' \
                  f'{slack_commands_list}'

        self.post_slack_message(msg=message)

    def stop_real_time_messaging(self):
        self.rtm_client.stop()

    """
    Ways to interact with Slack and the slack bot
    """

    @slack.RTMClient.run_on(event='message')
    def say_hello(**payload):
        data = payload['data']
        text = data['text']

    @slack.RTMClient.run_on(event='message')
    def command_list(**payload):
        data = payload['data']
        text = data['text']
        command_in_slack_commands = False
        for known_command in SLACK_COMMANDS:
            if known_command in text and SLACK_BOT.bot_name in text:
                command_in_slack_commands = True
        if command_in_slack_commands is False and SLACK_BOT.bot_name in text:
            SLACK_BOT.post_slack_message(
                msg=f"I didn't understand that command, try using another command",
            )
            SLACK_BOT.send_command_list()

    @slack.RTMClient.run_on(event='message')
    def pause_experiment(**payload):
        data = payload['data']
        text = data['text']
        if 'pause experiment' in text and SLACK_BOT.bot_name in text:
            SLACK_BOT.post_slack_message(
                msg=f"Pausing experiment",
            )
            LIQUID_LEVEL_APPLICATION.pause_the_experiment()

    @slack.RTMClient.run_on(event='message')
    def resume_experiment(**payload):
        data = payload['data']
        text = data['text']
        if 'resume experiment' in text and SLACK_BOT.bot_name in text:
            SLACK_BOT.post_slack_message(
                msg=f"Resuming experiment",
            )
            LIQUID_LEVEL_APPLICATION.resume_experiment()

    @slack.RTMClient.run_on(event='message')
    def end_experiment(**payload):
        data = payload['data']
        text = data['text']
        if 'end experiment' in text and SLACK_BOT.bot_name in text:
            SLACK_BOT.post_slack_message(
                msg=f"Sending command to end experiment",
            )
            LIQUID_LEVEL_APPLICATION.end_experiment()

    @slack.RTMClient.run_on(event='message')
    def say_hello(**payload):
        data = payload['data']
        text = data['text']
        if 'ABORT' in text and SLACK_BOT.bot_name in text:
            SLACK_BOT.post_slack_message(
                msg=f"Throwing an exception to end the run",
            )
            raise KeyboardInterrupt

    @slack.RTMClient.run_on(event='message')
    def show_current_image(**payload):
        data = payload['data']
        text = data['text']
        if 'current image' in text and SLACK_BOT.bot_name in text:
            SLACK_BOT.post_slack_message(
                msg=f"Showing you what the webcam sees right now",
            )
            curr_time = (datetime.now()).strftime('%Y_%m_%d_%H_%M_%S')
            curr_image = LIQUID_LEVEL_APPLICATION.liquid_level.camera.take_picture()
            tolerance_bool, percent_diff = LIQUID_LEVEL_APPLICATION.liquid_level.run(image=curr_image)
            image_with_lines = LIQUID_LEVEL_APPLICATION.liquid_level.all_images_with_lines[-1]

            current_image_path = LIQUID_LEVEL_APPLICATION.slack_images_folder.save_image_to_folder(
                image_name=f'slack_image_query_{curr_time}',
                image=curr_image,
            )
            SLACK_BOT.post_slack_file(
                filepath=current_image_path,
                title='The last image taken',
                comment='The last image taken',
            )

    '''allow user to dispense for a certain number of seconds'''
    @slack.RTMClient.run_on(event='message')
    def dispense_into_vial(**payload):
        data = payload['data']
        text = data['text']
        if 'dispense' in text and SLACK_BOT.bot_name in text:
            # extract an array of numbers from the sent message to know how long to pump for
            numbers_in_slack_message = [float(s) for s in text.split() if s.isdigit()]
            # there should only be one number in the message to know how long to pump for
            if len(numbers_in_slack_message) != 1:
                SLACK_BOT.post_slack_message(
                    msg=f"You must specify one number in the message to indicate how many seconds to dispense for",
                )
                return
            else:
                time_to_pump = numbers_in_slack_message[0]
                LIQUID_LEVEL_APPLICATION.pump_self_correct(time_to_pump=time_to_pump,
                                                           direction='dispense',
                                                           )

    ''' allow user to withdraw for a certain number of seconds'''
    @slack.RTMClient.run_on(event='message')
    def withdraw_from_vial(**payload):
        data = payload['data']
        text = data['text']
        if 'dispense' in text and SLACK_BOT.bot_name in text:
            # extract an array of numbers from the sent message to know how long to pump for
            numbers_in_slack_message = [float(s) for s in text.split() if s.isdigit()]
            # there should only be one number in the message to know how long to pump for
            if len(numbers_in_slack_message) != 1:
                SLACK_BOT.post_slack_message(
                    msg=f"You must specify one number in the message to indicate how many seconds to withdraw for",
                )
                return
            else:
                time_to_pump = numbers_in_slack_message[0]
                LIQUID_LEVEL_APPLICATION.pump_self_correct(time_to_pump=time_to_pump,
                                                           direction='withdraw',
                                                           )

    ''' return a graph of liquid level over time for the running application'''
    @slack.RTMClient.run_on(event='message')
    def get_liquid_level_data_graph(**payload):
        data = payload['data']
        text = data['text']
        if ('liquid level graph' or ('liquid level graph' and ('sec' or 'min' or 'hour'))) in text and \
                SLACK_BOT.bot_name in text:
            time = datetime.now()
            time_formatted = time.strftime(LIQUID_LEVEL_APPLICATION.datetime_format)
            save_folder_path = LIQUID_LEVEL_APPLICATION.save_folder.get_path()
            data_file_path = LIQUID_LEVEL_APPLICATION.application_liquid_level_data_save_file_path
            if data_file_path == None:
                SLACK_BOT.post_slack_message(
                    msg=f"Cannot return graph of liquid level data - data file path is None",
                )
                return

            graph_save_location = os.path.join(save_folder_path, f'liquid_level_graph_{time_formatted}.png')
            x_axis_label = 'Time (minutes)'
            y_axis_label = 'Liquid level height'
            graph_title = 'Liquid level height over time'
            datetime_format = LIQUID_LEVEL_APPLICATION.datetime_format
            tolerance_levels = LIQUID_LEVEL_APPLICATION.liquid_level.track_liquid_tolerance_levels.get_relative_tolerance_height()
            tolerance_levels = [abs(1 - value) for value in tolerance_levels]   # essentially need to inverty because
            #  axes of graph is different from axes of python image get path of one image in the raw images folder
            reference_image_path = os.path.join(LIQUID_LEVEL_APPLICATION.all_drawn_images_folder.get_path(), os.listdir(os.path.join(
                LIQUID_LEVEL_APPLICATION.all_drawn_images_folder.get_path()))[-1])
            reference_image = cv2.imread(reference_image_path)

            relative_x_axis = True
            relative_x_axis_units = None
            if 'sec' in text:
                relative_x_axis_units = _unit_seconds
            elif 'min' in text:
                relative_x_axis_units = _unit_minutes
            elif 'hour' in text:
                relative_x_axis_units = _unit_hours
            else:
                relative_x_axis = False
            plot_liquid_level_data_over_time_course_graph(data_file_path=data_file_path,
                                                          graph_save_location=graph_save_location,
                                                          x_axis_label=x_axis_label,
                                                          y_axis_label=y_axis_label,
                                                          relative_x_axis=relative_x_axis,
                                                          relative_x_axis_units=relative_x_axis_units,
                                                          relative_tolerance_levels=tolerance_levels,
                                                          reference_image=reference_image,
                                                          graph_title=graph_title,
                                                          datetime_format=datetime_format)
            SLACK_BOT.post_slack_file(
                filepath=graph_save_location,
                title='Liquid level over time graph',
                comment='Liquid level over time graph',
            )


def main():
    app = QApplication(sys.argv)
    # set consistent style across platforms
    app.setStyle('Fusion')
    form = LiquidLevelGui()

    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
