import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import os
import sys
from heinsight.vision.folder_of_images_to_video_converter_gui import folder_of_images_to_video_converter_gui_design
from heinsight.vision.image_analysis import folder_of_images_to_video


class FolderOfImagesToVideoConverterGui(QtWidgets.QMainWindow,
                                        folder_of_images_to_video_converter_gui_design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(FolderOfImagesToVideoConverterGui, self).__init__(parent)
        self.setupUi(self)

        # the valid experiment types
        self.mp4_format = 'mp4'
        self.avi_format = 'avi'
        # dictionary of experiment types with the index they are at in the combo box in the gui
        self.output_video_formats = {0: self.mp4_format,
                                     1: self.avi_format,
                                     }
        # reversed dictionary of experiment types
        self.reverse_output_video_formats = dict((v, k) for k, v in self.output_video_formats.items())

        # default values in the gui
        self.default_output_video_format = self.mp4_format
        self.default_output_video_fps = 30
        self.default_display_image_name = False

        # parameters to be set from the gui
        self.folder_of_images_directory = None
        self.output_video_fps = self.default_output_video_fps
        self.output_video_name = None
        self.output_video_file_location = None
        self.output_video_format = self.default_output_video_format
        self.display_image_name = self.default_display_image_name

        # useful for the gui attributes
        self.number_of_images_in_folder = 0

    def setupUi(self, MainWindow):
        super().setupUi(self)
        self.folder_of_images_directory_browse_button.clicked.connect(self.set_images_directory)
        self.output_video_fps_spinBox.valueChanged.connect(self.set_fps)
        self.output_video_file_location_browse_button.clicked.connect(self.set_output_video_file_location)
        self.output_video_file_format_comboBox.activated.connect(self.set_output_video_format)
        self.display_image_name_checkBox.toggled.connect(self.set_display_image_name)
        self.convert_pushButton.clicked.connect(self.convert)

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

    def set_images_directory(self,):
        try:
            folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select image directory"))
            self.folder_of_images_directory = folder
            print(f'folder of images to convert: {folder}')

            # then display the folder path on the selected folder to save to label
            self.folder_of_images_directory_LineEdit.setText(f'{folder}')

            # count number of files and folders (of which there should be 0) in the folder
            self.number_of_images_in_folder = len(os.listdir(folder))
            print(f'number of images in the folder: {self.number_of_images_in_folder}')

        except:  # if user clicks cancel there wouldn't be a directory to save or use
            return

    def set_fps(self):
        self.output_video_fps = self.output_video_fps_spinBox.value()
        print(f'video fps: {self.output_video_fps}')

    def set_output_video_file_location(self):
        """
        prompt user to choose a location and file name to save the video file
        :return:
        """
        try:
            output_video_file_location_tuple = QtWidgets.QFileDialog.getSaveFileName(self,
                                                                      "Select where to save video"
                                                                      "file",
                                                                      )
            path_to_output_video_file_location = output_video_file_location_tuple[0]
            split_path_tuple = os.path.split(path_to_output_video_file_location)
            output_video_name = split_path_tuple[-1]

            self.output_video_file_location = path_to_output_video_file_location
            self.output_video_name = output_video_name

            self.output_video_file_location_LineEdit.setText(f'{path_to_output_video_file_location}')

            print(f'video file save path: {path_to_output_video_file_location}')
            print(f'video file name: {output_video_name}')
        except:
            return

    def set_output_video_format(self, index):
        """
        :return:
        """
        self.output_video_format = self.output_video_formats[index]
        print(f'output video format: {self.output_video_format}')

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

    def set_display_image_name(self):
        self.display_image_name = self.display_image_name_checkBox.isChecked()
        print(f'display image name on frame in video: {self.display_image_name}')

    def set_up_completed(self):
        folder_of_images_directory = self.folder_of_images_directory
        output_video_name = self.output_video_name

        if folder_of_images_directory is None or output_video_name is None:
            return False
        else:
            return True

    def convert(self):
        # if self.set_up_completed() is False:
        #     return
        # else:
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Conversion progress")
        dialog.resize(350, 100)
        horizontal_layout = QtWidgets.QHBoxLayout(dialog)
        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(self.number_of_images_in_folder - 1)
        progress_bar.setValue(0)
        horizontal_layout.addWidget(progress_bar)
        dialog.show()

        output_video_file_path = self.output_video_file_location + '.' + self.output_video_format
        folder_of_images_to_video(folder_path=self.folder_of_images_directory,
                                  output_video_file_location=output_video_file_path,
                                  fps=self.output_video_fps,
                                  display_image_name=self.display_image_name,
                                  progress_bar=progress_bar
                                  )

        self.create_pop_up_message_box(message='Conversion complete!',
                                       window_title='Conversion progress')


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    form = FolderOfImagesToVideoConverterGui()

    form.show()
    app.exec_()


if __name__ == '__main__':
    main()

# main()
