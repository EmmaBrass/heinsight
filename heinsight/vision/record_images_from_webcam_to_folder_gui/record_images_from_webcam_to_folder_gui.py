import PyQt5
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import os
import shutil
from heinsight.vision.camera import Camera


"""
create a .py from the gui design.ui file
venv\Scripts\pyuic5.exe heinsight/vision/record_images_from_webcam_to_folder_gui/record_images_from_webcam_to_folder_gui_design.ui -o heinsight/vision/record_images_from_webcam_to_folder_gui/record_images_from_webcam_to_folder_gui_design.py

"""


from heinsight.vision.record_images_from_webcam_to_folder_gui import record_images_from_webcam_to_folder_gui_design


class RecordImagesGUI(QtWidgets.QMainWindow,
                      record_images_from_webcam_to_folder_gui_design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(RecordImagesGUI, self).__init__(parent)
        self.setupUi(self)

        self.save_folder_location = None
        self.save_folder_name = None
        self.camera_port = None
        self.camera = None

    def setupUi(self, MainWindow):
        super().setupUi(self)
        self.save_folder_location_browse_button.clicked.connect(self.set_save_folder_location)
        self.camera_port_spinBox.valueChanged.connect(self.set_camera_port)
        self.start_recording_pushButton.clicked.connect(self.start_recording)

    def set_camera_port(self):
        self.camera_port = self.camera_port_spinBox.value()
        print(f'camera port: {self.camera_port}')

    def set_up_camera(self):
        self.camera = Camera(cam=self.camera_port,
                             save_folder_bool=True,
                             save_folder_location=self.save_folder_location,
                             save_folder_name=self.save_folder_name)

    def set_save_folder_location(self):
        try:
            self.save_folder_location = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                                        "Select image directory"))
            print(f'folder to save to: {self.save_folder_location}')

            split_path = os.path.split(self.save_folder_location)
            self.save_folder_name = split_path[-1]

            shutil.rmtree(path=self.save_folder_location)  # need to remove because the camera class will create a
            # folder, but in the step to select a directory a folder also is created

            self.save_folder_location_LineEdit.setText(f'{self.save_folder_location}')

        except:  # if user clicks cancel there wouldn't be a directory to save or use
            raise

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

    def start_recording(self):
        self.set_up_camera()
        self.camera.record_images(crop_image=False)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    form = RecordImagesGUI()

    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
