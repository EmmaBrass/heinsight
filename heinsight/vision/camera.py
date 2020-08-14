import cv2
import threading
import tkinter
import numpy as np
from datetime import datetime
import pylab as pl
from PIL import ImageTk, Image
from matplotlib.collections import RegularPolyCollection
from heinsight.files import HeinsightFolder
from heinsight.vision.image_analysis import ImageAnalysis


_cv = {  # controlled variables for adjusting camera values
    'frame_width': 3,
    'frame_height': 4,
    'brightness': 10,
    'contrast': 11,
    'saturation': 12,
    'exposure': 15,
}


class Camera:
    """
    Way to control a laptop webcam or usb webcam
    """

    def __init__(self,
                 cam=0,
                 image_analysis: ImageAnalysis = None,
                 save_folder_bool: bool = False,
                 save_folder_name: str = None,
                 save_folder_location: str = None,
                 ):
        """

        :param int, cam: 0, or 1 (or a higher number depending on the number of caperas connected). The camera you
            want to use to take a picture with. 0 if its the only camera, 1 if it is a secondary camera.
        :param bool, save_folder_bool: True if at the end of the application you want to save all the images
            that were taken and used throughout the application run to use
        :param str, save_folder_name: Name of the save folder - generally it would be the experiment name
        :param str, save_folder_location: location to create the folder to save everything
        """

        self.cam = cam  # int, the camera to use
        # create temporary VideoCapture to get the width and height of the images from the camera
        self.connected_camera = None
        self.connect_to_camera()
        self.cam_width = self.connected_camera.get(3)  # float
        self.cam_height = self.connected_camera.get(4)  # float
        cv2.destroyAllWindows()

        self.image_analysis = image_analysis
        if image_analysis is None:
            self.image_analysis = ImageAnalysis()

        self.all_images = []  # list of list, list of [str, np.ndarray], that is the time the picture taken,
        # and the image that was taken images can be accessed via:  date_time_with_line, line_img =
        # camera.all_images[-1]

        self.save_folder_bool = save_folder_bool

        # create initial main folder to save all images to
        if self.save_folder_bool is True:
            self.save_folder = HeinsightFolder(
                folder_name=save_folder_name,
                folder_path=save_folder_location,
            )

        # initial parameters to be used by reset()
        self.initial_arguments = {
            'cam': self.cam,
            'image_analysis': self.image_analysis,
            'save_folder_bool': self.save_folder_bool,
            'save_folder_name': save_folder_name,
            'save_folder_location': save_folder_location,
        }

    def connect_to_camera(self):
        try:
            self.connected_camera = cv2.VideoCapture(self.cam, cv2.CAP_DSHOW)
        except:
            print(f'could not connect to camera')

    def disconnect_from_camera(self):
        self.connected_camera.release()
        self.connected_camera = None

    def reset(self):
        # if this is called, reset all the initial attributes, and delete the folder of images, if there was one that
        #  will be saved, and create a new folder for saved images

        # reset all attributes to initial values
        self.cam = self.initial_arguments['cam']
        self.image_analysis = self.initial_arguments['image_analysis']

        self.all_images = []
        self.save_folder_bool = self.initial_arguments['save_folder_bool']

        if self.save_folder_bool is True:  # recreate a folder of images to store new images that will be taken
            self.save_folder = HeinsightFolder(
                folder_name=self.initial_arguments['save_folder_name'],
                folder_path=self.initial_arguments['save_folder_location'],
            )

    def get_cam(self, camera_to_get):
        """
        Returns a VideoCapture object for the camera you want to get

        :param int, camera_to_get: the camera you want to get
        :return: VideoCamera, video
        """
        video = cv2.VideoCapture(camera_to_get, cv2.CAP_DSHOW)

        return video

    def take_picture(self):
        """
        Take a picture with the camera
        :return: frame is a numpy.ndarray, the image taken by the camera as BGR image
        """
        if self.connected_camera is None:
            self.connect_to_camera()
        frame = None

        time = datetime.now()

        while frame is None:
            # take a picture with a camera
            _, frame = self.connected_camera.read()
            _, frame = self.connected_camera.read()
            # cv2.destroyAllWindows()
            if frame is None:
                self.disconnect_from_camera()
                self.connect_to_camera()

        time_as_str = time.strftime('%Y_%m_%d_%H_%M_%S')
        self.all_images.append([time_as_str, frame])
        # to prevent using too much memory, delete all except the latest image taken after more than n images have be
        #  taken
        n = 5
        if len(self.all_images) >= n:
            self.all_images = self.all_images[-1:]

        if self.save_folder_bool is True:
            self.save_folder.save_image_to_folder(image_name=time_as_str,
                                                  image=frame)

        return frame

    def save_all_images(self):
        """
        save all images in memory, with the time as the name of the image
        :return:
        """
        for i in range(len(self.all_images)):
            date_time_str, image = self.all_images[i]
            self.save_folder.save_image_to_folder(image_name=date_time_str,
                                                  image=image)

    def select_rectangular_area(self, image):
        """
        Allows user to see an image image, and to draw
        a box on the image and use that as the selected frame of where to look for something. After drawing a box
        you can press 'r' to clear it to reselect a different box or press 'c' to choose the box that will be
        the frame.

        :return: float, the fraction relative to the image size for each side that you would have to crop from to get
            the rectangular region of interest.
        """

        crop_left, crop_right, crop_top, crop_bottom = self.image_analysis.select_rectangular_area(image=image)

        return crop_left, crop_right, crop_top, crop_bottom

    def video_run(self):
        self.disconnect_from_camera()
        # stream what the video sees
        video_capture = cv2.VideoCapture(self.cam, cv2.CAP_DSHOW)
        video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
        while True:
            # capture frame-by-frame
            ret, image = video_capture.read()

            if not ret:  # doesn't really matter here, it matters more for reading from an image, but for video
                # doesnt occur
                break

            cv2.imshow('Live video', image)

            # if press the q button exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    # todo should rename this
    def record_images(self,
                      crop_image=False):
        """

        :param bool, crop_image: Whether you want to crop the image that the video camera sees and records the
        pictures with or not (True to crop)

        :return:
        """
        # basically take constant stream of pics, and display and save them original image (user can select a region
        # to crop the image and only view and save that cropped portion)

        self.disconnect_from_camera()
        # first select the frame
        if crop_image is True:
            image = self.take_picture()
            crop_left, crop_right, crop_top, crop_bottom = self.select_rectangular_area(image=image)

            video_capture = self.connected_camera

            while True:
                # capture frame-by-frame
                ret, image = video_capture.read()
                time = datetime.now()
                time_as_str = time.strftime('%Y_%m_%d_%H_%M_%S')

                if not ret:  # doesn't really matter here, it matters more for reading from an image, but for video
                    # doesnt occur
                    break

                cropped_image = self.image_analysis.crop_horizontal(image=image,
                                                                    crop_left=crop_left,
                                                                    crop_right=crop_right)
                cropped_image = self.image_analysis.crop_vertical(image=cropped_image,
                                                                  crop_top=crop_top,
                                                                  crop_bottom=crop_bottom)

                cv2.imshow('Live cropped video', cropped_image)

                if self.save_folder_bool is True:
                    self.save_folder.save_image_to_folder(image_name=time_as_str,
                                                          image=cropped_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video_capture.release()
                    break
        else:
            video_capture = cv2.VideoCapture(self.cam, cv2.CAP_DSHOW)

            while True:
                ret, image = video_capture.read()
                time = datetime.now()
                time_as_str = time.strftime('%Y_%m_%d_%H_%M_%S')

                if not ret:  # doesn't really matter here, it matters more for reading from an image, but for video
                    # doesnt occur
                    break

                cv2.imshow('Live  video', image)

                if self.save_folder_bool is True:
                    self.save_folder.save_image_to_folder(image_name=time_as_str,
                                                          image=image)

                # if press the q button exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video_capture.release()
                    break

    def find_rectangular_area_rows_and_columns(self, image, show=False, crop_left=None, crop_right=None, crop_top=None, crop_bottom=None):
        left, right, top, bottom = self.image_analysis.find_rectangular_area_rows_and_columns(image=image,
                                                                                              show=show,
                                                                                              crop_left=crop_left,
                                                                                              crop_right=crop_right,
                                                                                              crop_top=crop_top,
                                                                                              crop_bottom=crop_bottom,
                                                                                              )

        return left, right, top, bottom


###################################################################################

def read_rgb(reads=5, section=25, webcam=0):
    """
    Reads the center section of a set of images captured by the webcam

    :param reads: number of reads to average
    :param section: the number of pixels of the center section to read (by default this will be 25x25 section)
    :param webcam: the webcam address number to access
    :return: averaged r,g,b value of the center section
    """
    frames = capture_image(
        reads,
        webcam=webcam,
    )
    # height, width
    h = len(frames[0])
    w = len(frames[0][0])

    # snip the desired sections
    snippets = np.asarray([[
        [frame[y][x] for x in range(int(w/2 - section / 2), int(w/2 + section / 2))]
        for y in range(int(h/2 - section / 2), int(h/2 + section / 2))] for frame in frames]
    )
    # flatten snippets and reverse bgr values
    snippets = [[rgb[::-1] for sublist in snippet for rgb in sublist] for snippet in snippets]
    col_lists = [
        [rgb[i] for lst in snippets for rgb in lst] for i in [0, 1, 2]
    ]
    return [int(sum(lst) / len(lst)) for lst in col_lists], frames[0]


def snippet(frame, section=25):
    """
    Extracts a centered snippet from a frame

    :param frame: frame of values
    :return: snippet of the frame
    """
    h = len(frame)
    w = len(frame[0])

    # snip the desired sections
    snippet = np.asarray([
        [frame[y][x] for x in range(int(w / 2 - section / 2), int(w / 2 + section / 2))]
        for y in range(int(h / 2 - section / 2), int(h / 2 + section / 2))]
    )
    return snippet


def average_rgb(frame, incoming='bgr'):
    """
    Determines the average RGB value of a frame

    :param frame: frame of values
    :return: average rgb value
    """
    flat = [rgb for sublist in frame for rgb in sublist]
    col_lists = [
        [rgb[i] for rgb in flat] for i in [0, 1, 2]
    ]
    if incoming.lower() == 'bgr':
        return [int(sum(lst) / len(lst)) for lst in col_lists][::-1]
    if incoming.lower() == 'rgb':
        return [int(sum(lst) / len(lst)) for lst in col_lists]


def plot_colour(colour):
    fig, ax = pl.subplots(
        1,
        # figsize=(9.6, 5.4),
    )
    col = RegularPolyCollection(
        sizes=(1000000,),
        numsides=4,
        facecolors=pl.matplotlib.colors.to_rgb([val / 255 for val in colour]),
        offsets=[0.5, 0.5],
        transOffset=ax.transData
    )
    ax.axis('off')
    ax.add_collection(col)

    pl.show()


def read_and_plot():
    plot_colour(read_rgb())


def capture_image(n=1, webcam=0, show=False, **settings):
    """
    Captures webcam image and overlays the target frame over the image (if specified)

    :param n: number of frames to capture
    :param webcam: webcam index
    :param show: show the captured frame
    :return:
    """
    video = cv2.VideoCapture(webcam)
    # todo fix this after Globe
    if len(settings) == 0:
        video.set(15, -5)
    for setting in settings:
        video.set(_cv[setting], settings[setting])
    video.grab()
    frames = [video.retrieve()[1] for i in range(n)]
    # frames = [video.read()[1] for i in range(n)]
    video.release()
    if show is True:
        show_image(np.hstack(frames))
    if n == 1:
        return frames[0]
    return frames


def box_frame(frame, section=40, linewidth=1):
    """
    Takes a frame and puts a highlight box in the center

    :param frame: frame to manipulate
    :param section: pixel width and height of the section
    :param linewidth: width of the line (pixels)
    :return:
    """
    h = len(frame)
    w = len(frame[0])
    midh = h / 2
    midw = w / 2
    l = int(midw - section / 2 - 1)
    r = int(midw + section / 2 + 1)
    t = int(midh + section / 2 + 1)
    b = int(midh - section / 2 - 1)
    for x in range(l, r):
        for i in range(linewidth):
            frame[t+i][x] = np.array([0, 255, 0], dtype='uint8')
            frame[b-i][x] = np.array([0, 255, 0], dtype='uint8')
    for y in range(b-i, t+i):
        for i in range(linewidth):
            frame[y][l-i] = np.array([0, 255, 0], dtype='uint8')
            frame[y][r+i] = np.array([0, 255, 0], dtype='uint8')
    return np.asarray(frame, dtype='uint8')


def show_image(image):
    cv2.imshow('', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def get_whitebalance(frame=None):
    """
    Record the white balance values for the current environment in the LAB colour space
    Based on https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption

    :param frame: optional frame to hand the function
    :return: scalars for applying white balance
    """
    if frame is None:
        frame = capture_image()
    final = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    avg_a = np.average(final[:, :, 1])
    avg_b = np.average(final[:, :, 2])
    return avg_a, avg_b


def apply_whitebalance(uncorr, a_avg, b_avg, show=False):
    """
    Applies a previous set of white balance parameters to a frame in an efficient manner (using numpy vectorization)

    :param uncorr: uncorrected frame
    :param a_avg: average a value
    :param b_avg: average b value
    :param show: show a comparison of before and after
    :return: corrected frame
    """
    # convert to lab colour space
    labspace = cv2.cvtColor(uncorr, cv2.COLOR_BGR2LAB)

    # extract luminosity
    lorig = labspace[:, :, 0]

    # create base luminosity scalar array for a and b
    l = np.copy(lorig) * (100 / 255.)
    # create scalar arrays
    ascale = np.copy(l) * (a_avg - 128) * 0.011
    bscale = np.copy(l) * (b_avg - 128) * 0.011

    # retrieve and manipulate a and b arrays
    a = labspace[:, :, 1] - ascale
    b = labspace[:, :, 2] - bscale

    # cast to appropriate data types for back conversion
    a = a.astype('uint8')
    b = b.astype('uint8')

    # combine into LAB arrays and reshape to the original form
    modified = np.dstack((lorig, a, b))
    shaped = np.reshape(modified, uncorr.shape)

    # convert back to BGR array and return
    out = cv2.cvtColor(shaped, cv2.COLOR_LAB2BGR)
    if show is True:
        show_image(np.hstack((uncorr, out)))
    return out


import os


def test():
    # example of setting up
    root_path = os.getcwd()
    test_folder_name = 'test_camera_class'
    test_folder_path = os.path.join(root_path, test_folder_name)

    camera = Camera(cam=0,
                    save_folder_bool=False,
                    save_folder_location=test_folder_path,
                    save_folder_name=test_folder_name,
                    )

    # # stream from the camera
    # camera.video_run()

    # # test selecting polygonal area from image analysis class
    # ia = ImageAnalysis()
    # image = camera.take_picture()
    # ia.select_polygonal_area(image=image)
    # edges = ia.find_maximum_edges_of_mask()
    # print(f'edges of polygonal area: {edges}')
    # # drawn_image = ia.draw_selected_polygonal_area_on_image(image=image)
    # # ia.display_image(image_name='drawn', image=drawn_image)
    # # drawn_image = ia.draw_mask_on_image(image=image)
    # # ia.display_image(image_name='drawn', image=drawn_image)

    # # testing selecting multiple polygonal areas
    # ia = ImageAnalysis()
    # image = camera.take_picture()
    # ia.select_multiple_polygonal_areas(image=image,
    #                                    number_of_areas_to_select=3)
    # # drawn_image = ia.draw_multiple_selected_polygonal_areas_on_image(image=image)
    # # ia.display_image(image_name='drawn', image=drawn_image)
    # # drawn_image = ia.draw_mask_on_image(image=image)
    # # ia.display_image(image_name='drawn', image=drawn_image)

    # # test of getting and saving images from live camera feed; need to set save_folder_bool to True
    # camera.record_images(crop_image=True)

    # # test cameragui
    # my_gui = CameraGUI(camera=camera)
    # my_gui.root.mainloop()


# test()


