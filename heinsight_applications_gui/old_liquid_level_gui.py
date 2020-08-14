"""
Basic gui for liquid level - this is an old gui just for seeing what a webcam sees and what the resulting image that
is used for liquid level detection looks like. to use this old gui, scroll to the bottom and comment the test_gui()
method out, then run the test_gui() method, making sure to use the correct camera port

Most important highlights to use the gui:

This will launch a window where you can see what the webcam sees and see what the effects of the liquid level algorithm
look like. From the gui, you can press the select region of interest button. This will launch a window where you can
click points on either window to draw any polygonal shape to set as a region of interest for the liquid level
algorithm to work within. when you are satisfied with your selection, press 'c', the press the 'enter' key. if at any
point while selecting the region of interest you are not satisfied, to reset all the current selections press the 'r'.

After selection is done, you will be returned to the main window. Then you will see the region of interest you
selected drawn on the image coming from the webcam. You can toggle the outline drawn on the webcam view by clicking
the 'show mask contour on image' button. If you only want to see what is within the selected region of interest,
toggle the 'show area in mask' button.

To see how the liquid level algorithm is working, toggle the 'show liquid level lines' button. You can select the
number of liquid leves that should be reported back using the sliding bar.
"""

import cv2
import tkinter
import threading
from datetime import datetime
from PIL import ImageTk, Image
from tkinter import messagebox
from heinsight.liquidlevel.liquid_level import LiquidLevel
from heinsight.liquidlevel.track_tolerance_levels import TrackTwoLiquidToleranceLevels
from heinsight.vision.camera import Camera


class LiquidLevelGUI:
    def __init__(self,
                 liquid_level: LiquidLevel,
                 slack_bot=None,
                 ):
        self.liquid_level = liquid_level
        self.slack_bot = slack_bot

        # initialize root window
        self.root = tkinter.Tk()
        self.root_background_colour = 'purple'
        self.root.configure(background=self.root_background_colour)
        self.root.title("Basic Liquid Level GUI")
        self.root_height = self.root.winfo_height()  # the root window height
        self.root_width = self.root.winfo_width()  # the root window height

        # create container frames for the gui
        # all these height values must add to 1
        self.top_frame_height = 0.2  # float, fraction of how much of the entire window size will be for the top frame
        self.center_frame_height = 0.5  # float, fraction of how much of the entire window size will be for the top
        # frame
        self.bottom_frame_height = 0.15  # float, fraction of how much of the entire window size will be for the top
        # frame
        self.bottom_frame_2_height = 0.15  # need another container to put things in the bottom until figure out how
        # to use grid and use a better frame
        self.top_frame = tkinter.Frame(self.root, bg=self.root_background_colour, height=self.root_height*self.top_frame_height,
                                       width=self.root.winfo_width())
        self.center_frame = tkinter.Frame(self.root, bg=self.root_background_colour, height=self.root_height * self.center_frame_height, width=self.root.winfo_width())
        self.bottom_frame = tkinter.Frame(self.root, bg=self.root_background_colour, height=self.root_height * self.bottom_frame_height, width=self.root.winfo_width())
        self.bottom_frame_2 = tkinter.Frame(self.root, bg=self.root_background_colour,
                                            height=self.root_height * self.bottom_frame_2_height,
                                            width=self.root.winfo_width())
        # layout main containers
        self.top_frame.grid(row=0, column=0)
        self.center_frame.grid(row=1, column=0)
        self.bottom_frame.grid(row=2, column=0)
        self.bottom_frame_2.grid(row=3, column=0)

        # initialize other attributes here
        self.show_mask_on_contour = True  # bool, used to control if the mask outline is displayed on the live stream
        #  or not
        self.show_area_in_mask = False  # bool, used to control if when the mask contour is showing, if it should
        # just be an outline, or only showing the part of the image that is allowed by the mask
        self.show_liquid_level_lines = False  # bool, used to control if want to display liquid level lines on the
        # live stream images or not

        # create button to toggle the live video stream on and off
        self.toggle_live_video_on_off_button = tkinter.Button(self.top_frame, text="Stop video live stream",
                                           command=self.toggle_live_video_on_off,)
        self.toggle_live_video_on_off_button.pack(side='left', padx=3, pady=3)

        # create button to let user select the region of interest
        self.select_region_of_interest_button = tkinter.Button(self.bottom_frame, text="Select region of interest (ROI)",
                                                               command=self.select_region_of_interest, )
        self.select_region_of_interest_button.pack(side='left', padx=3, pady=3)

        # create button to reset the reference image and reference row
        self.select_reference_row_button = tkinter.Button(self.bottom_frame,
                                                          text="Select reference row",
                                                          command=self.select_reference_row)
        self.select_reference_row_button.pack(side='left', padx=3, pady=3)

        # create button to let user select the tolerance
        self.select_tolerance_button = tkinter.Button(self.bottom_frame, text="Select tolerance",
                                                      command=self.select_tolerance, )
        self.select_tolerance_button.pack(side='left', padx=3, pady=3)

        # create button to let user display the outline of the contour or not, on both display image panels
        self.toggle_show_mask_contour_on_images_button = tkinter.Button(self.top_frame, text="Show mask contour on "
                                                                                             "images",
                                                               command=self.toggle_show_mask_contour_on_images, )
        self.toggle_show_mask_contour_on_images_button.pack(side='left', padx=3, pady=3)

        # create button to show the liquid level lines or not on both display image panels
        self.toggle_show_liquid_level_lines_on_images_button = tkinter.Button(self.top_frame, text="Show liquid level lines",
                                                                              command=self.toggle_show_liquid_level_lines_on_images, )
        self.toggle_show_liquid_level_lines_on_images_button.pack(side='left', padx=3, pady=3)

        # create button to reset the region of interest mask
        self.reset_region_of_interest_mask_button = tkinter.Button(self.bottom_frame,
                                                                   text="Reset region of interest (ROI)",
                                                                   command=self.reset_region_of_interest_mask, )
        self.reset_region_of_interest_mask_button.pack(side='left', padx=3, pady=3)

        # create button to let user display only show whats allowed to be searched in the mask; for this,
        # the mask contour show must be True, and will switch it between showing the mask outline, or only the image
        # within the allowed area in the mask
        self.toggle_show_mask_outline_or_area_button = tkinter.Button(self.top_frame, text='Show area in (ROI) mask',
                                                                      command=self.toggle_show_mask_outline_or_area, )
        self.toggle_show_mask_outline_or_area_button.pack(side='left', padx=3, pady=3)

        self.image_panel_one = tkinter.Label(self.center_frame)  # create image panel with the image
        self.image_panel_one.pack(side="left", padx=3,)
        self.image_panel_two = tkinter.Label(self.center_frame)  # create image panel with the image
        self.image_panel_two.pack(side="left", padx=3,)
        self.camera_cv_last_image = None  # to keep the most current image from the live stream, before converted
        # into tkinter image

        # slider for the number of rows to count to find the meniscus
        self.rows_to_count_label = tkinter.Label(self.bottom_frame_2, text='No. of rows to count')
        self.rows_to_count_label.pack(side='left', padx=5, pady=5)
        self.rows_to_count_scale = tkinter.Scale(self.bottom_frame, from_=1, to=15,
                                                 orient=tkinter.HORIZONTAL)
        self.rows_to_count_scale.set(self.liquid_level.rows_to_count)
        self.rows_to_count_scale.pack(side='left', padx=3, pady=3)

        # slider for the number of menisci to check
        self.menisci_to_check_label = tkinter.Label(self.bottom_frame_2, text='No. of menisci to check')
        self.menisci_to_check_label.pack(side='left', padx=3, pady=3)
        self.menisci_to_check_scale = tkinter.Scale(self.bottom_frame_2, from_=0, to=5,
                                                    orient=tkinter.HORIZONTAL)
        self.menisci_to_check_scale.set(self.liquid_level.number_of_liquid_levels_to_find)
        self.menisci_to_check_scale.pack(side='left', padx=3, pady=3)

        # slider for the minimum fraction of a slice of of a contour image that needs to be white pixels in order for
        #  that slice to be considered to have/be a liquid level
        self.find_meniscus_minimum_label = tkinter.Label(self.bottom_frame_2, text='Fraction of a slice needed to be a '
                                                                                   'liquid level')
        self.find_meniscus_minimum_label.pack(side='left', padx=3, pady=3)
        self.find_meniscus_minimum_scale = tkinter.Scale(self.bottom_frame_2, from_=0, to=100,
                                                         orient=tkinter.HORIZONTAL)
        self.find_meniscus_minimum_scale.set(self.liquid_level.find_meniscus_minimum*100)
        self.find_meniscus_minimum_scale.pack(side='left', padx=3, pady=3)

        # set values and stuff for video streaming to the gui
        self.video_capture = cv2.VideoCapture(self.liquid_level.camera.cam, cv2.CAP_DSHOW)  # create the video capture instance
        self.thread = None  # need a thread to pool video
        self.stop_event = None  # need a way to know when to stop pooling video
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.video_stream, args=())
        self.thread.start()

        # what to happen when you try to exit out of the gui window
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if tkinter.messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.camera_is_on():
                self.toggle_live_video_on_off()
            self.root.destroy()

    def video_stream(self):
        try:  # have a try except because sometimes stop event is changed while in the middle which can cause issues
            # to rise because it cannot continue to diplay the image, usually because it hasn't yet been converted to
            #  an appropriate type for tkinter to display
            while self.camera_is_on():  # keep video stream going until stop event if havent told it to stop
                # alter the attributes of liquid level to change the way find_contour works

                self.get_and_set_liquid_level_parameters_from_gui()

                _, image = self.video_capture.read()
                self.camera_cv_last_image = image
                # change from BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert image from BGR to RGB
                contour_image = self.liquid_level.find_contour(image)  # get the contour image

                # raw image in tkinter image format
                image_one_tkinter = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))  # cant pass numpy array
                # image so need to convert it first using Image.fromarray(), and then use that to make a tkinter
                # image, if the liquid level  instance start() function has already been run
                if self.liquid_level.mask_to_search_inside is not None:  # if an ROI has been chosen
                    liquid_level_image_copy = self.liquid_level.loaded_image.copy()
                    contour_image, _ = self.liquid_level.load_and_find_level(image)  # find the contour image and
                    #  get the row where the meniscus is
                    time = datetime.now()
                    self.liquid_level.add_image_to_memory(img=self.liquid_level.draw_lines(img=liquid_level_image_copy),
                                                          img_name=time.strftime('%d_%m_%Y_%H_%M_%S'),
                                                          array_to_save_to=self.liquid_level.all_images_with_lines,
                                                          )
                    if self.liquid_level.camera.save_folder_bool is True:
                        self.liquid_level.save_drawn_image()

                # contour image in tkinter format
                contour_image_tkinter = ImageTk.PhotoImage(image=Image.fromarray(contour_image))
                if self.show_liquid_level_lines is True:  # if want to show lines on the live stream, then create
                    # image with lines drawn on it
                    image_with_lines = self.liquid_level.draw_lines(img=image)
                else:
                    image_with_lines = image

                image_with_lines_rgb = cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB)  # convert image_with_lines
                # from BGR to RGB image with liquid lines drawn in tkinter format
                image_with_lines_tkinter = ImageTk.PhotoImage(image=Image.fromarray(image_with_lines_rgb))
                if (self.show_mask_on_contour is True) and (self.liquid_level.mask_to_search_inside is not None):
                    image_with_lines_rgb = self.draw_mask_on_image(image_with_lines_rgb)
                    image_with_lines_tkinter = ImageTk.PhotoImage(image=Image.fromarray(image_with_lines_rgb))
                if (self.show_mask_on_contour is True) and (self.liquid_level.mask_to_search_inside is not None):
                    contour_image = self.draw_mask_on_image(contour_image)
                    contour_image_tkinter = ImageTk.PhotoImage(image=Image.fromarray(contour_image))
                # display the images on the screen
                self.display_image(image_one=image_with_lines_tkinter, image_two=contour_image_tkinter)

        except Exception as e:
            print(f'ran in an error {e}')
            tkinter.messagebox.showinfo('error', f'ran in an error {e}')

    def display_image(self, image_one, image_two):
        self.image_panel_one.configure(image=image_one)
        self.image_panel_one.image = image_one
        self.image_panel_two.configure(image=image_two)
        self.image_panel_two.image = image_two

    def toggle_live_video_on_off(self):
        # toggle the live stream video on and off
        if self.camera_is_on():  # camera live stream is currently on, then set stop_event to set to stop the video
            # stream
            self.stop_event.set()
            self.turn_camera_off()  # stop video capture
            # change text on the button to prompt user to start the video stream again
            self.toggle_live_video_on_off_button.configure(text='Start video live stream')
        else:
            self.turn_camera_on()
            self.stop_event.clear()
            # set the thread again and start it
            self.thread = None
            self.thread = threading.Thread(target=self.video_stream, args=())
            self.thread.start()
            # change text on the button to prompt user to start the video stream again
            self.toggle_live_video_on_off_button.configure(text='Stop video live stream')

    def turn_camera_on(self):
        # turn the camera on by creating a video capture instance
        self.video_capture = cv2.VideoCapture(self.liquid_level.camera.cam, cv2.CAP_DSHOW)  # set video capture

    def turn_camera_off(self):
        # turn camera off by releasing the video capture
        self.video_capture.release()

    def select_reference_row(self):
        # set the last image as the reference image. if there is a region of interest that has already been selected,
        #  set the reference liquid level within the region of interest too
        if self.liquid_level.mask_to_search_inside is not None:
            if self.liquid_level.track_liquid_tolerance_levels is not None:
                self.liquid_level.track_liquid_tolerance_levels.select_reference_row(self.camera_cv_last_image)
                # make pop up box for information on how to set a reference row
                tkinter.messagebox.showinfo("Set a reference row", "Set the the height the liquid level should return to "
                                                                       "by clicking on the image at the desired height. "
                                                                       "Press 'c' then 'enter' to accept the selection."
                                                                       "Press 'r' to reset.")
        else:
            # todo is there much use for if the ROI hasnt been selected yet? doesnt seem to be atm...
            return

    def select_tolerance(self):
        if self.liquid_level.track_liquid_tolerance_levels is not None:
            self.liquid_level.track_liquid_tolerance_levels.select_tolerance(self.camera_cv_last_image)

            # make pop up box for information on how to select tolerance levels
            tkinter.messagebox.showinfo("Select tolerance levels", "Set the upper and lower tolerance bounds"
                                        "by clicking on the image at a height above an below the reference. "
                                        "Press 'c' then 'enter' to accept the selection."
                                        "Press 'r' to reset.")

    def camera_is_on(self):
        if not self.stop_event.is_set():  # if the camera is on live stream
            return True
        if self.stop_event.is_set():  # if the camera is not on live stream
            return False

    def select_region_of_interest(self):
        try:
            # function to allow user to select the reference liquid level
            # if video live stream is on then it first must be turned off
            if self.camera_is_on():  # if the camera is on live stream, turn it off
                print('turn off camera')
                self.toggle_live_video_on_off()
            print('reset liquid level')
            self.liquid_level.reset()  # reset the liquid level instance
            # set the liquid level attributes for find_contour()  that are from the gui
            self.get_and_set_liquid_level_parameters_from_gui()
            # make pop up box for information on how to draw the region of interest to find liquid level
            tkinter.messagebox.showinfo("Select region of interest (ROI)", "Define the area for the liquid level algorithm to search."
                                                                           "Use the mouse to click on the corners of a closed polygonal"
                                                                           "shape. Press 'c' then 'enter' to accept the selection."
                                                                           "Press 'r' to reset.")

            # let user select region of interest
            print('load and set reference')
            contour_image = self.liquid_level.load_image_and_select_and_set_parameters(img=self.camera_cv_last_image,
                                                                                       select_region_of_interest=True,
                                                                                       set_reference=False,
                                                                                       select_tolerance=False)
            time = datetime.now()
            if self.liquid_level.track_liquid_tolerance_levels is not None:
                self.liquid_level.all_images_with_lines.append([time.strftime('%d_%m_%Y_%H_%M_%S'),
                                                                self.liquid_level.draw_ref_on_loaded_image()])
            self.liquid_level.all_images_no_lines.append([time.strftime('%d_%m_%Y_%H_%M_%S'), self.liquid_level.loaded_image])
            self.liquid_level.all_images_edge.append([time.strftime('%d_%m_%Y_%H_%M_%S'), contour_image])
            if self.stop_event.is_set():  # if the live stream is off then turn it on
                self.toggle_live_video_on_off()
        except Exception as e:
            # sometimes there is a slight error because where the video live stream was so far, there is some data
            # that was not retrieved, so need to quickly start and stop the video live stream again and then try to
            # select the region of interest again
            print(f'ran in an error {e}')
            tkinter.messagebox.showinfo('error', f'ran in an error {e} - try again')

    def get_and_set_liquid_level_parameters_from_gui(self):
        # self.liquid_level.set_canny_threshold_1(self.canny_threshold_1_scale.get())
        # self.liquid_level.set_canny_threshold_2(self.canny_threshold_2_scale.get())
        self.liquid_level.rows_to_count = self.rows_to_count_scale.get()
        self.liquid_level.menisci_to_check = self.menisci_to_check_scale.get()
        self.liquid_level.find_meniscus_minimum = float(self.find_meniscus_minimum_scale.get() / 100)

    def toggle_show_mask_contour_on_images(self):
        if self.show_mask_on_contour is False:
            self.show_mask_on_contour = True
            self.toggle_show_mask_contour_on_images_button.configure(text="Stop showing mask contour on images")
        else:
            self.show_mask_on_contour = False
            self.toggle_show_mask_contour_on_images_button.configure(text="Show mask contour on images")

    def toggle_show_liquid_level_lines_on_images(self):
        if self.show_liquid_level_lines is False:
            self.show_liquid_level_lines = True
            self.toggle_show_liquid_level_lines_on_images_button.configure(text="Don't show liquid level lines")
        else:
            self.show_liquid_level_lines = False
            self.toggle_show_liquid_level_lines_on_images_button.configure(text="Show liquid level lines")

    def toggle_show_mask_outline_or_area(self):
        if self.show_area_in_mask is True:
            self.show_area_in_mask = False
            self.toggle_show_mask_outline_or_area_button.configure(text="Show area in mask")
        else:
            self.show_area_in_mask = True
            self.toggle_show_mask_outline_or_area_button.configure(text="Show outline of mask")

    def draw_mask_on_image(self, cv_image):
        # draw mask either outline or area on the cv image, then return the image
        if cv_image.shape is 2:  # if image is black and white, not rgb
            line_colour = (255, 255, 255)  # make line colour white
        else:
            # cv image is rgb
            line_colour = (0, 255, 0)  # make line colour green
        if self.show_area_in_mask is True:
            line_colour = (255, 255, 255)  # make line colour white
            cv2.drawContours(self.liquid_level.mask_to_search_inside,
                             [self.liquid_level.list_of_frame_points_frame_points_list], -1,
                             line_colour, -1, cv2.LINE_AA)
            # put mask on image, and black out anything not in the area you want to search in
            cv_image = cv2.bitwise_and(cv_image, cv_image, mask=self.liquid_level.mask_to_search_inside)

        else:
            cv2.drawContours(cv_image,
                             [self.liquid_level.list_of_frame_points_frame_points_list], -1,
                             line_colour, 1, cv2.LINE_AA)
        return cv_image

    def reset_region_of_interest_mask(self):
        if self.camera_is_on():  # if the camera is on live stream, turn it off
            print('turn off camera')
            self.toggle_live_video_on_off()
        self.liquid_level.reset_region_of_interest()
        self.liquid_level.reset_row()
        if not self.camera_is_on():  # if the camera is on live stream, turn it off
            print('turn on camera')
            self.toggle_live_video_on_off()


if __name__ == '__main__':
    def test_gui(cam_number=1):
        CAMERA_NUMBER = cam_number  # zero if the only webcam is the usb webcam, 1 if computer has its own webcam but you want to
        #  use
        # the usb webcam
        CREATE_SAVE_FOLDER = False  # whether or not to create a folder to save all the images in; these are to save all the
        # raw images taken by the camera
        ROWS_TO_COUNT = 5  # number of rows of pixels to look for the meniscus choice between 2-5 have worked well in the past

        # create the Camera object
        camera = Camera(cam=CAMERA_NUMBER,
                        save_folder_bool=CREATE_SAVE_FOLDER,
                        )

        track_liquid_tolerance_levels = TrackTwoLiquidToleranceLevels()

        # create the LiquidLevel object
        liquid_level = LiquidLevel(camera,
                                   track_liquid_tolerance_levels=track_liquid_tolerance_levels,
                                   rows_to_count=ROWS_TO_COUNT,
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


    test_gui()
