import os
import cv2
import numpy as np

from hein_utilities.files_gui import Folder


def load_images_from_folder(folder: str,
                            file_name_filter: str = None):
    """
    Takes a folder and loads images from the folder into an array, and also creates another array with the names of
    the images. if you want to only retrieve some files then use a filter, to get files that only contain the filter
    in the file name

    :param folder: the folder in the directory or the path to a folder from the directory to load images from
     :param str, file_name_filter: if you want to filter the files in the folder to must include a string in the name
        in, put that here
    :return: images is an array of the images for a folder, image_names is a list of the names of the images that
        were loaded
    """
    images = []
    image_names = []
    for filename_with_extension in os.listdir(folder):
        split_up_filename_with_extension = filename_with_extension.split('.')
        filename_without_file_type = split_up_filename_with_extension[0]
        filename = filename_without_file_type

        # if a file name filter was specified
        if file_name_filter is not None:
            if file_name_filter in filename:
                # if the filename includes what you want to the filename to contain then you can load the file
                pass
            else:
                continue

        img = cv2.imread(os.path.join(folder, filename_with_extension))
        img = np.asarray(img)

        if img is not None:
            images.append(img)
            image_names.append(f'{filename}')

    return images, image_names


class HeinsightFolder(Folder):
    """
        Class to create and store a folder hierarchy and to easily create folders and access paths to save files and
        folders in. Has convenience methods for image related files

        Example using the Folder class - run each line one at a time to see changes to disk:

        root_path = os.getcwd()

        test_folder_name = 'test'
        test_folder_path = os.path.join(root_path, test_folder_name)

        test_folder = Folder(folder_name=test_folder_name, folder_path=test_folder_path)

        sub_folder_one = test_folder.make_and_add_sub_folder(sub_folder_name='sub_folder_one')
        sub_folder_two = test_folder.make_and_add_sub_folder(sub_folder_name='sub_folder_two')
        sub_sub_folder_one = sub_folder_one.make_and_add_sub_folder(sub_folder_name='sub_sub_folder_one')

        sub_folder_one.delete_from_disk()
        sub_folder_two.delete_from_disk()

        test_folder_two = Folder(folder_name=test_folder_name, folder_path=test_folder_path)

        test_folder.delete_from_disk()
        test_folder_two.delete_from_disk()

        """

    def __init__(self,
                 folder_name: str,
                 folder_path: str,
                 ):
        """
        A folder can have files and folders in it, but for now just care about it containing folders. The name of the
        folder should be the same as the last part of the folder path.

        :param str, folder_path: path to save the folder on disk
        :param str, folder_name: Should be the same as the last part of the folder path
        """
        super().__init__(
            folder_name=folder_name,
            folder_path=folder_path
        )

    def save_image_to_folder(self,
                             image_name: str,
                             image,
                             file_format: str = 'jpg',
                             ):
        """
        Save an image to disk using cv2

        :param str, image_name: name of the file to save the image as in the folder
        :param image: image to save
        :param str, file_format: file format to save the image as. Is jpg by default.

        :return: str, path_to_save_image: the path the image was saved to
        """
        image_name = f'{image_name}.{file_format}'
        path_to_save_image = os.path.join(self.path, image_name)
        cv2.imwrite(path_to_save_image, image)

        return path_to_save_image

    def make_and_add_sub_folder(self,
                                sub_folder_name: str,
                                ):
        """
        Create a sub-folder with a given name under the main folder; the path of the sub-folder is the name of the
        sub-folder concatenated on to the end of the path of the main folder

        :return: HeinsightFolder, sub_folder: the sub-folder that was created
        """

        if sub_folder_name in self.children:
            raise Exception(f'Main folder already has a sub-folder called {sub_folder_name}')
        else:
            parent_folder_path = self.path
            sub_folder_path = os.path.join(
                parent_folder_path,
                sub_folder_name
            )
            sub_folder = HeinsightFolder(
                folder_name=sub_folder_name,
                folder_path=sub_folder_path,
            )
            sub_folder.set_parent(component=self)

            self.add_child_component(component=sub_folder)

        return sub_folder
