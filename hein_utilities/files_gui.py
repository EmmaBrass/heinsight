import os
import shutil
import tkinter


class Component:
    def __init__(self,
                 name: str,
                 path: str,
                 ):
        """
        Class that Folder inherits from, created in case there are other components to be tracked in folders in the
        future, such as files.

        :param str, name: name of the component
        :param str, path: path to save the component in. The last part of the path must be the name of the component
        """
        self.name = name
        self.path = path
        self.parent = None  # parent Component this component belongs to; can be set to None if it is the first
        # component being made

        try:
            self.save_to_disk()
        except FileExistsError as error:
            root = tkinter.Tk()
            yes_or_no_result = tkinter.messagebox.askyesno(
                f'Save to disk',
                f'File or folder at {path}. '
                f'Would you like to create one with "_copy_#" appended to the end of the file/folder name?')
            if yes_or_no_result is True:
                # rename the file something else
                for i in range(20):  # randomly put 20 here, just assuming there won't already be more than this
                    # number of folders that will be made with the same name
                    try:
                        old_name = self.get_name()
                        old_path = self.get_path()
                        new_name = old_name + f'_copy_{i}'
                        new_path = old_path + f'_copy_{i}'
                        self.set_name(name=new_name)
                        self.set_path(path=new_path)
                        self.save_to_disk()
                        root.destroy()
                        break
                    except FileExistsError as error:
                        self.set_name(name=old_name)
                        self.set_path(path=old_path)
            else:
                raise error

    def get_name(self):
        return self.name

    def get_path(self):
        return self.path

    def get_parent(self):
        return self.parent

    def set_name(self,
                 name: str,
                 ):
        self.name = name

    def set_path(self,
                 path: str,
                 ):
        self.path = path

    def set_parent(self,
                   component,
                   ):
        """
        Set the parent component for this component

        :param Component, component:
        :return:
        """
        self.parent = component

    def save_to_disk(self):
        """
        Actually create the component on the computer at its path with its name
        :return:
        """

        raise NotImplementedError

    def delete_from_disk(self):
        """
        Actually delete the component from the computer
        :return:
        """
        shutil.rmtree(path=self.path)


class Folder(Component):
    """
    Class to create and store a folder hierarchy and to easily create folders and access paths to save files and
    folders in

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
            name=folder_name,
            path=folder_path
        )
        self.children = set()

    def get_name(self):
        return super().get_name()

    def get_path(self):
        return super().get_path()

    def get_parent(self):
        return super().get_parent()

    def save_to_disk(self):
        os.makedirs(
            name=self.path,
        )

    def delete_from_disk(self):
        super().delete_from_disk()

    def set_parent(self,
                   component: Component,
                   ):
        super().set_parent(component=component)

    def get_parent(self):
        super().get_parent()

    def get_children(self):
        return self.children

    def add_child_component(self,
                            component: Component,
                            ):
        """
        Add a component to the set of children and set the parent of the child component to be this folder

        :param Component, component:
        :return:
        """
        self.children.add(component)
        component.set_parent(component=self)

    def remove_and_delete_component(self,
                                    component: Component,
                                    ):
        """
        Remove a child component from the children set and delete it from disk

        :param Component, component:
        :return:
        """
        self.children.remove(component)
        component.delete_from_disk()

    def make_and_add_sub_folder(self,
                                sub_folder_name: str,
                                ):
        """
        Create a sub-folder with a given name under the main folder; the path of the sub-folder is the name of the
        sub-folder concatenated on to the end of the path of the main folder

        :return: Folder, sub_folder: the sub-folder that was created
        """

        if sub_folder_name in self.children:
            raise Exception(f'Main folder already has a sub-folder called {sub_folder_name}')
        else:
            parent_folder_path = self.path
            sub_folder_path = os.path.join(
                parent_folder_path,
                sub_folder_name
            )
            sub_folder = Folder(
                folder_name=sub_folder_name,
                folder_path=sub_folder_path,
            )
            sub_folder.set_parent(component=self)

            self.add_child_component(component=sub_folder)

        return sub_folder
