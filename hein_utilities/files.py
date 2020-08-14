import os
import time
import warnings


class Watcher(object):
    def __init__(self,
                 path: str,
                 watchfor: str = '',
                 includesubfolders:bool = True,
                 subdirectory: str = None,
                 watch_type: str = 'file',
                 exclude_subfolders: list = None,
                 ):
        """
        Watches a folder for file changes.

        :param path: The folder path to watch for changes
        :param watchfor: Watch for this item. This can be a full filename, or an extension (denoted by *., e.g. "*.ext")
        :param bool includesubfolders: wehther to search subfolders
        :param str subdirectory: specified subdirectory
        :param str watch_type: The type of item to watch for ('file' or 'folder')
        :param exclude_subfolders: specific sub folders to exclude when looking for files. These subfolders will be
            globally excluded.
        """
        self._path = None
        self._subdir = None
        self.path = path
        self.subdirectory = subdirectory
        self.includesubfolders = includesubfolders
        self.watchfor = watchfor
        self.watch_type = watch_type
        if exclude_subfolders is None:
            exclude_subfolders = []
        self.exclude_subfolders = exclude_subfolders

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.contents)} {self.watchfor})'

    def __str__(self):
        return f'{self.__class__.__name__} with {len(self.contents)} matches of {self.watchfor}'

    def __len__(self):
        return len(self.contents)

    def __iter__(self):
        for file in self.contents:
            yield file

    @property
    def path(self) -> str:
        """path to watch"""
        return self._path

    @path.setter
    def path(self, newpath):
        if not os.path.isdir(newpath):
            raise ValueError(f'The specified path\n{newpath}\ndoes not exist.')
        self._path = newpath

    @property
    def subdirectory(self) -> str:
        """specific subdirectory to watch within the path"""
        return self._subdir

    @subdirectory.setter
    def subdirectory(self, newdir):
        if newdir is None:
            del self.subdirectory
            return
        if not os.path.isdir(
            os.path.join(self.path, newdir)
        ):
            raise ValueError(f'The subdirectory {newdir} does not exist in the path {self.path}.')
        if newdir in self.exclude_subfolders:
            raise ValueError(f'The subdirectory {newdir} is specifically excluded in the exclude_subfolders attribute.')
        self._subdir = newdir

    @subdirectory.deleter
    def subdirectory(self):
        self._subdir = None

    @property
    def contents(self) -> list:
        """Finds all instances of the watchfor item in the path"""
        # todo make this less arduous for large directories
        if self.subdirectory is not None:
            path = os.path.join(self.path, self.subdirectory)
        else:
            path = self._path
        contents = []
        if self.includesubfolders is True:
            for root, dirs, files in os.walk(path):  # walk through specified path
                dirs[:] = [d for d in dirs if d not in self.exclude_subfolders]
                if self.watch_type == 'file':
                    for filename in files:  # check each file
                        if self._condition_match(filename) is True:  # check condition match
                            file_path = os.path.join(root, filename)
                            if os.path.isfile(file_path) is True:  # ensure file
                                contents.append(file_path)
                elif self.watch_type == 'folder':
                    for directory in dirs:
                        if self._condition_match(directory) is True:  # match condition
                            dir_path = os.path.join(root, directory)
                            if os.path.isdir(dir_path) is True:  # ensure directory
                                contents.append(dir_path)
        else:
            for file in os.listdir(path):
                if self._condition_match(file):
                    file_path = os.path.join(path, file)
                    if self.watch_type == 'file' and os.path.isfile(file_path):
                        contents.append(file_path)
                    elif self.watch_type == 'folder' and os.path.isdir(file_path):
                        contents.append(file_path)
        return contents

    def _condition_match(self, name: str):
        """
        Checks whether the file name matches the conditions of the instance.

        :param name: file or folder name.
        :return: bool
        """
        # match extension
        if name.lower().endswith(self.watchfor[1:].lower()):
            return True
        elif name.lower() == self.watchfor.lower():
            return True
        return False

    def check_path_for_files(self):
        """Finds all instances of the watchfor item in the path"""
        warnings.warn('The check_path_for_files method has be depreciated, access .contents directly',
                      DeprecationWarning)
        return self.contents

    def find_subfolder(self):
        """returns the subdirectory path within the full path where the target file is"""
        if self.subdirectory is not None:
            path = os.path.join(self.path, self.subdirectory)
        else:
            path = self.path
        contents = []
        for root, dirs, files in os.walk(path):  # walk through specified path
            for filename in files:  # check each file
                if self._condition_match(filename):  # match conditions
                    # todo catch file/folder?
                    contents.append(root)
        return contents

    def wait_for_presence(self, waittime=1.):
        """waits for a specified match to appear in the watched path"""
        while len(self.contents) == 0:
            time.sleep(waittime)
        return True

    def oldest_instance(self, wait=False, **kwargs):
        """
        Retrieves the first instance of the watched files.

        :param wait: if there are no instances, whether to wait for one to appear
        :return: path to first instance (None if there are no files present)
        """
        if len(self.contents) == 0:  # if there are no files
            if wait is True:  # if waiting is specified
                self.wait_for_presence(**kwargs)
            else:  # if no wait and no files present, return None
                return None
        if len(self.contents) == 1:  # if there is only one file
            return os.path.join(self._path, self.contents[0])
        else:  # if multiple items in list
            return os.path.join(  # return path to oldest (last modified) file in directory
                self._path,
                min(
                    zip(
                        self.contents,  # files in directory
                        [  # last modifiation time for files in directory
                            os.path.getmtime(
                                os.path.join(self._path, filename)
                            ) for filename in self.contents
                        ]
                    ),
                    key=lambda x: x[1]
                )[0]
            )

    def newest_instance(self):
        """
        Retrieves the newest instance of the watched files.

        :return: path to newest instance
        :rtype: str
        """
        if len(self.contents) == 0:  # if there are no files
            # if wait is True:  # if waiting is specified
            #     self.wait_for_presence(**kwargs)
            # else:  # if no wait and no files present, return None
            return None
        if len(self.contents) == 1:  # if there is only one file
            return os.path.join(self._path, self.contents[0])
        else:  # if multiple items in list
            return os.path.join(  # return path to oldest (last modified) file in directory
                self._path,
                max(
                    zip(
                        self.contents,  # files in directory
                        [os.path.getmtime(  # last modifiation time for files in directory
                            os.path.join(self._path, filename)
                        ) for filename in self.contents]
                    ),
                    key=lambda x: x[1]
                )[0]
            )

    def update_path(self, newpath):
        """
        Updates the path to file of the instance.

        :param str newpath: path to new file
        """
        warnings.warn('The update_path method has been depreciated, modify .path directly', DeprecationWarning)
        self.path = newpath


