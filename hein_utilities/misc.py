import numpy as np
from itertools import islice, chain


def avg_stdev(lst):
    """
    calculates, the average, standard deviation, and relative standard deviation of a list of values

    :param lst: list of values
    :return: average, standard deviation, relative standard deviation expressed as a percent
    """
    if len(lst) == 0:
        return None, None
    avg = sum(lst) / len(lst)
    if len(lst) >= 3:
        stdev = np.sqrt(
            sum([(i - avg) ** 2 for i in lst]) / (len(lst) - 1)
        )
    else:
        stdev = None
    if avg == 0. or stdev is None:
        pstdev = None
    else:
        pstdev = stdev / avg * 100.
    return avg, stdev, pstdev



def find_nearest(lst, value, wiggle=None):
    """
    Finds the nearest value in a dictionary or list

    :param lst: sorted list or dictionary with keys that are values
    :param value: value to find
    :param wiggle: the wiggle room that the value needs to be within (the bounds are [value-wiggle, value+wiggle])
    :return: the nearest key in the dictionary to the value
    """
    if len(lst) == 0:  # if there are no values
        return None
    if type(lst) == dict:  # if handed a dictionary
        lst = sorted(lst.keys())
    val = lst[
        np.abs(  # array of absolute differences of each list value to the target
            [val - value for val in lst]
        ).argmin()  # index of the minimum value
    ]
    if wiggle is not None and abs(value - val) > wiggle:  # if it's outside the wiggle area
        return None
    return val


def front_pad(lst: list,
              length: int,
              pad_value=0.,
              ) -> list:
    """
    Front pads a list to the specified length with the provided value.

    :param lst: list to pad
    :param length: target length
    :param pad_value: value to pad the list with
    :return: padded list
    :rtype: list
    """
    return [pad_value] * (length - len(lst)) + lst


def chunk_iterable(iterable, chunk_size):
    """
    Iterates an iterable in chunks of specified size.
    https://stackoverflow.com/questions/24527006/split-a-generator-into-chunks-without-pre-walking-it

    :param iterable: iterable to iterate over
    :param chunk_size: list chunk size
    :return: iterable
    """
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, chunk_size - 1))
