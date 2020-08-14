"""
Class to track how long an experiment has been running, and to know when to stop an experiment. Also be able to set
intervals of time so that actions can be taken when an interval has just been passed.
"""

from datetime import datetime


class TimeManager:
    def __init__(self,
                 end_time: int,
                 time_interval: float,
                 ):
        """
        :param int, end_time: how many hours to allow something to run for
        :param float, time_interval: fraction of an hour you want to set as a time interval
        """
        self.start_time = datetime.now()  # time an instance was made
        self.end_time = end_time
        self.time_interval = time_interval

        self.intervals_elapsed = []  # keep track of the number of time intervals that have passed in an array

        self.initial_end_time = end_time
        self.initial_time_interval = time_interval

    def reset(self):
        """
        Reset the time manager so that the start time is the current time, and set the end time and time interval to
        what the time manager instance was initially instantiated with

        :return:
        """
        self.start_time = datetime.now()
        self.end_time = self.initial_end_time
        self.time_interval = self.initial_time_interval

    def set_start_time(self,
                       time: datetime,
                       ):
        """
        Set a start time for the manager
        :param time:
        :return:
        """
        self.start_time = time

    def set_end_time(self,
                     end_time: int,
                     ):
        """
        Set a how many hours until the end has been reached
        :param int, end_time:
        :return:
        """
        self.end_time = end_time

    def set_time_interval(self,
                          time_interval: float,
                          ):
        """
        Set the time interval
        :param float, time_interval: fraction of an hour you want to set as a time interval
        :return:
        """
        self.time_interval = time_interval

    def time_since_started(self,
                           time: datetime = None,
                           ):
        """
        return the number of hours that have elapsed since the start time from a given time, or the current time if
        no time was provided

        :return: float
        """
        if time is None:
            time = datetime.now()
        hours_elapsed = self.hours_elapsed(time=time)
        # normal division (/) instead of (//) to know the actual number of hours that have elapsed, not just an interval
        intervals_elapsed = hours_elapsed / self.time_interval
        return self.time_interval * intervals_elapsed

    def is_after_end_time(self,
                          time: datetime,
                          ):
        """
        Check if the time is given is after the start time + the number of hours until the end time

        :param datetime, time:
        :return: bool, return true if the number of hours until the end time has been exceeded since the start time
        """
        hours_elapsed = self.hours_elapsed(time=time)
        after_end_time = hours_elapsed >= self.end_time
        return after_end_time

    def hours_elapsed(self,
                      time: datetime,
                      ):
        """
        The number of hours that have elapsed since the given time and the start time
        :param datetime, time:
        :return: float, hours_elapse: the number of hours that have elapsed since starting the time manager
        """
        time_diff = time - self.start_time
        hours_elapsed = time_diff.total_seconds() / 3600
        return hours_elapsed

    def calculate_interval(self,
                           hours_elapsed: float,
                           ):
        """
        Calculate the number of intervals that have elapsed, given a number of hours that have elapsed since the
        start time
        :return: float, interval, the number of intervals that have elapsed since the start time
        """
        interval = hours_elapsed // self.time_interval
        return interval

    def interval_in_intervals_elapsed(self,
                                      interval: float
                                      ):
        """
        given an interval, check if it is in the list of intervals that have elapsed. return True if the interval is
        in the list of elapsed intervals

        :param float, interval:
        :return:
        """
        boolean = interval in self.intervals_elapsed
        return boolean

    def append_interval(self,
                        interval: float,
                        ):
        """
        append an interval to the list of intervals elapsed

        :param float, interval:
        :return:
        """
        self.intervals_elapsed.append(interval)

    def has_a_time_interval_elapsed(self,
                                    time: datetime = None,
                                    ):
        """
        Based on the time given, check if the interval based on the current time is already in the
        intervals_elapsed list. if it isn't in there already, return True. If no time was given, then use the current
        time
        :param datetime, time:
        :return: bool
        """
        if time is None:
            time = datetime.now()

        hours_elapsed = self.hours_elapsed(time=time)

        # number of intervals of the specified time interval has passed
        interval = self.calculate_interval(hours_elapsed=hours_elapsed)

        # if the interval is in the list of elapsed intervals then
        interval_in_intervals_elapsed = self.interval_in_intervals_elapsed(interval=interval)

        return not interval_in_intervals_elapsed

    def more_than_one_interval_has_elapsed(self):
        """
        return True if at least one interval has elapsed since the start time
        :return:
        """
        return len(self.intervals_elapsed) > 1



