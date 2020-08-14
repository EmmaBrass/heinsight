"""
Classes used to track something - the specific purpose of the try tracker is to set a maximum number for the number
of times an action should be done or tried, and to notify the user when the limit has been reached. The number of
tries can also be reset.
"""


class TryTracker:
    def __init__(self,
                 max_number_of_tries: int,
                 ):
        self.try_counter = 0
        self.max_number_of_tries = max_number_of_tries

    def reset_try_counter(self):
        self.try_counter = 0

    def get_try_counter(self):
        return self.try_counter

    def get_max_number_of_tries(self):
        return self.max_number_of_tries

    def set_max_number_of_tries(self,
                                max_number: int
                                ):
        self.max_number_of_tries = max_number

    def not_reached_maximum_number_of_tries(self):
        """
        Return True if try counter has not yet reached the maximum number of tries
        :return: bool,
        """
        if self.try_counter < self.max_number_of_tries:
            return True

    def reached_maximum_number_of_tries(self):
        """
        Return True if the maximum number of tries has been reached.

        :return: bool,
        """
        if self.try_counter is self.max_number_of_tries:
            return True

    def increment_try_counter(self):
        """
        Increment try counter by 1
        :return:
        """

        self.try_counter += 1


class CPCTracker(TryTracker):
    """
    Tracker for CPC applications. Specifically used to track the number of times the liquid level in a container was
    not successfully found.
    """

    def __init__(self,
                 max_number_of_tries: int,
                 ):
        super().__init__(max_number_of_tries=max_number_of_tries)




