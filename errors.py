# custom exceptions for this project. (when a csv file can't be loaded, when a test point is too far from the ideal function)

class DataLoadError(Exception):
    """raised when a csv file can't be loaded (missing or broken)"""
    pass


class TooFarError(Exception):
    """raised when a test point is too far from the ideal function"""
