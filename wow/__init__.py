from watroo import utils

__version__ = '0.0.1'
__all__ = ['wow']


def wow(*args, **kwargs):
    """
    Processes an image with the WOW algorithm
    :param args:
    :param kwargs:
    :return: processed image (ndarray)
    """
    wow_image, _ = utils.wow(*args, **kwargs)
    return wow_image
