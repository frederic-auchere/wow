from watroo import utils

__version__ = '0.0.1'
__all__ = ['wow']


def wow(image, *args, **kwargs):
    wow_image, _ = utils.wow(image, *args, **kwargs)
    return wow_image
