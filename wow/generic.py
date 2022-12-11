import os


def make_directory(name):
    if name:
        if os.path.isdir(name):
            return name
        try:
            os.mkdir(name)
            return name
        except FileNotFoundError:
            return ''
    else:
        return ''


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)