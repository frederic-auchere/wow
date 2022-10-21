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
