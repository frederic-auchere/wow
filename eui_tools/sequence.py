import os
import glob


def _ffmpeg_exists():
    pass


class Sequence:
    def __init__(self, source, **kwargs):
        self.files = glob.glob(os.path.join(source, '*.fits'))
        self.files.sort()
        self.output_directory = self._make_output_directory(kwargs['output_directory'])
        self.no_encoding = kwargs['no_encoding']

    @staticmethod
    def _make_output_directory(source):
        if source is None:
            return ''
        else:
            os.mkdir(source)
            return source

    def register(self):
        pass
