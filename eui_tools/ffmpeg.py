import argparse
import subprocess
import itertools


parser = argparse.ArgumentParser(description='Concatenate videos with FFMPEG, add "xfade" between segments.')
parser.add_argument('--segments_file', '-f', metavar='Segments file', type=str, nargs=1,
                    help='Segments text file for concatenating. e.g. "segments.txt"')
parser.add_argument('--output', '-o', dest='output_filename', type=str,
                    default='ffmpeg_concat_fade_out.mp4',
                    help='output filename to provide to ffmpeg. default="ffmpeg_concat_fade_out.mp4"')
parser.add_argument('-t', '--transition', help='Transition type (see https://www.ffmpeg.org/ffmpeg-filters.html#xfade)',
                    default='fade', type=str)
parser.add_argument('segments', nargs='+')


def main(args):

    if args.segments_file:
        with open(args.segments_file[0], 'r') as seg_file:
            # cut the `file '` prefix and `'` postfix
            segments = [line[6:-2] for line in seg_file.readlines() if len(line.strip()) > 0 and line[0] != "#"]
    else:
        segments = args.segments

    # Get the lengths of the videos in seconds
    file_lengths = [
        float(subprocess.run(['ffprobe',
                              '-v', 'error',
                              '-show_entries', 'format=duration',
                              '-of', 'default=noprint_wrappers=1:nokey=1',
                              f],
                             capture_output=True).stdout.splitlines()[0])
        for f in segments
    ]

    width = int(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries',
                                'stream=width', '-of', 'default=nw=1:nk=1', segments[0]],
                               capture_output=True).stdout.splitlines()[0])
    height = int(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries',
                                 'stream=height', '-of', 'default=nw=1:nk=1', segments[0]],
                                capture_output=True).stdout.splitlines()[0])

    files_input = [['-i', f, '-f', 'lavfi', '-t', '1', '-i', f'color=c=black:s={width}x{height}'] for f in segments]

    video_fades = ""
    last_fade_output = "0v"
    video_length = 0

    for i in range(1, 2*len(segments)+1):

        video_length += file_lengths[i//2 - 1] - 0.25
        next_fade_output = "v%d%d" % (i - 1, i)
        if i % 2:
            filt = "xfade=transition=%s:duration=0.5:offset=%.3f" % (args.transition, video_length - 1)
        else:
            filt = "null"
        video_fades += "[%s][%d:v]%s" % \
            (last_fade_output, i, filt)
        if i < 2*len(segments):
            video_fades += "[%s];" % next_fade_output

        last_fade_output = next_fade_output

    video_fades += f",format=yuv420p"

    ffmpeg_args = ['ffmpeg',
                   *itertools.chain(*files_input),
                   '-filter_complex', video_fades,
                   '-y',
                   args.output_filename]

    print(" ".join(ffmpeg_args))
    subprocess.run(ffmpeg_args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
