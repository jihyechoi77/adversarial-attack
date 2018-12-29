import argparse
import os
import shutil
import sys
import numpy as np

train_ratio = 0.75
val_ratio = 0.1


def move_files(abs_dir):
    """Move files into subdirectories."""

    """
    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]

    i = 0
    curr_subdir = None

    for f in files:
        # create new subdir if necessary
        if i % N == 0:
            subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i / N + 1))
            os.mkdir(subdir_name)
            curr_subdir = subdir_name

        # move file to current dir
        f_base = os.path.basename(f)
        shutil.move(f, os.path.join(subdir_name, f_base))
        i += 1
    """

    sbj_names = os.listdir(abs_dir)

    for s in sbj_names:
        if not os.path.exists(os.path.join(abs_dir+'-ready', 'train', s)):
            os.mkdir(os.path.join(abs_dir+'-ready', 'train', s))
        if not os.path.exists(os.path.join(abs_dir+'-ready', 'test', s)):
            os.mkdir(os.path.join(abs_dir+'-ready', 'test', s))
        if not os.path.exists(os.path.join(abs_dir+'-ready', 'validation', s)):
            os.mkdir(os.path.join(abs_dir+'-ready', 'validation', s))

        abs_subdir = os.path.join(abs_dir, s)
        files = os.listdir(abs_subdir)
        files_num = len(files)

        flag = 'train'
        for i, f in enumerate(files):
            if i == np.floor(files_num*train_ratio):
                flag = 'validation'
            if i == np.floor(files_num*(train_ratio+val_ratio)):
                flag = 'test'

            save_path = os.path.join(abs_dir+'-ready', flag, s, f)
            # move file
            shutil.copy(os.path.join(abs_subdir, f), save_path)


def main(args):
    """Module's main entry point (zopectl.command)."""
    src_dir = args.src_dir

    if not os.path.exists(src_dir):
        raise Exception('Directory does not exist ({0}).'.format(src_dir))

    move_files(os.path.abspath(src_dir))


def parse_arguments(argv):
    """Parse command line arguments passed to script invocation."""
    parser = argparse.ArgumentParser(description='Split files into multiple subfolders.')
    parser.add_argument('src_dir', help='source directory')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
