import os
import numpy as np

from openpose_extraction import extract_poses
from format_data import filter_poses_to_xy

TRIMMED_SEQUENCE_FLAG = "_T"


def walk_level(top_dir, level=1):
    """
    Functions to walk through all the sub-directories of a given directory

    Code from: https://stackoverflow.com/a/234329

    :param level:
    :type top_dir: object
    """


    curr_dir = top_dir.rstrip(os.path.sep)
    root, dirs, files = next(os.walk(curr_dir))
    print(dirs)

    assert os.path.isdir(curr_dir)
    num_sep = curr_dir.count(os.path.sep)
    for root, dirs, files in os.walk(curr_dir, topdown=True):
        print(dirs)
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def extract_from_sequence(root_dir, subject_name, sequence_name):
    """
    Extract the poses from a sequence and save it to JSON. A sequence can consist of many videos covering different
    angles.

    :param root_dir:
    :param subject_name:
    :param sequence_name:
    :return:
    """
    sequence_dir = os.path.join(root_dir, subject_name, sequence_name)

    if not os.path.exists(sequence_dir):
        return

    _, _, files = next(os.walk(sequence_dir))

    for file in sorted(files):
        path = os.path.join(sequence_dir, file)

        # Extract the keypoints using OpenPose
        extracted_poses = extract_poses(media_path=path, should_extract=True, should_display=True)

        filtered_poses = filter_poses_to_xy(extracted_poses)
        print(filtered_poses.shape)


def extract_from_subject(root_dir, subject_name, sequence_name):
    """
    Extract the poses from a single subject, either from all sequences or a specific one

    :param root_dir:
    :param subject_name:
    :param sequence_name:
    :return:
    """
    subject_dir = os.path.join(root_dir, subject_name)

    if not os.path.exists(subject_dir):
        return

    # If no sequence is specified, loop over all available sequences, else just extract the given sequence
    if sequence_name is None:
        _, sequence_names, _ = next(os.walk(subject_dir))

        # Remove un-trimmed sequences before extraction if they have trimmed counterparts
        # Some sequences are trimmed in the beginning and end to remove frames containing more than one individual
        for sequence_name in sequence_names:
            if TRIMMED_SEQUENCE_FLAG in sequence_name:
                deprecate_sequence_name = sequence_name.replace(TRIMMED_SEQUENCE_FLAG, '')
                sequence_names.remove(deprecate_sequence_name)

        for sequence_name in sorted(sequence_names):
            extract_from_sequence(root_dir=root_dir, subject_name=subject_name, sequence_name=sequence_name)
    else:
        extract_from_sequence(root_dir=root_dir, subject_name=subject_name, sequence_name=sequence_name)


def extract_from_subjects(root_dir, sequence_name=None):
    """
    Extract the poses from all the subjects

    :param root_dir:
    :param sequence_name:
    :return:
    """

    # Loop through the dir containing all the subjects
    _, subject_names, _ = next(os.walk(root_dir))
    for subject_name in sorted(subject_names):
        extract_from_subject(root_dir=root_dir, subject_name=subject_name, sequence_name=sequence_name)


if __name__ == "__main__":
    extract_from_subjects(root_dir=os.environ['DATASET_DIR'] + "/VIDEO/")
    #extract_from_sequence(root_dir=os.environ['DATASET_DIR'] + "/VIDEO/", subject_name="SUBJECT_0", sequence_name="SEQ_0")
