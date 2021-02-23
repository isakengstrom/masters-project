import os

from openpose_extraction import extract_keypoints


def walk_level(top_dir, level=1):
    """
    Functions to walk through all the sub-directories of a given directory

    Code from: https://stackoverflow.com/a/234329

    :param level:
    :type top_dir: object
    """

    curr_dir = top_dir.rstrip(os.path.sep)
    assert os.path.isdir(curr_dir)
    num_sep = curr_dir.count(os.path.sep)
    for root, dirs, files in os.walk(curr_dir, topdown=True):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def extract_from_sequence(root_dir, subject_name, sequence_name):
    """
    Extract the pose from a single video sequence and save it to JSON

    :param root_dir:
    :param subject_name:
    :param sequence_name:
    :return:
    """
    sequence_dir = os.path.join(root_dir, subject_name, sequence_name)
    for _, dirs, files in walk_level(sequence_dir, level=0):
        for file in sorted(files):
            path = os.path.join(sequence_dir, file)

            # Extract the keypoints using OpenPose
            keypoints = extract_keypoints(media_path=path, media_type='video', should_extract=False)

    print("extracted from sequence")


def extract_from_subject(root_dir, subject_name, sequence_name):
    """
    Extract the pose from a single subject, either from all sequences or a specific one

    :param root_dir:
    :param subject_name:
    :param sequence_name:
    :return:
    """
    subject_dir = os.path.join(root_dir, subject_name)
    if sequence_name is None:
        for _, local_dir_name, _ in walk_level(subject_dir, level=0):
            for sequence_name in sorted(local_dir_name):
                extract_from_sequence(root_dir=root_dir, subject_name=subject_name, sequence_name=sequence_name)
    else:
        seq_dir = os.path.join(subject_dir, sequence_name)
        extract_from_sequence(seq_dir)


def extract_from_subjects(root_dir, sequence_name=None):
    """
    Extract the pose from all the subjects

    :param root_dir:
    :param sequence_name:
    :return:
    """
    for _, local_dir_name, _ in walk_level(top_dir=root_dir, level=0):
        for subject_name in sorted(local_dir_name):
            extract_from_subject(root_dir=root_dir, subject_name=subject_name, sequence_name=sequence_name)


if __name__ == "__main__":
    extract_from_subjects(root_dir=os.environ['DATASET_DIR'] + "/VIDEO/", sequence_name=None)
