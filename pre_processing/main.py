"""
Starting point for the pose extraction, using OpenPose.
"""

import os

from helper_files.project_config import SHOULD_LIMIT, lower_lim_check, upper_lim_check, TRIMMED_SEQUENCE_FLAG
from pose_extraction.extraction_config import DATASET_PATH
from pose_extraction.FOI_extraction import extract_sequence
from network_modelling.sync_sequences import synchronise_sequence


def loop_over_sequence(sequence_dir, subject_idx, sequence_idx, action):
    """
    Extracts, processes and saves the poses from a sequence. A sequence can consist of many videos covering different angles.

    :param sequence_dir:
    :param subject_idx:
    :param sequence_idx:
    :param action:
    :return:
    """

    if not os.path.exists(sequence_dir):
        return

    # Get the angle names (child file names of a sequence)
    _, _, camera_angles = next(os.walk(sequence_dir))
    action(sequence_dir, subject_idx, sequence_idx, camera_angles)


def loop_over_subject(subject_dir, subject_idx, action=None):
    """
    Extract the poses from a single subject, either from all sequences or a specific one

    :param subject_dir:
    :param subject_idx:
    :param action:
    :return:
    """

    if not os.path.exists(subject_dir):
        return

    # Get the sequence names (child folder names of a subject)
    _, sequence_names, _ = next(os.walk(subject_dir))

    # Remove un-trimmed sequences before extraction if they have trimmed counterparts
    # Some sequences are trimmed in the beginning and end to remove frames containing more than one individual
    for sequence_name in sequence_names:
        if TRIMMED_SEQUENCE_FLAG in sequence_name:
            deprecate_sequence_name = sequence_name.replace(TRIMMED_SEQUENCE_FLAG, "")
            sequence_names.remove(deprecate_sequence_name)

    for sequence_idx in range(len(sorted(sequence_names))):
        if SHOULD_LIMIT and lower_lim_check(sequence_idx, "seq"):
            continue
        if SHOULD_LIMIT and upper_lim_check(sequence_idx, "seq"):
            break

        sequence_dir = os.path.join(subject_dir, sequence_names[sequence_idx])
        loop_over_sequence(sequence_dir, subject_idx, sequence_idx, action)


def loop_over_foi_dataset(root_dir, action=None):
    """
    Extract the poses from all the subjects
    :param root_dir:
    :param action:
    :return:
    """

    # Loop through the dir containing all the subjects
    _, subject_names, _ = next(os.walk(root_dir))

    for subject_idx in range(len(sorted(subject_names))):
        if SHOULD_LIMIT and lower_lim_check(subject_idx, "sub"):
            continue
        if SHOULD_LIMIT and upper_lim_check(subject_idx, "sub"):
            break

        subject_dir = os.path.join(root_dir, subject_names[subject_idx])
        loop_over_subject(subject_dir, subject_idx, action)


if __name__ == "__main__":
    """""""""""
    For extraction of the FOI dataset
    """""""""""

    #loop_over_foi_dataset(root_dir=DATASET_PATH, action=extract_sequence)

    """""""""""
    For syncing the sequences
    """""""""""

    loop_over_foi_dataset(root_dir=DATASET_PATH, action=synchronise_sequence)
