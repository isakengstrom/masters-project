import os
import numpy as np

from openpose_extraction import extract_poses
from extraction_config import DEV, TRIMMED_SEQUENCE_FLAG, SHOULD_DISPLAY, SHOULD_EXTRACT


def extract_from_sequence(sequence_dir):
    """
    Extract the poses from a sequence and save it to JSON. A sequence can consist of many videos covering different
    angles.

    :return:
    """

    if not os.path.exists(sequence_dir):
        return

    extracted_sequence = []
    angle_idx = 0
    _, _, camera_angles = next(os.walk(sequence_dir))
    for angle in sorted(camera_angles):

        if DEV:
            angle_idx += 1
            if angle_idx > 2:
                break

        path = os.path.join(sequence_dir, angle)

        # Extract the keypoints using OpenPose
        extracted_angle = extract_poses(media_path=path, should_extract=SHOULD_EXTRACT, should_display=SHOULD_DISPLAY)
        extracted_sequence.append(extracted_angle)

    return extracted_sequence


def extract_from_subject(subject_dir, sequence_name):
    """
    Extract the poses from a single subject, either from all sequences or a specific one

    :return:
    """

    if not os.path.exists(subject_dir):
        return

    extracted_subject = []

    # If no sequence is specified, loop over all available sequences, else just extract the given sequence
    if sequence_name is None:
        sequence_idx = 0
        _, sequence_names, _ = next(os.walk(subject_dir))

        # Remove un-trimmed sequences before extraction if they have trimmed counterparts
        # Some sequences are trimmed in the beginning and end to remove frames containing more than one individual
        for sequence_name in sequence_names:
            if TRIMMED_SEQUENCE_FLAG in sequence_name:
                deprecate_sequence_name = sequence_name.replace(TRIMMED_SEQUENCE_FLAG, "")
                sequence_names.remove(deprecate_sequence_name)

        for sequence_name in sorted(sequence_names):

            if DEV:
                sequence_idx += 1
                if sequence_idx > 2:
                    break

            sequence_dir = os.path.join(subject_dir, sequence_name)
            extracted_sequence = extract_from_sequence(sequence_dir)
            extracted_subject.append(extracted_sequence)
    else:
        sequence_dir = os.path.join(subject_dir, sequence_name)
        extracted_sequence = extract_from_sequence(sequence_dir)
        extracted_subject.append(extracted_sequence)

    return extracted_subject


def extract_from_foi_dataset(root_dir, sequence_name=None):
    """
    Extract the poses from all the subjects

    :return:
    """

    extracted_dataset = []
    subject_idx = 0

    # Loop through the dir containing all the subjects
    _, subject_names, _ = next(os.walk(root_dir))
    for subject_name in sorted(subject_names):

        if DEV:
            subject_idx += 1
            if subject_idx > 3:
                break

        subject_dir = os.path.join(root_dir, subject_name)
        extracted_subject = extract_from_subject(subject_dir, sequence_name)
        extracted_dataset.append(extracted_subject)

    return extracted_dataset


if __name__ == "__main__":
    extracted_subjects = extract_from_foi_dataset(root_dir=os.environ['DATASET_DIR'] + "/VIDEO/")
    #extract_from_sequence(root_dir=os.environ['DATASET_DIR'] + "/VIDEO/", subject_name="SUBJECT_0", sequence_name="SEQ_0")


