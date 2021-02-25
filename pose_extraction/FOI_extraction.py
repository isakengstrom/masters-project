import os
import numpy as np

from openpose_extraction import extract_poses
from extraction_config import DEV, in_dev_limits, TRIMMED_SEQUENCE_FLAG, SHOULD_DISPLAY, SHOULD_EXTRACT


def extract_from_sequence(sequence_dir):
    """
    Extract the poses from a sequence and save it to JSON. A sequence can consist of many videos covering different
    angles.

    :return:
    """

    if not os.path.exists(sequence_dir):
        return

    extracted_sequence = []

    _, _, camera_angles = next(os.walk(sequence_dir))
    for i in range(len(sorted(camera_angles))):
        # Move to the next camera angle if i is not in given limits during DEV
        if DEV and not in_dev_limits(i, "angle"):
            continue

        path = os.path.join(sequence_dir, camera_angles[i])

        # Extract the keypoints using OpenPose
        extracted_angle = extract_poses(media_path=path, should_extract=SHOULD_EXTRACT, should_display=SHOULD_DISPLAY)
        extracted_sequence.append(extracted_angle)

    return extracted_sequence


def extract_from_subject(subject_dir):
    """
    Extract the poses from a single subject, either from all sequences or a specific one

    :return:
    """

    if not os.path.exists(subject_dir):
        return

    extracted_subject = []

    # Get the sequence names (child folder names of a subject)
    _, sequence_names, _ = next(os.walk(subject_dir))

    # Remove un-trimmed sequences before extraction if they have trimmed counterparts
    # Some sequences are trimmed in the beginning and end to remove frames containing more than one individual
    for sequence_name in sequence_names:
        if TRIMMED_SEQUENCE_FLAG in sequence_name:
            deprecate_sequence_name = sequence_name.replace(TRIMMED_SEQUENCE_FLAG, "")
            sequence_names.remove(deprecate_sequence_name)

    for i in range(len(sorted(sequence_names))):
        # Move to the next sequence name if i is not in given limits during DEV
        if DEV and not in_dev_limits(i, "seq"):
            continue

        sequence_dir = os.path.join(subject_dir, sequence_names[i])
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

    for i in range(len(sorted(subject_names))):
        # Move to the next subject name if i is not in given limits during DEV
        if DEV and not in_dev_limits(i, "sub"):
            continue

        subject_dir = os.path.join(root_dir, subject_names[i])
        extracted_subject = extract_from_subject(subject_dir)
        extracted_dataset.append(extracted_subject)

    return extracted_dataset


if __name__ == "__main__":
    extracted_subjects = extract_from_foi_dataset(root_dir=os.environ['DATASET_DIR'] + "/VIDEO/")
    #extract_from_sequence(root_dir=os.environ['DATASET_DIR'] + "/VIDEO/", subject_name="SUBJECT_0", sequence_name="SEQ_0")


