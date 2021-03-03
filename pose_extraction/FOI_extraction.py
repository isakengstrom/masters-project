import os

from openpose_extraction import extract_poses
from extraction_config import SHOULD_LIMIT, TRIMMED_SEQUENCE_FLAG, SHOULD_DISPLAY, SHOULD_EXTRACT, SHOULD_SAVE
from extraction_config import lower_lim_check, upper_lim_check
from process_poses import process_poses, save_processed_poses


def extract_from_sequence(sequence_dir, subject_idx, sequence_idx):
    """
    Extracts, processes and saves the poses from a sequence. A sequence can consist of many videos covering different angles.

    :param sequence_dir:
    :param subject_idx:
    :param sequence_idx:
    """

    if not os.path.exists(sequence_dir):
        return

    # Get the angle names (child file names of a sequence)
    _, _, camera_angles = next(os.walk(sequence_dir))

    for angle_idx in range(len(sorted(camera_angles))):
        if SHOULD_LIMIT and lower_lim_check(angle_idx, "ang"):
            continue
        if SHOULD_LIMIT and upper_lim_check(angle_idx, "ang"):
            break

        path = os.path.join(sequence_dir, camera_angles[angle_idx])

        file_name = "SUB{}_SEQ{}_ANG{}".format(subject_idx, sequence_idx, angle_idx)

        print("\n----------- {} -----------\n".format(file_name))

        # Extract the poses using OpenPose
        print("Extracting..")
        extracted_poses = extract_poses(media_path=path, should_extract=SHOULD_EXTRACT, should_display=SHOULD_DISPLAY)

        # Process the poses
        print("Processing..")
        processed_poses = process_poses(extracted_poses)

        # Save the poses to json, one file for every subject's sequences and angles
        if SHOULD_EXTRACT and SHOULD_SAVE:
            print("Saving..")
            save_processed_poses(processed_poses, file_name)


def extract_from_subject(subject_dir, subject_idx):
    """
    Extract the poses from a single subject, either from all sequences or a specific one

    :param subject_dir:
    :param subject_idx:
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
        extract_from_sequence(sequence_dir, subject_idx, sequence_idx)


def extract_from_foi_dataset(root_dir):
    """
    Extract the poses from all the subjects

    :param root_dir:
    """

    # Loop through the dir containing all the subjects
    _, subject_names, _ = next(os.walk(root_dir))

    for subject_idx in range(len(sorted(subject_names))):

        if SHOULD_LIMIT and lower_lim_check(subject_idx, "sub"):
            continue
        if SHOULD_LIMIT and upper_lim_check(subject_idx, "sub"):
            break

        subject_dir = os.path.join(root_dir, subject_names[subject_idx])
        extract_from_subject(subject_dir, subject_idx=subject_idx)


if __name__ == "__main__":
    extract_from_foi_dataset(root_dir=os.environ['DATASET_DIR'] + "/VIDEO/")
    #extract_from_sequence(root_dir=os.environ['DATASET_DIR'] + "/VIDEO/", subject_name="SUBJECT_0", sequence_name="SEQ_0")


