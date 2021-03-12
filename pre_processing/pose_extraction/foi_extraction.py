import os

from .openpose_extraction import extract_poses
from .extraction_config import SHOULD_DISPLAY, SHOULD_EXTRACT, SHOULD_SAVE
from .pre_save_process import process_poses, save_processed_poses
from helper_files.limits_helper import SHOULD_LIMIT, lower_lim_check, upper_lim_check


def extract_session(session_dir, subject_idx, session_idx, camera_angles):
    for angle_idx in range(len(sorted(camera_angles))):
        if SHOULD_LIMIT and lower_lim_check(angle_idx, "ang"):
            continue
        if SHOULD_LIMIT and upper_lim_check(angle_idx, "ang"):
            break

        path = os.path.join(session_dir, camera_angles[angle_idx])

        file_name = "SUB{}_SESS{}_ANG{}".format(subject_idx, session_idx, angle_idx)

        print("\n----------- {} -----------\n".format(file_name))

        # Extract the poses using OpenPose
        print("Extracting..")
        extracted_poses = extract_poses(media_path=path, should_extract=SHOULD_EXTRACT, should_display=SHOULD_DISPLAY)

        # Process the poses
        print("Processing..")
        processed_poses = process_poses(extracted_poses)

        # Save the poses to json, one file for every subject's sessions and angles
        if SHOULD_EXTRACT and SHOULD_SAVE:
            print("Saving..")
            save_processed_poses(processed_poses, file_name)


if __name__ == "__main__":
    extract_session(session_dir=os.environ['DATASET_DIR'] + "/VIDEO/SUBJECT_0/SEQ_0", subject_idx=0, session_idx=0,
                    camera_angles=["above.MTS"])


