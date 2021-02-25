import numpy as np

from FOI_extraction import extract_from_foi_dataset
from process_poses import process_poses
from extraction_config import DATASET_PATH

# TODO fix the output in this function
def print_dataset_structure(dataset_poses):
    for subject_poses in dataset_poses:
        for sequence_poses in subject_poses:
            print("\t Subject {}, sequence {}, angle {}".format(len(dataset_poses), len(subject_poses), len(sequence_poses)))
            for angle_poses in sequence_poses:
                poses = np.array(angle_poses)

                print("\t\t{} frames | {} poses/frame | {} keypoints/pose | {} parameters/keypoint".format(poses.shape[0], poses.shape[1], poses.shape[2], poses.shape[3]))




if __name__ == "__main__":
    foi_dataset_poses = extract_from_foi_dataset(root_dir=DATASET_PATH)

    print_dataset_structure(foi_dataset_poses)

    process_poses(foi_dataset_poses)

