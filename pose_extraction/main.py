import numpy as np

from FOI_extraction import extract_from_foi_dataset
from process_poses import process_poses, save_processed_poses
from extraction_config import DATASET_PATH

# TODO fix the output in this function
def print_dataset_structure(dataset_poses, flag=1):
    for subject_poses in dataset_poses:
        for sequence_poses in subject_poses:
            if flag == 0:
                print("\t Subjects {}, sequences {}, angles {}".format(len(dataset_poses), len(subject_poses), len(sequence_poses)))
                for angle_poses in sequence_poses:
                    poses = np.array(angle_poses)

                    print("\t\t{} frames | {} poses/frame | {} keypoints/pose | {} parameters/keypoint".format(poses.shape[0], poses.shape[1], poses.shape[2], poses.shape[3]))

            elif flag == 1:
                sequence_poses = np.array(sequence_poses)
                print(sequence_poses.shape)
                #print("\t\t{} frames | {} keypoints/pose | {} parameters/keypoint".format(poses.shape[0], poses.shape[1], poses.shape[2]))



if __name__ == "__main__":
    '''
    test = {
        "hej": {
            "da": "två",
            "aa": "one"
        },
        "pos": 1
    }
    print(test["hej"]["da"])
    '''

    foi_dataset_poses = extract_from_foi_dataset(root_dir=DATASET_PATH)

    #print_dataset_structure(foi_dataset_poses)

    processed_poses = process_poses(foi_dataset_poses)
        #print_dataset_structure(processed_poses, flag=1)
    save_processed_poses(processed_poses)

