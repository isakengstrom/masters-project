import numpy as np
import math

from extraction_config import SAVE_PATH
from openpose_extraction import extract_poses  # Only for testing in main function
from json_helpfile import save_to_json


def save_processed_poses(poses):
    inputs = {}
    for sub_idx in range(len(poses)):
        subject_poses = poses[sub_idx]
        for seq_idx in range(len(subject_poses)):
            sequence_poses = subject_poses[seq_idx]
            for angle_idx in range(len(sequence_poses)):
                angle_poses = sequence_poses[angle_idx]
                name = "SUB{}_SEQ{}_ANG{}".format(sub_idx, seq_idx, angle_idx)
                json_name = SAVE_PATH + "/json_dumps_1/{}.json".format(name)

                for frame_idx in range(len(angle_poses)):
                    inputs[frame_idx] = angle_poses[frame_idx]

                save_to_json(inputs, json_name)
                inputs = {}


def filter_skeleton_to_xy(skeleton):
    """
    Filters the skeleton's keypoints from (x, y, c) to (x, y).
    Where x, y are the 2D coordinates and c is the confidence
    """

    skeleton = np.array(skeleton, dtype=object)

    # Discard the c (confidence) value
    filtered_skeleton = skeleton[:, 0:2]

    return filtered_skeleton


def euclid_dist(p, q):
    return math.dist(p, q)


def remove_skeletons(prev_skeleton, skeletons):
    distances = []

    for skel_idx in range(skeletons.shape[0]):
        p = skeletons[skel_idx, 8, 0:2].tolist()
        q = prev_skeleton[8, :].tolist()

        distances.append(euclid_dist(p, q))

    correct_idx = distances.index(min(distances))

    return skeletons[correct_idx, :, :]


def process_skeletons_to_one(dataset_poses):
    processed_dataset = []
    for subject_poses in dataset_poses:
        processed_subject = []
        for sequence_poses in subject_poses:
            processed_sequence = []
            for angle_poses in sequence_poses:
                processed_angle = []
                prev_skeleton = np.empty((25, 2))

                for frame_idx in range(len(angle_poses)):
                    skeletons = np.array(angle_poses[frame_idx])
                    if skeletons.shape[0] == 1:
                        skeleton = np.squeeze(skeletons, axis=0)
                    else:
                        skeleton = remove_skeletons(prev_skeleton, skeletons)

                    filtered_skeleton = filter_skeleton_to_xy(skeleton)
                    processed_angle.append(filtered_skeleton)
                    prev_skeleton = filtered_skeleton

                processed_sequence.append(processed_angle)
            processed_subject.append(processed_sequence)
        processed_dataset.append(processed_subject)
    return processed_dataset


def process_poses(dataset_poses):

    """

    :param dataset_poses:
    :return:
    """

    return process_skeletons_to_one(dataset_poses)

if __name__ == "__main__":
    #poses = extract_poses("/home/isaeng/Exjobb/media/mini.jpg", 'image')
    #poses = extract_poses("/home/isaeng/Exjobb/media/dir", 'images')
    poses = extract_poses("/home/isaeng/Exjobb/media/front.mp4", 'video', should_extract=True)
    #filter_poses_to_xy(poses=poses)
