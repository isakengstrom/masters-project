import numpy as np
import math

from extraction_config import SAVE_PATH
from openpose_extraction import extract_poses  # Only for testing in main function
from json_helpfile import save_to_json


def save_processed_poses(poses, subject_idx, sequence_idx, angle_idx):

    name = "SUB{}_SEQ{}_ANG{}".format(subject_idx, sequence_idx, angle_idx)
    json_name = SAVE_PATH + "/{}.json".format(name)

    save_to_json(poses, json_name)


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


def process_skeletons_to_one(media_poses):

    processed_angle = []
    prev_skeleton = np.empty((25, 2))

    for skeletons in media_poses:
        skeletons = np.array(skeletons)
        if skeletons.shape[0] == 1:
            skeleton = np.squeeze(skeletons, axis=0)
        else:
            skeleton = remove_skeletons(prev_skeleton, skeletons)

        filtered_skeleton = filter_skeleton_to_xy(skeleton)
        processed_angle.append(filtered_skeleton)
        prev_skeleton = filtered_skeleton

    return processed_angle


def process_poses(media_poses):

    """

    :param media_poses:
    :return:
    """

    return process_skeletons_to_one(media_poses)

if __name__ == "__main__":
    #poses = extract_poses("/home/isaeng/Exjobb/media/mini.jpg", 'image')
    #poses = extract_poses("/home/isaeng/Exjobb/media/dir", 'images')
    poses = extract_poses("/home/isaeng/Exjobb/media/front.mp4", 'video', should_extract=True)
    #filter_poses_to_xy(poses=poses)
