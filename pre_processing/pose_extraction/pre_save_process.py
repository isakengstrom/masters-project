import numpy as np
import math

from .extraction_config import SAVE_PATH
from .openpose_extraction import extract_poses  # Only for testing in main function
from helpers import write_to_json


def save_processed_poses(poses, file_name):
    json_name = SAVE_PATH + "/{}.json".format(file_name)
    write_to_json(poses, json_name)


def filter_skeleton_to_xy(skeleton):
    """
    Filters the skeleton's keypoints from (x, y, c) to (x, y).
    Where x, y are the 2D coordinates and c is the confidence
    """

    skeleton = np.array(skeleton, dtype=object)

    # Discard the c (confidence) value
    filtered_skeleton = skeleton[:, 0:2]

    return filtered_skeleton


def euclidean_distance(p, q):
    return math.dist(p, q)


def remove_skeletons(prev_skeleton, skeletons):
    """
    Removes the least likely skeletons from each frame, by comparing the distance from a skeletal joint in the previous
    frame's skeleton to the current frame's skeletons. The joint used is the central hip joint (Nr 8 from OpenPose).

    :param prev_skeleton:
    :param skeletons:
    :return:
    """

    distances = []

    # Extract the hip joint coordinates of the previous skeleton
    q = prev_skeleton[8, :].tolist()

    # Loop over all of the current frame's skeletons
    for skel_idx in range(skeletons.shape[0]):
        # Extract the hip joint of a current skeleton
        p = skeletons[skel_idx, 8, 0:2].tolist()

        distances.append(euclidean_distance(p, q))

    # Get the index of the skeleton who is closest to that of the previous frame
    correct_idx = distances.index(min(distances))

    return skeletons[correct_idx, :, :]


def process_poses(media_poses):
    """
    Process the poses in a media file.

    :param media_poses:
    :return:
    """
    processed_skeleton = []
    prev_skeleton = np.empty((25, 2))

    # For each
    for skeletons in media_poses:
        skeletons = np.array(skeletons)

        # Remove unnecessary dimension if there's only one skeleton per frame
        if skeletons.shape[0] == 1:
            skeleton = np.squeeze(skeletons, axis=0)
        # Else, remove the least likely skeletons from each frame
        else:
            skeleton = remove_skeletons(prev_skeleton, skeletons)

        # Remove the confidence measure of each joint in a skeleton
        filtered_skeleton = filter_skeleton_to_xy(skeleton)
        processed_skeleton.append(filtered_skeleton)
        prev_skeleton = filtered_skeleton

    return processed_skeleton


if __name__ == "__main__":
    #poses = extract_poses("/home/isaeng/Exjobb/media/mini.jpg", 'image')
    #poses = extract_poses("/home/isaeng/Exjobb/media/dir", 'images')
    poses = extract_poses("/home/isaeng/Exjobb/media/front.mp4", 'video', should_extract=True)
    #filter_poses_to_xy(poses=poses)
