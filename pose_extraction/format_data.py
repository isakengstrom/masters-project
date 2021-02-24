import numpy as np

# Only for testing in main function
from openpose_extraction import extract_poses


def filter_poses_to_xy(_poses):
    """
    Filters the poses' keypoints from (x, y, c) to (x, y).
    Where x, y are the 2D coordinates and c is the confidence
    """

    _poses = np.array(_poses, dtype=object)

    # Remove the 1:th dimension, corresponding to the different poses, as the maximum number of people is 1
    # due to the OpenPose flag "number_people_max = 1"
    _poses = np.squeeze(_poses, axis=1)

    # Discard the c (confidence) value
    filtered_poses = _poses[:, :, 0:2]

    return filtered_poses


if __name__ == "__main__":
    #poses = extract_poses("/home/isaeng/Exjobb/media/mini.jpg", 'image')
    #poses = extract_poses("/home/isaeng/Exjobb/media/dir", 'images')
    poses = extract_poses("/home/isaeng/Exjobb/media/front.mp4", 'video', should_extract=True)
    filter_poses_to_xy(poses=poses)
