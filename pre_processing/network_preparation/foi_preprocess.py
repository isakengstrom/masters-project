import os
import numpy as np

from helper_files.json_helper import read_from_json
from ..sequence_synchronisation.sync_config import DATA_PATH


def filter_file_names(unfiltered_names):
    filtered_names = []
    for unfiltered_name in unfiltered_names:
        _, file_extension = os.path.splitext(unfiltered_name)
        if file_extension == ".json":
            filtered_names.append(unfiltered_name)

    return filtered_names


def preprocess():
    print("preprocessing")

    if not os.path.exists(DATA_PATH):
        return

    dir_path, _, unfiltered_file_names = next(os.walk(DATA_PATH))
    file_names = filter_file_names(unfiltered_file_names)
    file_names = sorted(file_names)

    for file_name in file_names:
        keypoints = read_from_json(dir_path + file_name)
        keypoints = np.array(keypoints)
        print(file_name + ": " + str(keypoints.shape))


if __name__ == "__main__":
    preprocess()
