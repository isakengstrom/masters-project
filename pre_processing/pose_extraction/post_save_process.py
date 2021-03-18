import os
import numpy as np

from helpers import read_from_json
from helpers.paths import OP_EXTRACTED_PATH, OP_KEYPOINTS_PATH_1


def filter_file_names(unfiltered_names):
    filtered_names = []
    for unfiltered_name in unfiltered_names:
        _, file_extension = os.path.splitext(unfiltered_name)
        if file_extension == ".json":
            filtered_names.append(unfiltered_name)

    return filtered_names


def load_extracted_files(path=OP_EXTRACTED_PATH):
    print("preprocessing")

    if not os.path.exists(path):
        return

    dir_path, _, unfiltered_file_names = next(os.walk(path))
    file_names = filter_file_names(unfiltered_file_names)
    file_names = sorted(file_names)

    for file_name in file_names:
        keypoints = read_from_json(dir_path + file_name)

        #print(keypoints[0][0])
        keypoints = np.array(keypoints)
        file_name = file_name.split('.')[0]

        print(file_name + ": " + str(keypoints.shape))


if __name__ == "__main__":
    load_extracted_files()
    #preprocess(OP_KEYPOINTS_PATH_1)
