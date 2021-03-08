import os
import numpy as np

from pose_extraction.json_helpfile import read_from_json
from modelling_config import DATA_PATH

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

    # Get the sequence names (child folder names of a subject)
    dir_path, _, unfiltered_file_names = next(os.walk(DATA_PATH))
    data_names = filter_file_names(unfiltered_file_names)
    data_names = sorted(data_names)


    #keypoints = read_from_json(dir_path + "SUB0_SEQ0_ANG0.json")



if __name__ == "__main__":
    preprocess()
