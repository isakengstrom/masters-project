import os
import numpy as np

from helpers import read_from_json
from helpers.paths import OP_EXTRACTED_PATH, OP_SYNCED_PATH, TRIM_INTERVALS
from .pre_config import SHOULD_LOAD_TRIMMED, SHOULD_SYNC_SESSIONS
from .sync_sessions import sync_sessions


def filter_file_names(unfiltered_names):
    filtered_names = []
    for unfiltered_name in unfiltered_names:
        _, file_extension = os.path.splitext(unfiltered_name)
        if file_extension == ".json":
            filtered_names.append(unfiltered_name)

    return filtered_names


def trim_frames(data, interval_info):
    # Trim away frames at the beginning, amount specified by interval_info["lower_frames"]
    if interval_info["lower_frames"] is not None:
        data = data[interval_info["lower_frames"]:, :, :]

    # Trim away frames at the end, amount specified by interval_info["upper_frames"]
    # I.e output will be in the interval 0 : length - interval_info["upper_frames"]
    if interval_info["upper_frames"] is not None:
        data = data[:-interval_info["upper_frames"], :, :]

    return data


def process_extracted_files(path=OP_EXTRACTED_PATH):
    print("preprocessing")

    if not os.path.exists(path):
        return

    if SHOULD_SYNC_SESSIONS:
        sync_sessions(path)
        path = OP_SYNCED_PATH  # Change the path to load the synced files instead.

    dir_path, _, unfiltered_file_names = next(os.walk(path))
    file_names = filter_file_names(unfiltered_file_names)
    file_names = sorted(file_names)

    # Load the intervals at which to cut the untrimmed sessions
    intervals_data = read_from_json(TRIM_INTERVALS, use_dumps=True)

    for file_name in file_names:
        view_data = read_from_json(dir_path + file_name)  # Load all the data from one view
        view_data = np.array(view_data)  # Convert data to np array

        file_name = file_name.split('.')[0].lower()  # Remove ".json" and convert to lowercase
        subject_name, session_name, view_name = file_name.split('_')  # Split sub*_sess*_view* to sub*, sess* and view*
        interval_info = intervals_data[subject_name][session_name]

        # Crop the beginning and end frames if they need trimming
        if not SHOULD_LOAD_TRIMMED and interval_info["status"] == "trim":
            view_data = trim_frames(data=view_data, interval_info=interval_info)

        print(view_name + "::: " + str(view_data.shape))


if __name__ == "__main__":
    process_extracted_files()
