import os
import argparse

# Path constants
DATASET_PATH = os.environ['DATASET_DIR'] + "/VIDEO/"  # Path to the dataset that should be extracted from
SAVE_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/json_dumps_test"  # Path to the directory to save the JSON files

# Flags
TRIMMED_SEQUENCE_FLAG = "_T"  # Some sequences have trimmed versions, indicating by this flag in the name

# Settings
SHOULD_DISPLAY = True  # OpenPose: If the stream should be displayed during pose extraction
SHOULD_EXTRACT = False  # OpenPose: If extraction should take place
SHOULD_SAVE = False  # If the poses should be saved tp JSON
SHOULD_LIMIT = False # If false, runs through the whole FOI dataset, else it runs through the intervals specified in LIMIT_PARAMS


# Setting a limit to 'None' or '0<' will disable that limit
# These limits work like this: "lower_lim" <= param < "upper_lim"
LIMIT_PARAMS = {
    # Limits for the subjects
    "sub_lower_lim": None,
    "sub_upper_lim": None,
    # Limits for the sequences
    "seq_lower_lim": None,
    "seq_upper_lim": None,
    # Limits for the angles
    "ang_lower_lim": None,
    "ang_upper_lim": None,
    # Limits for the frames
    "frame_lower_lim": None,
    "frame_upper_lim": None,
}


def lower_lim_check(ind, param):
    if LIMIT_PARAMS[param + "_lower_lim"] is None or LIMIT_PARAMS[param + "_lower_lim"] < 0:
        return False
    else:
        return LIMIT_PARAMS[param + "_lower_lim"] >= ind


def upper_lim_check(ind, param):
    if LIMIT_PARAMS[param + "_upper_lim"] is None or LIMIT_PARAMS[param + "_upper_lim"] < 0:
        return False
    elif LIMIT_PARAMS[param + "_lower_lim"] > LIMIT_PARAMS[param + "_upper_lim"]:
        return False
    else:
        return LIMIT_PARAMS[param + "_upper_lim"] < ind


def get_openpose_params():
    """
    get the OpenPose flags/parameters

    A list over the main flags can be found here:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md#main-flags

    A full list of flags can be found here:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/advanced/demo_advanced.md#all-flags

    The maximum accuracy configuration can be found here:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md#maximum-accuracy-configuration

    :return: dict()
    """

    params = dict()

    # Misc params
    params["model_folder"] = os.environ['OPENPOSE_DIR'] + "/models"
    params["disable_blending"] = False
    params["display"] = 0
    params["num_gpu"] = -1
    params["num_gpu_start"] = 0
    #params["output_resolution"] = "-1x-1"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.25
    params["scale_number"] = 1
    params["render_threshold"] = 0.075


    #params["number_people_max"] = 1  # If the data contains more than one person,

    # params for body keypoints
    params["model_pose"] = "BODY_25"  # "BODY_25", "COCO", "MPI"
    params["net_resolution"] = "-1x368"  # Lower res needed for COCO and MPI?

    # params for face keypoints
    params["face"] = False
    params["face_net_resolution"] = "256x256"

    # params for hand keypoints
    params["hand"] = False
    params["hand_net_resolution"] = "256x256"

    # Init parser
    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()

    # Add potential OpenPose params from path, overrides previously set flags
    for i in range(0, len(args[1])):
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"

        curr_item = args[1][i]
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            params[key] = next_item

    return params
