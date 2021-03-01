import os
import argparse

SHOULD_DISPLAY = False  # OpenPose: If the stream should be displayed during pose extraction
SHOULD_EXTRACT = True  # OpenPose: If extraction should take place
SHOULD_SAVE = True  # If the poses should be saved tp JSON
DATASET_PATH = os.environ['DATASET_DIR'] + "/VIDEO/"  # Path to the dataset that should be extracted from
SAVE_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/json_dumps_1"  # Path to the directory to save the JSON files
TRIMMED_SEQUENCE_FLAG = "_T"  # Some sequences have trimmed versions, indicating by this flag in the name

# DEV parameters and functions
DEV = False
DEV_PARAMS = {
    # Run extraction on a specific subject/sequence/camera_angle/video_frame
    "sub_nr": 2,
    "seq_nr": 0,
    "angle_nr": 2,
    "frame_nr": None,

    # Set one of the following params to 'None' to disable the limits
    # E.g: "seq_lower_lim" being 'None' disables the seq limits, not the others (sub, angle, frame)

    # Used if 'sub_nr' is 'None'
    "sub_lower_lim": None,
    "sub_upper_lim": 11,

    # Used if 'seq_nr' is 'None'
    "seq_lower_lim": None,
    "seq_upper_lim": 10,

    # Used if 'angle_nr' is 'None'
    "angle_lower_lim": None,
    "angle_upper_lim": 6,

    # TODO: does not work well with frames, take a while if low limit is high
    # Used if 'frame_nr' is 'None'
    "frame_lower_lim": 0,
    "frame_upper_lim": 20,
}


def in_dev_limits(ind, param):
    """
    Check if dev index is in the specified DEV_PARAMS boundaries

    :param ind: Index to check
    :param param: Which limit to check against
    :return: bool
    """
    if DEV_PARAMS[param + "_nr"] is None:
        # If one of the limits connected to the param is 'None', those two limits will be disabled
        if DEV_PARAMS[param + "_lower_lim"] is None or DEV_PARAMS[param + "_upper_lim"] is None:
            return True

        return DEV_PARAMS[param + "_lower_lim"] <= ind < DEV_PARAMS[param + "_upper_lim"]
    else:
        return DEV_PARAMS[param + "_nr"] == ind


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
    params["render_threshold"] = 0.1


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
