import os

# Path to the foi dataset
DATASET_PATH = os.environ['DATASET_DIR'] + "VIDEO/"  # Path to the dataset that should be extracted from

# dir to save the openpose keypoints at
OP_SAVE_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/json_dumps_test"

# Name of folders used in paths below
SYNC_STATUS = "non_synced"  # non_synced / synced
TRIM_STATUS = "untrimmed_version"  # untrimmed_version / trimmed_version

# Paths to fetch the openpose keypoints from
OP_EXTRACTED_PATH = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/{}/{}/".format(TRIM_STATUS, SYNC_STATUS)

# Path to fetch/save the synced openpose keypoints from
OP_SYNCED_PATH = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/{}/synced/".format(TRIM_STATUS)

# Path to the json that stores at which frames the untrimmed extracted
TRIM_INTERVALS = os.environ['PROJECT_PATH'] + "users/isaeng/data_processing/pre_processing/frame_intervals.json"

# Path to the session offsets extracted using cross correlation
CC_OFFSETS_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/json_dumps_test/offsets.json"
