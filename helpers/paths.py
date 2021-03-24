import os

# Path to the foi dataset
DATASET_PATH = os.environ['DATASET_DIR'] + "VIDEO/"  # Path to the dataset that should be extracted from

# dir to save the openpose keypoints at
OP_SAVE_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/json_dumps_test"

# Name of folder used in paths below
SYNC_STATUS = "non_synced"  # non_synced / synced

# Paths to fetch the openpose keypoints from
OP_EXTRACTED_PATH = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/untrimmed_version/{}/".format(SYNC_STATUS)
OP_EXTRACTED_PATH_T = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/trimmed_version/{}/".format(SYNC_STATUS)

# Path to fetch/save the synced openpose keypoints from
OP_SYNCED_PATH = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/untrimmed_version/synced/"

# Path to the json that stores at which frames the untrimmed extracted
TRIM_INTERVALS = os.environ['PROJECT_PATH'] + "users/isaeng/data_processing/pre_processing/frame_intervals.json"

# Path to the session offsets extracted using cross correlation
CC_OFFSETS_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/json_dumps_test/offsets.json"

# Paths for the model:
SEQUENTIAL_DATA_PATH = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/trimmed_version/{}/sequential_data.json".format(SYNC_STATUS)
NON_SEQUENTIAL_DATA_PATH = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/trimmed_version/{}/sequential_data.json".format(SYNC_STATUS)
