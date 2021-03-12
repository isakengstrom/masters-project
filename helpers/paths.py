import os

# Path to the foi dataset
DATASET_PATH = os.environ['DATASET_DIR'] + "VIDEO/"  # Path to the dataset that should be extracted from

# dir to save the openpose keypoints at
OP_KP_SAVE_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/op_full"

# Path to fetch the openpose keypoints from
OP_KEYPOINTS_PATH = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/"

# Path to the session offsets extracted using cross correlation
CC_OFFSETS_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/json_dumps_test/offsets.json"