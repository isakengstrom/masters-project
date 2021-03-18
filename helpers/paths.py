import os

# Path to the foi dataset
DATASET_PATH = os.environ['DATASET_DIR'] + "VIDEO/"  # Path to the dataset that should be extracted from

# dir to save the openpose keypoints at
OP_SAVE_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/json_dumps_test"

# Path to fetch the openpose keypoints from
OP_EXTRACTED_PATH = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/untrimmed_version/"
OP_KEYPOINTS_PATH_1 = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/trimmed_version/"
# Path to the session offsets extracted using cross correlation
CC_OFFSETS_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/json_dumps_test/offsets.json"