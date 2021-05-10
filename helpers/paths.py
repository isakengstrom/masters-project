import os

# Naming Conventions:
# OP - OpenPose
# EXTR - Extracted
# KP - KeyPoints
# CC - Cross Correlation
#

# Unused: UT - untrimmed # T - trimmed # F - final

# Path to the foi dataset
DATASET_PATH = os.path.join(os.environ['DATASET_DIR'], "VIDEO/")  # Path to the dataset that should be extracted from

# dir to save the openpose keypoints at
OP_SAVE_PATH = os.path.join(os.environ['DATASET_DIR'], "/isaeng_extr/json_dumps_test")

# Paths to the openpose keypoints
EXTR_PATH = os.path.join(os.environ['DATASET_DIR'], "EXTRACTED_KEYPOINTS/isaeng/")
EXTR_PATH_SSD = os.path.join(os.environ['THESIS_PROJECT_PATH'], "data/")

# Path to the json that stores at which frames the untrimmed extracted
TRIM_INTERVALS = os.path.join(os.environ['REPO_PATH'], "users/isaeng/data_processing/pre_processing/frame_intervals.json")

# Path to the joints lookup table
JOINTS_LOOKUP_PATH = os.path.join(os.environ['REPO_PATH'], "users/isaeng/network_modelling/joints_lookup.json")

# Path to the session offsets extracted using cross correlation
CC_OFFSETS_PATH = os.path.join(os.environ['DATASET_DIR'], "/isaeng_extr/json_dumps_test/offsets.json")

# Path for TensorBoard's SummaryWriter
TB_RUNS_PATH = os.path.join(os.environ['STATES_PATH'], "runs/")


