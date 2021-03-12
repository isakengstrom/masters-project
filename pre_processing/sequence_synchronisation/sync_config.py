import os

DATASET_PATH = os.environ['DATASET_DIR'] + "VIDEO/"  # Path to the dataset that should be extracted from
DATA_PATH = os.environ['DATASET_DIR'] + "EXTRACTED_KEYPOINTS/isaeng/"
OFFSETS_SAVE_PATH = os.environ['DATASET_DIR'] + "/isaeng_extr/json_dumps_test/offsets.json"
EXTRACT_OFFSET = False
USE_OFFSET = False
SHOULD_DISPLAY = True
FIX_BACK_CAMERA = True


