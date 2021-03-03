"""
Starting point for the pose extraction, using OpenPose.
"""

from FOI_extraction import extract_from_foi_dataset
from extraction_config import DATASET_PATH


if __name__ == "__main__":
    extract_from_foi_dataset(root_dir=DATASET_PATH)
