"""

"""

import os
from scipy import fft
from scipy.io import wavfile
import numpy as np
import cv2
from glob import glob

from .sync_config import EXTRACT_OFFSET, USE_OFFSET, SHOULD_DISPLAY, FIX_BACK_CAMERA
from helpers import write_to_json, read_from_json, draw_label, SHOULD_LIMIT, LIMIT_PARAMS
from helpers.paths import CC_OFFSETS_PATH




