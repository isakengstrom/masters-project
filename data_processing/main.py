"""
Starting point for the pose extraction, using OpenPose.
"""

import os
import time
import numpy as np

from helpers import SHOULD_LIMIT, lower_lim_check, upper_lim_check, read_from_json
from helpers.display_helper import display_session
from helpers.json_helper import combine_json_files
from helpers.paths import DATASET_PATH, EXTR_PATH, EXTR_PATH_SSD
from pose_extraction.extraction_config import TRIMMED_SESSION_FLAG, SHOULD_USE_TRIMMED

from pose_extraction.foi_extraction import extract_session
from pre_processing.cc_sync_sessions import cc_session_sync
from pre_processing.post_extraction_processing import process_extracted_files


def loop_over_session(session_dir, subject_idx, session_idx, action):
    """
    Extracts, processes and saves the poses from a session. A session can consist of many videos covering different
    views.

    :param session_dir:
    :param subject_idx:
    :param session_idx:
    :param action:
    :return:
    """

    if not os.path.exists(session_dir):
        return

    # Get the view names (child file names of a session)
    _, _, views = next(os.walk(session_dir))

    action(session_dir, subject_idx, session_idx, views)


def loop_over_subject(subject_dir, subject_idx, action=None):
    """
    Extract the poses from a single subject

    :param subject_dir:
    :param subject_idx:
    :param action:
    :return:
    """

    if not os.path.exists(subject_dir):
        return

    # Get the session names (child folder names of a subject)
    _, sess_names, _ = next(os.walk(subject_dir))

    if SHOULD_USE_TRIMMED:
        # Remove un-trimmed sessions (sess) before extraction if they have trimmed counterparts
        # Some sessions are trimmed in the beginning and end to remove frames containing more than one individual
        for sess_name in sess_names:
            if TRIMMED_SESSION_FLAG in sess_name:
                deprecate_session_name = sess_name.replace(TRIMMED_SESSION_FLAG, "")
                sess_names.remove(deprecate_session_name)
    else:
        for sess_name in sess_names:
            if TRIMMED_SESSION_FLAG in sess_name:
                sess_names.remove(sess_name)

    for sess_idx in range(len(sorted(sess_names))):
        if SHOULD_LIMIT and lower_lim_check(sess_idx, "sess"):
            continue
        if SHOULD_LIMIT and upper_lim_check(sess_idx, "sess"):
            break

        sess_dir = os.path.join(subject_dir, sess_names[sess_idx])

        loop_over_session(sess_dir, subject_idx, sess_idx, action)


def loop_over_foi_dataset(root_dir, action=None):
    """
    Extract the poses from all the subjects
    :param root_dir:
    :param action:
    :return:
    """

    # Loop through the dir containing all the subjects
    _, subject_names, _ = next(os.walk(root_dir))

    for subject_idx in range(len(sorted(subject_names))):
        if SHOULD_LIMIT and lower_lim_check(subject_idx, "sub"):
            continue
        if SHOULD_LIMIT and upper_lim_check(subject_idx, "sub"):
            break

        subject_dir = os.path.join(root_dir, subject_names[subject_idx])
        loop_over_subject(subject_dir, subject_idx, action)


if __name__ == "__main__":
    start_time = time.time()
    """""""""""
    For extraction of the FOI dataset
    """""""""""

    #loop_over_foi_dataset(root_dir=DATASET_PATH, action=extract_session)

    """""""""""
    For syncing the sessions
    """""""""""

    #loop_over_foi_dataset(root_dir=DATASET_PATH, action=cc_session_sync)

    """""""""""
    For viewing session, synced or not
    """""""""""

    #loop_over_foi_dataset(root_dir=DATASET_PATH, action=display_session)

    """""""""""
    other
    """""""""""
    #process_extracted_files()
    #data_info = read_from_json(EXTR_PATH + "final_data_info.json")
    #print(data_info)

    #combine_json_files(EXTR_PATH + "final/")
    #data = read_from_json(EXTR_PATH_SSD + "final/combined/combined.json")
    #print(np.array(data["SUB5_SESS0_VIEW3.json"]).shape)

    #print(f"Main finished in {time.time()-start_time:0.1f}s")
