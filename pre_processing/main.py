"""
Starting point for the pose extraction, using OpenPose.
"""

import os

from helper_files.limits_helper import SHOULD_LIMIT, lower_lim_check, upper_lim_check, TRIMMED_SESSION_FLAG
from pose_extraction.extraction_config import DATASET_PATH
from pose_extraction.foi_extraction import extract_session
from session_synchronisation.sync_sessions import synchronise_session


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

    # Remove un-trimmed sessions (sess) before extraction if they have trimmed counterparts
    # Some sessions are trimmed in the beginning and end to remove frames containing more than one individual
    for sess_name in sess_names:
        if TRIMMED_SESSION_FLAG in sess_name:
            deprecate_session_name = sess_name.replace(TRIMMED_SESSION_FLAG, "")
            sess_names.remove(deprecate_session_name)

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
    """""""""""
    For extraction of the FOI dataset
    """""""""""

    #loop_over_foi_dataset(root_dir=DATASET_PATH, action=extract_session)

    """""""""""
    For syncing the sessions
    """""""""""

    loop_over_foi_dataset(root_dir=DATASET_PATH, action=synchronise_session)
