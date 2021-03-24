"""
Cross correlation (cc) syncing of sessions. This file is deprecate.

http://www.dsg-bielefeld.de/dsg_wp/wp-content/uploads/2014/10/video_syncing_fun.pdf
"""

import os
from scipy import fft
from scipy.io import wavfile
import numpy as np
import cv2
from glob import glob

from .pre_config import EXTRACT_OFFSET, USE_OFFSET, SHOULD_DISPLAY, FIX_BACK_CAMERA
from helpers import write_to_json, read_from_json, display_session, SHOULD_LIMIT, LIMIT_PARAMS
from helpers.paths import CC_OFFSETS_PATH


def cc_audio_offset(audio_file_1, audio_file_2):
    """
    Get the audio offset between two WAV files, uses cross correlation

    Code influenced by https://github.com/rpuntaie/syncstart/blob/main/syncstart.py

    :param audio_file_1:
    :param audio_file_2:
    :return:
    """

    sample_rate_1, audio_1 = wavfile.read(audio_file_1)
    sample_rate_2, audio_2 = wavfile.read(audio_file_2)
    assert sample_rate_1 == sample_rate_2, "Assert that the sample rate of WAV files is equal"

    fs = sample_rate_1  # Sampling frequency

    # If stereo, use one of the channels
    if audio_1.shape[1] == 2:
        audio_1 = audio_1[:, 0]

    # If stereo, use one of the channels
    if audio_2.shape[1] == 2:
        audio_2 = audio_2[:, 0]

    ls1 = len(audio_1)
    ls2 = len(audio_2)
    pad_size = ls1 + ls2 + 1
    pad_size = 2 ** (int(np.log(pad_size) / np.log(2)) + 1)

    s1pad = np.zeros(pad_size)
    s1pad[:ls1] = audio_1
    s2pad = np.zeros(pad_size)
    s2pad[:ls2] = audio_2

    # Calculate the cross correlation
    corr = fft.ifft(fft.fft(s1pad) * np.conj(fft.fft(s2pad)))
    ca = np.absolute(corr)
    x_max = np.argmax(ca)

    if x_max > pad_size // 2:
        file, offset = audio_file_2, (pad_size - x_max) / fs
    else:
        file, offset = audio_file_1, x_max / fs

    return file, offset


def save_cc_offset_to_json(session_dir, views, subject_idx, session_idx, file_path=CC_OFFSETS_PATH):

    offset_results = dict()

    if os.path.exists(file_path):
        offset_results = read_from_json(file_path)
    else:
        offset_results["offsets"] = {}

    name = "sub" + str(subject_idx) + "_sess" + str(session_idx)
    offset_results["offsets"][name] = {}
    offset_results["offsets"][name]["session_dir"] = session_dir
    offset_results["offsets"][name]["views"] = {}

    ref_view_name = None

    for view_idx, view in enumerate(views):

        view_dir = os.path.join(session_dir, view)
        view_dir = view_dir.replace(" ", '\ ')
        view_name = view.split(".")[0]

        if view_idx == 0:
            ref_view_name = view_name

        # Extract audio from a video
        print("Creating '{}' file..".format(view_name + ".wav"))
        cmd_create_wav = "ffmpeg -i {0} -map 0:1 -acodec pcm_s16le -ac 2 {1} -hide_banner -loglevel error".format(view_dir, view_name + ".wav")
        os.system(command=cmd_create_wav)

        # Use the audio to find the offset
        print("Retrieving audio offset between '{}' and '{}'..".format(ref_view_name + ".wav", view_name + ".wav"))
        relative_file, offset = cc_audio_offset(ref_view_name + ".wav", view_name + ".wav")

        relative_file_name = relative_file.split(".")[0]
        view_name = view.split(".")[0]

        offset_results["offsets"][name]["views"][view_name] = {}
        offset_results["offsets"][name]["views"][view_name]["video_name"] = view
        offset_results["offsets"][name]["views"][view_name]["relative_name"] = relative_file_name
        offset_results["offsets"][name]["views"][view_name]["offset_reference"] = ref_view_name
        offset_results["offsets"][name]["views"][view_name]["offset_msec"] = offset

    # Save the result of the session to json
    print("Saving session offsets..")
    write_to_json(offset_results, file_path)

    # Remove the audio file used for the session
    print("Removing audio files..")
    cmd_remove_wav_files = "rm *wav"
    os.system(cmd_remove_wav_files)


def cc_session_sync(session_dir, subject_idx, session_idx, views):
    """
    [FAULTY] Should find the offsets using cross correlation (cc) of the video audio.
    Found that this method is unsuccessful to use on the audio encoded over the FOI dataset.

    There is a method specifically designed for this encoding, use that one instead.

    :param session_dir:
    :param subject_idx:
    :param session_idx:
    :param views:
    :return:
    """
    print("\n-------- Subject {} - Session {} --------".format(subject_idx, session_idx))

    if glob('*wav'):
        # Remove any remaining WAV files in the dir if a previous process was interrupted
        print("Removing audio files..")
        cmd_remove_wav_files = "rm *wav"
        os.system(cmd_remove_wav_files)

    if EXTRACT_OFFSET:
        print("Extracting offsets..")
        save_cc_offset_to_json(session_dir, views, subject_idx, session_idx)

    if SHOULD_DISPLAY:
        display_session(session_dir, subject_idx, session_idx, views)


if __name__ == "__main__":
    cc_audio_offset("above.wav", "back.wav")
