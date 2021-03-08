"""
http://www.dsg-bielefeld.de/dsg_wp/wp-content/uploads/2014/10/video_syncing_fun.pdf
"""

import os
import subprocess
import glob
from scipy import fft
from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt


from pose_extraction.extraction_config import TRIMMED_SEQUENCE_FLAG
from modelling_config import DATASET_PATH


def audio_offset(audio_file_1, audio_file_2):
    """
    Get the audio offset between two WAV files

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

    #print("x_max {} other {}".format(x_max, pad_size//2))

    if x_max > pad_size // 2:
        file, offset = audio_file_2, (pad_size - x_max) / fs
    else:
        file, offset = audio_file_1, x_max / fs

    #print("file: " + str(file) + "  offset: " + str(offset))

    return file, offset

def synchronise_sequence(sequence_dir, sequence_angles):
    """"""

    print(sequence_dir + ",  " + str(sequence_angles))
    ref_clip_idx = 0
    ref_clip = os.path.join(sequence_dir, sequence_angles[ref_clip_idx])
    ref_clip = ref_clip.replace(" ", '\ ')
    print(sequence_angles)
    sequence_angles.pop(ref_clip_idx)
    print(sequence_angles)
    cmd_create_ref_wav = "ffmpeg -i {0} -map 0:1 -acodec pcm_s16le -ac 2 {1}".format(ref_clip, "ref.wav")
    os.system(command=cmd_create_ref_wav)

    results = []
    results.append((ref_clip, 0))
    for angle in sequence_angles:
        angle_dir = os.path.join(sequence_dir, angle)
        angle_dir = angle_dir.replace(" ", '\ ')
        angle_audio_name = angle.split(".")[0] + ".wav"

        cmd_create_wav = "ffmpeg -i {0} -map 0:1 -acodec pcm_s16le -ac 2 {1}".format(angle_dir, angle_audio_name)
        os.system(command=cmd_create_wav)

        file, offset = audio_offset("ref.wav", angle_audio_name)
        results.append((file, offset))

        '''
        cmd_find_audio_offset = "Praat crosscorrelate.praat ref.wav {}".format(angle_audio_name)
        result = subprocess.check_output(cmd_find_audio_offset, shell=True)
        results.append((angle, result.split('\n')[0]))
        '''

    cmd_remove_wav_files = "rm *wav"
    os.system(cmd_remove_wav_files)

    #for result in results:


    '''
    for result in results:
        print("File: {}, offset: {}".format(result[0], result[1]))
    '''

def visit_sequence(root_dir):
    _, subject_names, _ = next(os.walk(root_dir))

    for subject_idx in range(len(sorted(subject_names))):
        subject_dir = os.path.join(root_dir, subject_names[subject_idx])
        if not os.path.exists(subject_dir):
            return

        # Get the sequence names (child folder names of a subject)
        _, sequence_names, _ = next(os.walk(subject_dir))

        # Remove un-trimmed sequences before extraction if they have trimmed counterparts
        # Some sequences are trimmed in the beginning and end to remove frames containing more than one individual
        for sequence_name in sequence_names:
            if TRIMMED_SEQUENCE_FLAG in sequence_name:
                deprecate_sequence_name = sequence_name.replace(TRIMMED_SEQUENCE_FLAG, "")
                sequence_names.remove(deprecate_sequence_name)

        for sequence_idx in range(len(sorted(sequence_names))):
            sequence_dir = os.path.join(subject_dir, sequence_names[sequence_idx])
            if not os.path.exists(sequence_dir):
                return

            # Get the angle names (child file names of a sequence)
            _, _, camera_angles = next(os.walk(sequence_dir))

            synchronise_sequence(sequence_dir, camera_angles)

            '''
            print("Subject idx: " + str(subject_idx) + ", sequence idx: " + str(sequence_idx) + ", angles: " + str(camera_angles))

            for angle_idx in range(len(sorted(camera_angles))):

                path = os.path.join(sequence_dir, camera_angles[angle_idx])
            '''
        break


if __name__ == "__main__":
    visit_sequence(DATASET_PATH)
    #audio_offset("ref.wav", "back.wav")
    #audio_offset("back.wav", "back.wav")
    #audio_offset("back.wav", "ref.wav")
    #audio_offset("ref.wav", "front.wav")