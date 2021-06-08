"""
This file contains the code for pose extraction using OpenPose.
The code is heavily inspired by the Python examples that OpenPose provides:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/tutorial_api_python
"""

import sys
import cv2  # OpenCV installed for python
import os
import time
from sys import platform

from .extraction_config import get_openpose_params
from helpers import SHOULD_LIMIT, LIMIT_PARAMS, upper_lim_check


def extract_poses(media_path=None, media_type='video', should_extract=True, should_display=True):
    """
    Extract the poses of a media file or directory using OpenPose

    :param should_display:
    :param should_extract:
    :param media_type:
    :param media_path:
    :return:
    """
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            # Windows
            if platform == "win32":
                sys.path.append(dir_path + '/../../python/openpose/Release')
                os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            # Ubuntu/OSX
            else:
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu)

                sys.path.append(os.environ['OPENPOSE_DIR'] + '/build/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Default media path
        if media_path is None:
            media_path = os.environ['OPENPOSE_DIR'] + "/examples/media/video.avi"

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        # Starting OpenPose
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(get_openpose_params())
        op_wrapper.start()

        start = time.time()
        datum = op.Datum()
        extracted_poses = []

        # Check if the pose should be extracted
        def extractable():
            return should_extract and datum.poseKeypoints is not None and datum.poseKeypoints.size > 1

        extraction_status = 'successful'

        if media_type == 'video':
            stream = cv2.VideoCapture(media_path)

            if SHOULD_LIMIT and LIMIT_PARAMS["frame_lower_lim"] is not None and LIMIT_PARAMS["frame_lower_lim"] >= 0:
                stream.set(cv2.CAP_PROP_POS_FRAMES, LIMIT_PARAMS["frame_lower_lim"])

            frame_idx = int(stream.get(cv2.CAP_PROP_POS_FRAMES))

            if SHOULD_LIMIT and LIMIT_PARAMS["frame_upper_lim"] is not None:
                total_frames = min(int(stream.get(cv2.CAP_PROP_FRAME_COUNT)), LIMIT_PARAMS["frame_upper_lim"])
            else:
                total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

            while stream.isOpened():
                has_frame, frame = stream.read()

                if not has_frame:
                    break

                if SHOULD_LIMIT and upper_lim_check(frame_idx, "frame"):
                    break

                datum.cvInputData = frame
                op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

                if extractable():
                    extracted_poses.append(datum.poseKeypoints)

                if should_display:
                    cv2.imshow("OpenPose 1.7.0 - Video Stream", datum.cvOutputData)
                    key = cv2.waitKey(1)  # 1 for continuous stream, 0 for stills

                    # Press "Esc", 'q' or 'Q' to exit stream
                    if key == 27 or key == ord('q') or key == ord('Q'):
                        extraction_status = 'interrupted by user'
                        break

                if frame_idx % 250 == 0:
                    print('\r', "Progress: {:.1f}%, at frame {} / {}".format(frame_idx / total_frames * 100, frame_idx, total_frames), end='')

                frame_idx += 1

        elif media_type == 'image':
            frame = cv2.imread(media_path)
            datum.cvInputData = frame
            op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

            if extractable():
                extracted_poses.append(datum.poseKeypoints)

            if should_display:
                cv2.imshow("OpenPose 1.7.0 - Single Image", datum.cvOutputData)
                cv2.waitKey(0)

        elif media_type == 'images':
            # Read frames on directory
            image_paths = op.get_images_on_directory(media_path)

            # Process and display images
            for image_path in image_paths:
                frame = cv2.imread(image_path)
                datum.cvInputData = frame
                op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

                if extractable():
                    extracted_poses.append(datum.poseKeypoints)

                if should_display:
                    cv2.imshow("OpenPose 1.7.0 - Multiple Images", datum.cvOutputData)
                    key = cv2.waitKey(15)

                    # Press "Esc", 'q' or 'Q' to exit stream
                    if key == 27 or key == ord('q') or key == ord('Q'):
                        extraction_status = 'interrupted by user'
                        break

        end = time.time()
        print('\nPose extraction of {} was {}. Run time: {:.2f} seconds'.format(media_type, extraction_status, (end - start)))

        return extracted_poses

    except Exception as e:
        print(e)
        sys.exit(-1)


if __name__ == "__main__":
    #extract_poses()
    #extract_poses("/home/isaeng/Exjobb/media/mini.jpg", 'image')
    #extract_poses("/home/isaeng/Exjobb/media/tompa_flip_0_25.MOV", 'video')
    extract_poses("/home/isaeng/Exjobb/images/walking/sub0b_sess0.png", 'image')
    #extract_poses("/home/isaeng/Exjobb/media/dir", 'images')
    #extract_poses(media_path=os.environ['DATASET_DIR'] + "/VIDEO/SUBJECT_0/SEQ_0/skew.MTS", media_type='video')
