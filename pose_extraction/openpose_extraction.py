"""
This file contains the code for pose extraction using OpenPose.
The code is heavily inspired by the Python examples that OpenPose provides:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/tutorial_api_python
"""

import sys
import cv2  # OpenCV installed for python
import os
import time
import numpy as np
from sys import platform

from extraction_config import DEV, DEV_PARAMS
from extraction_config import get_openpose_params


def extract_poses(media_path=None, media_type='video', should_extract=True, should_display=True):
    """

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

        # TODO: decide when pose should be extracted, this seems to work
        # Check if the pose should be extracted
        def is_extractable():
            return should_extract and datum.poseKeypoints.size > 1 #and datum.poseKeypoints.size == 75

        extraction_status = 'successful'

        if media_type == 'image':
            frame = cv2.imread(media_path)
            datum.cvInputData = frame
            op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

            if is_extractable():
                extracted_poses.append(datum.poseKeypoints)

            if should_display:
                cv2.imshow("OpenPose 1.7.0 - Single Image", datum.cvOutputData)
                cv2.waitKey(0)

        elif media_type == 'video':
            frame_idx = 0
            stream = cv2.VideoCapture(media_path)
            while stream.isOpened():
                has_frame, frame = stream.read()
                if has_frame:
                    if DEV:
                        frame_idx += 1
                        if frame_idx < DEV_PARAMS["frame_lower_lim"]:
                            continue
                        if frame_idx > DEV_PARAMS["frame_upper_lim"]:
                            break

                    datum.cvInputData = frame
                    op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

                    if is_extractable():
                        extracted_poses.append(datum.poseKeypoints)


                    if should_display:
                        cv2.imshow("OpenPose 1.7.0 - Video Stream", datum.cvOutputData)
                        key = cv2.waitKey(1)

                        # Press "Esc", 'q' or 'Q' to exit stream
                        if key == 27 or key == ord('q') or key == ord('Q'):
                            extraction_status = 'interrupted by user'
                            break
                else:
                    break

        elif media_type == 'images':
            # Read frames on directory
            image_paths = op.get_images_on_directory(media_path)

            # Process and display images
            for image_path in image_paths:
                frame = cv2.imread(image_path)
                datum.cvInputData = frame
                op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

                if is_extractable():
                    extracted_poses.append(datum.poseKeypoints)

                if should_display:
                    cv2.imshow("OpenPose 1.7.0 - Multiple Images", datum.cvOutputData)
                    key = cv2.waitKey(15)

                    # Press "Esc", 'q' or 'Q' to exit stream
                    if key == 27 or key == ord('q') or key == ord('Q'):
                        extraction_status = 'interrupted by user'
                        break

        end = time.time()
        print('Pose extraction of {} was {}. Run time: {:.2f} seconds'.format(media_type, extraction_status, (end - start)))

        return extracted_poses

    except Exception as e:
        print(e)
        sys.exit(-1)


if __name__ == "__main__":
    #extract_poses()
    #extract_poses("/home/isaeng/Exjobb/media/mini.jpg", 'image')
    extract_poses("/home/isaeng/Exjobb/media/tompa_flip_0_25.MOV", 'video')
    #extract_poses("/home/isaeng/Exjobb/media/front.mp4", 'video')
    #extract_poses("/home/isaeng/Exjobb/media/dir", 'images')
    #extract_poses(media_path=os.environ['DATASET_DIR'] + "/VIDEO/SUBJECT_0/SEQ_0/skew.MTS", media_type='video')
