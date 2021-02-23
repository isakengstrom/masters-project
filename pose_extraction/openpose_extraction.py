"""
This file contains the code for pose extraction using OpenPose.
The code is heavily inspired by the examples from the Python examples that OpenPose provides:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/tutorial_api_python
"""

import sys
import cv2  # OpenCV installed for python
import os
from sys import platform
import argparse
import time


def get_openpose_params():
    """
    get the OpenPose flags/parameters

    A list over the main flags can be found here:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md#main-flags

    A full list of flags can be found here:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/advanced/demo_advanced.md#all-flags

    :return: params: Dict[str, Union[str, bool]] = dict()
    """

    params = dict()
    params["model_folder"] = os.environ['OPENPOSE_DIR'] + "/models"
    params["disable_blending"] = False
    params["display"] = 0

    params["num_gpu"] = -1
    params["num_gpu_start"] = 0

    #params["output_resolution"] = "-1x-1"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.25
    params["scale_number"] = 1
    params["render_threshold"] = 0.05

    params["number_people_max"] = -1

    # params for body keypoints
    params["model_pose"] = "BODY_25"  # "BODY_25", "COCO", "MPI"
    params["net_resolution"] = "-1x320"  # Lower res needed for COCO and MPI?

    # params for face keypoints
    #params["face"] = False
    #params["face_net_resolution"] = "368x368"

    # params for hand keypoints
    #params["hand"] = False
    #params["hand_net_resolution"] = "368x368"

    return params


def extract_keypoints(media_path=None, media_type='image', should_extract=False):
    """

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

        # Flags
        parser = argparse.ArgumentParser()

        default_media_path = media_path

        if media_path is None:
            default_media_path = os.environ['OPENPOSE_DIR'] + "/examples/media/COCO_val2014_000000000241.jpg"

        parser.add_argument("--media_path", default=default_media_path, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        parser.add_argument("--should_display", default=True, help="Disable to not display visually.")

        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = get_openpose_params()

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        # Starting OpenPose
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(params)
        op_wrapper.start()

        extraction_status = 'successful'

        start = time.time()
        datum = op.Datum()
        extracted_keypoints = []

        # Check if the pose should be extracted
        def is_extractable():
            return should_extract and datum.poseKeypoints.size > 1 and datum.poseKeypoints.size == 75

        if media_type == 'image':
            # Process Image
            frame = cv2.imread(args[0].media_path)
            datum.cvInputData = frame
            op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

            if is_extractable:
                extracted_keypoints.append(datum.poseKeypoints)
                #print(datum.poseKeypoints.size)
                #print("Body keypoints: \n" + str(datum.poseKeypoints))

            if args[0].should_display:
                cv2.imshow("OpenPose 1.7.0 - Single Image", datum.cvOutputData)
                cv2.waitKey(0)

        elif media_type == 'video':
            stream = cv2.VideoCapture(media_path)
            while stream.isOpened():
                has_frame, frame = stream.read()
                if has_frame:
                    datum.cvInputData = frame
                    op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

                    if is_extractable:
                        extracted_keypoints.append(datum.poseKeypoints)

                    if args[0].should_display:
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
            image_paths = op.get_images_on_directory(args[0].media_path)

            # Process and display images
            for image_path in image_paths:
                frame = cv2.imread(image_path)
                datum.cvInputData = frame
                op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

                if is_extractable:
                    extracted_keypoints.append(datum.poseKeypoints)

                if args[0].should_display:
                    cv2.imshow("OpenPose 1.7.0 - Multiple Images", datum.cvOutputData)
                    key = cv2.waitKey(15)

                    # Press "Esc", 'q' or 'Q' to exit stream
                    if key == 27 or key == ord('q') or key == ord('Q'):
                        extraction_status = 'interrupted by user'
                        break

        end = time.time()
        print('Pose extraction of {} was {}. Run time: {:.2f} seconds'.format(media_type, extraction_status, (end - start)))

        return extracted_keypoints

    except Exception as e:
        print(e)
        sys.exit(-1)


if __name__ == "__main__":
    #extract_keypoints()
    #extract_keypoints("/home/isaeng/Exjobb/media/mini.jpg", 'image')
    #extract_keypoints("/home/isaeng/Exjobb/media/tompa_flip_0_25.MOV", 'video')
    #extract_keypoints("/home/isaeng/Exjobb/media/front.mp4", 'video')
    #extract_keypoints("/home/isaeng/Exjobb/media/dir", 'images')
    extract_keypoints(os.environ['DATASET_DIR'] + "/VIDEO/SUBJECT_0/SEQ_0/skew.MTS", 'video')