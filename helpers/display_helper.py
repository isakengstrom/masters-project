import cv2
import os
from helpers import read_from_json, LIMIT_PARAMS, SHOULD_LIMIT
from helpers.paths import CC_OFFSETS_PATH
from data_processing.pre_processing.pre_config import USE_OFFSET, FIX_BACK_CAMERA

def draw_label(img, text, pos=(20, 20), bg_color=(200, 200, 200)):
    """
    Code from: https://stackoverflow.com/a/54616857/15354710
    """
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def display_session(session_dir, subject_idx, session_idx, views):
    offsets_data = read_from_json(CC_OFFSETS_PATH)
    print(offsets_data)

    session_paths = []
    for view in views:
        session_paths.append(os.path.join(session_dir, view))

    streams = []
    starting_frames = []
    starting_msec = []
    for view_idx in range(len(session_paths)):
        stream = cv2.VideoCapture(session_paths[view_idx])
        view_name = views[view_idx].split("/")[-1].split(".")[0]

        if USE_OFFSET:
            name = "sub" + str(subject_idx) + "_sess" + str(session_idx)
            offset = offsets_data["offsets"][name]["views"][view_name]["offset_msec"]
            stream.set(cv2.CAP_PROP_POS_MSEC, offset)
        elif FIX_BACK_CAMERA and view_name == "back":
            offset = 100
            stream.set(cv2.CAP_PROP_POS_FRAMES, offset)

        if SHOULD_LIMIT and LIMIT_PARAMS["frame_lower_lim"] is not None and LIMIT_PARAMS["frame_lower_lim"] >= 0:
            stream.set(cv2.CAP_PROP_POS_FRAMES, LIMIT_PARAMS["frame_lower_lim"])

        starting_frames.append(int(stream.get(cv2.CAP_PROP_POS_FRAMES)))
        starting_msec.append(stream.get(cv2.CAP_PROP_POS_MSEC))
        streams.append(stream)

    while True:

        rets = []
        imgs = []
        dims = []

        for stream_idx, stream in enumerate(streams):
            if not stream.isOpened():
                break

            ret, img = stream.read()

            if not ret:
                continue
            rets.append(ret)

            if stream_idx == 0:
                scale_percent = 50
            else:
                scale_percent = 25

            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            img = cv2.resize(img, dim)

            labels = [
                "View: {}".format(views[stream_idx]),
                "FPS: {}".format(stream.get(cv2.CAP_PROP_FPS)),
                "Start frame: {}".format(starting_frames[stream_idx]),
                "Curr frame {}".format(int(stream.get(cv2.CAP_PROP_POS_FRAMES))),
                "Start ms: {:.1f}".format(starting_msec[stream_idx]),
                "Curr ms: {:.0f}".format(stream.get(cv2.CAP_PROP_POS_MSEC)),
            ]

            for i, label in enumerate(labels):
                draw_label(img, text=label, pos=(20, 20*(i+1)))

            if stream_idx == 0:
                draw_label(img, text="- REF VIEW", pos=(130, 20))

            imgs.append(img)
            dims.append(dim)

        if not any(rets):
            print("Could not read from cameras")
            break

        row1 = cv2.hconcat([imgs[1], imgs[2]])
        row2 = cv2.hconcat([imgs[3], imgs[4]])
        col1_col2 = cv2.vconcat([row1, row2])
        col1_col2_col3 = cv2.vconcat([col1_col2, imgs[0]])

        cv2.imshow("Subject {} - Session {}".format(subject_idx, session_idx), col1_col2_col3)
        key = cv2.waitKey(1)

        # Press "Esc", 'q' or 'Q' to exit stream
        if key == 27 or key == ord('q') or key == ord('Q'):
            break

    for stream in streams:
        stream.release()

    cv2.destroyAllWindows()
