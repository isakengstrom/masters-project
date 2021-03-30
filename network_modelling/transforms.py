import numpy as np

from helpers import read_from_json
from helpers.paths import JOINTS_LOOKUP_PATH




#class ToTensor:



class NormalisePose(object):
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    def __call__(self, pose):

        pose = np.array(pose)

        min_x = min(pose[:, 0])
        max_x = max(pose[:, 0])
        min_y = min(pose[:, 1])
        max_y = max(pose[:, 1])

        max_dist_x = abs(max_x - min_x)
        max_dist_y = abs(max_y - min_y)

        if max_dist_x > max_dist_y:
            normalised = (pose - min_x) / max_dist_x
        else:
            normalised = (pose - min_y) / max_dist_y

        normalised = normalised * (self.high - self.low) + self.low

        return normalised


class FilterPose(object):
    def __init__(self, path=JOINTS_LOOKUP_PATH):
        joints_lookup = read_from_json(path, use_dumps=True)

        active_name = joints_lookup["activate_by"]
        filtered_indexes = []
        for joint in joints_lookup["joints"]:
            for trait in joints_lookup["active_" + active_name]:
                if joint[active_name] == trait:
                    filtered_indexes.append(joint["op_idx"])
        self.filtered_indexes = filtered_indexes
        print("filter init")

    def __call__(self, pose):
        assert len(pose) == 25
        pose = np.array(pose)
        pose = pose[self.filtered_indexes]
        return pose.tolist()

