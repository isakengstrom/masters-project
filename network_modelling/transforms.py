import numpy as np
import torch
from helpers import read_from_json
from helpers.paths import JOINTS_LOOKUP_PATH


class ToTensor:
    """
    Converts a sequence from a Numpy Array to a Torch Tensor
    """
    def __call__(self, item):
        seq_id, seq = item["id"], item["sequence"]

        return {"id": seq_id, "sequence": torch.from_numpy(seq)}


class ChangePoseOrigin(object):
    def __init__(self):
        raise NotImplementedError


class NormalisePose(object):
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    # Influenced by: https://stats.stackexchange.com/a/281164
    def __call__(self, pose):
        assert type(pose) is np.ndarray

        min_x = min(pose[:, 0])
        max_x = max(pose[:, 0])
        min_y = min(pose[:, 1])
        max_y = max(pose[:, 1])

        max_dist_x = abs(max_x - min_x)
        max_dist_y = abs(max_y - min_y)

        # Normalise to interval [0, 1]
        if max_dist_x > max_dist_y:
            normalised = (pose - min_x) / max_dist_x
        else:
            normalised = (pose - min_y) / max_dist_y

        # Normalise from [0,1] to [self.low, self.high]
        normalised = normalised * (self.high - self.low) + self.low

        return normalised


class FilterPose(object):
    def __init__(self, path=JOINTS_LOOKUP_PATH):
        """
        Initialise the pose filter. It filters out the unwanted joints from the OpenPose data. This is according to
        the "activate_by" and "active_*" parameters in the file 'joints_lookup.json'.

        E.g. if we want to filter according to OpenPose index (op_idx), the parameters are set as follows:
            "activate_by": "op_idx"
            "active_op_idx": [a, b, ...]

            Here, a, b and ... are the OpenPose indexes that want to be KEPT, the rest are discarded

        The indexes that should be KEPT are stored in 'self.filtered_indexes', which is used for every call of the
        filter.

        :param path:
        """
        joints_lookup = read_from_json(path, use_dumps=True)

        active_name = joints_lookup["activate_by"]
        self.filtered_indexes = []

        for joint in joints_lookup["joints"]:
            for trait in joints_lookup["active_" + active_name]:
                if joint[active_name] == trait:
                    self.filtered_indexes.append(joint["op_idx"])

    def __call__(self, pose):
        assert len(pose) == 25
        assert type(pose) is np.ndarray

        # Uses Numpy's extended slicing to return ONLY the indexes saved in the list 'self.filtered_indexes'
        return pose[self.filtered_indexes]


