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
        print("Init ChangePoseOrigin")


class NormalisePoses(object):
    """
    Transform which normalises the poses in a sequence in a given range, default being [0, 1].
    The normalisation is applied locally on each pose, compared to applying it globally to a sequence.
    """

    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    # Influenced by: https://stats.stackexchange.com/a/281164
    def __call__(self, item):
        seq_id, seq = item["id"], item["sequence"]
        assert type(seq) is np.ndarray

        # Get the min and max of the x and y coordinates separately for each pose in the sequence.
        # The vars are of shape (sequence_len,)
        min_x = np.amin(seq[:, :, 0], axis=1)
        max_x = np.amax(seq[:, :, 0], axis=1)
        min_y = np.amin(seq[:, :, 1], axis=1)
        max_y = np.amax(seq[:, :, 1], axis=1)

        # Calculate the distance for each dimension in which the coordinates lie.
        dist_x = np.abs(max_x - min_x)
        dist_y = np.abs(max_y - min_y)

        # Pick the min and distance to use for each pose in the sequence depending on the condition.
        # The condition checks which distance is larger of the x and y dimension.
        # It should normalise along the largest dimensional distance.
        condition = dist_x > dist_y
        seq_mins = np.where(condition, min_x, min_y)
        seq_dists = np.where(condition, dist_x, dist_y)

        # Tiles the vars from shape (seq_len,) to (seq_len, num_openpose_joints, coords), most likely (seq_len, 25, 2)
        # The values are unchanged, but need to be broadcast to the same dimensions as the input sequence for the
        # next step.
        tile_len = seq.shape[1] * seq.shape[2]
        seq_mins = np.tile(seq_mins, tile_len).reshape(seq.shape, order='F')
        seq_dists = np.tile(seq_dists, tile_len).reshape(seq.shape, order='F')

        # Normalise to interval [0, 1]
        normalised = (seq - seq_mins) / seq_dists

        # Normalise from [0,1] to [self.low, self.high]
        normalised = normalised * (self.high - self.low) + self.low

        return {"id": seq_id, "sequence": normalised}


class FilterJoints(object):
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

    def __call__(self, item):
        seq_id, seq = item["id"], item["sequence"]
        assert type(seq) is np.ndarray

        # Uses Numpy's extended slicing to return ONLY the indexes saved in the list 'self.filtered_indexes'
        return {"id": seq_id, "sequence": seq[:, self.filtered_indexes]}


