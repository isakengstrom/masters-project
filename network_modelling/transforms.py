import numpy as np
import torch
from helpers import read_from_json
from helpers.paths import JOINTS_LOOKUP_PATH


class ToTensor(object):
    """
    Converts a sequence from a Numpy Array to a Torch Tensor
    """
    def __call__(self, item):
        item["sequence"] = torch.from_numpy(item["sequence"])

        return item


class ApplyNoise(object):
    def __init__(self, loc=0.0, scale=0.1):
        self.loc = loc
        self.scale = scale

    def __call__(self, item):
        seq = item["sequence"]

        assert type(seq) is np.ndarray

        seq += np.random.normal(self.loc, self.scale, seq.shape)
        item["sequence"] = seq

        return item


class ChangePoseOrigin(object):
    """
    Changes the origins of the poses in a sequence to the specified joint.

    NOTE:
        Apply this transform before applying 'FilterJoints()', as it otherwise can use the incorrect joint.
    """
    def __init__(self, path=JOINTS_LOOKUP_PATH, origin_name="c_hip"):
        joints_lookup = read_from_json(path, use_dumps=True)
        self.origin_name = origin_name
        self.origin_idx = None

        # Get the openpose index corresponding to the origin_name, this index represent the origin joint.
        for joint in joints_lookup["joints"]:
            if self.origin_name == joint["name"]:
                self.origin_idx = joint["op_idx"]

        assert self.origin_idx is not None

    def __call__(self, item):
        seq = item["sequence"]

        assert type(seq) is np.ndarray

        # Small assertion to make sure origin_idx is in range.
        # Still needs to be applied before 'FilterJoints()' tough!
        assert self.origin_idx <= seq.shape[1]

        # Store the coordinate values of the origins, then tile them to the same dimensions as seq so subtraction
        # can be applied.
        origins = seq[:, self.origin_idx, :]
        origins = np.tile(origins, seq.shape[1]).reshape(seq.shape, order='F')

        # Transform all the coordinates with the origins.
        item["sequence"] = seq - origins

        return item


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
        seq = item["sequence"]

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

        item["sequence"] = normalised
        return item


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
        seq = item["sequence"]
        assert type(seq) is np.ndarray

        # Uses Numpy's extended slicing to return ONLY the indexes saved in the list 'self.filtered_indexes'
        item["sequence"] = seq[:, self.filtered_indexes]

        return item


