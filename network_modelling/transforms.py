
from helpers import read_from_json
from helpers.paths import JOINTS_LOOKUP_PATH

'''
class ToTensor(object):
    raise NotImplementedError


class NormalisePose(object):
    raise NotImplementedError
'''


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

        return pose[self.filtered_indexes]

