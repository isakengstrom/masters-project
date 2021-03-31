import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from helpers import read_from_json
from helpers.paths import JOINTS_LOOKUP_PATH


class DatasetElement:
    def __init__(self, element):
        self.file_name = element["file_name"]
        self.sub_name = element["sub_name"]
        self.sess_name = element["sess_name"]
        self.view_name = element["view_name"]

        self.sub = int(self.sub_name[-1])
        self.sess = int(self.sess_name[-1])
        self.view = int(self.view_name[-1])

        self.seq_id = "s{}s{}v{}".format(self.sub, self.sess, self.view)


class InfoElement(DatasetElement):
    def __init__(self, element):
        super().__init__(element)

        self.len = element["len"]
        self.shape = element["shape"]


class SeqElement(DatasetElement):
    def __init__(self, element):
        super().__init__(element)

        self.start = int(element["start"])
        self.end = int(element["end"])


'''
class ToTensor(object):
    raise NotImplementedError

class ChangePoseOrigin(object):
    raise NotImplementedError
'''


class Joint:
    def __init__(self, op_idx, name, coords):
        self.op_idx = op_idx
        self.name = name
        self.coords = coords
        self.x = self.coords[0]
        self.y = self.coords[1]


class FilterPose:
    """
    Transform to filter out the unwanted Joints
    """
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
        # print(joints[self.filtered_indexes])
        print("entered call")
        return pose[self.filtered_indexes]


class NormalisePose(object):


    def __call__(self, item):
        seq_id, keypoints = item["id"], item["keypoints"]

        print(keypoints[0,:,:].shape)



        return {"id": seq_id, "keypoints": keypoints}


# TODO: make sure this is called correctly, might need __call__ and separate pose and transform from init
#  needs to be called from FOIDataset Pose yadiyada..
class Pose:
    def __init__(self, pose, transform=None, path=JOINTS_LOOKUP_PATH):
        self.transform = transform

        joints_lookup = read_from_json(path, use_dumps=True)
        joints_info = joints_lookup["joints"]
        self.joints = []

        #print(pose)
        for idx, coords in enumerate(pose):
            #print(coords[0])
            op_idx = joints_info[idx]["op_idx"]
            name = joints_info[idx]["name"]

            joint = Joint(op_idx, name, coords)
            self.joints.append(joint)

    def __len__(self):
        return len(self.joints)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            return Pose(self.joints[idx])

        item = self.joints
        #if self.transform:
        #    print("run transforms")
        #    item = self.transform(item)

        #print(item)
        return item

'''
class Sequence:
    def __init__(self, sequence_len):
        self.sequence_len = sequence_len
        
        self.poses = []

    def __len__(self):
        return self.sequence_len

    def __call__(self, poses):
        self.poses = []
        for pose in poses:
            self.poses.append(Pose(pose))
'''


class FOIKineticPoseDataset(Dataset):
    def __init__(self, json_path, root_dir, sequence_len, transform=None, pose_transform=None):
        # Data loading
        self.json_path = json_path
        self.root_dir = root_dir
        self.sequence_len = sequence_len

        self.lookup = self.__create_lookup()

        #print(SeqElement(self.lookup[1000]).id)

        self.transform = transform

        self.pose_transform = pose_transform

        self.poses = []

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            raise NotImplementedError

        se = SeqElement(self.lookup[idx])

        file_path = os.path.join(self.root_dir, se.file_name)
        file_data = read_from_json(file_path)

        poses = file_data[se.start:se.end]
        poses = np.array(poses)

        self.poses = []
        for pose in poses:
            print(pose.shape)
            self.poses.append(Pose(pose, self.pose_transform))

        print(self.poses)
        item = {"id": se.seq_id, 'poses': self.poses}

        if self.transform:
            item = self.transform(item)

        return item

    def __create_lookup(self):
        data_info = read_from_json(self.json_path)

        lookup = []
        for element in data_info:
            el = InfoElement(element)
            seq_info = dict()
            seq_info["file_name"] = el.file_name
            seq_info["sub_name"] = el.sub_name
            seq_info["sess_name"] = el.sess_name
            seq_info["view_name"] = el.view_name
            seq_info["id"] = el.seq_id

            for i in range(0, el.len, self.sequence_len):
                seq_info["start"] = i
                seq_info["end"] = i + self.sequence_len - 1
                lookup.append(seq_info.copy())

            # Fix the last sequence's end frame
            lookup[-1]["end"] = min(lookup[-1]["end"], el.len)

        return lookup



