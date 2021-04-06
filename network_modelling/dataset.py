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

        self.label = self.sub
        self.key = "s{}s{}v{}".format(self.sub, self.sess, self.view)


class DimensionsElement(DatasetElement):
    def __init__(self, element):
        super().__init__(element)

        self.len = element["len"]
        self.shape = element["shape"]


class SequenceElement(DatasetElement):
    def __init__(self, element):
        super().__init__(element)

        self.start = int(element["start"])
        self.end = int(element["end"])


class Joint:
    def __init__(self, op_idx, name, coords):
        self.op_idx = op_idx
        self.name = name
        self.coords = coords
        self.x, self.y = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx]


class Pose:
    def __init__(self, pose, path=JOINTS_LOOKUP_PATH):
        joints_lookup = read_from_json(path, use_dumps=True)
        joints_info = joints_lookup["joints"]

        joints = []
        for idx, coords in enumerate(pose):
            op_idx = joints_info[idx]["op_idx"]
            name = joints_info[idx]["name"]

            joint = Joint(op_idx, name, coords)
            joints.append(joint)

        self.joints = joints

    def __len__(self):
        return len(self.joints)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.joints[idx]

        return item


class Sequence:
    def __init__(self, root_dir, seq, sequence_len):
        seq_info = SequenceElement(seq)

        self.__sequence_len = sequence_len
        self.id = seq_info.key
        self.label = seq_info.label

        file_path = os.path.join(root_dir, seq_info.file_name)
        file_data = read_from_json(file_path)

        seq_data = file_data[seq_info.start:seq_info.end]
        seq_data = np.array(seq_data)

        self.poses = np.array(seq_data)

    def __len__(self):
        return self.__sequence_len

    def __getitem__(self, idx):
        return self.poses[idx]


class FOIKineticPoseDataset(Dataset):
    def __init__(self, json_path, root_dir, sequence_len, transform=None):
        # Data loading
        self.json_path = json_path
        self.root_dir = root_dir
        self.sequence_len = sequence_len
        self.transform = transform

        self.lookup = self.__create_lookup()

        print("Init dataset")

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            raise NotImplementedError

        seq = Sequence(self.root_dir, self.lookup[idx], self.sequence_len)

        item = {"seq_idx": idx, "key": seq.id, "label": seq.label, "sequence": seq.poses}

        if self.transform:
            item = self.transform(item)

        return item

    def __create_lookup(self):
        data_info = read_from_json(self.json_path)

        lookup = []
        for element in data_info:
            element_info = DimensionsElement(element)
            seq_info = dict()
            seq_info["file_name"] = element_info.file_name
            seq_info["sub_name"] = element_info.sub_name
            seq_info["sess_name"] = element_info.sess_name
            seq_info["view_name"] = element_info.view_name
            seq_info["key"] = element_info.key

            for i in range(0, element_info.len, self.sequence_len):
                seq_info["start"] = i
                seq_info["end"] = i + self.sequence_len - 1
                lookup.append(seq_info.copy())

            # Fix the last sequence's end frame
            lookup[-1]["end"] = min(lookup[-1]["end"], element_info.len)

        return lookup