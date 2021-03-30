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

        self.len = element["len"]
        self.shape = element["shape"]


class Joint:
    def __init__(self):
        raise NotImplementedError

class Pose:
    def __init__(self):
        raise NotImplementedError

class Sequence:
    def __init__(self, root_dir, seq):
        seq_info = DatasetElement(seq)
        self.start = int(seq["start"])
        self.end = int(seq["end"])

        file_path = os.path.join(root_dir, seq_info.file_name)
        file_data = read_from_json(file_path)





class FOIKineticPoseDataset(Dataset):
    def __init__(self, json_path, root_dir, sequence_len, transform=None, pose_transform=None):
        # Data loading
        self.json_path = json_path
        self.root_dir = root_dir
        self.sequence_len = sequence_len

        self.lookup = self.__create_lookup()

        self.transform = transform
        self.pose_transform = pose_transform

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            raise NotImplementedError

        seq = Sequence(self.root_dir, self.lookup[idx])


    def __create_lookup(self):
        data_info = read_from_json(self.json_path)

        lookup = []
        for element in data_info:
            element_info = DatasetElement(element)
            seq_info = dict()
            seq_info["file_name"] = element_info.file_name
            seq_info["sub_name"] = element_info.sub_name
            seq_info["sess_name"] = element_info.sess_name
            seq_info["view_name"] = element_info.view_name
            seq_info["seq_id"] = element_info.seq_id

            for i in range(0, element_info.len, self.sequence_len):
                seq_info["start"] = i
                seq_info["end"] = i + self.sequence_len - 1
                lookup.append(seq_info.copy())

            # Fix the last sequence's end frame
            lookup[-1]["end"] = min(lookup[-1]["end"], element_info.len)

        return lookup