import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
#import pandas as pd
import os

from helpers import read_from_json


class DatasetElement:
    def __init__(self, element):
        self.file_name = element["file_name"]
        self.sub_name = element["sub_name"]
        self.sess_name = element["sess_name"]
        self.view_name = element["view_name"]

        self.sub = int(self.sub_name[-1])
        self.sess = int(self.sess_name[-1])
        self.view = int(self.view_name[-1])

        self.identifier = "s{}s{}v{}".format(self.sub, self.sess, self.view)

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


class FOIKineticPoseDataset(Dataset):
    def __init__(self, json_path, root_dir, sequence_len, transform=None):
        # Data loading
        self.json_path = json_path
        self.sequence_len = sequence_len

        self.lookup = self._create_lookup()

        #print(SeqElement(self.lookup[1000]).identifier)

        self.root_dir = root_dir
        self.transform = transform

    def _create_lookup(self):
        data_info = read_from_json(self.json_path)

        lookup = []
        for element in data_info:
            el = InfoElement(element)
            # print("File name {}, len {}, shape {}".format(el.file_name, el.len, el.shape))

            seq_info = dict()
            seq_info["file_name"] = el.file_name
            seq_info["sub_name"] = el.sub_name
            seq_info["sess_name"] = el.sess_name
            seq_info["view_name"] = el.view_name
            seq_info["identifier"] = el.identifier

            for i in range(0, el.len, self.sequence_len):
                seq_info["start"] = i
                seq_info["end"] = i + self.sequence_len - 1
                lookup.append(seq_info.copy())

            # Fix the last sequence's end frame
            lookup[-1]["end"] = min(lookup[-1]["end"], el.len)

        return lookup

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq_name = os.path.join(self.root_dir, self.lookup.iloc[idx, 0])

        print("Get item")





