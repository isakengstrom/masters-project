import torch
import torchvision
from torch.utils.data import Dataset
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


class Sequences:
    def __init__(self, root_dir, sequence_len):
        self.__root_dir = root_dir
        self.__sequence_len = sequence_len

    def __len__(self):
        return self.__sequence_len

    def __call__(self, item):
        seq_info = SequenceElement(item)

        file_path = os.path.join(self.__root_dir, seq_info.file_name)
        file_data = read_from_json(file_path)

        seq_data = file_data[seq_info.start:seq_info.end]

        return np.array(seq_data), seq_info.label


# TODO: get positives and negatives correctly, they are all the same sequence atm
class FOIKineticPoseDataset(Dataset):
    def __init__(self, json_path, root_dir, sequence_len, network_type="triplet", data_limiter=None, transform=None):
        # Data loading
        self.json_path = json_path
        #self.root_dir = root_dir
        self.sequence_len = sequence_len
        self.network_type = network_type
        self.data_limiter = data_limiter
        self.transform = transform

        self.sequences = Sequences(root_dir, sequence_len)
        self.lookup = self.__create_lookup()

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            raise TypeError

        if self.network_type == "siamese":
            item = self.__get_siamese(idx)
        elif self.network_type == "triplet":
            item = self.__get_triplet(idx)
        else:
            item = self.__get_single(idx)

        return item

    def __get_single(self, idx):
        sequence, label = self.sequences(self.lookup[idx])

        if self.transform:
            sequence = self.transform(sequence)

        return sequence, label

    def __get_siamese(self, idx):
        positive_sequence, positive_label = self.sequences(self.lookup[idx])
        negative_sequence, _ = self.sequences(self.lookup[idx])

        if self.transform:
            positive_sequence = self.transform(positive_sequence)
            negative_sequence = self.transform(negative_sequence)

        return positive_sequence, negative_sequence, positive_label

    def __get_triplet(self, idx):
        anchor_sequence, anchor_label = self.sequences(self.lookup[idx])
        positive_sequence, _ = self.sequences(self.lookup[idx])
        negative_sequence, _ = self.sequences(self.lookup[idx])

        if self.transform:
            anchor_sequence = self.transform(anchor_sequence)
            positive_sequence = self.transform(positive_sequence)
            negative_sequence = self.transform(negative_sequence)

        return anchor_sequence, positive_sequence, negative_sequence, anchor_label

    def __create_lookup(self):
        data_info = read_from_json(self.json_path)

        lookup = []
        for element in data_info:
            element_info = DimensionsElement(element)

            # Skip adding a sequence to the lookup if it is not specified in the data_limiter.
            if self.__should_skip_sequence(element_info):
                continue

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

    def __should_skip_sequence(self, info):
        if self.data_limiter is None:
            return False

        if not isinstance(self.data_limiter, dict):
            print("The data limiter is not a dict or 'None'")
            raise TypeError

        def contains(limit, element):
            if self.data_limiter[limit] is None:
                return True

            if element in self.data_limiter[limit]:
                return True

            return False

        if not contains("subjects", info.sub):
            return True

        if not contains("sessions", info.sess):
            return True

        if not contains("views", info.view):
            return True

