import torch
from torch.utils.data import Dataset
import numpy as np
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


class DataLimiter:
    """
    Used to specify which sequences to extract from the dataset.
    Values can either be 'None' or a list of indices.

    If 'None', don't limit that parameter, e.g.
        "subjects": None, "sessions": None, "views": None
        will get all sequences, from s0_s0_v0 to s9_s0_v4

    If indices, get the corresponding sequences, e.g.
        "subjects": [0], "sessions": [0,1], "views": [0,1,2]
        will get s0_s0_v0, s0_s0_v1, s0_s0_v2, s0_s1_v0, s0_s1_v1, s0_s1_v2
    """

    def __init__(self, subjects: list = None, sessions: list = None, views: list = None):
        self.subjects = subjects
        self.sessions = sessions
        self.views = views

    def skip_sequence(self, info: DimensionsElement) -> bool:

        if self.subjects is not None and info.sub not in self.subjects:
            return True

        if self.sessions is not None and info.sess not in self.sessions:
            return True

        if self.views is not None and info.view not in self.views:
            return True


class Sequences:
    def __init__(self, root_dir, sequence_len):
        self.__root_dir = root_dir
        self.__sequence_len = sequence_len

    def __len__(self):
        return self.__sequence_len

    def __call__(self, item: dict) -> tuple:
        seq_info = SequenceElement(item)

        file_path = os.path.join(self.__root_dir, seq_info.file_name)
        file_data = read_from_json(file_path)

        seq_data = file_data[seq_info.start:seq_info.end]

        return np.array(seq_data), seq_info.label


# TODO: get positives and negatives correctly, they are all the same sequence atm
class FOIKineticPoseDataset(Dataset):
    def __init__(self, json_path, root_dir, sequence_len, is_train=False, network_type="triplet", data_limiter=None, transform=None):
        # Data loading
        self.json_path = json_path
        #self.root_dir = root_dir
        self.sequence_len = sequence_len
        self.network_type = network_type
        self.data_limiter = data_limiter
        self.transform = transform
        self.is_train = is_train

        self.sequences = Sequences(root_dir, sequence_len)
        self.lookup = self.__create_lookup()

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx) -> tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            print("Slicing is not available.")
            raise TypeError

        if self.network_type == "single" or not self.is_train:
            sequence, label = self.sequences(self.lookup[idx])

            if self.transform:
                sequence = self.transform(sequence)

            return sequence, label

        elif self.network_type == "siamese":
            positive_sequence, positive_label = self.sequences(self.lookup[idx])
            negative_sequence, _ = self.sequences(self.lookup[idx])

            if self.transform:
                positive_sequence = self.transform(positive_sequence)
                negative_sequence = self.transform(negative_sequence)

            return positive_sequence, negative_sequence, positive_label

        elif self.network_type == "triplet":
            anchor_sequence, anchor_label = self.sequences(self.lookup[idx])
            positive_sequence, _ = self.sequences(self.lookup[idx])
            negative_sequence, _ = self.sequences(self.lookup[idx])

            if self.transform:
                anchor_sequence = self.transform(anchor_sequence)
                positive_sequence = self.transform(positive_sequence)
                negative_sequence = self.transform(negative_sequence)

            return anchor_sequence, positive_sequence, negative_sequence, anchor_label

        else:
            raise Exception("If not loading training dataset, make sure the is_train flag is set to True, "
                            "Otherwise, the network_type is invalid, should be 'single', 'siamese' or 'triplet'.")

    def __create_lookup(self) -> list:
        data_info = read_from_json(self.json_path)

        lookup = []
        for element in data_info:
            element_info = DimensionsElement(element)

            # Skip adding a sequence to the lookup if it is not specified in the data_limiter.
            if self.data_limiter and self.data_limiter.skip_sequence(element_info):
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

            # TODO: Solve so this isn't necessary. For it to work, needs to add zeros to end of last seq, which is
            #   shorter than the rest of the seqs
            del lookup[-1]

        return lookup


