import torch
from torch.utils.data import Dataset
import numpy as np
import os
import multiprocessing

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
        # The original FOI Gait dataset contains subjects [0-9], sessions [0-1], views [0-4]
        if subjects is None:
            subjects = range(10)
        if sessions is None:
            sessions = range(2)
        if views is None:
            views = range(5)

        self.subjects = subjects
        self.sessions = sessions
        self.views = views

    def skip_sequence(self, info: DimensionsElement) -> bool:

        if info.sub not in self.subjects:
            return True

        if info.sess not in self.sessions:
            return True

        if info.view not in self.views:
            return True


class LoadData:
    """
    Loads the data from json into memory.

    The data is converted to NumPy arrays with values of type: as_type.

    Loading of the files can be threaded with the num_workers argument.
    """
    def __init__(self, root_dir, data_limiter, num_workers=0, as_type='float64'):
        data_dict = {}

        all_file_names = []
        all_file_dirs = []
        for subject in data_limiter.subjects:
            for session in data_limiter.sessions:
                for view in data_limiter.views:
                    file_name = f"SUB{subject}_SESS{session}_VIEW{view}.json"
                    file_dir = os.path.join(root_dir, file_name)
                    all_file_names.append(file_name)
                    all_file_dirs.append(file_dir)

        # If not threaded or if num workers <= 0, load files using the main process
        if num_workers is None or num_workers <= 0:
            num_files = len(all_file_names)
            for idx in sorted(range(len(all_file_names))):
                file_name = all_file_names[idx]
                file_dir = all_file_dirs[idx]
                data_dict[file_name] = self.read_json_as_numpy(file_dir).astype(as_type)
                print('\r|', f"Loading file {idx+1} of {num_files}..", end='')
            print("")

        # Threaded file loading
        else:
            pool = multiprocessing.Pool(processes=num_workers)
            data_list = pool.map(self.read_json_as_numpy, all_file_dirs)

            for idx in sorted(range(len(all_file_names))):
                file_name = all_file_names[idx]
                data_dict[file_name] = data_list[idx].astype(as_type)

        self.data = data_dict

    @staticmethod
    def read_json_as_numpy(file_dir):
        return np.array(read_from_json(file_dir))


class Sequences:
    def __init__(self, instantiated_data):
        self.instantiated_data = instantiated_data

    def __call__(self, item: dict) -> tuple:
        seq_info = SequenceElement(item)

        sequence = self.instantiated_data.data[seq_info.file_name][seq_info.start:seq_info.end]

        return sequence, seq_info.label


# TODO: get positives and negatives correctly, they are all the same sequence atm
class FOIKineticPoseDataset(Dataset):
    def __init__(self, data, json_path, root_dir, sequence_len, is_train=False, network_type="triplet", data_limiter=None, transform=None):
        # Data loading
        self.json_path = json_path
        self.root_dir = root_dir
        self.sequence_len = sequence_len
        self.network_type = network_type
        self.data_limiter = data_limiter
        self.transform = transform
        self.is_train = is_train

        self.instantiated_data = data
        self.sequences = Sequences(self.instantiated_data)

        self.lookup, self.keys_set = self.create_lookup()

        self.idx_lookup = self.create_idx_lookup()

        print(self.lookup[0])

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx) -> dict:
        """
        The main sequence is always called 'anchor'.

        Therefore, in cases of siamese, the sequences will be 'anchor' and 'negative', this would commonly be 'positive'
        and 'negative'.

        However, the current naming conventions makes the implementation more generalizable.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            print("Slicing is not available.")
            raise TypeError

        item = dict()
        dummy_idx = 0

        if self.network_type == "single" or not self.is_train:
            anchor_sequence, anchor_label = self.sequences(self.lookup[idx])

            if self.transform:
                anchor_sequence = self.transform(anchor_sequence)

            item["anchor"] = {}
            item["anchor"]["sequence"] = anchor_sequence
            item["anchor"]["label"] = anchor_label

        elif self.network_type == "siamese":
            anchor_sequence, anchor_label = self.sequences(self.lookup[idx])
            negative_sequence, negative_label = self.sequences(self.lookup[dummy_idx])

            if self.transform:
                anchor_sequence = self.transform(anchor_sequence)
                negative_sequence = self.transform(negative_sequence)

            item["anchor"] = {}
            item["anchor"]["sequence"] = anchor_sequence
            item["anchor"]["label"] = anchor_label

            item["negative"] = {}
            item["negative"]["sequence"] = negative_sequence
            item["negative"]["label"] = negative_label

        elif self.network_type == "triplet":
            anchor_sequence, anchor_label = self.sequences(self.lookup[idx])
            positive_sequence, positive_label = self.sequences(self.lookup[dummy_idx])
            negative_sequence, negative_label = self.sequences(self.lookup[dummy_idx])

            if self.transform:
                anchor_sequence = self.transform(anchor_sequence)
                positive_sequence = self.transform(positive_sequence)
                negative_sequence = self.transform(negative_sequence)

            item["anchor"] = {}
            item["anchor"]["sequence"] = anchor_sequence
            item["anchor"]["label"] = anchor_label

            item["positive"] = {}
            item["positive"]["sequence"] = positive_sequence
            item["positive"]["label"] = positive_label

            item["negative"] = {}
            item["negative"]["sequence"] = negative_sequence
            item["negative"]["label"] = negative_label

        else:
            raise Exception("If not loading training dataset, make sure the is_train flag is set to True, "
                            "Otherwise, the network_type is invalid, should be 'single', 'siamese' or 'triplet'.")

        return item

    def create_lookup(self) -> tuple:
        data_info = read_from_json(self.json_path)

        lookup = []
        key_set = set()
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
            key_set.add(element_info.key)

            for i in range(0, element_info.len, self.sequence_len):
                seq_info["start"] = i
                seq_info["end"] = i + self.sequence_len
                lookup.append(seq_info.copy())

            # Fix the last sequence's end frame
            lookup[-1]["end"] = min(lookup[-1]["end"], element_info.len)

            # TODO: Solve so this isn't necessary. For it to work, needs to add zeros to end of last seq, which is
            #   shorter than the rest of the seqs
            del lookup[-1]

        return lookup, sorted(key_set)

    def create_idx_lookup(self):
        idx_lookup = dict()

        for key in self.keys_set:
            idx_lookup[key] = []

        for seq_idx, seq in enumerate(self.lookup):
            idx_lookup[seq['key']].append(seq_idx)

        return idx_lookup
