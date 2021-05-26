import numpy as np
import os
import multiprocessing

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from helpers import read_from_json


def create_samplers(dataset_len, train_split=.8, val_split=.2, val_from_train=True, shuffle=True, split_limit_factor=None):
    """
    Influenced by: https://stackoverflow.com/a/50544887

    This is not (as of yet) stratified sampling,
    read more about it here: https://stackoverflow.com/a/52284619
    or here: https://github.com/ncullen93/torchsample/blob/master/torchsample/samplers.py#L22


    split_limit_factor - a hack to lower the number of samples by a factor. Can be used to compare different sequence
        lengths with the same number of samples. The factor should be in the interval [0,1].

    """

    indices = list(range(dataset_len))

    if split_limit_factor is not None:
        if split_limit_factor >= 1 or split_limit_factor <= 0:
            raise Exception(f'The split_limit_factor must be in interval [0, 1], it was {split_limit_factor}.')

        dataset_len = dataset_len * split_limit_factor

    # Randomize the splits
    if shuffle:
        random_seed = 22  # 42
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if val_from_train:
        train_test_split = int(np.floor(train_split * dataset_len))
        train_val_split = int(np.floor((1 - val_split) * train_test_split))

        temp_indices = indices[:train_test_split]

        train_indices = temp_indices[:train_val_split]
        val_indices = temp_indices[train_val_split:]
        test_indices = indices[train_test_split:]
    else:
        test_split = 1 - (train_split + val_split)

        first_split = int(np.floor(train_split * dataset_len))
        second_split = int(np.floor((train_split + test_split) * dataset_len))

        train_indices = indices[:first_split]
        test_indices = indices[first_split:second_split]
        val_indices = indices[second_split:]

    return SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices), SubsetRandomSampler(val_indices)


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
        self.key = "{}{}{}".format(self.sub, self.sess, self.view)


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


class FOIKinematicPoseDataset(Dataset):
    def __init__(self, data, json_path, sequence_len, data_limiter=None, transform=None):
        # Data loading
        self.json_path = json_path
        self.sequence_len = sequence_len
        self.data_limiter = data_limiter
        self.transform = transform
        self.instantiated_data = data

        self.sequences = Sequences(self.instantiated_data)
        self.lookup = self.create_lookup()

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx) -> tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            raise Exception("Slicing is not available.")

        sequence, label = self.sequences(self.lookup[idx])

        if self.transform:
            sequence = self.transform(sequence)

        return sequence, label

    def create_lookup(self) -> list:
        data_info = read_from_json(self.json_path)

        lookup = []
        for element in data_info:
            element_info = DimensionsElement(element)

            # Skip adding a sequence to the lookup if it is not specified in the data_limiter.
            if self.data_limiter and self.data_limiter.skip_sequence(element_info):
                continue

            seq_info = {"file_name": element_info.file_name, "sub_name": element_info.sub_name,
                        "sess_name": element_info.sess_name, "view_name": element_info.view_name,
                        "key": element_info.key}

            for i in range(0, element_info.len, self.sequence_len):
                seq_info["start"] = i
                seq_info["end"] = i + self.sequence_len
                lookup.append(seq_info.copy())

            # Fix the last sequence's end frame
            lookup[-1]["end"] = min(lookup[-1]["end"], element_info.len)

            # TODO: Solve so this isn't necessary. For it to work, needs to add zeros to end of last seq, which is
            #   shorter than the rest of the seqs.
            #   - Is there a way to do that efficiently, without checking for the last element at every iteration?
            del lookup[-1]

        return lookup

