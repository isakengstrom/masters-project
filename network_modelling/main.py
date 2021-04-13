import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np

from models.LSTM import LSTM
from train import train
from test import test
from dataset import FOIKineticPoseDataset
from helpers.paths import EXTR_PATH, JOINTS_LOOKUP_PATH
from helpers import read_from_json
from sequence_transforms import FilterJoints, ChangePoseOrigin, ToTensor, NormalisePoses, AddNoise, ReshapePoses


def create_samplers(dataset_len, train_split=.8, val_split=.2, val_from_train=True, shuffle=True):
    """

    Influenced by: https://stackoverflow.com/a/50544887

    This is not (as of yet) stratified sampling,
    read more about it here: https://stackoverflow.com/a/52284619
    or here: https://github.com/ncullen93/torchsample/blob/master/torchsample/samplers.py#L22

    :param dataset_len:
    :param train_split:
    :param val_split:
    :param val_from_train:
    :param shuffle:
    :return:
    """

    indices = list(range(dataset_len))

    if shuffle:
        # TODO: Look into if this truly generates random, see this:
        #  https://www.reddit.com/r/MachineLearning/comments/mocpgj/p_using_pytorch_numpy_a_bug_that_plagues/
        random_seed = 42
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

        # Check that there is a somewhat reasonable split left for testing
        assert test_split >= 0.1

        first_split = int(np.floor(train_split * dataset_len))
        second_split = int(np.floor((train_split + test_split) * dataset_len))

        train_indices = indices[:first_split]
        test_indices = indices[first_split:second_split]
        val_indices = indices[second_split:]

    return SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices), SubsetRandomSampler(val_indices)


def check_dataset_item(item):
    """"""
    print(len(item))
    '''
    if len(item)
    seq = item["sequences"]["anchor_sequence"]
    print(seq.shape)
    dim = ""

    if isinstance(seq, np.ndarray):
        dim = seq.shape
    elif isinstance(seq, list):
        dim = len(seq)
    elif isinstance(seq, torch.Tensor):
        dim = seq.size()

    print("Dataset instance with index {} and key '{}'\n\ttype: {}, \n\tDimensions: {}"
          .format(item["seq_idx"], item["key"], type(seq), dim))

    '''


if __name__ == "__main__":
    # Get the active number of OpenPose joints from the joints lookup. For full kinetic pose, this will be 25,
    # but with the FilterJoints() transform, it can be a lower amount.
    # (See if the filter is applied below, when calling the dataset)
    joints_lookup = read_from_json(JOINTS_LOOKUP_PATH, use_dumps=True)
    active_name = joints_lookup["activate_by"]
    num_joints = len(joints_lookup["active_" + active_name])

    # The number of coordinates for each of the OpenPose joints, equal to 2 if using both x and y
    num_joint_coords = 2

    ####################################################################
    # Hyper parameters #################################################
    ####################################################################

    # There are 10 people in the dataset that we want to classify correctly.
    num_classes = 10

    # Number of epochs - The number of times the dataset is worked through during training
    num_epochs = 2

    # Batch size - tightly linked with gradient descent.
    # The number of samples worked through before the params of the model are updated
    #   - Batch Gradient Descent: batch_size = len(dataset)
    #   - Stochastic Gradient descent: batch_size = 1
    #   - Mini-Batch Gradient descent: 1 < batch_size < len(dataset)
    batch_size = 32

    # Learning rate
    learning_rate = 0.001

    # Number of features
    input_size = num_joints * num_joint_coords

    # Length of a sequence, the length represent the number of frames.
    # The FOI dataset is captured at 50 fps
    sequence_len = 100

    # Layers for the RNN
    num_layers = 1  # Number of stacked RNN layers
    hidden_size = 2  # Number of features in hidden state

    # Loss function
    margin = 0.2  # The margin used if margin loss is used

    # Other params
    json_path = EXTR_PATH + "final_data_info.json"
    root_dir = EXTR_PATH + "final/"
    network_type = "single"

    use_cuda = torch.cuda.is_available()

    # Add checkpoint dir if it doesn't exist
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')

    # Add saved_models dir if it doesn't exist
    if not os.path.isdir('./models/saved_models'):
        os.mkdir('./models/saved_models')

    # Limiter #################################################################
    #   Used to specify which sequences to extract from the dataset.
    #   Values can either be 'None' or a list of indices.
    #
    #   If 'None', don't limit that parameter, e.g.
    #       "subjects": None, "sessions": None, "views": None
    #        will get all sequences, from s0_s0_v0 to s9_s0_v4
    #
    #   If indices, get the corresponding sequences, e.g.
    #       "subjects": [0], "sessions": [0,1], "views": [0,1,2]
    #       will get s0_s0_v0, s0_s0_v1, s0_s0_v2, s0_s1_v0, s0_s1_v1, s0_s1_v2
    #
    ###########################################################################
    data_limiter = {
        "subjects": None,
        "sessions": [0],
        "views": [0],
    }

    # Transforms
    composed = transforms.Compose([
        ChangePoseOrigin(),
        FilterJoints(),
        NormalisePoses(),
        ReshapePoses(),
        ToTensor()
    ])

    train_dataset = FOIKineticPoseDataset(
        json_path=json_path,
        root_dir=root_dir,
        sequence_len=sequence_len,
        is_train=True,
        network_type=network_type,
        data_limiter=data_limiter,
        transform=composed
    )

    test_dataset = FOIKineticPoseDataset(
        json_path=json_path,
        root_dir=root_dir,
        sequence_len=sequence_len,
        is_train=False,
        data_limiter=data_limiter,
        transform=composed
    )

    train_sampler, test_sampler, val_sampler = create_samplers(len(train_dataset), train_split=.8, val_split=.2, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size, sampler=test_sampler, num_workers=4)
    val_loader = DataLoader(train_dataset, batch_size, sampler=val_sampler, num_workers=4)

    if network_type == "single":
        loss_function = nn.CrossEntropyLoss()
    elif network_type == "siamese":
        raise NotImplementedError
    elif network_type == "triplet":
        loss_function = nn.TripletMarginLoss(margin)
    else:
        raise NameError

    device = torch.device('cuda' if use_cuda else 'cpu')

    model = LSTM(input_size, hidden_size, num_layers, num_classes, device)

    if use_cuda:
        model.cuda()
        cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model, loss_log, acc_log = train(model, train_loader, optimizer, loss_function, num_epochs, device, network_type)

    test_acc = test(model, test_loader, device)
