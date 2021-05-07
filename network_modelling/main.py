import json
import os
import time
import datetime
import numpy as np
import math
import subprocess

import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# To start board, type the following in the terminal: tensorboard --logdir=runs
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from learn import learn
from evaluate import evaluate

from models.LSTM import LSTM, LSTM_2, BDLSTM, AladdinLSTM
from models.RNN import GenNet
from losses.margin_losses import TripletMarginLoss

from dataset import FOIKineticPoseDataset, DataLimiter, LoadData
from sequence_transforms import FilterJoints, ChangePoseOrigin, ToTensor, NormalisePoses, AddNoise, ReshapePoses
from helpers.paths import EXTR_PATH, EXTR_PATH_SSD, JOINTS_LOOKUP_PATH, TB_RUNS_PATH


def create_samplers(dataset_len, train_split=.8, val_split=.2, val_from_train=True, shuffle=True):
    """
    Influenced by: https://stackoverflow.com/a/50544887

    This is not (as of yet) stratified sampling,
    read more about it here: https://stackoverflow.com/a/52284619
    or here: https://github.com/ncullen93/torchsample/blob/master/torchsample/samplers.py#L22
    """

    indices = list(range(dataset_len))

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

        # Check that there is a somewhat reasonable split left for testing
        assert test_split >= 0.1

        first_split = int(np.floor(train_split * dataset_len))
        second_split = int(np.floor((train_split + test_split) * dataset_len))

        train_indices = indices[:first_split]
        test_indices = indices[first_split:second_split]
        val_indices = indices[second_split:]

    return SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices), SubsetRandomSampler(val_indices)


def main():
    now = str(datetime.datetime.now()).split('.')[0]  # Save Date and time of run, split to remove microseconds
    run_info = dict()
    run_info["at"] = now
    params = dict()

    writer = SummaryWriter(TB_RUNS_PATH)
    #cmd_start_tensorboard = ["tensorboard", "--logdir", TB_RUNS_PATH]
    #subprocess.Popen(cmd_start_tensorboard)

    # Pick OpenPose joints for the model,
    # these are used in the FilterPose() transform, as well as when deciding the input_size/number of features
    joints_lookup_activator = "op_idx"

    # OpenPose indices, same as in the OpenPose repo.
    if joints_lookup_activator == "op_idx":
        params["joint_filter"] = [1, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]  # Select OpenPose indices
    elif joints_lookup_activator == "name":
        params["joint_filter"] = ["nose", "c_hip", "neck"]  # Joint names, see more in the 'joints_lookup.json' file
    else:
        NotImplementedError(f"The Joint filter of the '{joints_lookup_activator}' activator is not implemented.")

    ####################################################################
    # Hyper parameters #################################################
    ####################################################################

    data_limiter = DataLimiter(
        subjects=None,
        sessions=[0],
        views=[3]
    )

    # There are 10 people in the dataset that we want to classify correctly. Might be limited by data_limiter though
    num_classes = len(data_limiter.subjects)

    # Number of epochs - The number of times the dataset is worked through during learning
    params["num_epochs"] = 5

    # Batch size - tightly linked with gradient descent.
    # The number of samples worked through before the params of the model are updated
    #   - Batch Gradient Descent: batch_size = len(dataset)
    #   - Stochastic Gradient descent: batch_size = 1
    #   - Mini-Batch Gradient descent: 1 < batch_size < len(dataset)
    # From Yann LeCun, batch_size <= 32: arxiv.org/abs/1804.07612
    params["batch_size"] = 32

    # Learning rate
    params["learning_rate"] = 5e-4  # 0.05 5e-4 5e-8

    # Get the active number of OpenPose joints from the joint_filter. For full kinetic pose, this will be 25,
    # The joint_filter will also be applied further down, in the FilterJoints() transform.
    num_joints = len(params["joint_filter"])

    # The number of coordinates for each of the OpenPose joints, equal to 2 if using both x and y
    num_joint_coords = 2

    # Number of features
    params["input_size"] = num_joints * num_joint_coords  # 28

    # Length of a sequence, the length represent the number of frames.
    # The FOI dataset is captured at 50 fps
    params["sequence_len"] = 150

    # Network / Model params
    params["num_layers"] = 2  # Number of stacked RNN layers
    params["hidden_size"] = 256 * 2  # Number of features in hidden state
    params["net_type"] = "lstm"
    params["bidirectional"] = False

    # Loss function
    params["loss_type"] = "single"
    params["loss_margin"] = 1  # The margin for certain loss functions

    # Other params
    json_path = EXTR_PATH + "final_data_info.json"
    root_dir = EXTR_PATH + "final/"
    json_path_ssd = EXTR_PATH_SSD + "final_data_info.json"
    root_dir_ssd = EXTR_PATH_SSD + "final/"

    use_cuda = torch.cuda.is_available()

    # Add checkpoint dir if it doesn't exist
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')

    # Add saved_models dir if it doesn't exist
    if not os.path.isdir('./models/saved_models'):
        os.mkdir('./models/saved_models')

    # Transforms
    composed = transforms.Compose([
        NormalisePoses(low=1, high=100),
        ChangePoseOrigin(),
        FilterJoints(activator=joints_lookup_activator, joint_filter=params["joint_filter"]),
        ReshapePoses(),
        #AddNoise(scale=1),
        ToTensor()
    ])

    # Load the data into memory
    print(f"| Loading data into memory..")
    load_start_time = time.time()
    data = LoadData(root_dir=root_dir_ssd, data_limiter=data_limiter, num_workers=8)
    print(f"| Loading finished in {time.time() - load_start_time:0.1f}s")

    train_dataset = FOIKineticPoseDataset(
        data=data,
        json_path=json_path_ssd,
        root_dir=root_dir_ssd,
        sequence_len=params["sequence_len"],
        is_train=True,
        loss_type=params["loss_type"],
        data_limiter=data_limiter,
        transform=composed
    )

    test_dataset = FOIKineticPoseDataset(
        data=data,
        json_path=json_path_ssd,
        root_dir=root_dir_ssd,
        sequence_len=params["sequence_len"],
        is_train=False,
        data_limiter=data_limiter,
        transform=composed
    )

    train_sampler, test_sampler, val_sampler = create_samplers(
        dataset_len=len(train_dataset),
        train_split=.7,
        val_split=.15,
        val_from_train=False,
        shuffle=True
    )

    train_loader = DataLoader(train_dataset, params["batch_size"], sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, params["batch_size"], sampler=test_sampler, num_workers=4)
    val_loader = DataLoader(train_dataset, params["batch_size"], sampler=val_sampler, num_workers=4)

    if params["loss_type"] == "single":
        loss_function = nn.CrossEntropyLoss()
    elif params["loss_type"] == "siamese":
        raise NotImplementedError
    elif params["loss_type"] == "triplet":
        loss_function = TripletMarginLoss(margin=params["loss_margin"])
    else:
        raise Exception("Invalid network_type")

    device = torch.device('cuda' if use_cuda else 'cpu')
    run_info["device"] = str(device)

    model = GenNet(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        num_classes=num_classes,
        device=device,
        bidirectional=params["bidirectional"],
        net_type=params["net_type"]
    )

    if use_cuda:
        model.cuda()
        cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    #optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=0.9)

    run_info["model_name"] = str(type(model)).split('.')[-1][:-2]
    run_info["optimizer_name"] = str(type(optimizer)).split('.')[-1][:-2]
    run_info["loss_function_name"] = str(type(loss_function)).split('.')[-1][:-2]
    run_info["transforms"] = [transform.split(' ')[0].split('.')[1] for transform in str(composed).split('<')[1:]]

    split = dict()
    split["tot_num_seqs"] = len(train_dataset)
    split["batch_size"] = params["batch_size"]
    split["train_split"] = len(train_sampler)
    split["val_split"] = len(val_sampler)
    split["test_split"] = len(test_sampler)
    split["num_train_batches"] = len(train_loader)
    split["num_val_batches"] = len(val_loader)
    split["num_test_batches"] = len(test_loader)
    run_info["split"] = split

    def print_setup():
        print('-' * 32, 'Setup', '-' * 33)
        print(f"| Model: {run_info['model_name']}\n"
              f"| Optimizer: {run_info['optimizer_name']}\n"
              f"| Loss type: {params['loss_type']}\n"
              f"| Loss function: {run_info['loss_function_name']}\n"
              f"| Device: {run_info['device']}\n"
              f"|")

        print(f"| Sequence transforms:")
        [print(f"| {name_idx + 1}: {name}") for name_idx, name in enumerate(run_info["transforms"])]
        print(f"|")

        print(f"| Total sequences: {len(train_dataset)}\n"
              f"| Train split: {len(train_sampler)}\n"
              f"| Val split: {len(val_sampler)}\n"
              f"| Test split: {len(test_sampler)}\n"
              f"|")

        print(f"| Learning phase:\n"
              f"| Epochs: {params['num_epochs']}\n"
              f"| Batch size: {params['batch_size']}\n"
              f"| Train batches: {len(train_loader)}\n"
              f"| Val batches: {len(val_loader)}\n"
              f"|")

        print(f"| Testing phase:\n"
              f"| Batch size: {params['batch_size']}\n"
              f"| Test batches: {len(test_loader)}")

    print_setup()

    start_time = time.time()

    print('-' * 28, 'Learning phase', '-' * 28)
    model = learn(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        num_epochs=params["num_epochs"],
        device=device,
        classes=data_limiter.subjects,
        tb_writer=writer,
        loss_type=params["loss_type"]
    )

    print('-' * 28, 'Testing phase', '-' * 29)
    test_accuracy = evaluate(
        data_loader=test_loader,
        model=model,
        device=device,
        is_test=True
    )

    print(f'| Finished testing | Accuracy: {test_accuracy:.6f} | Total time: {time.time() - start_time:.2f}s ')

    writer.close()
    run_info["params"] = params
    json_info = json.dumps(run_info)
    print(json_info)


if __name__ == "__main__":
    main()


