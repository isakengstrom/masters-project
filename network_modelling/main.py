import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

# To start board, type the following in the terminal: tensorboard --logdir=runs
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import time

from models.LSTM import LSTM, LSTM_2, BRNN
from train import train

from learn import learn
from evaluate import evaluate

from dataset import FOIKineticPoseDataset, DataLimiter
from helpers.paths import EXTR_PATH, JOINTS_LOOKUP_PATH, TB_RUNS_PATH
from helpers import read_from_json
from sequence_transforms import FilterJoints, ChangePoseOrigin, ToTensor, NormalisePoses, AddNoise, ReshapePoses


def create_samplers(dataset_len, train_split=.8, val_split=.2, val_from_train=True, shuffle=True):
    """
    Influenced by: https://stackoverflow.com/a/50544887

    This is not (as of yet) stratified sampling,
    read more about it here: https://stackoverflow.com/a/52284619
    or here: https://github.com/ncullen93/torchsample/blob/master/torchsample/samplers.py#L22
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


if __name__ == "__main__":
    writer = SummaryWriter(TB_RUNS_PATH)

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

    data_limiter = DataLimiter(
        subjects=[0,1,2,3,4],
        sessions=[0],
        views=[2]
    )

    # There are 10 people in the dataset that we want to classify correctly. Might be limited by data_limiter though
    if data_limiter.subjects is None:
        num_classes = 10
    else:
        num_classes = len(data_limiter.subjects)

    # Number of epochs - The number of times the dataset is worked through during learning
    num_epochs = 100

    # Batch size - tightly linked with gradient descent.
    # The number of samples worked through before the params of the model are updated
    #   - Batch Gradient Descent: batch_size = len(dataset)
    #   - Stochastic Gradient descent: batch_size = 1
    #   - Mini-Batch Gradient descent: 1 < batch_size < len(dataset)
    batch_size = 2

    # Learning rate
    learning_rate = 0.05

    # Number of features
    input_size = num_joints * num_joint_coords

    # Length of a sequence, the length represent the number of frames.
    # The FOI dataset is captured at 50 fps
    sequence_len = 1000

    # Layers for the RNN
    num_layers = 2  # Number of stacked RNN layers
    hidden_size = 256  # Number of features in hidden state

    # Loss function
    margin = 0.2  # The margin for certain loss functions

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

    # Transforms
    composed = transforms.Compose([
        ChangePoseOrigin(),
        FilterJoints(),
        #NormalisePoses(),
        ReshapePoses(),
        ToTensor()]
    )

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

    train_sampler, test_sampler, val_sampler = create_samplers(
        dataset_len=len(train_dataset),
        train_split=.8,
        val_split=.2,
        val_from_train=True,
        shuffle=True
    )

    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size, sampler=test_sampler, num_workers=0)
    val_loader = DataLoader(train_dataset, batch_size, sampler=val_sampler, num_workers=0)

    if network_type == "single":
        loss_function = nn.CrossEntropyLoss()
    elif network_type == "siamese":
        raise NotImplementedError
    elif network_type == "triplet":
        loss_function = nn.TripletMarginLoss(margin)
    else:
        raise Exception("Invalid network_type")

    device = torch.device('cuda' if use_cuda else 'cpu')

    model = BRNN(input_size, hidden_size, num_layers, num_classes, device)

    if use_cuda:
        model.cuda()
        cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()

    model_name = str(type(model)).split('.')[-1][:-2]
    optimizer_name = str(type(optimizer)).split('.')[-1][:-2]
    loss_function_name = str(type(loss_function)).split('.')[-1][:-2]

    transform_names = [transform.split(' ')[0].split('.')[1] for transform in str(composed).split('<')[1:]]

    print('-' * 32, 'Setup', '-' * 33)
    print(f"| Model: {model_name}\n"
          f"| Optimizer: {optimizer_name}\n"
          f"| Network type: {network_type}\n"
          f"| Loss function: {loss_function_name}\n"
          f"| Device: {device}\n"
          f"|")

    print(f"| Sequence transforms:")
    [print(f"| {name_idx+1}: {name}") for name_idx, name in enumerate(transform_names)]
    print(f"|")

    print(f"| Total sequences: {len(train_dataset)}\n"
          f"| Train split: {len(train_sampler)}\n"
          f"| Val split: {len(val_sampler)}\n"
          f"| Test split: {len(test_sampler)}\n"
          f"|")

    print(f"| Learning phase:\n"
          f"| Epochs: {num_epochs}\n"
          f"| Batch size: {batch_size}\n"
          f"| Train batches: {len(train_loader)}\n"
          f"| Val batches: {len(val_loader)}\n"
          f"|")

    print(f"| Testing phase:\n"
          f"| Batch size: {batch_size}\n"
          f"| Test batches: {len(test_loader)}\n"
          f"|")

    print('-' * 28, 'Learning phase', '-' * 28)
    model = learn(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        num_epochs=num_epochs,
        device=device,
        network_type=network_type
    )

    print('-' * 28, 'Testing phase', '-' * 29)
    test_accuracy = evaluate(
        data_loader=test_loader,
        model=model,
        device=device
    )

    print(f'| Finished testing | Accuracy: {test_accuracy:.3f} | Total time: {time.time() - start_time:.3f}s ')
