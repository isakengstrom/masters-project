import os
import time
import numpy as np

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
from dataset import FOIKineticPoseDataset, DataLimiter
from sequence_transforms import FilterJoints, ChangePoseOrigin, ToTensor, NormalisePoses, AddNoise, ReshapePoses
from helpers.paths import EXTR_PATH, JOINTS_LOOKUP_PATH, TB_RUNS_PATH


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
        #np.random.seed(random_seed)
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

    # Pick OpenPose joints for the model,
    # these are used in the FilterPose() transform, as well as when deciding the input_size/number of features
    joints_lookup_activator = "op_idx"
    joint_filter = []

    # OpenPose indices, same as in the OpenPose repo.
    if joints_lookup_activator == "op_idx":
        joint_filter = [1, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]  # Select OpenPose indices
        #joint_filter = list(range(25))  # All OpenPose indices

    # Joint names, see more in the 'joints_lookup.json' file
    elif joints_lookup_activator == "name":
        joint_filter = ["nose", "c_hip", "neck"]

    else:
        NotImplementedError(f"The Joint filter of the '{joints_lookup_activator}' activator is not implemented.")

    ####################################################################
    # Hyper parameters #################################################
    ####################################################################

    data_limiter = DataLimiter(
        subjects=[0,1,2, 3, 4, 5, 6, 7, 8, 9],
        sessions=[0],
        views=None
    )

    # There are 10 people in the dataset that we want to classify correctly. Might be limited by data_limiter though
    if data_limiter.subjects is None:
        num_classes = 10
    else:
        num_classes = len(data_limiter.subjects)

    # Number of epochs - The number of times the dataset is worked through during learning
    num_epochs = 30

    # Batch size - tightly linked with gradient descent.
    # The number of samples worked through before the params of the model are updated
    #   - Batch Gradient Descent: batch_size = len(dataset)
    #   - Stochastic Gradient descent: batch_size = 1
    #   - Mini-Batch Gradient descent: 1 < batch_size < len(dataset)
    batch_size = 16

    # Learning rate
    learning_rate = 5e-4  # 0.05 5e-8

    # Get the active number of OpenPose joints from the joint_filter. For full kinetic pose, this will be 25,
    # The joint_filter will also be applied further down, in the FilterJoints() transform.
    num_joints = len(joint_filter)

    # The number of coordinates for each of the OpenPose joints, equal to 2 if using both x and y
    num_joint_coords = 2

    # Number of features
    input_size = num_joints * num_joint_coords  # 28

    # Length of a sequence, the length represent the number of frames.
    # The FOI dataset is captured at 50 fps
    sequence_len = 350

    # Layers for the RNN
    num_layers = 2  # Number of stacked RNN layers
    hidden_size = 256*2  # Number of features in hidden state

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
        #NormalisePoses(),
        ChangePoseOrigin(),
        FilterJoints(activator=joints_lookup_activator, joint_filter=joint_filter),
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

    '''
    print("Dataset length: ", len(train_dataset))
    set_len = len(train_dataset)
    for idx, seq in enumerate(train_dataset):
        print(f'At {idx} / {set_len}, shape: {seq[0].size()}')
    '''

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
        train_split=.7,
        val_split=.15,
        val_from_train=False,
        shuffle=True
    )

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
        raise Exception("Invalid network_type")

    device = torch.device('cuda' if use_cuda else 'cpu')

    model = LSTM(input_size, hidden_size, num_layers, num_classes, device)

    if use_cuda:
        model.cuda()
        cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    start_time = time.time()

    model_name = str(type(model)).split('.')[-1][:-2]
    optimizer_name = str(type(optimizer)).split('.')[-1][:-2]
    loss_function_name = str(type(loss_function)).split('.')[-1][:-2]

    transform_names = [transform.split(' ')[0].split('.')[1] for transform in str(composed).split('<')[1:]]

    def print_setup():
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
              f"| Test batches: {len(test_loader)}")

    print_setup()

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

