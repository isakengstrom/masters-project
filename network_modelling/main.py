import itertools
import json
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# To start board, type the following in the terminal: tensorboard --logdir=runs
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


from learn import learn
from evaluate import evaluate

from models.RNN import GenNet
from losses.margin_losses import TripletMarginLoss

from dataset import FOIKinematicPoseDataset, DataLimiter, LoadData, create_samplers
from sequence_transforms import FilterJoints, ChangePoseOrigin, ToTensor, NormalisePoses, AddNoise, ReshapePoses

from helpers import write_to_json
from helpers.paths import EXTR_PATH_SSD, TB_RUNS_PATH


def print_setup(setup: dict):
    params = setup['params']
    split = setup['split']

    print('-' * 32, 'Setup', '-' * 33)
    print(f"| Model: {setup['model_name']}\n"
          f"| Optimizer: {setup['optimizer_name']}\n"
          f"| Loss type: {params['loss_type']}\n"
          f"| Loss function: {setup['loss_function_name']}\n"
          f"| Device: {setup['device']}\n"
          f"|")

    print(f"| Sequence transforms:")
    [print(f"| {name_idx + 1}: {name}") for name_idx, name in enumerate(setup["transforms"])]
    print(f"|")

    print(f"| Total sequences: {split['tot_num_seqs']}\n"
          f"| Train split: {split['train_split']}\n"
          f"| Val split: {split['val_split']}\n"
          f"| Test split: {split['test_split']}\n"
          f"|")

    print(f"| Learning phase:\n"
          f"| Epochs: {params['num_epochs']}\n"
          f"| Batch size: {params['batch_size']}\n"
          f"| Train batches: {split['num_train_batches']}\n"
          f"| Val batches: {split['num_val_batches']}\n"
          f"|")

    print(f"| Testing phase:\n"
          f"| Batch size: {params['batch_size']}\n"
          f"| Test batches: {split['num_test_batches']}")


def make_save_dirs():
    if not os.path.isdir('./saves'):
        os.mkdir('./saves')

    # Add checkpoint dir if it doesn't exist
    if not os.path.isdir('./saves/checkpoints'):
        os.mkdir('./saves/checkpoints')

    # Add saved_models dir if it doesn't exist
    if not os.path.isdir('./saves/models'):
        os.mkdir('./saves/models')

    if not os.path.isdir('./saves/runs'):
        os.mkdir('./saves/runs')


def parameters():
    """
    Initialise the hyperparameters (+ some other params)

    - Batch size - tightly linked with gradient descent. The number of samples worked through before the params of the
      model are updated.
      - Batch Gradient Descent: batch_size = len(dataset)
      - Stochastic Gradient descent: batch_size = 1
      - Mini-Batch Gradient descent: 1 < batch_size < len(dataset)
      - Advice from Yann LeCun, batch_size <= 32: arxiv.org/abs/1804.07612


    :return: params: dict()
    """

    params = dict()

    # Pick OpenPose joints for the model,
    # these are used in the FilterPose() transform, as well as when deciding the input_size/number of features
    params['joints_activator'] = "op_idx"

    # OpenPose indices, same as in the OpenPose repo.
    if params['joints_activator'] == "op_idx":
        params['joint_filter'] = [1, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]  # Select OpenPose indices
    elif params['joints_activator'] == "name":
        params['joint_filter'] = ["nose", "c_hip", "neck"]  # Joint names, see more in the 'joints_lookup.json' file
    else:
        NotImplementedError(f"The Joint filter of the '{params['joints_activator']}' activator is not implemented.")

    params['num_epochs'] = 5
    params['batch_size'] = 32
    params['learning_rate'] = 5e-4  # 0.05 5e-4 5e-8

    # Get the active number of OpenPose joints from the joint_filter. For full kinetic pose, this will be 25,
    # The joint_filter will also be applied further down, in the FilterJoints() transform.
    num_joints = len(params['joint_filter'])

    # The number of coordinates for each of the OpenPose joints, equal to 2 if using both x and y
    num_joint_coords = 2

    # Number of features
    params['input_size'] = num_joints * num_joint_coords  # 28

    # Length of a sequence, the length represent the number of frames.
    # The FOI dataset is captured at 50 fps
    params['sequence_len'] = 150

    # Network / Model params
    params['num_layers'] = 2  # Number of stacked RNN layers
    params['hidden_size'] = 256 * 2  # Number of features in hidden state
    params['net_type'] = "lstm"
    params['bidirectional'] = False

    # Loss function
    params['loss_type'] = "single"
    params['loss_margin'] = 1  # The margin for certain loss functions

    return params


def multi_run():
    runnables = {
        'bidirectional': [False, True],
        'net_type': ['rnn', 'gru', 'lstm'],
        'sequence_len': [50, 100, 200, 400, 800],
    }

    # Create every combination from the lists in runnables
    runnable_products = [dict(zip(runnables, value)) for value in itertools.product(*runnables.values())]

    # Run the network by firstly overwriting the params with the runnables.
    params = parameters()
    for overwrite in runnable_products:
        params.update(overwrite)

        run_network(params)


def run_network(params: dict):
    now = datetime.datetime.now()  # Save Date and time of run

    # Dict to save all the run info. When learning and evaluating is finished, this will be saved to disk.
    run_info = dict()
    run_info['at'] = str(now).split('.')[0]
    run_info['params'] = params

    print(run_info['at'])
    # Formatting of run_info save file name
    run_name = f'd{now.strftime("%y")}{now.strftime("%m")}{now.strftime("%d")}' \
               f'_h{now.strftime("%H")}m{now.strftime("%M")}.json'
    print(run_name)

    # Data limiter: Go to definition for more info
    data_limiter = DataLimiter(
        subjects=None,
        sessions=[0],
        views=[3]
    )

    # There are 10 people in the dataset that we want to classify correctly. Might be limited by data_limiter though
    num_classes = len(data_limiter.subjects)

    # Transforms
    composed = transforms.Compose([
        NormalisePoses(low=1, high=100),
        ChangePoseOrigin(),
        FilterJoints(activator=params['joints_activator'], joint_filter=params['joint_filter']),
        ReshapePoses(),
        #AddNoise(scale=1),
        ToTensor()
    ])

    make_save_dirs()

    # Other params
    json_path_ssd = os.path.join(EXTR_PATH_SSD, "final_data_info.json")
    root_dir_ssd = os.path.join(EXTR_PATH_SSD, "final/")

    # Load the data into memory
    print(f"| Loading data into memory..")
    load_start_time = time.time()
    data = LoadData(root_dir=root_dir_ssd, data_limiter=data_limiter, num_workers=8)
    print(f"| Loading finished in {time.time() - load_start_time:0.1f}s")

    train_dataset = FOIKinematicPoseDataset(
        data=data,
        json_path=json_path_ssd,
        sequence_len=params['sequence_len'],
        is_train=True,
        loss_type=params['loss_type'],
        data_limiter=data_limiter,
        transform=composed
    )

    test_dataset = FOIKinematicPoseDataset(
        data=data,
        json_path=json_path_ssd,
        sequence_len=params['sequence_len'],
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

    train_loader = DataLoader(train_dataset, params['batch_size'], sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, params['batch_size'], sampler=test_sampler, num_workers=4)
    val_loader = DataLoader(train_dataset, params['batch_size'], sampler=val_sampler, num_workers=4)

    # Store all the info of how the data is split into train, val and test
    run_info['split'] = {
        'tot_num_seqs': len(train_dataset), 'batch_size': params['batch_size'], 'train_split': len(train_sampler),
        'val_split': len(val_sampler), 'test_split': len(test_sampler), 'num_train_batches': len(train_loader),
        'num_val_batches': len(val_loader), 'num_test_batches': len(test_loader)
    }

    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    run_info['device'] = str(device)

    # The recurrent neural net model, RNN, GRU or LSTM
    model = GenNet(
        input_size=params['input_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        num_classes=num_classes,
        device=device,
        bidirectional=params['bidirectional'],
        net_type=params['net_type']
    ).to(device)

    # The optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    #optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=0.9)

    if params['loss_type'] == "single":
        loss_function = nn.CrossEntropyLoss()
    elif params['loss_type'] == "siamese":
        raise NotImplementedError
    elif params['loss_type'] == "triplet":
        loss_function = TripletMarginLoss(margin=params['loss_margin'])
    else:
        raise Exception("Invalid network_type")

    # Get the names as strings for the pytorch objects of interest
    run_info['model_name'] = str(type(model)).split('.')[-1][:-2]
    run_info['optimizer_name'] = str(type(optimizer)).split('.')[-1][:-2]
    run_info['loss_function_name'] = str(type(loss_function)).split('.')[-1][:-2]
    run_info['transforms'] = [transform.split(' ')[0].split('.')[1] for transform in str(composed).split('<')[1:]]

    print_setup(setup=run_info)

    writer = SummaryWriter(TB_RUNS_PATH)  # TensorBoard writer
    start_time = time.time()

    print('-' * 28, 'Learning phase', '-' * 28)
    model, learn_info = learn(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        num_epochs=params['num_epochs'],
        device=device,
        classes=data_limiter.subjects,
        tb_writer=writer,
        loss_type=params['loss_type']
    )

    print('-' * 28, 'Testing phase', '-' * 29)
    test_info = evaluate(
        data_loader=test_loader,
        model=model,
        device=device,
        is_test=True
    )

    writer.close()

    run_info['tot_time'] = time.time() - start_time
    run_info['learn_info'] = learn_info
    run_info['test_info'] = test_info
    save_path = os.path.join('./saves/runs', run_name)
    write_to_json(run_info, save_path)
    json_info = json.dumps(run_info)
    print(json_info)

    print(f"| Finished testing | Accuracy: {test_info['accuracy']:.6f} | Total time: {run_info['tot_time'] :.2f}s")


if __name__ == "__main__":
    multi_run()
    #run_network(parameters())


