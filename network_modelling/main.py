import itertools
import math
import os
import time
import datetime
from statistics import mean

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter # tensorboard --logdir /home/isaeng/Exjobb/states/runs/
from torch.utils.data import DataLoader

from learn import learn
from evaluate import evaluate
from models.RNN import GenNet
from dataset import FOIKinematicPoseDataset, DataLimiter, LoadData, create_samplers
from sequence_transforms import FilterJoints, ChangePoseOrigin, ToTensor, NormalisePoses, AddNoise, ReshapePoses

from helpers import write_to_json
from helpers.paths import EXTR_PATH_SSD, TB_RUNS_PATH
from helpers.result_formatter import mean_confidence_interval


# Const paths
JSON_PATH_SSD = os.path.join(EXTR_PATH_SSD, "final_data_info.json")
ROOT_DIR_SSD = os.path.join(EXTR_PATH_SSD, "final/")


# TODO: Bug - subjects can only be in order and if, for example, we want sub3, then subsequent subs must be used too,
#  - i.e. [sub0, sub1, sub2, sub3], cannot be [sub2, sub3] or [sub3]
# Data limiter: Go to definition for more info
DATA_LIMITER = DataLimiter(
    subjects=None,
    sessions=[0],
    views=None,
)

DATA_LIMITER_EVAL = DataLimiter(
    subjects=[0, 1],
    sessions=[1],
    views=[0],
)


# Load the data into memory
print(f"| Loading data into memory..")
load_start_time = time.time()
LOADED_DATA = LoadData(root_dir=ROOT_DIR_SSD, data_limiter=DATA_LIMITER, num_workers=8)
print(f"| Loading finished in {time.time() - load_start_time:0.1f}s")
print('-' * 72)

print(f"| Loading data into memory..")
load_start_time = time.time()
LOADED_DATA_EVAL = LoadData(root_dir=ROOT_DIR_SSD, data_limiter=DATA_LIMITER_EVAL, num_workers=8)
print(f"| Loading finished in {time.time() - load_start_time:0.1f}s")
print('-' * 72)


def print_setup(setup: dict, params):
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

    params['num_epochs'] = 250
    params['batch_size'] = 32
    params['learning_rate'] = 0.0005  #5e-4  # 0.05 5e-4 5e-8
    params['learning_rate_lim'] = None #5.1e-7
    params['step_size'] = 100

    # Get the active number of OpenPose joints from the joint_filter. For full kinetic pose, this will be 25,
    # The joint_filter will also be applied further down, in the FilterJoints() transform.
    num_joints = len(params['joint_filter'])

    # The number of coordinates for each of the OpenPose joints, is 2 if using x, y coords
    num_joint_coords = 2

    # Number of features
    params['input_size'] = num_joints * num_joint_coords  # 28

    # Length of a sequence, the length represent the number of frames.
    # The FOI dataset is captured at 50 fps
    params['sequence_len'] = 100
    params['simulated_len'] = 800

    # Network / Model params
    params['num_layers'] = 2  # Number of stacked RNN layers
    params['hidden_size'] = 256*2  # Number of features in hidden state
    params['net_type'] = 'gru'
    params['bidirectional'] = False
    params['max_norm'] = 1

    params['task'] = 'metric'  # 'classification'/'metric'
    # Loss function
    params['loss_type'] = 'single'  # 'single'/'triplet'
    params['loss_margin'] = 25  # The margin for certain loss functions

    # Settings for the network run
    params['should_learn'] = True
    params['should_write'] = False
    params['should_load_checkpoints'] = False
    params['should_test_unseen_sessions'] = False  # Test the unseen sessions (sess1) for sub 0 and 1
    params['should_val_unseen_sessions'] = False  # Val split from unseen sessions, otherwise uses seen session (sess0)

    return params


def multi_grid(num_repeats=1):
    """"""
    
    grids = [
        {
            'loss_margin': 0.1,
            'loss_type': 'triplet',
            'task': 'metric',
            'step_size': 100,
            'num_epochs': 250,
        },
        {
            'loss_margin': 0.5,
            'loss_type': 'triplet',
            'task': 'metric',
            'step_size': 100,
            'num_epochs': 250,
        },
        {
            'loss_margin': 1,
            'loss_type': 'triplet',
            'task': 'metric',
            'step_size': 100,
            'num_epochs': 250,
        },
        {
            'loss_margin': 2,
            'loss_type': 'triplet',
            'task': 'metric',
            'step_size': 100,
            'num_epochs': 250,
        },
    ]

    num_grids = len(grids)
    for grid_idx, grid in enumerate(grids):
        print(f"| Grid {grid_idx+1}/{num_grids}")
        grid_search(num_repeats, grid_idx, grid)
        print(f"| ", "_____-----"*10)


def grid_search(num_repeats=1, outer_grid_idx=-1, grid=None):
    multi_start = datetime.datetime.now()  # Date and time of start
    multi_start_time = time.time()  # Time of start

    if grid is None:
        # Override any parameter in parameter()
        grid = {
            #'bidirectional': [False, True],
            #'net_type': ['gru'],
            #'sequence_len': [5, 10, 15, 20, 25],
            #'hidden_size': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            #'max_norm': [0.01, 0.1, 1]
            #'num_epochs': 2
            #'loss_margin': [50, 100]
        }

    # Wrap every value in a list if it isn't already the case
    for key, value in grid.items():
        if not isinstance(value, list):
            grid[key] = [value]

    # Create every combination from the lists in grid
    all_grid_combinations = [dict(zip(grid, value)) for value in itertools.product(*grid.values())]

    num_runs = len(all_grid_combinations)
    run_formatter = int(math.log10(num_runs)) + 1  # Used for printing spacing

    # Store runtime information
    multi_info = {'at': str(multi_start).split('.')[0], 'duration': None, 'num_runs': num_runs, 'num_reps': num_repeats}
    multi_runs = dict()
    multi_results = dict()

    params = parameters()
    params['num_runs'] = num_runs

    # Run the network by firstly overriding the params with the grid.
    for grid_idx, override_params in enumerate(all_grid_combinations):
        if outer_grid_idx != -1:
            print(f"| Grid {grid_idx+1}")
        # Print the current run index and the current notable params
        print(f"| Run {grid_idx+1:{run_formatter}.0f}/{num_runs}")
        [print(f"| {idx+1}. {key}: {val}", end=' ') for idx, (key, val) in enumerate(override_params.items())]
        print('\n')

        params.update(override_params)  # Override the params
        params['run_idx'] = grid_idx

        # Run the network num_reps times
        reps_info = repeat_run(params, num_repeats=num_repeats)

        # Store runtime information
        multi_runs[grid_idx] = dict()
        multi_runs[grid_idx] = reps_info
        multi_runs[grid_idx]['notable_params'] = override_params
        multi_runs[grid_idx]['params'] = params

        # Store runtime information
        multi_results[grid_idx] = {
            'setup': override_params,
            'duration': reps_info['duration'],
            'accuracy': reps_info['accuracies_mean'],
            'confidence_scores': reps_info['confidence_scores']
        }

        print('---*' * 20)

    # Store runtime information
    multi_info['multi_runs'] = multi_runs
    multi_info['duration'] = time.time() - multi_start_time
    multi_results['duration'] = multi_info['duration']

    # Formatting of run_info save file name
    run_name = f'd{multi_start.strftime("%y")}{multi_start.strftime("%m")}{multi_start.strftime("%d")}_h{multi_start.strftime("%H")}m{multi_start.strftime("%M")}.json'
    full_info_path = os.path.join('./saves/runs', run_name)
    result_info_path = os.path.join('./saves/runs', 'r_' + run_name)

    # Save the results
    write_to_json(multi_info, full_info_path)  # Naming format: dYYMMDD_hHHmMM.json
    write_to_json(multi_results, result_info_path)  # Naming format: r_dYYMMDD_hHHmMM.json

    # Print the results
    print(f"| Total time {multi_results['duration']:.2f}")
    for grid_idx, grid_result in multi_results.items():
        if isinstance(grid_idx, int):
            print(f"| Notable parameters -", end='')
            [print(f" {param_name}: {param} ", end='|') for param_name, param in grid_result['setup'].items()]
            print(f"\n| Scores:")
            [print(f"| - {name}: {score:.3f} +/- {conf:.3f} ") for name, (score, conf) in grid_result['confidence_scores'].items()]
            print(f"|---------")


def repeat_run(params: dict = None, num_repeats: int = 2) -> dict:
    reps_start = str(datetime.datetime.now()).split('.')[0]
    reps_start_time = time.time()

    if params is None:
        params = parameters()

    params['num_reps'] = num_repeats
    repetitions = dict()
    test_scores = dict()
    confusion_matrices = dict()
    test_accs = []
    for rep_idx in range(num_repeats):
        params['rep_idx'] = rep_idx

        run_info = run_network(params)

        # Store runtime information
        test_accs.append(run_info['test_info']['accuracy'])
        confusion_matrices[rep_idx] = run_info['test_info'].pop('confusion_matrix', None)
        test_scores[rep_idx] = run_info['test_info']
        repetitions[rep_idx] = run_info

    # Add the scores from each rep into lists, one list for each score type
    scores_concat = next(iter(test_scores.values()))
    for key in scores_concat:
        score_list = []
        for rep_scores in test_scores.values():
            score_list.append(rep_scores[key])
        scores_concat[key] = score_list

    # Calculate the confidence interval for the different scores
    confidence_scores = dict()
    for score_name, score in scores_concat.items():
        confidence_scores[score_name] = mean_confidence_interval(score)

    # Store runtime information
    reps_info = {
        'at': reps_start,
        'duration': time.time() - reps_start_time,
        'accuracies_mean': mean(test_accs),
        'accuracies': test_accs,
        'repetitions': repetitions,
        'scores': scores_concat,
        'confidence_scores': confidence_scores,
        'confusion_matrices': confusion_matrices
    }

    return reps_info


def run_network(params: dict = None) -> dict:
    """
    Run the network. Either called directly, or from repeat_run()
    """

    if params is None:
        params = parameters()

    run_start = datetime.datetime.now()  # Save Date and time of run

    # There are 10 people in the dataset that we want to classify correctly. Might be limited by data_limiter though
    num_classes = len(DATA_LIMITER.subjects)

    # Transforms
    composed = transforms.Compose([
        NormalisePoses(low=1, high=100),
        ChangePoseOrigin(),
        FilterJoints(activator=params['joints_activator'], joint_filter=params['joint_filter']),
        ReshapePoses(),
        #AddNoise(scale=1),
        ToTensor()
    ])

    # Create save dir if they don't exist
    make_save_dirs()

    train_dataset = FOIKinematicPoseDataset(
        data=LOADED_DATA,
        json_path=JSON_PATH_SSD,
        sequence_len=params['sequence_len'],
        data_limiter=DATA_LIMITER,
        transform=composed
    )

    train_sampler, test_sampler, val_sampler = create_samplers(
        dataset_len=len(train_dataset),
        train_split=.70,
        val_split=.15,
        val_from_train=False,
        shuffle=True,
        #split_limit_factor=params['sequence_len']/params['simulated_len']
    )

    train_loader = DataLoader(train_dataset, params['batch_size'], sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(train_dataset, params['batch_size'], sampler=test_sampler, num_workers=4)
    val_loader = DataLoader(train_dataset, params['batch_size'], sampler=val_sampler, num_workers=4)

    if params['should_test_unseen_sessions']:
        test_dataset = FOIKinematicPoseDataset(
            data=LOADED_DATA_EVAL,
            json_path=JSON_PATH_SSD,
            sequence_len=params['sequence_len'],
            data_limiter=DATA_LIMITER_EVAL,
            transform=composed
        )

        _, test_sampler, temp_sampler = create_samplers(
            dataset_len=len(test_dataset),
            train_split=.0,
            val_split=.3,
            val_from_train=False,
            shuffle=True,
            # split_limit_factor=params['sequence_len']/params['simulated_len']
        )

        test_loader = DataLoader(test_dataset, params['batch_size'], sampler=test_sampler, num_workers=4)

        if params['should_val_unseen_sessions']:
            val_sampler = temp_sampler
            val_loader = DataLoader(test_dataset, params['batch_size'], sampler=val_sampler, num_workers=4)


    # Use cuda if possible
    # TODO: Bug - Not everything is being sent to the cpu, fix in other parts of the scripts
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cudnn.benchmark = torch.cuda.is_available()

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

    # Pick loss function depending on params
    if params['loss_type'] == "single":
        loss_function = nn.CrossEntropyLoss()
    elif params['loss_type'] == "siamese":
        raise NotImplementedError
    elif params['loss_type'] == "triplet":
        loss_function = nn.TripletMarginLoss(margin=params['loss_margin'])
    else:
        raise Exception("Invalid network_type")

    #print_setup(setup=run_info, params=params)

    writer = SummaryWriter(TB_RUNS_PATH) if params['should_write'] else None

    start_time = time.time()

    model, learn_info = learn(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        num_epochs=params['num_epochs'],
        device=device,
        classes=DATA_LIMITER.subjects,
        lr_lim=params['learning_rate_lim'],
        loss_type=params['loss_type'],
        task=params['task'],
        max_norm=params['max_norm'],
        step_size=params['step_size'],
        tb_writer=writer,
        params=params
    )

    if params['should_load_checkpoints']:
        print('Loading network checkpoints for testing..')
        # network_modelling/saves/checkpoints/checkpoint_margin_5_epoch_65.pth
        checkpoint = torch.load('./saves/checkpoints/checkpoint_margin_5_epoch_65.pth')
        model.load_state_dict(checkpoint['net'])

    test_info, _ = evaluate(
        train_loader=train_loader,
        eval_loader=val_loader,
        model=model,
        task=params['task'],
        device=device,
        classes=DATA_LIMITER.subjects,
        is_test=False
    )

    # Close TensorBoard writer if it exists
    if writer is not None:
        writer.close()

    # Dict to save all the run info. When learning and evaluating is finished, this will be saved to disk.
    run_info = {
        'at': str(run_start).split('.')[0],
        'duration': time.time() - start_time,
        'device': str(device),
        'model_name': str(type(model)).split('.')[-1][:-2],
        'optimizer_name': str(type(optimizer)).split('.')[-1][:-2],
        'loss_function_name': str(type(loss_function)).split('.')[-1][:-2],
        'transforms': [transform.split(' ')[0].split('.')[1] for transform in str(composed).split('<')[1:]],
        'split': {
            'tot_num_seqs': len(train_dataset), 'batch_size': params['batch_size'], 'train_split': len(train_sampler),
            'val_split': len(val_sampler), 'test_split': len(test_sampler), 'num_train_batches': len(train_loader),
            'num_val_batches': len(val_loader), 'num_test_batches': len(test_loader)
        },
        'learn_info': learn_info,
        'test_info': test_info
    }

    print(f"| Finished testing  | Accuracy: {test_info['accuracy']:.6f} | Run time: {run_info['duration'] :.2f}s\n\n")

    return run_info


if __name__ == "__main__":

    multi_grid(num_repeats=1)
    #grid_search(num_repeats=1)
    #run_network()
    #result_info_path = os.path.join('./saves/runs', 'r_' + run_name)
    #res = read_from_json('./saves/runs/r_d210511_h17m18.json')
    #print(res)



