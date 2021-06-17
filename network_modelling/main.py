import itertools
import math
import os
import random
import time
import datetime
from statistics import mean

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir /home/isaeng/Exjobb/states/runs/
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses, miners, distances, reducers

from learn import learn
from evaluate import evaluate
from models.RNN import GenRNNNet
from dataset import FOIKinematicPoseDataset, DataLimiter, LoadData, create_samplers
from sequence_transforms import FilterJoints, ChangePoseOrigin, ToTensor, NormalisePoses, ReshapePoses  # AddNoise

from helpers import write_to_json, print_setup, make_save_dirs_for_net
from helpers.paths import EXTR_PATH_SSD, TB_RUNS_PATH, BACKUP_MODELS_PATH, RUNS_INFO_PATH
from helpers.result_formatter import mean_confidence_interval


# Const paths
JSON_PATH_SSD = os.path.join(EXTR_PATH_SSD, "final_data_info.json")
ROOT_DIR_SSD = os.path.join(EXTR_PATH_SSD, "final/")


# Data limiter: Go to definition for more info
DATA_LOAD_LIMITER = DataLimiter(
    subjects=None,
    sessions=[0],
    views=None,
)


# Load the data of the dataset into memory from json
print(f"| Loading data into memory..")
LOAD_START_TIME = time.time()
LOADED_DATA = LoadData(root_dir=ROOT_DIR_SSD, data_limiter=DATA_LOAD_LIMITER, num_workers=8)
print(f"| Loading finished in {time.time() - LOAD_START_TIME:0.1f}s")
print('-' * 72)


def parameters():
    """
    Initialise the hyperparameters (+ some other params)

    - Batch size - tightly linked with gradient descent. The number of samples worked through before the params of the
      model are updated.
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

    params['views'] = list(DATA_LOAD_LIMITER.views)

    params['num_epochs'] = 250
    params['batch_size'] = 32
    params['learning_rate'] = 0.0005  # 5e-4  # 0.05 5e-4 5e-8

    params['learning_rate_lim'] = 5.1e-6  # The network starts its breaking phase when this LR is reached
    params['load_best_post_lr_step'] = False  # If the best performing model should be loaded before a LR step
    params['step_size'] = 1  # Used in the lr_scheduler of Torch
    params['bad_val_lim_first'] = 1  # How many un-increasing validations to allow before taking the FIRST step
    params['bad_val_lim'] = 1  # How many un-increasing validations to allow before taking the REMAINING steps

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
    params['simulated_len'] = 800  # A limiter to how many sequences can be created (weird param used to evaluate RNNs)

    # Network / Model params
    params['num_layers'] = 2  # Number of stacked RNN layers
    params['hidden_size'] = 256*2  # Number of features in hidden state
    params['net_type'] = 'gru'
    params['bidirectional'] = False
    params['max_norm'] = 1

    # If the fc layer is used, the network will apply a fully connected layer to transform
    #  the embedding space to the dimensionality of embedding_dims
    params['use_fc_layer'] = True

    # Reduce the embedding space if fully connected layer is used, otherwise, the embedding space will have the same
    #  dimensionality as the hidden_size of the RNN network
    if params['use_fc_layer']:
        params['embedding_dims'] = 10
    else:
        params['embedding_dims'] = params['hidden_size']

    # Loss settings
    params['task'] = 'metric'  # 'classification'/'metric'
    params['loss_type'] = 'triplet'  # 'single'/'triplet'/'contrastive'
    params['loss_margin'] = 0.1  # The margin for certain loss functions

    # PyTorch Deep metric learning specific params
    params['use_musgrave'] = True  # Use the PyTorch deep metric library?
    params['metric_distance'] = 'lp_distance'  # Which distance metric to use

    # Settings for running double losses, one for subject, the other for view
    params['penalise_view'] = True  # Run a second loss function for the deep metric learning to penalise the view?
    params['label_type'] = 'sub'  # How to label each sequence of the dataset. See more in
    params['class_loss_margin'] = 0.1

    # Settings for the network run
    params['num_repeats'] = 1  # Run a network setup multiple times? Will save the confidence score

    params['should_learn'] = True  # Learn/train the network?
    params['should_write'] = True  # Write to TensorBoard?
    params['should_test_unseen_sessions'] = False  # Test the unseen sessions (sess1) for sub 0 and 1
    params['should_val_unseen_sessions'] = False  # Val split from unseen sessions, otherwise uses seen session (sess0)
    params['should_test_unseen_subjects'] = False  # Test
    params['should_test_unseen_views'] = True
    params['num_unseen_sub'] = 3

    params['checkpoint_to_load'] = os.path.join(RUNS_INFO_PATH, 'backups/512_dim/d210601_h09m46_ṛun1_rep1_e58_best.pth')

    if params['should_learn']:
        params['should_load_checkpoints'] = False
    else:
        params['should_load_checkpoints'] = True

    return params


def multi_grid():
    """
    Run multiple grid searches. Specify the grids in the grids list.

    Overrides parameter(), however,  not all params are cleared to work, might e.g. be problematic if a param is a list.
    """

    grids = [
        {
            'num_repeats': 3,
            'loss_margin': 0.5,
            'class_loss_margin': 0.1,
            'penalise_view': True,
            'batch_size': 64,
            'loss_type': 'triplet',
            'num_epochs': 250,
            'use_fc_layer': True,
            'embedding_dims': 10,
            'bad_val_lim_first': 5,
            'bad_val_lim': 3,
            'label_type': 'full'
        },
    ]

    # Loop over all grids
    num_grids = len(grids)
    for grid_idx, grid in enumerate(grids):
        print(f"| Grid {grid_idx+1}/{num_grids}")
        multi_results = grid_search(grid_idx, grid)
        print(f"| ", "_____-----"*10)


def grid_search(outer_grid_idx=-1, grid=None):
    multi_start = datetime.datetime.now()  # Date and time of start
    multi_start_time = time.time()  # Time of start

    # Set a grid if this function wasn't called from multi_grid()
    if grid is None:
        # Override any parameter in parameter()
        grid = {
            # 'bidirectional': [False, True],
            # 'net_type': ['gru'],
            # 'sequence_len': [5, 10, 15, 20, 25],
            # 'hidden_size': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            # 'max_norm': [0.01, 0.1, 1]
            # 'num_epochs': 2
            # 'loss_margin': [50, 100]
        }

    # Wrap every value in a list if it isn't already the case
    for key, value in grid.items():
        grid[key] = [value]

    # Create every combination from the lists in grid
    all_grid_combinations = [dict(zip(grid, value)) for value in itertools.product(*grid.values())]

    num_runs = len(all_grid_combinations)
    run_formatter = int(math.log10(num_runs)) + 1  # Used for printing spacing

    params = parameters()
    params['num_runs'] = num_runs

    # Store runtime information
    multi_info = {'at': str(multi_start).split('.')[0], 'duration': None, 'num_runs': num_runs, 'num_reps': params['num_repeats']}
    multi_runs = dict()
    multi_results = dict()

    # Formatting of run_info save file name
    run_name = f'd{multi_start.strftime("%y")}{multi_start.strftime("%m")}{multi_start.strftime("%d")}_h{multi_start.strftime("%H")}m{multi_start.strftime("%M")}.json'
    params['run_name'] = run_name

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
        reps_info = repeat_run(params)

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

    return multi_results


def repeat_run(params: dict = None) -> dict:
    """
    Run a network of the same settings a number of times. Used to get the confidence interval scores of several runs
    """

    reps_start = str(datetime.datetime.now()).split('.')[0]
    reps_start_time = time.time()

    if params is None:
        params = parameters()

    repetitions = dict()
    test_scores = dict()
    confusion_matrices = dict()
    test_accs = []

    for rep_idx in range(params['num_repeats']):
        params['rep_idx'] = rep_idx

        # Run the network
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


def generate_random_class_split(num_classes=10, test_split=3):
    """
    Generates a random class split used when training on some subjects and tested on others
    """
    classes = set(range(num_classes))
    assert test_split < num_classes
    test_classes = set(random.sample(classes, test_split))
    learn_classes = list([x for x in classes if x not in test_classes])
    test_classes = list(test_classes)
    return learn_classes, test_classes


def run_network(params: dict = None) -> dict:
    """
    Run the network. Either called directly, or from repeat_run()
    """

    if params is None:
        params = parameters()

    run_start = datetime.datetime.now()  # Save Date and time of run

    # Instantiate new data limiter
    data_limiter = DataLimiter(
        subjects=None,
        sessions=[0],
        views=params['views'],
    )

    # Transforms
    composed = transforms.Compose([
        NormalisePoses(low=1, high=100),
        ChangePoseOrigin(),
        FilterJoints(activator=params['joints_activator'], joint_filter=params['joint_filter']),
        ReshapePoses(),
        # AddNoise(scale=1),
        ToTensor()
    ])

    # Create save dir if they don't exist
    make_save_dirs_for_net()

    train_dataset = FOIKinematicPoseDataset(
        data=LOADED_DATA,
        json_path=JSON_PATH_SSD,
        sequence_len=params['sequence_len'],
        data_limiter=data_limiter,
        transform=composed,
        label_type=params['label_type']
    )

    # Create samplers, see definition for details
    train_sampler, test_sampler, val_sampler = create_samplers(
        dataset_len=len(train_dataset),
        train_split=.70,
        val_split=.15,
        val_from_train=False,
        shuffle=True,
        # split_limit_factor=params['sequence_len']/params['simulated_len']
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, params['batch_size'], sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(train_dataset, params['batch_size'], sampler=test_sampler, num_workers=4)
    val_loader = DataLoader(train_dataset, params['batch_size'], sampler=val_sampler, num_workers=4)
    comparison_loader = train_loader  # Used for comparing embeddings

    # This block is run when the task is to test unseen sessions
    if params['should_test_unseen_sessions']:
        print("| Getting unseen sessions")

        unseen_sessions_limiter = DataLimiter(
            subjects=[0, 1],
            sessions=[1],
            views=DATA_LOAD_LIMITER.views,
        )

        print(f"| Loading data into memory..")
        load_start_time = time.time()
        unseen_sessions_data = LoadData(root_dir=ROOT_DIR_SSD, data_limiter=unseen_sessions_limiter, num_workers=8)
        print(f"| Loading finished in {time.time() - load_start_time:0.1f}s")
        print('-' * 72)

        test_dataset = FOIKinematicPoseDataset(
            data=unseen_sessions_data,
            json_path=JSON_PATH_SSD,
            sequence_len=params['sequence_len'],
            data_limiter=unseen_sessions_limiter,
            transform=composed,
            label_type=params['label_type']
        )

        _, test_sampler, temp_sampler = create_samplers(
            dataset_len=len(test_dataset),
            train_split=.0,
            val_split=.0,
            val_from_train=False,
            shuffle=True,
        )

        test_loader = DataLoader(test_dataset, params['batch_size'], sampler=test_sampler, num_workers=4)

        if params['should_val_unseen_sessions']:
            val_sampler = temp_sampler
            val_loader = DataLoader(test_dataset, params['batch_size'], sampler=val_sampler, num_workers=4)

    # This block is run when the task is to test unseen subjects
    if params['should_test_unseen_subjects']:
        learn_classes, test_classes = generate_random_class_split(len(DATA_LOAD_LIMITER.subjects), params['num_unseen_sub'])

        data_limiter = DataLimiter(
            subjects=learn_classes,
            sessions=[0],
            views=params['views'],
        )

        test_limiter = DataLimiter(
            subjects=test_classes,
            sessions=[0],
            views=params['views'],
        )

        print("| Running on unseen subjects")
        print(f'| Learning classes: {data_limiter.subjects} | Testing Classes: {test_limiter.subjects}')

        train_dataset = FOIKinematicPoseDataset(
            data=LOADED_DATA,
            json_path=JSON_PATH_SSD,
            sequence_len=params['sequence_len'],
            data_limiter=data_limiter,
            transform=composed,
            label_type=params['label_type']
        )

        test_dataset = FOIKinematicPoseDataset(
            data=LOADED_DATA,
            json_path=JSON_PATH_SSD,
            sequence_len=params['sequence_len'],
            data_limiter=test_limiter,
            transform=composed,
            label_type=params['label_type']
        )

        train_sampler, _, val_sampler = create_samplers(
            dataset_len=len(train_dataset),
            train_split=.85,
            val_split=.15,
            val_from_train=False,
            shuffle=True,
        )

        comparison_sampler, test_sampler, _ = create_samplers(
            dataset_len=len(test_dataset),
            train_split=.7,
            val_split=.0,
            val_from_train=False,
            shuffle=True,
        )

        train_loader = DataLoader(train_dataset, params['batch_size'], sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(train_dataset, params['batch_size'], sampler=val_sampler, num_workers=4)
        test_loader = DataLoader(test_dataset, params['batch_size'], sampler=test_sampler, num_workers=4)
        comparison_loader = DataLoader(test_dataset, params['batch_size'], sampler=comparison_sampler, num_workers=4)

    # This block is run when the task is to test unseen views
    if params['should_test_unseen_views']:

        learn_views, test_view = generate_random_class_split(len(DATA_LOAD_LIMITER.views), 1)

        data_limiter = DataLimiter(
            subjects=data_limiter.subjects,
            sessions=[0],
            views=learn_views,
        )

        test_limiter = DataLimiter(
            subjects=data_limiter.subjects,
            sessions=[0],
            views=test_view,
        )

        print("| Running on unseen subjects")
        print(f'| Learning Views: {data_limiter.views} | Testing View: {test_limiter.views}')

        train_dataset = FOIKinematicPoseDataset(
            data=LOADED_DATA,
            json_path=JSON_PATH_SSD,
            sequence_len=params['sequence_len'],
            data_limiter=data_limiter,
            transform=composed,
            label_type=params['label_type']
        )

        test_dataset = FOIKinematicPoseDataset(
            data=LOADED_DATA,
            json_path=JSON_PATH_SSD,
            sequence_len=params['sequence_len'],
            data_limiter=test_limiter,
            transform=composed,
            label_type=params['label_type']
        )

        train_sampler, _, val_sampler = create_samplers(
            dataset_len=len(train_dataset),
            train_split=.85,
            val_split=.15,
            val_from_train=False,
            shuffle=True,
        )

        comparison_sampler, test_sampler, _ = create_samplers(
            dataset_len=len(test_dataset),
            train_split=.7,
            val_split=.0,
            val_from_train=False,
            shuffle=True,
        )

        train_loader = DataLoader(train_dataset, params['batch_size'], sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(train_dataset, params['batch_size'], sampler=val_sampler, num_workers=4)
        test_loader = DataLoader(test_dataset, params['batch_size'], sampler=test_sampler, num_workers=4)
        comparison_loader = DataLoader(test_dataset, params['batch_size'], sampler=comparison_sampler, num_workers=4)

    # Use cuda if possible
    # TODO: Bug - Not everything is being sent to the cpu, fix in other parts of the scripts
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cudnn.benchmark = torch.cuda.is_available()

    # The recurrent neural net model, RNN, GRU or LSTM
    model = GenRNNNet(
        input_size=params['input_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        use_fc_layer=params['use_fc_layer'],
        embedding_dims=params['embedding_dims'],
        device=device,
        bidirectional=params['bidirectional'],
        net_type=params['net_type'],
    ).to(device)

    # The optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    loss_function = None
    mining_function = None

    class_loss_function = None
    class_mining_function = None

    #reducer = None


    if params['use_musgrave']:
        reducer = reducers.ThresholdReducer(low=0)

        if params['metric_distance'] == 'cosine_similarity':
            distance = distances.CosineSimilarity()
        elif params['metric_distance'] == 'lp_distance':
            distance = distances.LpDistance()
        else:
            raise Exception("Invalid metric distance")

        if params['loss_type'] == "single":
            raise NotImplementedError
        elif params['loss_type'] == 'contrastive':
            mining_function = miners.PairMarginMiner(pos_margin=0, neg_margin=params['loss_margin'])
            loss_function = losses.ContrastiveLoss(pos_margin=0, neg_margin=params['loss_margin'])

        elif params['loss_type'] == "triplet":
            print('Using Musgrave triplet')
            mining_function = miners.TripletMarginMiner(margin=params['loss_margin'], distance=distance, type_of_triplets="semihard")
            loss_function = losses.TripletMarginLoss(margin=params['loss_margin'], distance=distance, reducer=reducer)

            if params['penalise_view']:
                print(f"| Penalising views")
                class_mining_function = miners.TripletMarginMiner(margin=params['class_loss_margin'], distance=distance, type_of_triplets="semihard")
                class_loss_function = losses.TripletMarginLoss(margin=params['class_loss_margin'], distance=distance, reducer=reducer)

        elif params['loss_type'] == 'n_pairs':
            #loss_function = losses.NPairsLoss()
            raise NotImplementedError
    else:
        if params['loss_type'] == "single":
            loss_function = nn.CrossEntropyLoss()

        elif params['loss_type'] == "triplet":
            loss_function = nn.TripletMarginLoss(margin=params['loss_margin'])
        else:
            raise Exception("Invalid loss type when not using musgrave implementation")

    # print_setup(setup=run_info, params=params)
    writer = SummaryWriter(TB_RUNS_PATH) if params['should_write'] else None

    start_time = time.time()

    model, learn_info = learn(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        class_loss_function=class_loss_function,
        mining_function=mining_function,
        class_mining_function=class_mining_function,
        num_epochs=params['num_epochs'],
        device=device,
        classes=data_limiter.subjects,
        task=params['task'],
        tb_writer=writer,
        params=params,
    )

    if params['should_load_checkpoints']:
        print('| Loading network checkpoints for testing..')
        checkpoint = torch.load(params['checkpoint_to_load'])
        model.load_state_dict(checkpoint['net'])

    test_info, _ = evaluate(
        train_loader=comparison_loader,
        eval_loader=test_loader,
        model=model,
        task=params['task'],
        device=device,
        tb_writer=writer,
        is_test=False,
        embedding_dims=params['embedding_dims'],
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

    multi_grid()
    # grid_search()
    # run_network()
    # result_info_path = os.path.join('./saves/runs', 'r_' + run_name)
    # res = read_from_json('./saves/runs/r_d210511_h17m18.json')
    # print(res)
