import os


def print_setup(setup: dict, params: dict):
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


def make_save_dirs_for_net():
    if not os.path.isdir('./saves'):
        os.mkdir('./saves')

    # Add checkpoint dir if it doesn't exist
    if not os.path.isdir('./saves/checkpoints'):
        os.mkdir('./saves/checkpoints')

    if not os.path.isdir('./saves/checkpoints/final'):
        os.mkdir('./saves/checkpoints/final')

    if not os.path.isdir('./saves/checkpoints/last_net_run'):
        os.mkdir('./saves/checkpoints/last_net_run')

    if not os.path.isdir('./saves/runs'):
        os.mkdir('./saves/runs')
