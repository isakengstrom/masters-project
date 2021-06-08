import math
import torch
import torch.nn as nn
import torch.nn.utils as utils


def get_all_triplets_indices(labels):
    """
    Simplified function by Musgrave. Source:
    https://github.com/KevinMusgrave/pytorch-metric-learning/blob/9559b21559ca6fbcb46d2d51d7953166e18f9de6/src/pytorch_metric_learning/utils/loss_and_miner_utils.py#L82
    """

    ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    matches.fill_diagonal_(0)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)


#  https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
def train(data_loader, model, optimizer, loss_function, device, epoch_idx, num_epochs, classes, max_norm, task,
          tb_writer, mining_function, params, class_loss_function=None, class_mining_function=None):

    num_triplets = 200

    # Total values are concatenated for the whole epoch.
    total_accuracy, total_count, total_loss = 0, 0, 0
    curr_accuracy, curr_loss = 0, 0

    num_classes = len(classes)
    num_batches = len(data_loader)
    log_interval = max(math.floor(num_batches / 3), 1)
    is_verbose = False

    # Used for spacing in the formatting of the status prints
    epoch_formatter = int(math.log10(num_epochs)) + 1
    batch_formatter = int(math.log10(num_batches)) + 1

    # Tensorboard variables
    global_step = (epoch_idx - 1) * num_batches  # Global step, unique for each combination of epoch and batch index.
    embeddings = torch.zeros(0, params['embedding_dims']).to(device)

    fake_classes = [0,1,2,3,4,5,6,7,8,9]
    tb_classes = [f"sub{class_idx}" for class_idx in fake_classes]
    tb_class_labels = []

    # Store runtime information
    train_info = {
        'num_batches': num_batches,
        'num_classes': num_classes,
        'accuracy': None,
        'loss': None,
        'global_step': global_step
    }

    model.train()

    for batch_idx, (sequences, labels) in enumerate(data_loader):
        global_step += 1  # Update to make it unique

        full_labels = None
        if params['label_type'] == 'full':
            full_labels, labels = labels
            full_labels = full_labels.to(device)

        sequences, labels = sequences.to(device), labels.to(device)

        # Clear the gradients of all variables
        optimizer.zero_grad()

        if params['use_musgrave']:
            sequences_out = model(sequences)
            indices_tuple = mining_function(sequences_out, labels)
            loss = loss_function(sequences_out, labels, indices_tuple)

            if params['penalise_view']:
                class_sequence_out = model(sequences)
                class_indices_tuple = class_mining_function(class_sequence_out, full_labels)
                class_loss = class_loss_function(class_sequence_out, full_labels, class_indices_tuple)

                # Feed Backward
                loss.backward()
                class_loss.backward()
            else:
                # Feed Backward
                loss.backward()

        elif params['loss_type'] == "single":

            # Feed the network forward
            sequences_out = model(sequences)

            # Calculate the loss
            loss = loss_function(sequences_out, labels)

            # Feed Backward
            loss.backward()

        elif params['loss_type'] == 'triplet':
            # Get all possible triplets
            anc_indices, pos_indices, neg_indices = get_all_triplets_indices(labels)
            # Select a number of triplets at random, amount defined by num_triplets
            random_indices = torch.randint(len(anc_indices), (num_triplets,))

            # Select the random triplets' indices
            anc_indices = anc_indices[random_indices]
            pos_indices = pos_indices[random_indices]
            neg_indices = neg_indices[random_indices]

            # Select the sequences and labels for the triplets
            anc_sequences, anc_labels = sequences[anc_indices, :, :], labels[anc_indices]
            pos_sequences, pos_labels = sequences[pos_indices, :, :], labels[pos_indices]
            neg_sequences, neg_labels = sequences[neg_indices, :, :], labels[neg_indices]

            # Feed the network forward
            anc_sequences_out = model(anc_sequences)
            pos_sequences_out = model(pos_sequences)
            neg_sequences_out = model(neg_sequences)

            # Calculate the loss
            loss = loss_function(anc_sequences_out, pos_sequences_out, neg_sequences_out)

            sequences_out, labels = anc_sequences_out, anc_labels

            # Feed Backward
            loss.backward()
        else:
            raise Exception("Invalid network_type, should be 'single', 'siamese' or 'triplet'")

        # Clip norms, to avoid exploding gradient problems
        if max_norm is not None:
            utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # Update the weights
        optimizer.step()

        if task == 'classification':
            _, predicted_labels = torch.max(sequences_out, 1)
            total_accuracy += (predicted_labels == labels).sum().item()
            total_count += labels.size(0)

        total_loss += loss.item()

        if tb_writer is not None:
            # Store the information for the tensorboard embeddings
            embeddings = torch.cat((embeddings, sequences_out), 0)
            if params['penalise_view']:
                tb_class_labels.extend([tb_classes[label // 10 ** 0 % 10] for label in full_labels])
            else:
                tb_class_labels.extend([tb_classes[label] for label in full_labels])
        # Don't want an update of the status or of tensorboard at first batch_idx, therefore continue
        if batch_idx <= 0:
            continue

        if batch_idx % log_interval == 0:
            #  curr_accuracy = total_accuracy / total_count
            curr_loss = total_loss / (batch_idx + 1)

            print(f"| Epoch {epoch_idx:{epoch_formatter}.0f}/{num_epochs} "
                  f"| Batch {batch_idx + 1:{batch_formatter}.0f}/{num_batches} "
                  f"| Loss: {curr_loss:9.6f} "
                  f"{f'| Accuracy: {curr_accuracy:.6f}' if task == 'classification' else ''} "
                  # f"| Global step {global_step} "
                  )

            if tb_writer is not None:
                # Add scalars to Tensorboard
                tb_writer.add_scalar('Train Accuracy', curr_accuracy, global_step=global_step)
                tb_writer.add_scalar('Train Loss', curr_loss, global_step=global_step)

    # Store runtime information
    train_info['accuracy'] = curr_accuracy
    train_info['loss'] = curr_loss

    if tb_writer is not None:
        # Add embeddings to tensorboard and flush everything in the writer to disk
        tb_writer.add_embedding(embeddings, metadata=tb_class_labels, global_step=global_step)
        tb_writer.flush()

    return model, train_info
