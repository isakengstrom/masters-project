import math
import torch
from collections import Counter

import torch.nn.utils as utils
from torch.utils.tensorboard import SummaryWriter


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
def train(data_loader, model, optimizer, loss_function, device, loss_type, epoch_idx, num_epochs, classes, max_norm,
          tb_writer: SummaryWriter = None):

    # Total values are concatenated for the whole epoch.
    total_accuracy, total_count, total_loss = 0, 0, 0
    curr_accuracy, curr_loss = 0, 0

    num_classes = len(classes)
    num_batches = len(data_loader)
    log_interval = max(math.floor(num_batches/3), 1)
    is_verbose = False

    # Used for spacing in the formatting of the status prints
    epoch_formatter = int(math.log10(num_epochs)) + 1
    batch_formatter = int(math.log10(num_batches)) + 1

    # Tensorboard variables
    global_step = (epoch_idx - 1) * num_batches  # Global step, unique for each combination of epoch and batch index.
    tb_classes = [f"sub{class_idx}" for class_idx in classes]
    tb_features = torch.zeros(0, num_classes).to(device)
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

        sequences, labels = sequences.to(device), labels.to(device)

        if loss_type == "single":

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            sequences_out = model(sequences)

            # Calculate the loss
            loss = loss_function(sequences_out, labels)

        elif loss_type == "siamese":
            raise NotImplementedError

        elif loss_type == "triplet":
            anc_indices, pos_indices, neg_indices = get_all_triplets_indices(labels)
            random_indices = torch.randint(len(anc_indices), (200,))

            anc_indices = anc_indices[random_indices]
            pos_indices = pos_indices[random_indices]
            neg_indices = neg_indices[random_indices]

            anc_labels, anc_sequences = labels[anc_indices], sequences[anc_indices, :, :]
            pos_labels, pos_sequences = labels[pos_indices], sequences[pos_indices, :, :]
            neg_labels, neg_sequences = labels[neg_indices], sequences[neg_indices, :, :]

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            anc_sequences_out = model(anc_sequences)
            pos_sequences_out = model(pos_sequences)
            neg_sequences_out = model(neg_sequences)

            # Calculate the loss
            loss = loss_function(anc_sequences_out, pos_sequences_out, neg_sequences_out)

            sequences_out, labels = anc_sequences_out, anc_labels
        else:
            raise Exception("Invalid network_type, should be 'single', 'siamese' or 'triplet'")

        # Feed Backward
        loss.backward()

        # Clip norms, to avoid exploding gradient problems
        if max_norm is not None:
            utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # Update the weights
        optimizer.step()

        _, predicted_labels = torch.max(sequences_out, 1)

        total_accuracy += (predicted_labels == labels).sum().item()
        total_count += labels.size(0)
        total_loss += loss.item()

        if tb_writer is not None:
            # Store the information for the tensorboard embeddings
            tb_features = torch.cat((tb_features, sequences_out), 0)
            tb_class_labels.extend([tb_classes[prediction] for prediction in labels])

        # Don't want an update of the status or of tensorboard at first batch_idx, therefore continue
        if batch_idx <= 0:
            continue

        if batch_idx % log_interval == 0:
            curr_accuracy = total_accuracy/total_count
            curr_loss = total_loss/(batch_idx+1)

            print(f"| Epoch {epoch_idx:{epoch_formatter}.0f}/{num_epochs} "
                  f"| Batch {batch_idx+1:{batch_formatter}.0f}/{num_batches} "
                  f"| Accuracy: {curr_accuracy:.6f} "
                  f"| Loss: {curr_loss:9.6f} |")
                  #f"| Global step {global_step} "

            if tb_writer is not None:
                # Add scalars to Tensorboard
                tb_writer.add_scalar('Train Accuracy', curr_accuracy, global_step=global_step)
                tb_writer.add_scalar('Train Loss', curr_loss, global_step=global_step)

            if is_verbose:
                print("Actual labels:", labels.data.cpu().numpy())

                lst = (predicted_labels == labels).cpu().numpy().astype(int)
                print("  Predicted labels:", predicted_labels.data.cpu().numpy(), "\n",
                      "    Actual labels:", labels.data.cpu().numpy(), "\n",
                      "True/False labels:", lst, " ", Counter(lst))
                print('-' * 72)

    # Store runtime information
    train_info['accuracy'] = curr_accuracy
    train_info['loss'] = curr_loss

    if tb_writer is not None:
        # Add embeddings to tensorboard and flush everything in the writer to disk
        tb_writer.add_embedding(tb_features, metadata=tb_class_labels, global_step=global_step)
        tb_writer.flush()

    return model, train_info
