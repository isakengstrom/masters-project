import math
import torch
from collections import Counter

import torch.nn.utils as utils
from torch.utils.tensorboard import SummaryWriter


def get_all_triplets_indices(labels, ref_labels=None):
    """
    Function from Musgrave:
    https://github.com/KevinMusgrave/pytorch-metric-learning/blob/9559b21559ca6fbcb46d2d51d7953166e18f9de6/src/pytorch_metric_learning/utils/loss_and_miner_utils.py#L82
    """
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
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

    for batch_idx, batch_samples in enumerate(data_loader):
        global_step += 1  # Update to make it unique

        if loss_type == "single":
            main_sequences, main_labels = batch_samples["main"]

            main_sequences = main_sequences.to(device)  # .squeeze(1) # Squeeze is for MNIST tests
            main_labels = main_labels.to(device)

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            anchor_sequences_out = model(main_sequences)

            # Calculate the loss
            loss = loss_function(anchor_sequences_out, main_labels)

        elif loss_type == "siamese":
            main_sequences, main_labels = batch_samples["main"]
            negative_sequences, _ = batch_samples["negative"]

            main_sequences, main_labels = main_sequences.to(device), main_labels.to(device)
            negative_sequences = negative_sequences.to(device)

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            anchor_sequences_out = model(main_sequences)
            negative_sequences_out = model(negative_sequences)

            # Calculate the loss
            loss = loss_function(anchor_sequences_out, negative_sequences_out)

        elif loss_type == "triplet":

            main_sequences, main_labels = batch_samples["main"]
            main_sequences, main_labels = main_sequences.to(device), main_labels.to(device)

            anc_indices, pos_indices, neg_indices = get_all_triplets_indices(main_labels)
            random_indices = torch.randint(len(anc_indices), (200,))

            anc_indices = anc_indices[random_indices]
            pos_indices = pos_indices[random_indices]
            neg_indices = neg_indices[random_indices]

            anc_labels, anc_sequences = main_labels[anc_indices], main_sequences[anc_indices, :, :]
            pos_labels, pos_sequences = main_labels[pos_indices], main_sequences[pos_indices, :, :]
            neg_labels, neg_sequences = main_labels[neg_indices], main_sequences[neg_indices, :, :]

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            anchor_sequences_out = model(anc_sequences)
            positive_sequences_out = model(pos_sequences)
            negative_sequences_out = model(neg_sequences)

            # Calculate the loss
            loss = loss_function(anchor_sequences_out, positive_sequences_out, negative_sequences_out)

        else:
            raise Exception("Invalid network_type, should be 'single', 'siamese' or 'triplet'")

        # Feed Backward
        loss.backward()

        # Clip norms, to avoid exploding gradient problems
        if max_norm is not None:
            utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # Update the weights
        optimizer.step()

        _, predicted_labels = torch.max(anchor_sequences_out, 1)

        total_accuracy += (predicted_labels == anc_labels).sum().item()
        total_count += anc_labels.size(0)
        total_loss += loss.item()

        if tb_writer is not None:
            # Store the information for the tensorboard embeddings
            tb_features = torch.cat((tb_features, anchor_sequences_out), 0)
            tb_class_labels.extend([tb_classes[prediction] for prediction in anc_labels])

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
                print("Actual labels:", main_labels.data.cpu().numpy())

                '''
                anchor_sequences = torch.zeros(0).to(device)
                positive_sequences = torch.zeros(0).to(device)
                negative_sequences = torch.zeros(0).to(device)
                for class_idx in classes:
                    if class_idx == 0 or class_idx == 1:

                        print('class idx', class_idx)
                        #print(anchor_sequences.size())
                        all_positives = torch.where(main_labels == class_idx)[0]
                        negatives = torch.where(main_labels != class_idx)[0]

                        # Shuffle indices of negatives
                        idx = torch.randperm(negatives.nelement())
                        negatives = negatives.view(-1)[idx].view(negatives.size())

                        # Anchor/positive index combinations
                        anchors_and_positives = torch.combinations(all_positives, r=2)

                        max_amount = min(anchors_and_positives.size()[0], negatives.size()[0])
                        anchors_and_positives = anchors_and_positives[:max_amount]
                        negative_idxs = negatives[:max_amount]
                        anchor_idxs, positive_idxs = anchors_and_positives.split(1, 1)
                        anchor_idxs, positive_idxs = anchor_idxs.squeeze(), positive_idxs.squeeze()
                        print(main_sequences[anchor_idxs].size())
                        print(main_sequences.size())
                        print('anchor idxs size: ', anchor_idxs.size())
                        

                        anchor_sequences = torch.cat((anchor_sequences, main_sequences[anchor_idxs]), 0).squeeze()
                        positive_sequences = torch.cat((positive_sequences, main_sequences[positive_idxs]), 0).squeeze()
                        negative_sequences = torch.cat((negative_sequences, main_sequences[negative_idxs]), 0).squeeze()

                        print(anchor_sequences.size())
                '''
                #print(anchors)
                        #print(all_positives)
                        #print(negatives[:max_amount])



                        #print(negatives[0].size())
                '''
                lst = (predicted_labels == main_labels).cpu().numpy().astype(int)
                print("  Predicted labels:", predicted_labels.data.cpu().numpy(), "\n",
                      "    Actual labels:", main_labels.data.cpu().numpy(), "\n",
                      "True/False labels:", lst, " ", Counter(lst))
                print('-' * 72)
                '''

    # Store runtime information
    train_info['accuracy'] = curr_accuracy
    train_info['loss'] = curr_loss

    if tb_writer is not None:
        # Add embeddings to tensorboard and flush everything in the writer to disk
        tb_writer.add_embedding(tb_features, metadata=tb_class_labels, global_step=global_step)
        tb_writer.flush()

    return model, train_info
