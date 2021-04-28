import math
import time
import torch
import torch.nn.utils as utils
from collections import Counter
from torch.utils.tensorboard import SummaryWriter


#  https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
def train(data_loader, model, optimizer, loss_function, device, network_type, epoch_idx, num_epochs, classes, tb_writer:SummaryWriter):

    # Total values are concatenated for the whole epoch.
    total_accuracy, total_count, total_loss = 0, 0, 0

    num_classes = len(classes)
    num_batches = len(data_loader)
    log_interval = max(math.floor(num_batches/10), 1)

    # Used for spacing in the formatting of the status prints
    epoch_formatter = int(math.log10(num_epochs)) + 1
    batch_formatter = int(math.log10(num_batches)) + 1

    # Tensorboard variables
    global_step = (epoch_idx - 1) * num_batches  # Global step, unique for each combination of epoch and batch index.
    embedding_interval = log_interval * 4  # How often the embeddings should be updated

    tb_classes = []
    for class_idx in classes:
        tb_classes.append(f"sub{class_idx}")

    tb_features = torch.zeros(0, num_classes).to(device)
    tb_class_labels = []

    model.train()

    for batch_idx, batch_samples in enumerate(data_loader):
        global_step += 1  # Update to make it unique

        if network_type == "single":
            main_sequences, main_labels = batch_samples["main"]

            main_sequences = main_sequences.to(device).squeeze(1)  # Squeeze(1) is for MNIST
            main_labels = main_labels.to(device)

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            main_sequences_out = model(main_sequences)

            # Calculate the loss
            loss = loss_function(main_sequences_out, main_labels)

        elif network_type == "siamese":
            main_sequences, main_labels = batch_samples["main"]
            negative_sequences, _ = batch_samples["negative"]

            main_labels = main_labels.to(device)
            main_sequences = main_sequences.to(device)
            negative_sequences = negative_sequences.to(device)

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            main_sequences_out = model(main_sequences)
            negative_sequences_out = model(negative_sequences)

            # Calculate the loss
            loss = loss_function(main_sequences_out, negative_sequences_out)

        elif network_type == "triplet":
            main_sequences, main_labels = batch_samples["main"]
            positive_sequences, _ = batch_samples["positive"]
            negative_sequences, _ = batch_samples["negative"]

            main_labels = main_labels.to(device)
            main_sequences = main_sequences.to(device)
            positive_sequences = positive_sequences.to(device)
            negative_sequences = negative_sequences.to(device)

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            main_sequences_out = model(main_sequences)
            positive_sequences_out = model(positive_sequences)
            negative_sequences_out = model(negative_sequences)

            # Calculate the loss
            loss = loss_function(main_sequences_out, positive_sequences_out, negative_sequences_out)

        else:
            raise Exception("Invalid network_type, should be 'single', 'siamese' or 'triplet'")

        # Feed Backward
        loss.backward()

        # Clip norms
        utils.clip_grad_norm_(model.parameters(), max_norm=.1)

        # Update the weights
        optimizer.step()

        _, predicted_labels = torch.max(main_sequences_out, 1)

        total_accuracy += (predicted_labels == main_labels).sum().item()
        total_count += main_labels.size(0)
        total_loss += loss.item()

        # Store the information for the tensorboard embeddings
        tb_features = torch.cat((tb_features, main_sequences_out), 0)
        tb_class_labels.extend([tb_classes[prediction] for prediction in predicted_labels])

        # Don't want update of status or of tensorboard at first batch_idx, therefore continue
        if batch_idx <= 0:
            continue

        if batch_idx % log_interval == 0:
            accuracy = total_accuracy/total_count
            loss = total_loss/(batch_idx+1)

            print(f"| Epoch {epoch_idx:{epoch_formatter}.0f}/{num_epochs} "
                  f"| Batch {batch_idx+1:{batch_formatter}.0f}/{num_batches} "
                  f"| Accuracy: {accuracy:.6f} "
                  f"| Loss: {loss:9.6f} |")
                  #f"| Global step {global_step} "

            # Add scalars to Tensorboard
            tb_writer.add_scalar('Train Accuracy', accuracy, global_step=global_step)
            tb_writer.add_scalar('Train Loss', loss, global_step=global_step)

        # Add embeddings and reset variables
        #if batch_idx % embedding_interval == 0:
    tb_writer.add_embedding(tb_features, metadata=tb_class_labels, global_step=global_step)
    tb_features = torch.zeros(0, num_classes).to(device)
    tb_class_labels = []

    return model
