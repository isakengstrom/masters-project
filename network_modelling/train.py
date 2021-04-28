import math
import time
import torch
import torch.nn.utils as utils
from collections import Counter
from torch.utils.tensorboard import SummaryWriter


#  https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
def train(data_loader, model, optimizer, loss_function, device, network_type, epoch_idx, num_epochs, tb_writer:SummaryWriter):

    total_accuracy, total_count, total_loss = 0, 0, 0
    epoch_formatter = int(math.log10(num_epochs)) + 1

    num_batches = len(data_loader)
    log_interval = max(math.floor(num_batches/10), 1)
    batch_formatter = int(math.log10(num_batches)) + 1
    global_step = (epoch_idx - 1) * num_batches

    model.train()

    classes = ["sub0", "sub1", "sub2", "sub3", "sub4", "sub5", "sub6", "sub7", "sub8", "sub9"]
    features = torch.zeros(0, 10).to(device)
    class_labels = []
    embedding_interval = log_interval * 4

    for batch_idx, batch_samples in enumerate(data_loader):
        global_step += 1
        if network_type == "single":
            sequences, labels = batch_samples

            labels = labels.to(device)
            sequences = sequences.to(device).squeeze(1)  # Squeeze(1) is for MNIST

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            sequences_out = model(sequences)

            # Calculate the loss
            loss = loss_function(sequences_out, labels)

        elif network_type == "siamese":
            sequences, negative_sequences, labels = batch_samples

            labels = labels.to(device)
            sequences = sequences.to(device)
            negative_sequences = negative_sequences.to(device)

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            sequences_out = model(sequences)
            negative_sequences_out = model(negative_sequences)

            # Calculate the loss
            loss = loss_function(sequences_out, negative_sequences_out)

        elif network_type == "triplet":
            sequences, positive_sequences, negative_sequences, labels = batch_samples

            labels = labels.to(device)
            sequences = sequences.to(device)
            positive_sequences = positive_sequences.to(device)
            negative_sequences = negative_sequences.to(device)

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            sequences_out = model(sequences)
            positive_sequences_out = model(positive_sequences)
            negative_sequences_out = model(negative_sequences)

            # Calculate the loss
            loss = loss_function(sequences_out, positive_sequences_out, negative_sequences_out)

        else:
            raise Exception("Invalid network_type, should be 'single', 'siamese' or 'triplet'")

        # Feed Backward
        loss.backward()

        # Clip norms
        utils.clip_grad_norm_(model.parameters(), max_norm=.1)

        # Update the weights
        optimizer.step()

        _, predicted_labels = torch.max(sequences_out, 1)

        total_accuracy += (predicted_labels == labels).sum().item()
        total_count += labels.size(0)
        total_loss += loss.item()

        accuracy = total_accuracy/total_count
        loss = total_loss/(batch_idx+1)

        features = torch.cat((features, sequences_out), 0)
        class_labels.extend([classes[prediction] for prediction in predicted_labels])

        if batch_idx > 0:
            if batch_idx % log_interval == 0:
                print(f"| Epoch {epoch_idx:{epoch_formatter}.0f}/{num_epochs} "
                      f"| Batch {batch_idx+1:{batch_formatter}.0f}/{num_batches} "
                      f"| Accuracy: {accuracy:.6f} "
                      f"| Loss: {loss:9.6f} |")
                      #f"| Global step {global_step} "

                tb_writer.add_scalar('Train Accuracy', accuracy, global_step=global_step)
                tb_writer.add_scalar('Train Loss', loss, global_step=global_step)

            if batch_idx % embedding_interval == 0:
                tb_writer.add_embedding(features, metadata=class_labels, global_step=global_step)
                features = torch.zeros(0, 10).to(device)
                class_labels = []

    return model
