import math
import torch
import torch.nn.utils as utils
from collections import Counter


#  https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
def train(data_loader, model, optimizer, loss_function, device, network_type, epoch_idx, num_epochs):

    total_accuracy, total_count, total_loss = 0, 0, 0
    num_epochs_digits = int(math.log10(num_epochs)) + 1

    num_batches = len(data_loader)
    log_interval = max(math.floor(num_batches/10), 1)
    num_batches_digits = int(math.log10(num_batches)) + 1

    model.train()

    for batch_idx, batch_samples in enumerate(data_loader):

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

        if batch_idx % log_interval == 0:
            print(f"| Epoch {epoch_idx:{num_epochs_digits}.0f}/{num_epochs} "
                  f"| Batch {batch_idx+1:{num_batches_digits}.0f}/{num_batches} "
                  f"| Accuracy: {total_accuracy/total_count:.6f} "
                  f"| Loss: {(total_loss/(batch_idx+1)):9.6f} |")

            if True:
                lst = (predicted_labels == labels).cpu().numpy().astype(int)
                print("  Predicted labels:", predicted_labels.data.cpu().numpy(), "\n",
                      "    Actual labels:", labels.data.cpu().numpy(), "\n",
                      "True/False labels:", lst, " ", Counter(lst))
                print('-' * 72)

            #print(total_accuracy, total_count, total_loss)
            #total_accuracy, total_count = 0, 0

        '''
        loss_log.append((train_loss / (batch_idx + 1)))
        acc_log.append(100. * correct / total)
        '''

    return model
