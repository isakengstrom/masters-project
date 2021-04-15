import torch
import torch.nn.utils as utils


#https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
def train(data_loader, model, optimizer, loss_function, device, network_type, epoch_idx, num_epochs):
    model.train()

    total_accuracy, total_count, total_loss = 0, 0, 0
    log_interval = 1
    num_batches = len(data_loader)

    for batch_idx, batch_sample in enumerate(data_loader):

        if network_type == "single":
            sequence, label = batch_sample

            label = label.to(device)
            sequence = sequence.to(device)

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            sequence_out = model(sequence)

            # Calculate the loss
            loss = loss_function(sequence_out, label)

        elif network_type == "siamese":
            sequence, negative_sequence, label = batch_sample

            label = label.to(device)
            sequence = sequence.to(device)
            negative_sequence = negative_sequence.to(device)

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            sequence_out = model(sequence)
            negative_sequence_out = model(negative_sequence)

            # Calculate the loss
            loss = loss_function(sequence_out, negative_sequence_out)

        elif network_type == "triplet":
            sequence, positive_sequence, negative_sequence, label = batch_sample

            label = label.to(device)
            sequence = sequence.to(device)
            positive_sequence = positive_sequence.to(device)
            negative_sequence = negative_sequence.to(device)

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed the network forward
            sequence_out = model(sequence)
            positive_sequence_out = model(positive_sequence)
            negative_sequence_out = model(negative_sequence)

            # Calculate the loss
            loss = loss_function(sequence_out, positive_sequence_out, negative_sequence_out)

        else:
            raise Exception("Invalid network_type, should be 'single', 'siamese' or 'triplet'")

        # Feed Backward
        loss.backward()

        # Clip norms
        utils.clip_grad_norm_(model.parameters(), max_norm=.1)

        # Update the weights
        optimizer.step()

        _, predicted_label = torch.max(sequence_out, 1)

        total_accuracy += (predicted_label == label).sum().item()
        total_count += label.size(0)
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(f"| Epoch {epoch_idx}/{num_epochs} "
                  f"| Batch {batch_idx+1}/{num_batches} "
                  f"| Accuracy: {total_accuracy/total_count:.6f} "
                  f"| Loss: {total_loss/(batch_idx+1):.6f} ")

        '''
        loss_log.append((train_loss / (batch_idx + 1)))
        acc_log.append(100. * correct / total)
        '''

    return model
