import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
def train(model, train_loader, optimizer, loss_function, num_epochs, device, network_type="triplet"):
    loss_log = []
    acc_log = []

    for epoch in range(num_epochs):
        model.train()

        train_loss = []
        correct = 0
        total = 0

        batch_len = len(train_loader)

        for batch_idx, sample_batched in enumerate(train_loader):
            if network_type == "single":
                sequence, label = sample_batched

                label = label.to(device)
                sequence = sequence.to(device)

                # Clear the gradients of all variables
                optimizer.zero_grad()

                # Feed the network forward
                sequence_out = model(sequence)

                # Calculate the loss
                loss = loss_function(sequence_out, label)

            elif network_type == "siamese":
                positive_sequence, negative_sequence, positive_label = sample_batched

                label = positive_label.to(device)
                positive_sequence = positive_sequence.to(device)
                negative_sequence = negative_sequence.to(device)

                # Clear the gradients of all variables
                optimizer.zero_grad()

                # Feed the network forward
                positive_out = model(positive_sequence)
                negative_out = model(negative_sequence)

                # Calculate the loss
                loss = loss_function(positive_out, negative_out)

            elif network_type == "triplet":
                anchor_sequence, positive_sequence, negative_sequence, anchor_label = sample_batched

                label = anchor_label.to(device)
                anchor_sequence = anchor_sequence.to(device)
                positive_sequence = positive_sequence.to(device)
                negative_sequence = negative_sequence.to(device)

                # Clear the gradients of all variables
                optimizer.zero_grad()

                # Feed the network forward
                anchor_out = model(anchor_sequence)
                positive_out = model(positive_sequence)
                negative_out = model(negative_sequence)

                # Calculate the loss
                loss = loss_function(anchor_out, positive_out, negative_out)

            else:
                raise AssertionError("Invalid network_type, should be 'single', 'siamese' or 'triplet'")

            # Feed Backward
            loss.backward()

            # Update the weights
            optimizer.step()

            # Update the training status
            train_loss.append(loss.cpu().detach().numpy())

            # Find the class with the highest output
            #_, predicted = torch.max(embeddings.data, 1)

            # Count number total number of images trained so far and the correctly classified ones
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Iteration {batch_idx+1}/{batch_len}: Loss = {np.mean(train_loss)}")


        #loss_log.append((train_loss / (batch_idx + 1)))
        #acc_log.append(100. * correct / total)

        # Save a checkpoint when the epoch finishes
        state = {
            'epoch': epoch,
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        file_path = f'./checkpoints/checkpoint_{epoch}.pth'
        torch.save(state, file_path)
    '''
    # Save the final model
    torch.save({
        "model_state_dict": model.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict()
    })
        model.state_dict(), './models/saved_models/lstm_triplet_loss_trained_foi_final')
    '''
    print('Training finished')

    train_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        batch_len = len(train_loader)
        for batch_idx, (seq, label) in enumerate(train_loader):
            train_results.append(model(seq.to(device)).cpu().numpy())
            labels.append(label)
            if batch_idx % 20 == 0:
                print(f"Iteration {batch_idx+1}/{batch_len}")

    train_results = np.concatenate(train_results)
    labels = np.concatenate(labels)
    print(train_results.shape)

    #vis
    plt.figure(figsize=(15, 10), facecolor="azure")
    for label in np.unique(labels):
        tmp = train_results[labels == label]
        plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

    plt.legend()
    plt.show()

    return model, loss_log, acc_log
