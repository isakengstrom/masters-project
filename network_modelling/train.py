import os
import torch
import torch.nn as nn
import numpy as np


#https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
def train(model, train_loader, optimizer, loss_function, num_epochs, device, network_type="triplet"):
    loss_log = []
    acc_log = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")

        model.train()

        train_loss = []
        correct = 0
        total = 0

        batch_len = len(train_loader)

        for batch_idx, sample_batched in enumerate(train_loader):
            label = None
            if network_type == "siamese":
                raise NotImplementedError

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
                raise NotImplementedError

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

            if batch_idx % 2 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Iteration {batch_idx+1}/{batch_len}: Loss = {np.mean(train_loss)}")


        #loss_log.append((train_loss / (batch_idx + 1)))
        #acc_log.append(100. * correct / total)

        # Save a checkpoint when the epoch finishes
        state = {
            'epoch': epoch,
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        file_path = f'./checkpoints/checkpoint_{epoch}.ckpt'
        torch.save(state, file_path)

    # Save the final model
    torch.save(model.state_dict(), './models/saved_models/lstm_triplet_loss_trained_foi_final')
    print('Training finished')
    return model, loss_log, acc_log
