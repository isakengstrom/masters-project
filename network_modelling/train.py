import os
import torch
import torch.nn as nn


def train(model, train_loader, optimizer, triplet_loss, device, start_epoch, num_epochs):
    loss_log = []
    acc_log = []

    for epoch in range(start_epoch, num_epochs+1):
        print("\nEpoch {}".format(epoch))

        model.train()

        train_loss = 0
        correct = 0
        total = 0

        batch_len = len(train_loader)

        for batch_idx, sample_batched in enumerate(train_loader):
            data = sample_batched["sequence"], labels = sample_batched["label"]

            data, labels = data.to(device), labels.to(device)

            #data, labels = torch.tensor(data, requires_grad=True), torch.tensor(labels).long()

            # Clear the gradients of all variables
            optimizer.zero_grad()

            # Feed forward the network
            #embeddings = model(data)
            #print(len(embeddings))

            #TODO: Fix this with triplet loss
            # Should be something like: loss = triplet_loss(anchor, positive, negative)

            anchor = torch.randn(100, 128, requires_grad=True)
            positive = torch.randn(100, 128, requires_grad=True)
            negative = torch.randn(100, 128, requires_grad=True)
            # Calculate the loss
            loss = triplet_loss(anchor, positive, negative)

            # Feed Backward
            loss.backward()

            # Update the weights
            optimizer.step()

            # Update the training status
            train_loss += loss.item()

            # Find the class with the highest output
            #_, predicted = torch.max(embeddings.data, 1)

            # Count number total number of images trained so far and the correctly classified ones
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()

            if batch_idx % 2 == 0:
                print("Epoch {}/{}, Iteration {}/{}: Loss = {}".format(epoch, num_epochs, batch_idx, batch_len, loss))

        #loss_log.append((train_loss / (batch_idx + 1)))
        #acc_log.append(100. * correct / total)

        # Save a checkpoint when the epoch finishes
        state = {
            'epoch': epoch,
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        file_path = './checkpoints/checkpoint_{}.ckpt'.format(epoch)
        torch.save(state, file_path)

    # Save the final model
    torch.save(model.state_dict(), './models/saved_models/lstm_triplet_loss_trained_foi_final')
    print('Training finished')
    return model, loss_log, acc_log
