import torch
import torch.nn as nn


def test(model, test_loader, device):
    """"""

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for (sequence, label) in test_loader:

            label = label.to(device)
            sequence = sequence.to(device)

            sequence_out = model(sequence)

            _, predicted = torch.max(sequence_out.data, 1)
            correct += (predicted == label).sum().item()

            total += label.size(0)

    accuracy = 100 * correct / total

    print(f'Accuracy of the network when testing on the test sequences: {accuracy:.3f}')

    return accuracy


