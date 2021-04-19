import torch


# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
def evaluate(data_loader, model, device):
    model.eval()

    total_accuracy, total_count = 0, 0

    with torch.no_grad():
        for batch_idx, (sequence, label) in enumerate(data_loader):
            label = label.to(device)
            sequence = sequence.to(device).squeeze(1)

            sequence_out = model(sequence)

            _, predicted_label = torch.max(sequence_out, 1)

            total_accuracy += (predicted_label == label).sum().item()
            total_count += label.size(0)

    return total_accuracy / total_count
