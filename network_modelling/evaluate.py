import torch


# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
def evaluate(data_loader, model, device):
    model.eval()

    total_accuracy, total_count = 0, 0

    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(data_loader):
            labels = labels.to(device)
            sequences = sequences.to(device).squeeze(1)

            sequences_out = model(sequences)

            _, predicted_labels = torch.max(sequences_out, 1)

            total_accuracy += (predicted_labels == labels).sum().item()
            total_count += labels.size(0)

    return total_accuracy / total_count
