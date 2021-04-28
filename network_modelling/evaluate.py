import torch


# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
def evaluate(data_loader, model, device):
    model.eval()

    total_accuracy, total_count = 0, 0

    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(data_loader):
            sequences, labels = batch_samples["main"]

            sequences = sequences.to(device).squeeze(1)
            labels = labels.to(device)

            sequences_out = model(sequences)

            _, predicted_labels = torch.max(sequences_out, 1)

            total_accuracy += (predicted_labels == labels).sum().item()
            total_count += labels.size(0)

    return total_accuracy / total_count
