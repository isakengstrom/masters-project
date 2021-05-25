import torch


# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
def evaluate(data_loader, model, device, is_test, classes=None):

    model.eval()

    total_accuracy, total_count = 0, 0
    eval_info = dict()
    with torch.no_grad():
        if is_test and classes is not None:
            classes = list(classes)
            num_classes = len(classes)
            conf = torch.zeros([num_classes, num_classes], dtype=torch.int32)

        for batch_idx, (sequences, labels) in enumerate(data_loader):

            sequences, labels = sequences.to(device).squeeze(1), labels.to(device)

            sequences_out = model(sequences)

            _, predicted_labels = torch.max(sequences_out, 1)

            total_accuracy += (predicted_labels == labels).sum().item()
            total_count += labels.size(0)

            if is_test:
                conf[predicted_labels, labels] += 1

        if is_test:
            conf_list = conf.tolist()
            eval_info['confusion_matrix'] = conf_list

        eval_info['accuracy'] = total_accuracy / total_count
    return eval_info

