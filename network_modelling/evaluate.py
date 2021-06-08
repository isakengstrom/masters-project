import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from sklearn import metrics


def evaluate(train_loader, eval_loader, model, task, device, embedding_dims, is_test, tb_writer=None):

    if task == 'classification':
        val_info = evaluate_classification(
            eval_loader=eval_loader,
            model=model,
            device=device,
            is_test=is_test
        )

        val_message = f"| Val accuracy: {val_info['accuracy']:.6f} "

    elif task == 'metric':
        val_info = evaluate_metric(
            train_loader=train_loader,
            eval_loader=eval_loader,
            model=model,
            device=device,
            embedding_dims=embedding_dims,
            is_test=is_test,
            tb_writer=tb_writer
        )

        val_message = f"| Pre@1: {val_info['precision_at_1']:.6f} " \
                      f"| R-Pre: {val_info['r_precision']:.6f} " \
                      f"| MAP@R: {val_info['mean_average_precision_at_r']:.6f} " \
                      f"| Sil: {val_info['silhouette']:.3f} " \
                      f"| CH: {val_info['ch']:.0f} "

    else:
        raise Exception("Invalid task type, should either by 'classification' or 'metric'")

    return val_info, val_message


# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
def evaluate_classification(eval_loader, model, device, is_test, classes=None):

    model.eval()

    total_accuracy, total_count = 0, 0
    eval_info = dict()
    with torch.no_grad():
        if is_test and classes is not None:
            classes = list(classes)
            num_classes = len(classes)
            conf = torch.zeros([num_classes, num_classes], dtype=torch.int32)

        for batch_idx, (sequences, labels) in enumerate(eval_loader):

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


def get_all_embeddings(data_loader, model, device, embedding_dims):
    embeddings = torch.zeros(0, embedding_dims).to(device)
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(data_loader):
            if isinstance(labels, tuple) or isinstance(labels, list):
                labels = labels[0].to(device)
            else:
                labels = labels.to(device)

            sequences, labels = sequences.to(device),  labels.to(device)

            sequences_out = model(sequences)
            embeddings = torch.cat((embeddings, sequences_out), 0)
            all_labels.extend(labels)

    return embeddings, torch.Tensor(all_labels).to(device)


def evaluate_metric(train_loader, eval_loader, model, device, embedding_dims, is_test, tb_writer):
    """
    Calculates Precision@1 (Recall@1), R-Precision and MAP@R directly from the embedding space

    Read more about the three accuracies in the following paper by Musgrave et al.
        https://arxiv.org/pdf/2003.08505.pdf


    https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index
    """
    model.eval()
    with torch.no_grad():
        accuracy_calculator = AccuracyCalculator(
            include=("precision_at_1", "r_precision", "mean_average_precision_at_r"), k=None
        )

        train_embeddings, train_labels = get_all_embeddings(train_loader, model, device, embedding_dims)
        eval_embeddings, eval_labels = get_all_embeddings(eval_loader, model, device, embedding_dims)

        scores = accuracy_calculator.get_accuracy(
            query=eval_embeddings,
            reference=train_embeddings,
            query_labels=eval_labels,
            reference_labels=train_labels,
            embeddings_come_from_same_source=False
        )

        scores['accuracy'] = scores['mean_average_precision_at_r']
        scores['silhouette'] = metrics.silhouette_score(eval_embeddings.cpu(), eval_labels.cpu(), metric='euclidean')
        scores['ch'] = metrics.calinski_harabasz_score(eval_embeddings.cpu(), eval_labels.cpu())

        if tb_writer is not None:
            eval_labels += 10
            # Add embeddings to tensorboard and flush everything in the writer to disk
            tb_writer.add_embedding(train_embeddings, metadata=train_labels.tolist(), global_step=0)
            tb_writer.add_embedding(eval_embeddings, metadata=eval_labels.tolist(), global_step=1)
            tb_writer.add_embedding(torch.cat((train_embeddings, eval_embeddings), 0), metadata=torch.cat((train_labels, eval_labels)).tolist(), global_step=2)
            tb_writer.flush()

    return scores
