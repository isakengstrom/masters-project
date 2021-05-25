import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from sklearn import metrics
from sklearn.cluster import KMeans


def get_all_embeddings(data_loader, model, device, num_classes):
    embeddings = torch.zeros(0, num_classes).to(device)
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(data_loader):
            sequences, labels = sequences.to(device), labels.to(device)

            sequences_out = model(sequences)
            embeddings = torch.cat((embeddings, sequences_out), 0)
            all_labels.extend(labels)

    return embeddings, torch.Tensor(all_labels).to(device)


def evaluate_metric(train_loader, eval_loader, model, device, classes, is_test, tb_writer=None):
    """
    Calculates Precision@1 (Recall@1), R-Precision and MAP@R directly from the embedding space

    Read more about the three accuracies in the following paper by Musgrave et al.
        https://arxiv.org/pdf/2003.08505.pdf


    https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index
    """
    model.eval()

    accuracy_calculator = AccuracyCalculator(
        include=("precision_at_1", "r_precision", "mean_average_precision_at_r"), k=None
    )

    num_classes = len(classes)
    train_embeddings, train_labels = get_all_embeddings(train_loader, model, device, num_classes)
    eval_embeddings, eval_labels = get_all_embeddings(eval_loader, model, device, num_classes)

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

