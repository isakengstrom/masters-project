import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from sklearn import metrics
from sklearn.cluster import KMeans


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


def evaluate_metric(train_loader, eval_loader, model, device, classes, is_test):
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
    '''
    centroids = torch.zeros((0, eval_embeddings.size(1))).to(device)

    for class_idx in range(num_classes):
        class_labels = torch.where(eval_labels == class_idx)
        class_centroid = torch.median(eval_embeddings[class_labels], dim=0).values

        class_centroid = torch.unsqueeze(class_centroid, 0)
        centroids = torch.cat((centroids, class_centroid))

    #centroids = torch.transpose(centroids, 0, 1)

    #kmeans_model = KMeans(n_clusters=num_classes, init=centroids.cpu(), n_init=1, random_state=1).fit(eval_embeddings.cpu())
    #kmeans_labels = kmeans_model.labels_
    '''

    scores['silhouette'] = metrics.silhouette_score(eval_embeddings.cpu(), eval_labels.cpu(), metric='euclidean')
    scores['ch'] = metrics.calinski_harabasz_score(eval_embeddings.cpu(), eval_labels.cpu())

    '''
    print(f"| Precision@1: {scores['precision_at_1']:.6f} "
          f"| R-Precision: {scores['r_precision']:.6f} "
          f"| MAP@R: {scores['mean_average_precision_at_r']:.6f} "
          f"| Sil: {scores['silhouette']:.6f} "
          f"| CH: {scores['ch']:.0f}"
          )
    '''

    return scores

