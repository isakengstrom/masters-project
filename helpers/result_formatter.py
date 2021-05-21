import numpy as np
import matplotlib.pyplot as plt
import json


def confusion_to_scores(confusion_matrix=None, actual_axis=0):
    """

    :param confusion_matrix:
    :param actual_axis: 0 if actual is along x-axis, else 1
    :return:
    """

    # Create dummy confusion matrix if none was sent in
    if confusion_matrix is None:
        '''
        confusion_matrix = [
            [10, 5, 6, 7, 8],
            [1, 11, 0, 0, 0],
            [2, 0, 12, 0, 0],
            [3, 0, 0, 13, 0],
            [4, 0, 0, 0, 14]
        ]
        '''
        confusion_matrix = [
            [7, 8, 9],
            [1, 2, 3],
            [3, 2, 1],
        ]

    confusion_matrix = np.array(confusion_matrix)

    # Calculations of the scores are made as if the actual classes along the y-axis (1)
    if actual_axis == 0:
        confusion_matrix = confusion_matrix.transpose()

    #
    preds = confusion_matrix.sum(axis=0)

    # Total number of values in the confusion matrix
    mat_sum = confusion_matrix.sum()

    # Support, how many actual samples there was for a class
    support = confusion_matrix.sum(axis=1)

    # Metrics
    # True positives - tps, True negatives - tns, False positives - fps, False negatives - fns
    tps = confusion_matrix.diagonal()
    fps = preds - tps
    fns = support - tps
    tns = mat_sum - tps - fps - fns

    # Precisions, recalls and f1s scores
    precisions = tps / (tps + fps)
    recalls = tps / (tps + fns)
    f1s = 2 * (precisions * recalls) / (precisions + recalls)

    # Total Metics
    tot_tps = tps.sum()
    tot_fps = fps.sum()
    tot_fns = fns.sum()
    #tot_tns = tns.sum()

    # Micro averages, should all be the same as accuracy
    micro_precision = tot_tps / (tot_tps + tot_fps)
    micro_recall = tot_tps / (tot_tps + tot_fns)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    accuracy = tot_tps / mat_sum

    assert micro_precision == micro_recall == micro_f1 == accuracy

    # Macro averages
    macro_precision = precisions.sum() / len(precisions)
    macro_recall = recalls.sum() / len(recalls)
    macro_f1 = f1s.sum() / len(f1s)

    # Weighted averages
    weighted_precision = (precisions * support).sum() / support.sum()
    weighted_recall = (recalls * support).sum() / support.sum()
    weighted_f1 = (f1s * support).sum() / support.sum()

    scores = {
        'tps': tps.tolist(),
        'tns': tns.tolist(),
        'fps': fps.tolist(),
        'fns': fns.tolist(),
        'precisions': precisions.tolist(),
        'recalls': recalls.tolist(),
        'f1s': f1s.tolist(),
        'support': support.tolist(),
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

    print(scores)
    print(json.dumps(scores))

    return scores

def barplot():

    # width of the bars
    bar_width = 0.3

    # Choose the height of the blue bars
    bars1 = [10, 9, 2]

    # Choose the height of the cyan bars
    bars2 = [10.8, 9.5, 4.5]

    # Choose the height of the error bars (bars1)
    yer1 = [0.5, 0.4, 0.5]

    # Choose the height of the error bars (bars2)
    yer2 = [1, 0.7, 1]

    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]

    # Create blue bars
    plt.bar(r1, bars1, width=bar_width, color='blue', edgecolor='black', yerr=yer1, capsize=7, label='poacee')

    # Create cyan bars
    plt.bar(r2, bars2, width=bar_width, color='cyan', edgecolor='black', yerr=yer2, capsize=7, label='sorgho')

    # general layout
    plt.xticks([r + bar_width for r in range(len(bars1))], ['cond_A', 'cond_B', 'cond_C'])
    plt.ylabel('height')
    plt.legend()

    # Show graphic
    plt.show()



if __name__ == "__main__":
    #barplot()
    confusion_to_scores()
