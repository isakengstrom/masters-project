import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import scipy.stats

from helpers import read_from_json
from helpers.paths import RUNS_INFO_PATH


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
    # tot_tns = tns.sum()

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


def mean_confidence_interval(data, confidence=0.95):
    if len(data) <= 1:
        return data[0], 0

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def bar_plot_rnns_seqs(settings, intervals):
    bar_groups = []
    conf_groups = []
    bar = []
    conf = []
    for idx in range(len(settings)):

        bar.append(intervals[idx][0])
        conf.append(intervals[idx][1])
        if (idx + 1) % 6 == 0:
            # print(idx)
            bar_groups.append(bar)
            conf_groups.append(conf)
            bar = []
            conf = []

    bar1 = bar_groups[0]
    bar2 = bar_groups[3]
    bar3 = bar_groups[2]
    bar4 = bar_groups[5]
    bar5 = bar_groups[1]
    bar6 = bar_groups[4]

    conf1 = conf_groups[0]
    conf2 = conf_groups[3]
    conf3 = conf_groups[2]
    conf4 = conf_groups[5]
    conf5 = conf_groups[1]
    conf6 = conf_groups[4]

    # width of the bars
    bar_width = 0.1

    '''
    r1 = [1, 7, 13, 19, 25, 31]
    r2 = [2, 8, 14, 20, 26, 32]
    r3 = [3, 9, 15, 21, 27, 33]
    r4 = [4, 10, 16, 22, 28, 34]
    r5 = [5, 11, 17, 23, 29, 35]
    r6 = [6, 12, 18, 24, 30, 36]
    r = r1 + r2 + r3 + r4 + r5 + r6
    '''

    r1 = np.arange(len(bar1))
    r2 = [x + bar_width+0.01 for x in r1]
    r3 = [x + bar_width+0.01 for x in r2]
    r4 = [x + bar_width+0.01 for x in r3]
    r5 = [x + bar_width+0.01 for x in r4]
    r6 = [x + bar_width+0.01 for x in r5]

    plt.bar(r1, bar1, width=bar_width, yerr=conf1, capsize=2, color='#8cc63f', label='RNN')
    plt.bar(r2, bar2, width=bar_width, yerr=conf2, capsize=2, color='#c0d2a7', label='B-RNN')
    plt.bar(r3, bar3, width=bar_width, yerr=conf3, capsize=2, color='#8fc0e8', label='LSTM')
    plt.bar(r4, bar4, width=bar_width, yerr=conf4, capsize=2, color='#bbd8ed', label='B-LSTM')
    plt.bar(r5, bar5, width=bar_width, yerr=conf5, capsize=2, color='#eb757c', label='GRU')
    plt.bar(r6, bar6, width=bar_width, yerr=conf6, capsize=2, color='#e9adb2', label='B-GRU')

    labels = ['25', '50', '100', '200', '400', '800']

    plt.hlines(y=1.0, xmin=-0.5, xmax=6, linestyles='dashed', color=(0.4, 0.4, 0.4, 0.4))
    plt.xlim([-0.1, 6.5])
    plt.ylim([0, 1.02])
    plt.xlabel('Sequence length')
    plt.ylabel('Accuracy')
    plt.xticks([r + (bar_width+0.01) * 2.5 for r in range(len(bar1))], labels, rotation=0)
    #plt.xticks([r_ + bar_width for r_ in range(len(r))], labels, rotation=0)

    #plt.subplots_adjust(bottom=0.2, top=0.98)
    plt.legend()
    # Show graphic
    #plt.show()


def get_settings_and_confidence_intervals(info):
    settings = []
    confidence_intervals = []
    for run_idx, run in info['multi_runs'].items():
        '''
        if int(run_idx) > 0:
            break

        for key in run:
            print(key)
        '''
        settings.append(run['notable_params'])
        confidence_intervals.append(mean_confidence_interval(run['accuracies']))

    # print(settings_and_intervals)
    return settings, confidence_intervals


def to_latex_rnns_seqs(settings, intervals):
    bar_groups = []
    conf_groups = []
    bar = []
    conf = []
    for idx in range(len(settings)):

        bar.append(intervals[idx][0])
        conf.append(intervals[idx][1])
        if (idx + 1) % 6 == 0:
            # print(idx)
            bar_groups.append(bar)
            conf_groups.append(conf)
            bar = []
            conf = []

    bar1 = bar_groups[0]
    bar2 = bar_groups[3]
    bar3 = bar_groups[2]
    bar4 = bar_groups[5]
    bar5 = bar_groups[1]
    bar6 = bar_groups[4]

    conf1 = conf_groups[0]
    conf2 = conf_groups[3]
    conf3 = conf_groups[2]
    conf4 = conf_groups[5]
    conf5 = conf_groups[1]
    conf6 = conf_groups[4]

    # width of the bars
    bar_width = 0.1

    # Get result in latex format
    for idx in range(len(bar1)):
        # print(round(bar6[idx]*100, 2), round(conf6[idx]*100, 2))
        print(f"{round(bar6[idx] * 100, 2)} $\pm$ {round(conf6[idx] * 100, 2)} &", end=' ')


def to_latex_embedding_scores(info):
    settings = []
    confs = []
    for run_idx, run in info['multi_runs'].items():
        if int(run_idx) > 0:
            break

        for key in run:
            print(key)
        '''
        '''
        settings.append(run['notable_params'])
        confs.append(run['confidence_scores'])
    print(confs)
    scores_to_latex = [
        confs[0]['precision_at_1'],
        confs[0]['r_precision'],
        confs[0]['mean_average_precision_at_r'],
        confs[0]['silhouette'],
        confs[0]['ch']
    ]

    for idx, score in enumerate(scores_to_latex):
        if idx < 3:
            print(f"{round(score[0] * 100, 2)} $\pm$ {round(score[1] * 100, 2)} &", end=' ')
        else:
            print(f"{round(score[0], 2)} $\pm$ {round(score[1], 2)} &", end=' ')
    print()
    print(scores_to_latex)


def fetch_run_info(relative_path, info_file_name):
    result_file_name = 'r_' + info_file_name

    info_path = os.path.join(RUNS_INFO_PATH, relative_path, info_file_name)
    result_path = os.path.join(RUNS_INFO_PATH, relative_path, result_file_name)

    info = read_from_json(info_path)
    result = read_from_json(result_path)

    #print(json.dumps(info))
    # print(json.dumps(result))

    return info, result


def main():
    # opti splits
    #info, result = fetch_run_info(relative_path='backups/gs-seq_nets_opt_split', info_file_name='d210511_h17m18.json')
    # equal split
    #info, result = fetch_run_info(relative_path='backups/gs-seq_nets_equal_split', info_file_name='d210514_h17m14.json')

    #settings, intervals = get_settings_and_confidence_intervals(info)
    #print(settings)
    #print(intervals)
    #bar_plot_rnns_seqs(settings, intervals)
    #to_latex_rnns_seqs(settings, intervals)


    # Embedding scores fetch
    #info, result = fetch_run_info(relative_path='backups/metric/final_all-classes/single', info_file_name='d210520_h18m32.json')
    #info, result = fetch_run_info(relative_path='backups/metric/final_all-classes/triplet_margin50', info_file_name='d210520_h21m58.json')
    #info, result = fetch_run_info(relative_path='backups/metric/final_all-classes/triplet_margin100', info_file_name='d210522_h17m01.json')
    #to_latex_embedding_scores(info)


    # Embedding scores fetch for unseen sessions
    #info, result = fetch_run_info(relative_path='backups/unseen_sessions', info_file_name='d210524_h17m45.json')
    #info, result = fetch_run_info(relative_path='backups/unseen_sessions', info_file_name='d210524_h18m21.json')
    info, result = fetch_run_info(relative_path='backups/unseen_sessions', info_file_name='d210524_h22m53.json')
    #to_latex_embedding_scores(info)
    # confusion_to_scores()

    best_epoch_acc = {'epoch': -1, 'accuracy': -1}
    best_epoch_loss = {'epoch': -1, 'loss': sys.maxsize}

    print(best_epoch_acc, best_epoch_loss)


if __name__ == "__main__":
    main()
