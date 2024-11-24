import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

root_dir = './'
models = ['GATGCNFG', 'GCNFG', 'GINFG', 'GATFG']
baselines = ['GAT_GCN', 'GCNNet', 'GINConvNet', 'GATNet']
datasets = ['kiba', 'bindingdb']

for j in range(len(datasets)):
    # model_prefix = '_'.join((models[0], datasets[1]))
    models_prefix = ['_'.join((model, datasets[j])) for model in models]
    model_true_paths = [root_dir + '_'.join((model_prefix, 'true')) for model_prefix in models_prefix]
    model_pred_paths = [root_dir + '_'.join((model_prefix, 'pred')) for model_prefix in models_prefix]
    baselines_prefix = ['_'.join((baseline, datasets[j])) for baseline in baselines]
    baseline_true_paths = [root_dir + '_'.join((baseline_prefix, 'true')) for baseline_prefix in baselines_prefix]
    baseline_pred_paths = [root_dir + '_'.join((baseline_prefix, 'pred')) for baseline_prefix in baselines_prefix]

    KIBA_THRESHOLD = 12.1
    BINDINGDB_THRESHOLD = 8.0
    threshold = [KIBA_THRESHOLD, BINDINGDB_THRESHOLD][j]

    for i in range(len(models)):
        model_pred = np.loadtxt(model_pred_paths[i])
        baseline_pred = np.loadtxt(baseline_pred_paths[i])
        model_true = np.loadtxt(model_true_paths[i])
        baseline_true = np.loadtxt(baseline_true_paths[i])

        model_bin_true = np.where(model_true >= threshold, 1, 0)
        fpr_m, tpr_m, _ = roc_curve(model_bin_true, model_pred)
        roc_auc_m = auc(fpr_m, tpr_m)

        baseline_bin_true = np.where(baseline_true >= threshold, 1, 0)
        fpr_b, tpr_b, _ = roc_curve(baseline_bin_true, baseline_pred)
        roc_auc_b = auc(fpr_b, tpr_b)

        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr_m, tpr_m, color='red', lw=2, label=f'{models[i]} (AUC = {roc_auc_m:.4f})')
        plt.plot(fpr_b, tpr_b, color='darkorange', lw=2, label=f'GraphDTA_{baselines[i]} (AUC = {roc_auc_b:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        if not os.path.exists(root_dir + 'figures'):
            os.makedirs(root_dir + 'figures')
        plt.savefig(root_dir + 'figures/' + models[i] + '_' + datasets[j] + '_roc_curve.svg', format='svg')
        # plt.show()
