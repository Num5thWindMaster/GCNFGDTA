# -*- coding: utf-8 -*-
# @Time    : 2024/5/16 0:02
# @Author  : HaiqingSun
# @OriginalFileName: test_auroc
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

root_dir = '../'
models = ['GATGCNFG']
baselines = ['GAT_GCN']
datasets = ['davis', 'kiba']
model_prefix = '_'.join((models[0], datasets[1]))
model_true_path = root_dir + '_'.join((model_prefix, 'true'))
model_pred_path = root_dir + '_'.join((model_prefix, 'pred'))
baseline_prefix = '_'.join((baselines[0], datasets[1]))
baseline_true_path = root_dir + '_'.join((baseline_prefix, 'true'))
baseline_pred_path = root_dir + '_'.join((baseline_prefix, 'pred'))

DAVIS_THRESHOLD = 7.0
KIBA_THRESHOLD = 12.1
threshold = KIBA_THRESHOLD

model_pred = np.loadtxt(model_pred_path)
baseline_pred = np.loadtxt(baseline_pred_path)
model_true = np.loadtxt(model_true_path)
baseline_true = np.loadtxt(baseline_true_path)

model_bin_true = np.where(model_true >= threshold, 1, 0)
fpr_m, tpr_m, _ = roc_curve(model_bin_true, model_pred)
roc_auc_m = auc(fpr_m, tpr_m)

baseline_bin_true = np.where(baseline_true >= threshold, 1, 0)
fpr_b, tpr_b, _ = roc_curve(baseline_bin_true, baseline_pred)
roc_auc_b = auc(fpr_b, tpr_b)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr_m, tpr_m, color='red', lw=2, label=f'GATGCNFG (AUC = {roc_auc_m:.4f})')
plt.plot(fpr_b, tpr_b, color='darkorange', lw=2, label=f'GraphDTA_GAT_GCN (AUC = {roc_auc_b:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.svg', format='svg')
plt.show()
