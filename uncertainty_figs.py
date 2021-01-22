import os
import warnings

import h5py

warnings.filterwarnings("ignore")
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import normalize
from dataset.dataset import Data as data
from utils.visualize_utils import visualize
from utils.eval_utils import compute_accuracy
from models import *
from config import options
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.eval_utils import mutual_info, predictive_entropy, uncertainty_fraction_removal, \
    normalized_uncertainty_toleration_removal, combo_uncertainty_fraction_removal

# importing all folds
baseline_unc = []
aug_unc = []
baseline_labs = []
aug_labs = []
y_labs = []
unc_fracs_base = []
unc_fracs_aug = []
acc_random_fracs = []
base_uncertainty_acc = []
aug_uncertainty_acc = []
for i in range(5):
    baseline = np.load('<your_selected_np_file>'.format(i))
    base_unc = baseline['unc']
    base_mc = baseline['data']
    base_mean_prob = base_mc.mean(axis=0)
    base_labs = []
    for j in range(len(base_mean_prob)):
        base_labs.append(np.argmax(base_mean_prob[j]))
    base_unc = (base_unc - np.abs(base_unc).min(axis=0)) / (
                np.abs(base_unc).max(axis=0) - np.abs(base_unc).min(axis=0))
    baseline_unc = np.append(baseline_unc, base_unc)
    baseline_labs = np.append(baseline_labs, base_labs)

    test = np.load('<your_selected_test_np_file>'
                   .format(i))
    labels = test['y'][:].astype(int)
    y_labs = np.append(y_labs, labels)

    aug = np.load('<your_selected_aug_np_file>'.format(i))
    aug_uncer = aug['unc']
    aug_mc = aug['data']
    aug_mean_prob = aug_mc.mean(axis=0)
    aug_labels = []
    for j in range(len(aug_mean_prob)):
        aug_labels.append(np.argmax(aug_mean_prob[j]))
    aug_uncer = (aug_uncer - np.abs(aug_uncer).min(axis=0)) / (np.abs(aug_uncer).max(axis=0) - np.abs(aug_uncer).min(axis=0))
    aug_unc = np.append(aug_unc, aug_uncer)
    aug_labs = np.append(aug_labs, aug_labels)

    unc_frac_base, acc_random_frac = uncertainty_fraction_removal(labels.astype(int), np.array(base_labs), np.array(base_unc), 50, 20, save=False)
    unc_fracs_base.append([unc_frac_base])
    acc_random_fracs.append([np.mean(acc_random_frac, axis=0)])
    unc_frac_aug, temp = uncertainty_fraction_removal(labels.astype(int), np.array(aug_labels), np.array(aug_uncer), 50, 20, save=False)
    unc_fracs_aug.append([unc_frac_aug])

    base_uncertainty = normalized_uncertainty_toleration_removal(labels.astype(int), np.array(base_labs), np.array(base_unc), 50, save=False)
    base_uncertainty_acc.append([base_uncertainty])

    aug_uncertainty = normalized_uncertainty_toleration_removal(labels.astype(int), np.array(aug_labels), np.array(aug_uncer), 50, save=False)
    aug_uncertainty_acc.append([aug_uncertainty])

base_mean = np.mean(unc_fracs_base, axis=0).squeeze()
base_std = (np.std(unc_fracs_base, axis=0)/2).squeeze()
aug_mean = np.mean(unc_fracs_aug, axis=0).squeeze()
aug_std = (np.std(unc_fracs_aug, axis=0)/2).squeeze()
rand_mean = np.mean(acc_random_fracs, axis=0).squeeze()
rand_std = (np.std(acc_random_fracs, axis=0)/2).squeeze()
base_acc_mean = np.mean(base_uncertainty_acc, axis=0).squeeze()
base_acc_std = (np.std(base_uncertainty_acc, axis=0)/2).squeeze()
aug_acc_mean = np.mean(aug_uncertainty_acc, axis=0).squeeze()
aug_acc_std = (np.std(aug_uncertainty_acc, axis=0)/2).squeeze()

fig, ax = plt.subplots(nrows=1, ncols=1)
line1, = ax.plot(np.linspace(1 / 50, 1, 50), base_acc_mean, '-o', lw=1, label='Baseline', markersize=3, color='royalblue')
ax.fill_between(np.linspace(1 / 50, 1, 50),
                base_acc_mean - base_acc_std,
                base_acc_mean + base_acc_std,
                color='blue', alpha=0.3)
line2, = ax.plot(np.linspace(1 / 50, 1, 50), aug_acc_mean, '-v', lw=1, label='ST Augmented', markersize=3, color='darkred')
ax.fill_between(np.linspace(1 / 50, 1, 50),
                aug_acc_mean - aug_acc_std,
                aug_acc_mean + aug_acc_std,
                color='red', alpha=0.3)
plt.xlabel('Normalized Tolerated Model Uncertainty')
plt.ylabel('Prediction Accuracy')
ax.legend()
ax.set_xlim([0.1,1])
ax.set_ylim([0.82,1])
fig.show()

fig, ax = plt.subplots(nrows=1, ncols=1)
line1, = ax.plot(np.linspace(1 / 50, 1, 50), base_mean, '-o', lw=1, label='Baseline', markersize=3, color='royalblue')
ax.fill_between(np.linspace(1 / 50, 1, 50),
                base_mean - base_std,
                base_mean + base_std,
                color='blue', alpha=0.3)
line2, = ax.plot(np.linspace(1 / 50, 1, 50), aug_mean, '-v', lw=1, label='ST Augmented', markersize=3, color='darkred')
ax.fill_between(np.linspace(1 / 50, 1, 50),
                aug_mean - aug_std,
                aug_mean + aug_std,
                color='red', alpha=0.3)
line3, = ax.plot(np.linspace(1 / 50, 1, 50), rand_mean, '-^', lw=1, label='Random', markersize=3, color='black')
ax.fill_between(np.linspace(1 / 50, 1, 50),
                rand_mean - rand_std,
                rand_mean + rand_std,
                color='grey', alpha=0.3)
ax.set_xlabel('Fraction of Retained Data')
ax.set_ylabel('Prediction Accuracy')
ax.legend()
ax.set_xlim([0.2,1])
ax.set_ylim([0.82,1])
fig.show()


# Generating the basline unc figure

uncertainty_fraction_removal(y_labs, baseline_labs, baseline_unc, 50, 20, save=True, save_dir='<your_selected_baseline_file>')

# Generating the augmented unc figure

uncertainty_fraction_removal(y_labs, aug_labs, aug_unc, 50, 20, save=True, save_dir='<your_selected_frac_file>')

# Generating the combined unc figure

combo_uncertainty_fraction_removal(y_labs, baseline_labs, baseline_unc, aug_labs, aug_unc, 50, 20, save=True, save_dir='<your_selected_frac_file>')

