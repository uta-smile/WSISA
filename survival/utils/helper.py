import os
import torch
import shutil
import numpy as np
from itertools import cycle
import time

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import interp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def accuracy(output, target, is_multilabel=False):
    """Computes the accuracy. Return the num of correct"""
    if is_multilabel:
        pred = output.clone()
        thresh = 0.5
        pred[pred < thresh] = 0
        pred[pred >= thresh] = 1
        correct = (pred == target).sum().float() / target.shape[1] /target.shape[0]
    else:
        pred = torch.argmax(pred, 1)
        correct = (pred == target).sum().float()
    return correct


def save_checkpoint(state, is_best, dst_path='.', filename='ckpt.pth.tar'):
    dst_path = os.path.join(dst_path,)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    filename = os.path.join(dst_path, filename)
    current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
    best_name = os.path.join(dst_path, 'best@ep{}_{}.pth.tar'.format(state['epoch'], current_time))
    torch.save(state, filename)
    if is_best:
        print('\n\n=> Best val @epoch {}, saving model'.format(state['epoch']))
        shutil.move(filename, best_name)
        # shutil.copyfile(filename, best_name)


def format_time(secs):
    """Given seconds, return hours:minutes:seconds"""
    hours, rem = divmod(secs, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds


def generate_cm_figure(cm, labels, title, fig_size=10):
    font = {
        # 'family': 'serif',
        'weight': 'normal',
        'size': 18,
    }
    plt.rc('font', **font)
    h, w = cm.shape
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap=plt.get_cmap('viridis'))
    ax.set_title(title, fontdict=font)
    xlocations = np.array(range(len(labels)))
    ax.set_ylabel('True label', fontdict=font)
    ax.set_xlabel('Predicted label', fontdict=font)
    ax.set_xticks(xlocations)
    ax.set_yticks(xlocations)
    ax.set_yticklabels(labels, fontdict=font)
    ax.set_xticklabels(labels, fontdict=font)

    # ind_array = np.arange(len(labels))
    # x, y = np.meshgrid(ind_array, ind_array)
    # for x_val, y_val in zip(x.flatten(), y.flatten()):
    for x in range(h):
        for y in range(w):
            c = cm[x][y]
            ax.text(y, x, "%0.2f" % (c,), color='white', fontsize=18, va='center', ha='center')

    # offset the tick
    tick_marks = np.array(range(len(labels))) + 0.5
    ax.set_xticks(tick_marks, minor=True)
    ax.set_yticks(tick_marks, minor=True)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.grid(True, which='minor', linestyle='-')
    #     plt.gcf().subplots_adjust(bottom=0.15)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)
    fig.tight_layout()  # remove paddings
    # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    # plt.margins(0,0）
    #     plt.show()
    # plt.savefig('/media/newhd/ysong/project/LGP/summary/t.png', dpi=200)
    return fig


def plot_confusion_matrix(y_true, y_pred, writer, phase='train', epoch=0, labels=['0', '1', '2', '3']):
    """Computer confusion_matrix and plot it into writer every epoch
    Args:
        cm: the confusion matrix to plot using matplotlib
        """
    if y_true.shape != y_pred.shape:
        y_pred = labels.reshape(y_true.shape)
    assert y_true.size == y_pred.size, 'Sizes not match! y_true: {}, y_pred: {}'.format(y_true.size, y_pred.size)

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # h, _ = cm.shape
    # confusion matrix symmetric along the diagonal
    # symmetric_cm = cm + np.transpose(cm)
    # diag = np.diag_indices(h)
    # cm[diag] -= cm[diag]
    fig_cm = generate_cm_figure(cm, labels, '[{}] confusion matrix@ep{}'.format(phase, epoch))
    fig_cm_normalized = generate_cm_figure(cm_normalized, labels,
                                           '[{}] normalized confusion matrix@ep{}'.format(phase, epoch))

    writer.add_figure('Confusion Matrix/{}/epoch_{}'.format(phase, epoch), fig_cm, global_step=epoch)
    writer.add_figure('Normalized Confusion Matrix/{}/epoch_{}'.format(phase, epoch),
                      fig_cm_normalized,
                      global_step=epoch)
    writer.flush()
    return cm, cm_normalized


def plot_confusion_matrix_multilabel(y_true, y_pred, writer, phase='train', epoch=0, labels=['0', '1', '2', '3']):
    """Computer confusion_matrix and plot it into writer every epoch
    Args:
        cm: the confusion matrix to plot using matplotlib
        """
    if y_true.shape != y_pred.shape:
        y_pred = labels.reshape(y_true.shape)
    assert y_true.size == y_pred.size, 'Sizes not match! y_true: {}, y_pred: {}'.format(y_true.size, y_pred.size)

    # cm = multilabel_confusion_matrix(y_true[:, label_col], y_pred[:, label_col])
    cm = multilabel_confusion_matrix(y_true, y_pred)
    for label_col in range(len(labels)):
        fig_cm = generate_cm_figure(cm[label_col], ['0', '1'], f'confusion matrix of label{label_col}', fig_size=6)
        writer.add_figure(f'Confusion Matrix/{phase}/epoch_{epoch}/label_{label_col}', fig_cm, global_step=epoch)

    cls_report = classification_report(y_true, y_pred)
    writer.add_text(tag='{}/{}'.format(phase, epoch), text_string=cls_report, global_step=epoch)
    writer.flush()
    return cm


# def compute_auc(y_true, y_pred, labels=[0, 1, 2, 3]):
#     y_true_one_hot = label_binarize(y_true)
#     auc = roc_auc_score(y_true_one_hot, y_pred)
#     return auc


def plot_roc_curve_and_compute_auc(y_true, y_pred, writer, phase, epoch, labels=[0, 1, 2, 3], is_multilabel=False):
    """Plot roc_curve for each class and writes into writer.
    From: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    Args:
        labels: The list for the ground truth labels
    """
    if not is_multilabel:
        y_true_bin = label_binarize(y_true, classes=labels)
    n_classes = len(labels)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
    # equal to: roc_auc = roc_auc_score(y_true_one_hot, y_pred)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    """Plot ROC curves for the multilabel"""
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    font = {
        # 'family': 'serif',
        'weight': 'normal',
        'size': 18,
    }
    plt.rc('font', **font)
    fig = plt.figure(figsize=(10, 10))
    linewidth = 6
    # Plot all ROC curves
    plt.plot(fpr["micro"],
             tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
             ''.format(roc_auc["micro"]),
             color='deeppink',
             linestyle=':',
             linewidth=linewidth)

    plt.plot(fpr["macro"],
             tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
             ''.format(roc_auc["macro"]),
             color='purple',
             linestyle=':',
             linewidth=linewidth)

    colors = cycle(['aqua', 'darkorange', 'limegreen', 'cornflowerblue', 'red', 'blue', 'gold', 'darkkhaki'])
    for i, color in zip(range(n_classes), colors):
        # show the original 1/2/3/4 label
        plt.plot(fpr[i],
                 tpr[i],
                 color=color,
                 lw=linewidth,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                 ''.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=linewidth)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict=font)
    plt.ylabel('True Positive Rate', fontdict=font)
    plt.title('[{}] ROC of multi-class@{}'.format(phase, epoch), fontdict=font)
    plt.legend(loc="lower right")
    fig.tight_layout()  # remove paddings
    writer.add_figure('ROC_all_lable/{0}/epoch_{1}'.format(phase, epoch), fig, global_step=epoch)
    writer.flush()
    return roc_auc["micro"], roc_auc["macro"]
