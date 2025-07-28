"""
Metrics Module

This module provides functions to compute common classification metrics and generate related plots for model evaluation.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             classification_report, auc, precision_recall_curve, matthews_corrcoef,
                             balanced_accuracy_score, roc_curve)


def compute_accuracy(y_true, y_pred):
    """
    Compute the accuracy of predictions.

    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.

    Returns:
        float: Accuracy score as the fraction of correct predictions.
    """
    return accuracy_score(y_true, y_pred)


def classification_report_dict(y_true, y_pred, labels=None, target_names=None):
    """
    Generate a classification report as a dictionary.

    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        labels (list, optional): List of label indices to include.
        target_names (list of str, optional): Display names for each label.

    Returns:
        dict: Classification metrics including precision, recall, f1-score, and support.
    """
    return classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True,
                                 zero_division=0)


def compute_confusion_matrix(y_true, y_pred, normalize=False):
    """
    Compute a confusion matrix for classifier predictions.

    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        normalize (bool): If True, normalize counts to proportions.

    Returns:
        np.ndarray: Confusion matrix array, normalized if requested.
    """
    # Compute raw confusion matrix counts.
    cm = confusion_matrix(y_true, y_pred)
    # Normalize the confusion matrix to proportions per true class.
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def plot_confusion_matrix(cm, classes=('non-toxic', 'toxic'), normalize=False):
    """
    Plot a confusion matrix using Matplotlib.

    Parameters:
        cm (np.ndarray): Confusion matrix array.
        classes (tuple of str): Class names for display on axes.
        normalize (bool): Whether the matrix contains proportions.

    Returns:
        matplotlib.figure.Figure: Figure object containing the plot.
    """
    fig, ax = plt.subplots()
    # Display confusion matrix as a heatmap.
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]);
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes);
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'

    # Annotate each cell with the count or proportion value.
    for i in range(2):

        for j in range(2):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center")

    ax.set_ylabel('True label');
    ax.set_xlabel('Predicted label')
    plt.tight_layout()

    return fig


def compute_roc_pr(y_true, y_score):
    """
    Compute ROC and Precision-Recall curve data along with their AUCs.

    Parameters:
        y_true (array-like): True binary class labels.
        y_score (array-like): Predicted probability scores for the positive class.

    Returns:
        tuple: ((fpr, tpr, roc_auc), (precision, recall, pr_auc)).
    """
    # Compute false positive rate and true positive rate for ROC curve.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    # Compute precision and recall for Precision-Recall curve.
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return (fpr, tpr, roc_auc), (precision, recall, pr_auc)


def plot_roc_pr(fpr_tpr, pr, name='model'):
    """
    Plot ROC and Precision-Recall curves side by side.

    Parameters:
        fpr_tpr (tuple): (fpr, tpr, roc_auc) from compute_roc_pr.
        pr (tuple): (precision, recall, pr_auc) from compute_roc_pr.
        name (str): Identifier for the model to display in titles.

    Returns:
        matplotlib.figure.Figure: Figure object containing both subplots.
    """
    (fpr, tpr, roc_auc) = fpr_tpr
    (precision, recall, pr_auc) = pr

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    # Plot ROC curve with AUC.
    ax.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
    # Plot diagonal reference line for random classifier.
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'ROC Curve — {name}')
    ax.set_xlabel('FPR');
    ax.set_ylabel('TPR');
    ax.legend()

    ax = axes[1]
    # Plot Precision-Recall curve with AUC.
    ax.plot(recall, precision, label=f'AUC={pr_auc:.2f}')
    ax.set_title(f'PR Curve — {name}')
    ax.set_xlabel('Recall');
    ax.set_ylabel('Precision');
    ax.legend()

    plt.tight_layout()
    return fig


def compute_additional_metrics(y_true, y_pred, y_score):
    """
    Compute additional classification metrics.

    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        y_score (array-like): Predicted scores for positive class.

    Returns:
        dict: Metrics including Matthews correlation coefficient and balanced accuracy.
    """
    # Compute Matthews correlation coefficient and balanced accuracy.
    return {'mcc': matthews_corrcoef(y_true, y_pred), 'balanced_acc': balanced_accuracy_score(y_true, y_pred), }


def calibration_plot(y_true, y_score, n_bins=10):
    """
    Create a calibration plot comparing predicted probabilities to observed frequencies.

    Parameters:
        y_true (array-like): True binary class labels.
        y_score (array-like): Predicted probability scores for the positive class.
        n_bins (int): Number of bins to use for calibration curve.

    Returns:
        matplotlib.figure.Figure: Figure object containing the calibration plot.
    """
    # Compute true and predicted probabilities for calibration.
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    fig, ax = plt.subplots()

    # Plot calibration curve against perfect calibration line.
    ax.plot(prob_pred, prob_true, marker='o', linewidth=1)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('Calibration curve')
    ax.set_xlabel('Mean predicted prob')
    ax.set_ylabel('Fraction of positives')

    plt.tight_layout()

    return fig


def threshold_metrics_curve(y_true, y_score, name='model', n_thresholds=50, return_df=False):
    """
    Evaluate precision, recall, and F1-score over different classification thresholds.

    Parameters:
        y_true (array-like): True binary class labels.
        y_score (array-like): Predicted probability scores for the positive class.
        name (str): Identifier for the model to display in plot title.
        n_thresholds (int): Number of thresholds to evaluate between 0 and 1.
        return_df (bool): If True, return a DataFrame of metrics instead of plotting.

    Returns:
        pandas.DataFrame or matplotlib.figure.Figure: DataFrame if return_df is True; otherwise, a Figure.
    """
    # Generate evenly spaced threshold values.
    thresholds = np.linspace(0, 1, n_thresholds)
    # Compute precision, recall, and F1 at each threshold.
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        yp = (y_score >= t).astype(int)
        precisions.append(precision_score(y_true, yp, zero_division=0))
        recalls.append(recall_score(y_true, yp, zero_division=0))
        f1s.append(f1_score(y_true, yp, zero_division=0))

    # If requested, return a DataFrame of metrics per threshold.
    if return_df:
        df = pd.DataFrame({
            'threshold': thresholds,
            'precision': precisions,
            'recall': recalls,
            'f1': f1s
        })
        return df

    # Plot threshold-based precision, recall, and F1 curves.
    fig, ax = plt.subplots()
    ax.plot(thresholds, precisions, label='precision')
    ax.plot(thresholds, recalls, label='recall')
    ax.plot(thresholds, f1s, label='f1')
    ax.set_title(f'Threshold metrics — {name}')
    ax.set_xlabel('Threshold')
    ax.legend()
    plt.tight_layout()
    return fig
