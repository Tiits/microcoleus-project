"""
src/metrics.py

Module de calcul des métriques d'évaluation pour les modèles de classification.
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
    Calcul de l'exactitude (accuracy).
    Paramètres :
        y_true (array-like) : étiquettes réelles.
        y_pred (array-like) : étiquettes prédites.
    Retour :
        float : score d'exactitude.
    """
    return accuracy_score(y_true, y_pred)


def classification_report_dict(y_true, y_pred, labels=None, target_names=None):
    """
    Génération d'un rapport de classification au format dictionnaire.
    Paramètres :
        y_true (array-like) : étiquettes réelles.
        y_pred (array-like) : étiquettes prédites.
        labels (list, optionnel) : liste des étiquettes à inclure.
        target_names (list, optionnel) : noms associés aux étiquettes.
    Retour :
        dict : rapport de classification.
    """
    return classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True,
        zero_division=0)


def compute_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def plot_confusion_matrix(cm, classes=('non-toxic', 'toxic'), normalize=False):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]);
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes);
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'

    for i in range(2):

        for j in range(2):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center")

    ax.set_ylabel('True label');
    ax.set_xlabel('Predicted label')
    plt.tight_layout()

    return fig


def compute_roc_pr(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return (fpr, tpr, roc_auc), (precision, recall, pr_auc)


def plot_roc_pr(fpr_tpr, pr, name='model'):
    (fpr, tpr, roc_auc) = fpr_tpr
    (precision, recall, pr_auc) = pr

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'ROC Curve — {name}')
    ax.set_xlabel('FPR');
    ax.set_ylabel('TPR');
    ax.legend()

    ax = axes[1]
    ax.plot(recall, precision, label=f'AUC={pr_auc:.2f}')
    ax.set_title(f'PR Curve — {name}')
    ax.set_xlabel('Recall');
    ax.set_ylabel('Precision');
    ax.legend()

    plt.tight_layout()
    return fig


def compute_additional_metrics(y_true, y_pred, y_score):
    return {'mcc': matthews_corrcoef(y_true, y_pred), 'balanced_acc': balanced_accuracy_score(y_true, y_pred), }


def calibration_plot(y_true, y_score, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    fig, ax = plt.subplots()

    ax.plot(prob_pred, prob_true, marker='o', linewidth=1)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('Calibration curve')
    ax.set_xlabel('Mean predicted prob')
    ax.set_ylabel('Fraction of positives')

    plt.tight_layout()

    return fig

def threshold_metrics_curve(y_true, y_score, name='model', n_thresholds=50, return_df=False):
    """
    Trace precision, recall et f1 en fonction du seuil.
    Si return_df=True, renvoie également un DataFrame avec les colonnes
    ['threshold', 'precision', 'recall', 'f1'].
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        yp = (y_score >= t).astype(int)
        precisions.append(precision_score(y_true, yp, zero_division=0))
        recalls.append(recall_score(y_true, yp, zero_division=0))
        f1s.append(f1_score(y_true, yp, zero_division=0))

    if return_df:
        df = pd.DataFrame({
            'threshold': thresholds,
            'precision': precisions,
            'recall': recalls,
            'f1': f1s
        })
        return df

    fig, ax = plt.subplots()
    ax.plot(thresholds, precisions, label='precision')
    ax.plot(thresholds, recalls,    label='recall')
    ax.plot(thresholds, f1s,        label='f1')
    ax.set_title(f'Threshold metrics — {name}')
    ax.set_xlabel('Threshold')
    ax.legend()
    plt.tight_layout()
    return fig


