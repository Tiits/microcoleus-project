"""
src/metrics.py

Module de calcul des métriques d'évaluation pour les modèles de classification.
"""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


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


def compute_precision(y_true, y_pred, average='binary'):
    """
    Calcul de la précision.
    Paramètres :
        y_true (array-like) : étiquettes réelles.
        y_pred (array-like) : étiquettes prédites.
        average (str) : méthode de moyennage ('binary', 'micro', 'macro', 'weighted').
    Retour :
        float : score de précision.
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true, y_pred, average='binary'):
    """
    Calcul du rappel (recall).
    Paramètres :
        y_true (array-like) : étiquettes réelles.
        y_pred (array-like) : étiquettes prédites.
        average (str) : méthode de moyennage ('binary', 'micro', 'macro', 'weighted').
    Retour :
        float : score de rappel.
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(y_true, y_pred, average='binary'):
    """
    Calcul du score F1.
    Paramètres :
        y_true (array-like) : étiquettes réelles.
        y_pred (array-like) : étiquettes prédites.
        average (str) : méthode de moyennage ('binary', 'micro', 'macro', 'weighted').
    Retour :
        float : score F1.
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_confusion_matrix(y_true, y_pred, labels=None):
    """
    Calcul de la matrice de confusion.
    Paramètres :
        y_true (array-like) : étiquettes réelles.
        y_pred (array-like) : étiquettes prédites.
        labels (list, optionnel) : liste des étiquettes pour indexer la matrice.
    Retour :
        ndarray : matrice de confusion.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


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
    return classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
