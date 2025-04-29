"""
src/data_loader.py

Module de gestion des jeux de données à partir de fichiers de splits.
"""
import os
import yaml
import random
import numpy as np
import tensorflow as tf
from PIL import Image

# Charger la configuration
with open('configs/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Fixer la seed pour reproductibilité
random.seed(cfg['seed'])
np.random.seed(cfg['seed'])
tf.random.set_seed(cfg['seed'])


def preprocess_and_load(path, label, img_size):
    """
    Charge, redimensionne et normalise une image à partir de son chemin.
    Args:
        path (str): Chemin vers l'image.
        label (int): Indice de la classe.
        img_size (tuple): Taille cible (H, W).
    Returns:
        (tensor, label)
    """
    # Lecture et traitement
    img = Image.open(path).convert('RGB').resize(img_size)
    x = np.array(img) / 255.0
    return x, label


def load_from_splits(split_file, batch_size, img_size):
    """
    Charge un tf.data.Dataset à partir d'un fichier de splits.
    Args:
        split_file (str): Chemin vers le fichier split (.txt).
        batch_size (int): Taille du batch.
        img_size (tuple): Taille des images.
    Returns:
        tf.data.Dataset, dict: Dataset et mapping classe->indice.
    """
    paths, labels = [], []
    # Lire les chemins et labels
    with open(split_file, 'r') as f:
        for line in f:
            p, lbl = line.strip().split('\t')
            paths.append(p)
            labels.append(lbl)
    # Créer mapping
    classes = sorted(set(labels))
    class_indices = {cls: i for i, cls in enumerate(classes)}
    y = [class_indices[l] for l in labels]

    # tf.data pipeline
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.map(lambda p, y: tf.py_function(func=lambda p, y: preprocess_and_load(p.numpy().decode(), y.numpy(), img_size),
                                            inp=[p, y], Tout=(tf.float32, tf.int32)),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(len(paths), seed=cfg['seed']).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, class_indices
