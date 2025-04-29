"""
src/data_loader.py

Module de gestion des jeux de données à partir de fichiers de splits et configuration YAML.
"""
import random
from src.config_utils import load_config
import numpy as np
import tensorflow as tf
from PIL import Image

# Charger la configuration et fixer les seeds
cfg = load_config()
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
        tuple: (image_array, label)
    """
    img = Image.open(path).convert('RGB').resize(img_size)
    x = np.array(img, dtype=np.float32) / 255.0
    return x, np.int32(label)


def _load_py(path, label, img_size):
    """
    Fonction interne pour tf.py_function : convertit path et label en numpy, appelle preprocess_and_load.
    """
    # Convertir en Python types
    path_str = path.numpy().decode('utf-8')
    label_int = int(label.numpy())
    x, lab = preprocess_and_load(path_str, label_int, img_size)
    return x, lab


def load_from_splits(split_file, batch_size, img_size):
    """
    Charge un tf.data.Dataset à partir d'un fichier de splits avec shapes explicites.
    Args:
        split_file (str): Chemin vers le fichier split (.txt).
        batch_size (int): Taille du batch.
        img_size (tuple): Taille des images (H, W).
    Returns:
        tf.data.Dataset, dict: Dataset et mapping classe -> indice.
    """
    # Lecture des chemins et labels
    paths, labels = [], []
    with open(split_file, 'r') as f:
        for line in f:
            p, lbl = line.strip().split('\t')
            paths.append(p)
            labels.append(lbl)
    # Mapping classe->indice
    classes = sorted(set(labels))
    class_indices = {cls: i for i, cls in enumerate(classes)}
    y = [class_indices[l] for l in labels]

    # Construction du tf.data pipeline
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.map(
        lambda p, y: tf.py_function(
            func=lambda p, y: _load_py(p, y, img_size),
            inp=[p, y],
            Tout=(tf.float32, tf.int32)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Fixer les shapes pour l'inférence du modèle
    ds = ds.map(
        lambda x, y: (
            tf.ensure_shape(x, (*img_size, 3)),
            tf.ensure_shape(y, ())
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.shuffle(len(paths), seed=cfg['seed'])
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_indices
