"""
Data Loader Module

This module provides functions to load and preprocess image data for training and evaluation using TensorFlow.
"""
import random

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config_utils import load_config

cfg = load_config()
random.seed(cfg['seed'])
np.random.seed(cfg['seed'])
tf.random.set_seed(cfg['seed'])


def preprocess_and_load(path, label, img_size):
    """
    Preprocess an image and return the image array and label.

    Parameters:
        path (str): Filesystem path to the image.
        label (int): Numeric label associated with the image.
        img_size (tuple of int): Target image size as (width, height).

    Returns:
        np.ndarray: Normalized image array of shape (height, width, 3).
        np.int32: Label as a 32-bit integer.
    """
    img = Image.open(path).convert('RGB').resize(img_size)
    # Convert image to a normalized float32 numpy array with values in [0, 1].
    x = np.array(img, dtype=np.float32) / 255.0
    # Cast the label to a 32-bit integer.
    return x, np.int32(label)


def _load_py(path, label, img_size):
    """
    Wrapper for tf.py_function to load and preprocess image data.

    Parameters:
        path (tf.Tensor): Byte string tensor of the image file path.
        label (tf.Tensor): Numeric tensor label.
        img_size (tuple of int): Target image dimensions.

    Returns:
        np.ndarray: Processed image array.
        np.int32: Processed label tensor.
    """
    path_str = path.numpy().decode('utf-8')
    label_int = int(label.numpy())
    x, lab = preprocess_and_load(path_str, label_int, img_size)
    return x, lab


def load_from_splits(split_file, batch_size, img_size):
    """
    Load image paths and labels from a split file and prepare a tf.data.Dataset.

    Parameters:
        split_file (str): Path to tab-separated file with image paths and labels.
        batch_size (int): Number of samples per batch.
        img_size (tuple of int): Target image dimensions as (width, height).

    Returns:
        tf.data.Dataset: Dataset yielding batches of (image, label).
        dict: Mapping of class names to numeric indices.
    """
    # Read file paths and labels from the split file.
    paths, labels = [], []
    with open(split_file, 'r') as f:
        for line in f:
            p, lbl = line.strip().split('\t')
            paths.append(p)
            labels.append(lbl)
    classes = sorted(set(labels))
    class_indices = {cls: i for i, cls in enumerate(classes)}
    y = [class_indices[l] for l in labels]

    # Create a TensorFlow dataset from the file paths and labels.
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    # Use tf.py_function to apply Python-based loading and preprocessing.
    ds = ds.map(
        lambda p, y: tf.py_function(
            func=lambda p, y: _load_py(p, y, img_size),
            inp=[p, y],
            Tout=(tf.float32, tf.int32)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(
        lambda x, y: (
            tf.ensure_shape(x, (*img_size, 3)),
            tf.ensure_shape(y, ())
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Shuffle the dataset with a fixed seed for reproducibility.
    ds = ds.shuffle(len(paths), seed=cfg['seed'])
    # Batch the dataset.
    ds = ds.batch(batch_size)
    # Prefetch batches for performance optimization.
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_indices
