"""
Preprocessing Finetune Module

This module provides functions for decoding TIFF images, applying cutout augmentation, and building TensorFlow datasets with optional augmentation pipelines.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Sequential, layers

import io


def _decode_tiff_and_resize(image_bytes, img_size):
    """
    Decode TIFF image bytes, convert to RGB, and resize to the specified dimensions.

    Parameters:
        image_bytes (bytes): Byte content of the TIFF image.
        img_size (tuple of int): Target image size as (height, width).

    Returns:
        np.ndarray: Resized image array with dtype uint8 and shape (height, width, 3).
    """
    # Load image from bytes buffer.
    img = Image.open(io.BytesIO(image_bytes))
    # Convert image to RGB channels.
    img = img.convert('RGB')
    # Resize image with bilinear interpolation.
    img = img.resize((img_size[1], img_size[0]), resample=Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def cutout(image, mask_size):
    """
    Apply Cutout augmentation by masking a random rectangle of the image.

    Parameters:
        image (tf.Tensor): Input image tensor of shape (H, W, C).
        mask_size (tuple of int): Size of the mask as (height, width).

    Returns:
        tf.Tensor: Augmented image tensor with a random region masked out.
    """
    img_h = tf.shape(image)[0]
    img_w = tf.shape(image)[1]
    mask_h = mask_size[0]
    mask_w = mask_size[1]

    # Randomly choose center coordinates for the mask.
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)

    # Compute the mask bounds and clip to image dimensions.
    y1 = tf.clip_by_value(cy - mask_h // 2, 0, img_h)
    x1 = tf.clip_by_value(cx - mask_w // 2, 0, img_w)
    y2 = tf.clip_by_value(y1 + mask_h, 0, img_h)
    x2 = tf.clip_by_value(x1 + mask_w, 0, img_w)

    # Create a mask of ones with the mask region size.
    mask = tf.ones((y2 - y1, x2 - x1), dtype=image.dtype)
    # Pad the mask to the full image size.
    mask = tf.pad(mask,
                  [[y1, img_h - y2],
                   [x1, img_w - x2]])
    # Expand mask to have same channels as image.
    mask = tf.expand_dims(mask, axis=-1)

    # Apply mask to the image by zeroing out the region.
    return image * (1.0 - mask)


def get_train_datagen(config):
    """
    Build a Keras Sequential model for training data augmentation based on the provided configuration.

    Parameters:
        config (dict): Configuration dictionary containing augmentation parameters under key 'augmentation'.

    Returns:
        tensorflow.keras.Sequential: Sequential model applying rescaling, geometric and color augmentations, conditional flip, and cutout.
    """
    aug = config['augmentation']

    contrast_delta = max(1 - aug['contrast_range'][0], aug['contrast_range'][1] - 1)
    brightness_delta = max(1 - aug['brightness_range'][0], aug['brightness_range'][1] - 1)

    return Sequential([
        # Normalize pixel values to [0,1].
        layers.Rescaling(1. / 255),

        # Apply geometric augmentations: rotation, translation, zoom.
        layers.RandomRotation(aug['rotation_range'] / 360.0),
        layers.RandomTranslation(aug['height_shift_range'], aug['width_shift_range']),
        layers.RandomZoom(aug['zoom_range']),

        # Apply random contrast and brightness adjustments.
        layers.RandomContrast(contrast_delta),
        layers.RandomBrightness(brightness_delta),

        # Optionally apply horizontal flip.
        layers.RandomFlip("horizontal") if aug['horizontal_flip'] else layers.Layer(),

        # Apply Cutout augmentation via Lambda layer.
        layers.Lambda(
            lambda x: cutout(x, tuple(aug['cutout_size']))
        )

    ])


def get_val_datagen():
    """
    Build a Keras Sequential model for validation data preprocessing.

    Returns:
        tensorflow.keras.Sequential: Sequential model applying only rescaling.
    """
    return Sequential([
        layers.Rescaling(1. / 255)
    ])


def make_dataset(
        df,
        img_size=(224, 224),
        batch_size=32,
        shuffle=True,
        augment_fn=None,
        weights=None
):
    """
    Create a tf.data.Dataset from a DataFrame of file paths and classes, with optional augmentation and sample weights.

    Parameters:
        df (pandas.DataFrame): DataFrame containing 'filename' and 'class' columns, plus optional weight column.
        img_size (tuple of int): Target image dimensions as (height, width).
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.
        augment_fn (callable, optional): Augmentation function or layer to apply to each image.
        weights (str, optional): Column name for sample weights in df.

    Returns:
        tf.data.Dataset: Dataset yielding tuples (image, label, weight).
    """
    # Extract file paths and integer-encoded labels from DataFrame.
    paths = df['filename'].values
    labels = pd.Categorical(df['class']).codes

    # Use provided sample weights or default to zeros.
    if weights is not None:
        sample_weights = df[weights].values
    else:
        sample_weights = None

    # Create dataset of file paths, labels, and weights.
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels, sample_weights if sample_weights is not None else tf.zeros_like(labels)))

    @tf.autograph.experimental.do_not_convert
    def _load_img(path, label, weight, img_size=img_size):
        """
        Load and resize an image from disk using tf.numpy_function and the TIFF decoder.

        Parameters:
            path (tf.Tensor): File path tensor.
            label (tf.Tensor): Label tensor.
            weight (tf.Tensor): Sample weight tensor.
            img_size (tuple of int): Target image size.

        Returns:
            tf.Tensor: Decoded and resized image tensor.
            tf.Tensor: Label tensor unchanged.
            tf.Tensor: Weight tensor unchanged.
        """
        # Read image file as bytes.
        img_bytes = tf.io.read_file(path)
        # Decode and resize using numpy-based function.
        img = tf.numpy_function(
            _decode_tiff_and_resize,
            [img_bytes, img_size],
            tf.uint8
        )
        # Ensure static shape for TensorFlow graph.
        img.set_shape([img_size[0], img_size[1], 3])
        return img, label, weight

    # Apply the image loading and resizing function to the dataset.
    ds = ds.map(_load_img, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation function if provided.
    if augment_fn is not None:
        ds = ds.map(lambda x, y, w: (augment_fn(x, training=True), y, w), num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle the dataset for training.
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)

    # Batch the dataset.
    ds = ds.batch(batch_size)
    # Prefetch data for performance.
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
