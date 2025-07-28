"""
Preprocessing Module

This module provides functions to load, augment, and preprocess image datasets organized in folder structures.
"""
import os
import shutil

import numpy as np
from PIL import Image, ImageOps

from src.config_utils import load_config

cfg = load_config()
TARGET_SIZE = tuple(cfg['preprocessing']['target_size'])
NORMALIZE = cfg['preprocessing']['normalize']
AUG = cfg['preprocessing']['augmentations']


def load_image(img_path):
    """
    Load an image from a file path, resize to target size, and optionally normalize.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: Image array, normalized to [0,1] if NORMALIZE is True, otherwise raw pixel values.
    """
    # Open image and convert to RGB format.
    img = Image.open(img_path).convert('RGB')
    # Resize image to target dimensions.
    img = img.resize(TARGET_SIZE)
    # Convert image to numpy array.
    arr = np.array(img)
    # Normalize pixel values to [0,1] if configured.
    return arr / 255.0 if NORMALIZE else arr


def augment_image(img):
    """
    Apply random augmentations to a PIL image based on configuration.

    Parameters:
        img (PIL.Image.Image): Input image to augment.

    Returns:
        PIL.Image.Image: Augmented image.
    """
    # Apply horizontal flip if enabled in configuration.
    if AUG.get('horizontal_flip', False):
        img = ImageOps.mirror(img)
    rot = AUG.get('rotation_range', 0)
    if rot:
        angle = np.random.uniform(-rot, rot)
        # Apply random rotation within specified range.
        img = img.rotate(angle)
    return img


def preprocess_folder(input_dir, output_dir):
    """
    Process all images in an input directory, applying resize, augmentation, and normalization, and save to an output directory.

    Parameters:
        input_dir (str): Root directory containing class subdirectories with raw images.
        output_dir (str): Directory where processed images will be saved; existing directory will be replaced.
    """
    # Remove existing output directory to regenerate processed data.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Create fresh output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each class folder in the input directory.
    for cls in os.listdir(input_dir):
        in_cls_path = os.path.join(input_dir, cls)
        out_cls_path = os.path.join(output_dir, cls)
        # Skip non-directory entries and hidden folders.
        if not os.path.isdir(in_cls_path) or cls.startswith('.'):
            continue
        # Create output folder for the current class.
        os.makedirs(out_cls_path, exist_ok=True)
        # Iterate over files in the class folder.
        for fname in os.listdir(in_cls_path):
            # Skip hidden files.
            if fname.startswith('.'):
                continue
            # Construct full input file path.
            in_path = os.path.join(in_cls_path, fname)
            try:
                # Open image file.
                img = Image.open(in_path).convert('RGB')
                # Resize image to target size.
                img = img.resize(TARGET_SIZE)
                # Apply augmentation pipeline.
                img = augment_image(img)
                # Convert augmented image to numpy array.
                arr = np.array(img)
                # Normalize pixel values if configured.
                if NORMALIZE:
                    arr = arr / 255.0
                # Convert array back to PIL Image for saving.
                out_img = Image.fromarray((arr * 255).astype(np.uint8)) if NORMALIZE else img
                # Save processed image to the output directory.
                out_img.save(os.path.join(out_cls_path, fname))
            # Log errors encountered during processing.
            except Exception as e:
                print(f"Erreur sur {in_path}: {e}")


if __name__ == '__main__':
    cfg = load_config()
    input_dir = cfg['data']['raw_dir']
    output_dir = cfg['data']['processed_dir']
    preprocess_folder(input_dir, output_dir)
