"""
Split Data Module

This module provides functionality to split a dataset of image file paths and labels into training, validation, and test sets based on specified ratios, and to save the splits to text files.
"""
import os
import random
from pathlib import Path

from src.config_utils import load_config


def generate_splits(cfg, out_dir='../splits', train_ratio=0.7, val_ratio=0.15):
    """
    Generate dataset splits and save to text files.

    Parameters:
        cfg (dict): Configuration dictionary containing seed and data directories.
        out_dir (str): Output directory for split files.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.

    Returns:
        None
    """
    # Set random seed for reproducibility.
    random.seed(cfg['seed'])
    # Load processed data directory from configuration.
    data_dir = Path(cfg['data']['processed_dir'])
    # Initialize list to hold (file path, label) tuples.
    file_label_pairs = []
    # Iterate over each class directory in the data directory.
    for cls in sorted(data_dir.iterdir()):
        if cls.is_dir():
            for img in cls.iterdir():
                # Include only supported image files based on extension.
                if img.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                    file_label_pairs.append((str(img), cls.name))
    # Shuffle all path-label pairs.
    random.shuffle(file_label_pairs)
    n = len(file_label_pairs)
    # Compute number of training samples.
    n_train = int(train_ratio * n)
    # Compute number of validation samples.
    n_val = int(val_ratio * n)
    # Partition data into train, validation, and test splits.
    splits = {
        'train': file_label_pairs[:n_train],
        'val': file_label_pairs[n_train:n_train + n_val],
        'test': file_label_pairs[n_train + n_val:],
    }
    # Ensure output directory exists.
    os.makedirs(out_dir, exist_ok=True)
    for split, pairs in splits.items():
        # Write each split file with tab-separated path and label.
        with open(f'{out_dir}/{split}.txt', 'w') as f:
            for path, label in pairs:
                f.write(f"{path}\t{label}\n")
    # Notify user of completion and locations of saved files.
    print(f"Splits saved in {out_dir}/train.txt, val.txt, test.txt")


if __name__ == "__main__":
    # Load configuration settings.
    cfg = load_config()
    # Generate dataset splits using default ratios.
    generate_splits(cfg)
