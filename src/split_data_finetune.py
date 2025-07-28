"""
Split Data Finetune Module

This module splits a directory of class-organized images into training, validation, and test sets
using stratified sampling, and saves the resulting file paths to text files for fine-tuning.
"""
import argparse
import os

from sklearn.model_selection import train_test_split


def main():
    """
    Parse command-line arguments and generate stratified train/val/test splits.

    Steps:
    1. Collect image file paths and labels from the raw data directory.
    2. Perform stratified splitting into train, validation, and test sets.
    3. Save each split's file paths to separate text files.
    """
    # Create argument parser for command-line interface.
    parser = argparse.ArgumentParser()
    # Define CLI arguments for input directory, output directory, and split ratios.
    parser.add_argument('--raw_dir', type=str, default='../data/raw/Switzerland',
                        help='Répertoire des images prétraitées')
    parser.add_argument('--split_dir', type=str, default='../splits/Switzerland/finetune_resnet50',
                        help='Répertoire de sortie pour les splits')
    parser.add_argument('--train_size', type=float, default=0.7, help='Proportion du train set')
    parser.add_argument('--val_size', type=float, default=0.15, help='Proportion du validation set')
    parser.add_argument('--test_size', type=float, default=0.15, help='Proportion du test set')
    args = parser.parse_args()
    # Parse the provided command-line arguments.

    # List class subdirectories in the raw data directory.
    classes = [d for d in os.listdir(args.raw_dir) if os.path.isdir(os.path.join(args.raw_dir, d))]
    # Initialize lists to hold image file paths and corresponding labels.
    file_paths, labels = [], []
    # Iterate over each class to collect image files.
    for cls in classes:
        cls_dir = os.path.join(args.raw_dir, cls)
        for fname in os.listdir(cls_dir):
            # Filter to include only supported image extensions.
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls)

    # Split data into training set and a temporary set using stratified sampling.
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        train_size=args.train_size,
        stratify=labels,
        random_state=42
    )

    # Compute relative validation size from validation and test ratios.
    rel_val = args.val_size / (args.val_size + args.test_size)
    # Split the temporary set into validation and test sets.
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        train_size=rel_val,
        stratify=temp_labels,
        random_state=42
    )

    # Ensure the output split directory exists.
    os.makedirs(args.split_dir, exist_ok=True)
    # Iterate over each split type and its file paths.
    for split, paths in zip(['train', 'val', 'test'], [train_paths, val_paths, test_paths]):
        out_txt = os.path.join(args.split_dir, f"{split}.txt")
        # Open the split file for writing.
        with open(out_txt, 'w') as f:
            # Write each file path on a new line.
            for p in paths:
                f.write(p + "\n")


# Entry point: execute main when script is run directly.
if __name__ == '__main__':
    main()
