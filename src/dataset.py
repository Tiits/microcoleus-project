"""
Dataset Module

Provides functions to list image files organized by class directories and to verify image file integrity.
"""
import os

from PIL import Image

from src.config_utils import load_config

cfg = load_config()
DATA_RAW_DIR = cfg['data']['raw_dir']
VALID_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def list_images(data_dir=DATA_RAW_DIR, valid_exts=VALID_EXTS):
    """
    List valid image files organized by class directories.

    Parameters:
        data_dir (str): Path to the root data directory containing subdirectories for each class.
        valid_exts (set of str): Set of allowed image file extensions.

    Returns:
        dict: Mapping from class names to lists of image filenames.
    """
    # Iterate over class subdirectories in the data directory.
    class_dict = {}
    for cls in os.listdir(data_dir):
        class_path = os.path.join(data_dir, cls)
        if os.path.isdir(class_path) and not cls.startswith('.'):
            # Collect files with valid image extensions in the current class directory.
            images = [f for f in os.listdir(class_path)
                      if os.path.splitext(f)[-1].lower() in valid_exts]
            class_dict[cls] = images
    return class_dict


def check_image_integrity(data_dir=DATA_RAW_DIR, valid_exts=VALID_EXTS):
    """
    Check integrity of image files by attempting to open and verify each image.

    Parameters:
        data_dir (str): Path to the root data directory containing subdirectories for each class.
        valid_exts (set of str): Set of allowed image file extensions.

    Returns:
        list: List of file paths that failed integrity verification.
    """
    # Iterate over class subdirectories to examine each image file.
    corrupt_files = []
    for cls in os.listdir(data_dir):
        class_path = os.path.join(data_dir, cls)
        if os.path.isdir(class_path) and not cls.startswith('.'):
            # Iterate over files in the class directory.
            for img_file in os.listdir(class_path):
                # Only process files with valid image extensions.
                if os.path.splitext(img_file)[-1].lower() in valid_exts:
                    img_path = os.path.join(class_path, img_file)
                    # Attempt to open and verify the image file for corruption.
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                    except Exception:
                        corrupt_files.append(img_path)
    return corrupt_files


if __name__ == '__main__':
    print("Listing des images :", list_images())
    corrupts = check_image_integrity()
    if corrupts:
        print("Images corrompues trouvées :", corrupts)
    else:
        print("Aucune image corrompue détectée.")
