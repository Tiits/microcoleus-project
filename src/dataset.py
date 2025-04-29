"""
src/dataset.py

Module utilitaire pour le listing et la vérification d'intégrité des images selon la configuration YAML.
"""
import os
from PIL import Image
from src.config_utils import load_config

# Charger la configuration
cfg = load_config()
DATA_RAW_DIR = cfg['data']['raw_dir']
VALID_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def list_images(data_dir=DATA_RAW_DIR, valid_exts=VALID_EXTS):
    """
    Retourne un dict {classe: [liste de fichiers images valides]}.
    """
    class_dict = {}
    for cls in os.listdir(data_dir):
        class_path = os.path.join(data_dir, cls)
        if os.path.isdir(class_path) and not cls.startswith('.'):
            images = [f for f in os.listdir(class_path)
                      if os.path.splitext(f)[-1].lower() in valid_exts]
            class_dict[cls] = images
    return class_dict


def check_image_integrity(data_dir=DATA_RAW_DIR, valid_exts=VALID_EXTS):
    """
    Vérifie que toutes les images sont ouvrables. Retourne une liste des fichiers corrompus.
    """
    corrupt_files = []
    for cls in os.listdir(data_dir):
        class_path = os.path.join(data_dir, cls)
        if os.path.isdir(class_path) and not cls.startswith('.'):
            for img_file in os.listdir(class_path):
                if os.path.splitext(img_file)[-1].lower() in valid_exts:
                    img_path = os.path.join(class_path, img_file)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                    except Exception:
                        corrupt_files.append(img_path)
    return corrupt_files


if __name__ == '__main__':
    # Exemple d'utilisation avec la configuration
    print("Listing des images :", list_images())
    corrupts = check_image_integrity()
    if corrupts:
        print("Images corrompues trouvées :", corrupts)
    else:
        print("Aucune image corrompue détectée.")
