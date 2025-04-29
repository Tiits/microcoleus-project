"""
src/preprocessing.py

Module de prétraitement des images basé sur la configuration définie dans configs/config.yaml.
"""
import os
import shutil
from PIL import Image, ImageOps
import numpy as np
from src.config_utils import load_config

# Charger la configuration\

cfg = load_config()
TARGET_SIZE = tuple(cfg['preprocessing']['target_size'])
NORMALIZE = cfg['preprocessing']['normalize']
AUG = cfg['preprocessing']['augmentations']


def load_image(img_path):
    """
    Charge une image, la redimensionne et restitue un tableau numpy normalisé si configuré.
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize(TARGET_SIZE)
    arr = np.array(img)
    return arr / 255.0 if NORMALIZE else arr


def augment_image(img):
    """
    Applique des transformations d'augmentation selon la configuration.
    """
    # Flip horizontal
    if AUG.get('horizontal_flip', False):
        img = ImageOps.mirror(img)
    # Rotation aléatoire
    rot = AUG.get('rotation_range', 0)
    if rot:
        angle = np.random.uniform(-rot, rot)
        img = img.rotate(angle)
    # Autres augmentations à ajouter ici si besoin
    return img


def preprocess_folder(input_dir, output_dir):
    """
    Applique le pipeline de prétraitement à toutes les images d'un dossier.

    Args:
        input_dir (str): chemin vers les images brutes.
        output_dir (str): chemin vers les images prétraitées.
    """
    # Nettoyage du dossier de sortie
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for cls in os.listdir(input_dir):
        in_cls_path = os.path.join(input_dir, cls)
        out_cls_path = os.path.join(output_dir, cls)
        if not os.path.isdir(in_cls_path) or cls.startswith('.'):
            continue
        os.makedirs(out_cls_path, exist_ok=True)
        for fname in os.listdir(in_cls_path):
            if fname.startswith('.'):
                continue
            in_path = os.path.join(in_cls_path, fname)
            try:
                img = Image.open(in_path).convert('RGB')
                img = img.resize(TARGET_SIZE)
                img = augment_image(img)
                arr = np.array(img)
                if NORMALIZE:
                    arr = arr / 255.0
                # Reconstruction et sauvegarde
                out_img = Image.fromarray((arr * 255).astype(np.uint8)) if NORMALIZE else img
                out_img.save(os.path.join(out_cls_path, fname))
            except Exception as e:
                print(f"Erreur sur {in_path}: {e}")


if __name__ == '__main__':
    # Exemple d'utilisation
    cfg = load_config()
    input_dir = cfg['data']['raw_dir']
    output_dir = cfg['data']['processed_dir']
    preprocess_folder(input_dir, output_dir)
