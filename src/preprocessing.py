import os
from PIL import Image, ImageOps
import numpy as np

def load_image(img_path, target_size=(128, 128)):
    """
    Ouvre une image et la redimensionne au format voulu.
    """
    img = Image.open(img_path)
    img = img.convert('RGB')  # Assure la cohérence des canaux
    img = img.resize(target_size)
    return np.array(img)

def normalize_image(img_array):
    """
    Normalise les pixels dans [0, 1].
    """
    return img_array / 255.0

def augment_image(img):
    """
    Applique une transformation d'augmentation de données simple (ex: flip horizontal).
    """
    # Exemples simples : à ajouter selon les besoins
    img_aug = ImageOps.mirror(img)  # Flip horizontal
    return img_aug

def preprocess_folder(input_dir, output_dir, target_size=(128,128), do_augment=False):
    """
    Applique le prétraitement à toutes les images d'un dossier et sauvegarde le résultat.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for cls in os.listdir(input_dir):
        in_cls_path = os.path.join(input_dir, cls)
        out_cls_path = os.path.join(output_dir, cls)
        if not os.path.isdir(in_cls_path): continue
        if not os.path.exists(out_cls_path):
            os.makedirs(out_cls_path)
        for fname in os.listdir(in_cls_path):
            if fname.startswith('.'): continue
            in_path = os.path.join(in_cls_path, fname)
            try:
                img = Image.open(in_path)
                img = img.convert('RGB')
                img = img.resize(target_size)
                if do_augment:
                    img = augment_image(img)
                img.save(os.path.join(out_cls_path, fname))
            except Exception as e:
                print(f"Erreur sur {in_path}: {e}")

if __name__ == '__main__':
    img_input_dir = '../data/raw'
    img_output_dir = '../data/processed'
    preprocess_folder(img_input_dir, img_output_dir, do_augment=False)
