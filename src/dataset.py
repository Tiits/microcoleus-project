import os
from PIL import Image

def list_images(img_data_dir, valid_exts=None):
    """
    Retourne un dict {classe : [liste de fichiers images valides]}
    """
    if valid_exts is None:
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

    class_dict = {}
    for cls in os.listdir(img_data_dir):
        class_path = os.path.join(img_data_dir, cls)
        if os.path.isdir(class_path) and not cls.startswith('.'):
            images = [f for f in os.listdir(class_path)
                      if os.path.splitext(f)[-1].lower() in valid_exts]
            class_dict[cls] = images
    return class_dict

def check_image_integrity(img_data_dir, valid_exts=None):
    """
    Vérifie que toutes les images sont ouvrables. Retourne une liste des fichiers corrompus.
    """
    if valid_exts is None:
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

    corrupt_files = []
    for cls in os.listdir(img_data_dir):
        class_path = os.path.join(img_data_dir, cls)
        if os.path.isdir(class_path) and not cls.startswith('.'):
            for img_file in os.listdir(class_path):
                if os.path.splitext(img_file)[-1].lower() in valid_exts:
                    img_path = os.path.join(class_path, img_file)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # Vérifie sans charger l'image en mémoire
                    except Exception:
                        corrupt_files.append(img_path)
    return corrupt_files

# Exemple d’utilisation
if __name__ == '__main__':
    data_dir = '../data/raw'
    print("Listing des images :", list_images(data_dir))
    corrupts = check_image_integrity(data_dir)
    if corrupts:
        print("Images corrompues trouvées :", corrupts)
    else:
        print("Aucune image corrompue détectée.")
