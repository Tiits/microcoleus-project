import os, random, yaml
from pathlib import Path
from src.config_utils import load_config

def generate_splits(cfg, out_dir='../splits', train_ratio=0.7, val_ratio=0.15):
    random.seed(cfg['seed'])
    data_dir = Path(cfg['data']['processed_dir'])
    # Récupérer tous les fichiers et leurs labels
    file_label_pairs = []
    for cls in sorted(data_dir.iterdir()):
        if cls.is_dir():
            for img in cls.iterdir():
                if img.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff'):
                    file_label_pairs.append((str(img), cls.name))
    # Mélange
    random.shuffle(file_label_pairs)
    n = len(file_label_pairs)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio   * n)
    splits = {
        'train': file_label_pairs[:n_train],
        'val'  : file_label_pairs[n_train:n_train+n_val],
        'test' : file_label_pairs[n_train+n_val:],
    }
    # Sauvegarde
    os.makedirs(out_dir, exist_ok=True)
    for split, pairs in splits.items():
        with open(f'{out_dir}/{split}.txt', 'w') as f:
            for path, label in pairs:
                f.write(f"{path}\t{label}\n")
    print(f"Splits saved in {out_dir}/train.txt, val.txt, test.txt")

if __name__ == "__main__":
    # Chargement de la config
    cfg = load_config()

    generate_splits(cfg)
