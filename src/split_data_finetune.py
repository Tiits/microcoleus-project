import os
import argparse
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='../data/processed/all', help='Répertoire des images prétraitées')
    parser.add_argument('--split_dir', type=str, default='../splits/all/finetune', help='Répertoire de sortie pour les splits')
    parser.add_argument('--train_size', type=float, default=0.7, help='Proportion du train set')
    parser.add_argument('--val_size', type=float, default=0.15, help='Proportion du validation set')
    parser.add_argument('--test_size', type=float, default=0.15, help='Proportion du test set')
    args = parser.parse_args()

    classes = [d for d in os.listdir(args.raw_dir) if os.path.isdir(os.path.join(args.raw_dir, d))]
    file_paths, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(args.raw_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        train_size=args.train_size,
        stratify=labels,
        random_state=42
    )

    rel_val = args.val_size / (args.val_size + args.test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        train_size=rel_val,
        stratify=temp_labels,
        random_state=42
    )

    os.makedirs(args.split_dir, exist_ok=True)
    for split, paths in zip(['train', 'val', 'test'], [train_paths, val_paths, test_paths]):
        out_txt = os.path.join(args.split_dir, f"{split}.txt")
        with open(out_txt, 'w') as f:
            for p in paths:
                f.write(p + "\n")

if __name__ == '__main__':
    main()
