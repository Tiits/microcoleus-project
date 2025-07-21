# app/model.py

import yaml
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn


class Classifier:
    """
    Charge le modèle entraîné et fournit une méthode predict(img)
    pour renvoyer la probabilité d'une cyanobactérie toxique.
    """
    def __init__(
        self,
        config_path: str = "../configs/config_finetune_resnet18.yaml",
        model_path: str = "../models/best_model.pth",
        device: str = None,
    ):
        # --- 1. Load config ---
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # --- 2. Device ---
        if device:
            self.device = torch.device(device)
        else:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # --- 3. Preprocessing pipeline ---
        aug = self.config['augmentation']
        self.preprocess = transforms.Compose([
            transforms.Resize(tuple(aug['resize'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug['normalization']['mean'],
                std=aug['normalization']['std']
            )
        ])

        # --- 4. Build model architecture ---
        model_cfg = self.config['model']
        # Ici on reprend ResNet18 pré-entrainé pour récupérer la bonne archi
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remplace la couche de sortie pour ton nombre de classes (2)
        in_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feats, model_cfg['num_classes'])

        # --- 5. Charger les poids du .pth ---
        checkpoint = torch.load(model_path, map_location=self.device)
        # Si tu as enregistré tout le model.state_dict(),
        # sinon adapte selon ton saving (ex: checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # --- 6. Classes éventuelles (facultatif) ---
        # Tu peux définir dans ta config ['data']['classes'] = ['non_toxique','toxique']
        self.classes = self.config.get('data', {}).get(
            'classes', ['Non-toxic', 'Toxic']
        )

    def predict(self, image: Image.Image, return_dict: bool = True):
        """
        Prend une PIL.Image, applique le preprocessing, et renvoie :
          - si return_dict=True : {'non_toxique': float, 'toxique': float}
          - sinon : (pred_idx: int, probas: List[float])
        """
        # 1. Prétraitement + batch dim
        x = self.preprocess(image).unsqueeze(0).to(self.device)

        # 2. Inférence
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # 3. Format de sortie
        if return_dict:
            return {cls: float(p) for cls, p in zip(self.classes, probs)}
        else:
            pred_idx = int(probs.argmax())
            return pred_idx, [float(p) for p in probs]


# Exemple d'utilisation rapide
if __name__ == "__main__":
    clf = Classifier(
        config_path="../configs/config_finetune_resnet18.yaml",
        model_path="../models/best_model.pth",
        device="cpu"
    )
    img = Image.open("tests/sample.jpg")
    print(clf.predict(img))
