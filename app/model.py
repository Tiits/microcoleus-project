# app/model.py

import yaml
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from app.crossval_model import CrossValClassifier

# Import des poids pré-entrainés pour chaque archi
from torchvision.models import ResNet18_Weights, ResNet50_Weights

class Classifier:
    """
    Charge un modèle finetuné (ResNet18 ou ResNet50 selon la config),
    et fournit une méthode predict(img) retournant les probabilités.
    """
    def __init__(self, config_path: str, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        # --- Chargement de la config YAML ---
        cfg = yaml.safe_load(open(config_path, "r"))
        self.input_size = cfg.get("input_size", 224)
        self.mean = cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = cfg.get("std", [0.229, 0.224, 0.225])
        self.classes = cfg.get("classes", ["Non toxique", "Toxique"])

        # Pipeline de transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # --- Sélection de l'architecture selon le nom du fichier de config ---
        arch = Path(config_path).stem.split("_")[-1]  # ex: 'resnet18' ou 'resnet50'
        if arch == "resnet18":
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif arch == "resnet50":
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Architecture '{arch}' non supportée.")

        # Adaptation de la couche fully-connected
        in_feats = self.model.fc.in_features
        out_feats = len(self.classes)
        self.model.fc = nn.Linear(in_feats, out_feats)

        # --- Chargement du checkpoint ---
        ckpt = torch.load(model_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)

        new_state = {}
        for k, v in state.items():
            if k == "fc.weight":
                new_state["fc.1.weight"] = v
            elif k == "fc.bias":
                new_state["fc.1.bias"] = v
            else:
                new_state[k] = v

        self.model.load_state_dict(new_state, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, img: Image.Image, return_dict: bool = True):
        """
        Prédit sur une seule image PIL.
        return_dict=True  -> renvoie {classe: prob}
        return_dict=False -> renvoie (idx_pred, [probs])
        """
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0)

        if return_dict:
            return {cls: float(probs[i]) for i, cls in enumerate(self.classes)}
        else:
            pred_idx = int(probs.argmax())
            return pred_idx, [float(p) for p in probs]


class CVWrapper:
    """Usable dans app.py pour choisir fold ou ensemble"""

    def __init__(self, config_path, checkpoint_dir, n_folds=2, device="cpu"):
        self.cv = CrossValClassifier(config_path, checkpoint_dir, n_folds, device)
        self.classes = self.cv.classes

    def predict(self, img, mode="ensemble", fold=None):
        if mode == "ensemble":
            return self.cv.predict_ensemble(img)
        elif mode == "fold" and fold is not None:
            return self.cv.predict_fold(img, fold)
        else:
            raise ValueError("Choose mode 'ensemble' or 'fold' with fold index")

# Exemple de test rapide
if __name__ == "__main__":
    clf = Classifier(
        config_path="../configs/config_finetune_resnet50.yaml",
        model_path="../models/resnet50_2025-07-23_4d75f97.pth",
        device="cpu"
    )
    img = Image.open("tests/sample.jpg")
    print(clf.predict(img))