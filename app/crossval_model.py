# app/crossval_model.py
import yaml
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from torchvision.models import ResNet18_Weights


class CrossValClassifier:
    """
    Charge plusieurs checkpoints (un par fold) et permet de prédire par fold ou en ensemble (majority vote).
    """
    def __init__(self, config_path: str, checkpoint_dir: str, n_folds: int = 2, device: str = "cpu"):
        self.device = torch.device(device)
        # --- Chargement config YAML ---
        cfg = yaml.safe_load(open(config_path, "r"))
        self.threshold = 0.49
        aug = cfg['augmentation']
        # Transformation d'évaluation
        self.transform = transforms.Compose([
            transforms.Resize(tuple(aug['resize'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=aug['normalization']['mean'],
                                 std=aug['normalization']['std'])
        ])
        self.classes = cfg.get("classes", ["Non toxique", "Toxique"])
        self.models = []
        # Charger chaque fold
        for fold in range(1, n_folds+1):
            model = models.resnet18(
                weights=ResNet18_Weights.DEFAULT if cfg['model']['pretrained'] else None
            )
            # adapter la tête
            in_feats = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_feats, cfg['model']['num_classes'])
            )
            ckpt_path = Path(checkpoint_dir) / f"fold{fold}_model.pth"
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device).eval()
            self.models.append(model)

    def predict_fold(self, img: Image.Image, fold: int):
        """
        Retourne les probabilités du fold indiqué (1-indexed).
        """
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.models[fold - 1](x)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0)

            pos_idx = self.classes.index("Toxique")
            vote = int(probs[pos_idx] >= self.threshold)

            return {**{cls: float(probs[i]) for i, cls in enumerate(self.classes)}, "vote": vote}

    def predict_ensemble(self, img: Image.Image):
        """
        Moyenne des probabilités par classe + verdict sur la classe 'Toxique'.
        Retourne un dict {classe: prob, 'vote': 0|1}.
        """
        probs_list = []
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for m in self.models:
                logits = m(x)
                probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
                probs_list.append(probs)

        mean_probs = torch.stack(probs_list).mean(dim=0)
        # index de la classe "Toxique"
        pos_idx = self.classes.index("Toxique")
        # verdict selon seuil optimal
        vote = int(mean_probs[pos_idx] >= self.threshold)

        return {
            **{cls: float(mean_probs[i]) for i, cls in enumerate(self.classes)},
            "vote": vote
        }