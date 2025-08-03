"""
Module for building and loading image classification models (ResNet18, ResNet50, or EfficientNetB0),
and a cross-validation wrapper for ensemble/fold-based inference.
"""

import yaml
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
import math
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.models import EfficientNet_B0_Weights


# Helper functions: transform pipeline construction, model builder, and checkpoint loader
def build_transform(cfg: dict):
    """
    Build an evaluation image transform pipeline from config dict.
    Uses cfg['augmentation'] if present, else falls back to input_size, mean, std.
    """
    aug = cfg.get("augmentation", {})
    resize = tuple(aug.get("resize", [cfg.get("input_size", 224)] * 2))
    norm = aug.get("normalization", {})
    mean = norm.get("mean", cfg.get("mean", [0.485, 0.456, 0.406]))
    std = norm.get("std", cfg.get("std", [0.229, 0.224, 0.225]))
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a model given architecture name and number of output classes.
    Supported architectures: resnet18, resnet50, efficientnet_b0.
    """
    arch_map = {
        "resnet18": (models.resnet18, ResNet18_Weights),
        "resnet50": (models.resnet50, ResNet50_Weights),
        "efficientnet_b0": (models.efficientnet_b0, EfficientNet_B0_Weights)
    }
    if arch not in arch_map:
        raise ValueError(f"Unsupported architecture '{arch}'.")
    builder, weights_enum = arch_map[arch]
    weights = weights_enum.DEFAULT if pretrained else None
    model = builder(weights=weights)
    # Replace classifier head
    if hasattr(model, "fc"):
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif hasattr(model, "classifier"):
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"Model architecture '{arch}' has no recognizable classifier head.")
    return model



def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device, strict: bool = False):
    """
    Load model weights from checkpoint. Supports checkpoints with 'model_state_dict' key.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=strict)
    return model


# Helper to infer architecture from config or file name
def infer_architecture(config_path: str, cfg: dict) -> str:
    """
    Infer the architecture name from config dict or file name.
    """
    model_cfg = cfg.get("model", {})
    if 'arch' in model_cfg:
        return model_cfg['arch']
    stem = Path(config_path).stem.lower()
    if 'resnet18' in stem:
        return 'resnet18'
    if 'resnet50' in stem:
        return 'resnet50'
    if 'efficientnet_b0' in stem or 'best' in stem:
        return 'efficientnet_b0'
    raise ValueError(f"Cannot infer architecture from config file name '{stem}'. Please specify 'model.arch' in config.")


# ImageClassifier: loads a fine-tuned model and provides prediction capability
class ImageClassifier:
    """
    Loads a fine-tuned model (ResNet18, ResNet50, or EfficientNetB0 based on config)
    and provides a predict(img) method returning class probabilities.
    """
    def __init__(self, config_path: str, model_path: str, device: str = "cpu"):
        cfg = yaml.safe_load(open(config_path, "r"))
        self.classes = cfg.get("classes", ["Non-toxic", "Toxic"])

        # Build transform pipeline from config
        self.transform = build_transform(cfg)

        self.device = torch.device(device)

        # Determine architecture and pretrained flag
        arch = infer_architecture(config_path, cfg)
        pretrained = cfg.get("model", {}).get("pretrained", True)

        # Build model
        self.model = build_model(arch, num_classes=len(self.classes), pretrained=pretrained)

        # Load weights
        load_checkpoint(self.model, model_path, self.device)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, img: Image.Image, return_dict: bool = True):
        """
        Predict on a single PIL image.
        return_dict=True  -> returns {class: prob}
        return_dict=False -> returns (pred_idx, [probs])
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


# Cross-validation classes: fold-specific and ensemble inference
# CrossValImageClassifier: loads multiple fold checkpoints for CV inference
class CrossValImageClassifier:
    """
    Loads multiple checkpoints (one per fold) and enables predictions by fold or ensemble (majority vote).
    """
    def __init__(self, config_path: str, checkpoint_dir: str, n_folds: int = 2, device: str = "cpu"):
        cfg = yaml.safe_load(open(config_path, "r"))
        self.threshold = cfg.get("threshold", 0.5)

        # Build transform pipeline from config
        self.transform = build_transform(cfg)

        self.classes = cfg.get("classes", ["Non-toxic", "Toxic"])
        # Determine architecture and pretrained flag
        arch = infer_architecture(config_path, cfg)
        pretrained = cfg.get("model", {}).get("pretrained", True)

        self.device = torch.device(device)

        self.models = []
        for fold in range(1, n_folds + 1):
            model = build_model(arch, num_classes=len(self.classes), pretrained=pretrained)
            ckpt_path = Path(checkpoint_dir) / f"fold_{fold}.pth"
            load_checkpoint(model, str(ckpt_path), self.device)
            model.to(self.device)
            model.eval()
            self.models.append(model)

    def predict_fold(self, img: Image.Image, fold: int):
        """
        Returns probabilities for the specified fold (1-indexed) and a binary vote.
        """
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.models[fold - 1](x)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
        pos_idx = self.classes.index("Toxic")
        vote = int(probs[pos_idx] >= self.threshold)
        return {**{cls: float(probs[i]) for i, cls in enumerate(self.classes)}, "vote": vote}

    def predict_ensemble(self, img: Image.Image):
        """
        Returns mean probabilities across folds and a majority vote for the 'Toxic' class.
        """
        probs_list = []
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for model in self.models:
                probs_list.append(torch.softmax(model(x), dim=1).cpu().squeeze(0))
        mean_probs = torch.stack(probs_list).mean(dim=0)
        pos_idx = self.classes.index("Toxic")
        # Compute fold-level votes and perform majority vote
        fold_votes = [int(probs[pos_idx] >= self.threshold) for probs in probs_list]
        vote = int(sum(fold_votes) >= math.ceil(len(self.models) / 2))
        return {**{cls: float(mean_probs[i]) for i, cls in enumerate(self.classes)}, "vote": vote}


# CVWrapper: simplifies selection between fold and ensemble predictions
class CVWrapper:
    """Used in app.py for selecting fold or ensemble prediction."""

    def __init__(self, config_path, checkpoint_dir, n_folds=2, device="cpu"):
        self.cv = CrossValImageClassifier(config_path, checkpoint_dir, n_folds, device)
        self.classes = self.cv.classes

    def predict(self, img, mode="ensemble", fold=None):
        if mode == "ensemble":
            return self.cv.predict_ensemble(img)
        elif mode == "fold" and fold is not None:
            return self.cv.predict_fold(img, fold)
        else:
            raise ValueError("Choose mode 'ensemble' or 'fold' with a valid fold index.")

# Example usage (quick test of ImageClassifier)
if __name__ == "__main__":
    clf = ImageClassifier(
        config_path="../configs/config_finetune_resnet50.yaml",
        model_path="../models/resnet50_2025-07-23_4d75f97.pth",
        device="cpu"
    )
    img = Image.open("tests/sample.jpg")
    print(clf.predict(img))