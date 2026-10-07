"""Segmentasyon, görüntü-temelli grade tahmini ve hibrit özellik çıkarımı modelleri."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_segmentation_model(name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """smp model fabrikası; başarılı checkpoint adı config ile birlikte kaydedilir."""
    import segmentation_models_pytorch as smp

    encoder_weights = "imagenet" if pretrained else None
    models = {
        "unet_resnet34": lambda: smp.Unet(
            encoder_name="resnet34", encoder_weights=encoder_weights, in_channels=3,
            classes=num_classes, activation=None,
        ),
        "unetpp_resnet34": lambda: smp.UnetPlusPlus(
            encoder_name="resnet34", encoder_weights=encoder_weights, in_channels=3,
            classes=num_classes, activation=None,
        ),
        "fpn_resnet34": lambda: smp.FPN(
            encoder_name="resnet34", encoder_weights=encoder_weights, in_channels=3,
            classes=num_classes, activation=None,
        ),
    }
    if name not in models:
        raise ValueError(f"Bilinmeyen segmentasyon modeli: {name}. Seçenekler: {list(models)}")
    return models[name]()


class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=1)
        n_classes = probabilities.shape[1]
        one_hot = F.one_hot(target.long(), num_classes=n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class MulticlassFocalLoss(nn.Module):
    def __init__(self, weights: list[float], gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.long(), weight=self.weights, reduction="none")
        p_t = torch.exp(-ce)
        return ((1 - p_t).pow(self.gamma) * ce).mean()


class CompoundSegmentationLoss(nn.Module):
    """0.40 CE + 0.40 Dice + 0.20 Focal; weights modul buffer'ı olarak GPU'ya taşınır."""
    def __init__(self, class_weights: list[float], loss_weights: dict[str, float]):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.dice = MulticlassDiceLoss()
        self.focal = MulticlassFocalLoss(class_weights)
        self.loss_weights = loss_weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.long(), weight=self.class_weights)
        dice = self.dice(logits, target)
        focal = self.focal(logits, target)
        return (
            self.loss_weights["cross_entropy"] * ce
            + self.loss_weights["dice"] * dice
            + self.loss_weights["focal"] * focal
        )


def _resnet50_weights(pretrained: bool):
    from torchvision.models import ResNet50_Weights
    return ResNet50_Weights.IMAGENET1K_V2 if pretrained else None


class ImageGradeClassifier(nn.Module):
    """ResNet50 tabanlı dört-sınıflı görüntü-only grade sınıflandırıcı."""
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.35):
        super().__init__()
        from torchvision.models import resnet50
        backbone = resnet50(weights=_resnet50_weights(pretrained))
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class HybridOrdinalClassifier(nn.Module):
    """ResNet50 image embedding + 12 morfometrik feature, CORAL ordinal head."""
    def __init__(self, n_morph_features: int, num_grades: int = 4, pretrained: bool = True):
        super().__init__()
        base = ImageGradeClassifier(num_classes=num_grades, pretrained=pretrained)
        self.image_encoder = base.feature_extractor
        self.morph_encoder = nn.Sequential(
            nn.Linear(n_morph_features, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.20),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(2048 + 32, 256), nn.ReLU(inplace=True), nn.Dropout(0.35),
            nn.Linear(256, num_grades - 1),
        )

    def forward(self, image: torch.Tensor, morph: torch.Tensor) -> torch.Tensor:
        image_features = torch.flatten(self.image_encoder(image), 1)
        return self.head(torch.cat([image_features, self.morph_encoder(morph)], dim=1))


def coral_targets(target: torch.Tensor, num_thresholds: int) -> torch.Tensor:
    threshold_ids = torch.arange(num_thresholds, device=target.device)[None, :]
    return (target[:, None] > threshold_ids).float()


def coral_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, coral_targets(target, logits.shape[1]))


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()
