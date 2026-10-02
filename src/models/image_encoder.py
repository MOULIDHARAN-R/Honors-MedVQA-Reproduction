"""
image_encoder.py

VGG-16 Image Encoder for Medical VQA.
Paper: "Visual Question Answering in the Medical Domain" (Canepa et al., 2023)
  * Uses VGG-16 pretrained on ImageNet.
  * Extracts a 1-dimensional visual encoding (4096-dimensional fc7 feature vector).
"""
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

try:
    from ..config import Config, CFG
except (ImportError, ValueError):
    from src.config import Config, CFG


class VGG16ImageEncoder(nn.Module):
    """
    Image encoder using ImageNet-pretrained VGG-16.
    Extracts the 4096-d feature representation from the second dense layer (fc7).
    """
    def __init__(self, cfg: Config = CFG):
        super().__init__()
        self.cfg = cfg
        weights = VGG16_Weights.IMAGENET1K_V1 if cfg.imagenet_pretrained else None
        backbone = vgg16(weights=weights)

        # 13 convolutional layers + 5 max-pooling layers
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # fc6 (Linear 25088 -> 4096) + ReLU + Dropout + fc7 (Linear 4096 -> 4096) + ReLU
        # Using inplace=False on ReLUs ensures safe gradient flow and hook compatibility (GradCAM).
        fc_layers = []
        for i, layer in enumerate(backbone.classifier):
            if i >= 5:  # stop after fc7 + ReLU (exclude fc7 dropout and fc8 1000-class head)
                break
            if isinstance(layer, nn.ReLU):
                fc_layers.append(nn.ReLU(inplace=False))
            else:
                fc_layers.append(layer)

        self.fc = nn.Sequential(*fc_layers)
        self.out_dim = cfg.image_feature_dim  # 4096

        if cfg.freeze_image_encoder:
            self.freeze()

    def freeze(self):
        """Freeze all parameters in the image encoder."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters in the image encoder."""
        for p in self.parameters():
            p.requires_grad = True

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns the final convolutional layer of VGG-16 (features[28])
        for Grad-CAM evidence verification as described in Section III-D / IV-B.
        """
        # In torchvision VGG-16: features[28] is the 13th Conv2d(512, 512, 3), followed by ReLU(29) and MaxPool(30)
        return self.features[28]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] normalized image tensor
        Returns:
            [B, 4096] image feature vector
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
