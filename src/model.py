"""Model — ResNet-50 pretrained + FC head for N cells.

4050-safe: frozen backbone by default, last block fine-tune only in L2,
AMP-compatible (no in-place ops that break autocast), K=200-500 (~300).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class GeoClassifier(nn.Module):
    """ResNet-50 backbone + single FC → N cells.

    Output: logits (B, N) → softmax → cell centroid lat/lon + confidence.
    Region derived from cell metadata (not predicted separately).

    Args:
        num_cells: number of geographic cells (200-500, ~300 for 4050).
        pretrained: whether to load ImageNet weights (True for L1/L2).
                   For Places365 comparison in L2, load via pretrained=False
                   then load_state_dict from Places365 checkpoint externally.
        freeze_backbone: if True, freeze all backbone params (L1 default).
                         L2 regime will unfreeze layer4.
        dropout: dropout before FC head (0.0-0.5).
    """

    def __init__(
        self,
        num_cells: int = 300,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_cells = num_cells
        self.freeze_backbone = freeze_backbone

        # torchvision 0.15+ uses weights= API; keep backward compat
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
        except AttributeError:
            # Fallback for older torchvision
            backbone = models.resnet50(pretrained=pretrained)

        # Keep all layers except final fc as feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        # resnet50 in_features = 2048
        in_features = backbone.fc.in_features

        # Optional dropout before head
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(in_features, num_cells)

        # Init head (torch default kaiming is fine; explicit xavier for clarity)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        # Cache backbone for freeze/unfreeze helpers
        self._backbone_modules = list(self.backbone.modules())

        if freeze_backbone:
            self.freeze_backbone_params()

    def freeze_backbone_params(self) -> None:
        """Freeze backbone, keep head trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.freeze_backbone = True

    def unfreeze_backbone(self, last_block_only: bool = True) -> None:
        """Unfreeze for L2 last-block fine-tune.

        Args:
            last_block_only: if True, unfreeze only layer4 (Bottleneck blocks).
                             If False, unfreeze entire backbone.
        """
        if last_block_only:
            # backbone is Sequential: [conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool]
            # layer4 is index 7 in the Sequential
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Sequential children: find layer4 by name
            # The Sequential contains named children; layer4 is 7th element
            try:
                layer4 = self.backbone[7]
                for p in layer4.parameters():
                    p.requires_grad = True
            except Exception:
                # Fallback: unfreeze all if structure unexpected
                for p in self.backbone.parameters():
                    p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True
        self.freeze_backbone = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: (B, 3, H, W) normalized with ImageNet stats, H=W=224 (or 384)

        Returns:
            logits: (B, num_cells)
        """
        # Backbone: (B, 2048, 1, 1) after avgpool
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)  # (B, 2048)
        feats = self.dropout(feats)
        logits = self.head(feats)
        return logits

    def predict(
        self, x: torch.Tensor, cells: Optional[list] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper — returns (pred_cell_ids, confidences).

        Args:
            x: batch of images
            cells: unused in L0; L1 will map cell->centroid using cells_to_centroids

        Returns:
            pred_ids: (B,) long tensor
            conf: (B,) float tensor (max softmax prob)
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return pred, conf


def build_model(
    num_cells: int = 300,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.0,
) -> GeoClassifier:
    """Factory used by train.py / eval.py.

    AMP-compatible: model itself has no autocast-unfriendly ops.
    Caller wraps forward+loss in torch.cuda.amp.autocast (train.py).
    """
    return GeoClassifier(
        num_cells=num_cells,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
