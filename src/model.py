"""Model — ResNet-50 pretrained + FC head for N cells.

4050-safe: frozen backbone by default, last block fine-tune only in L2,
AMP-compatible (no in-place ops that break autocast), K=200-500 (~300).
"""

from __future__ import annotations

from pathlib import Path
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


def load_places365_checkpoint(
    model: GeoClassifier,
    checkpoint_path: str | Path,
    strict_backbone: bool = False,
) -> dict:
    """Load Places365 pretrained weights into a GeoClassifier's backbone.

    The official checkpoint (resnet50_places365.pth.tar) is a legacy format
    whose state may be wrapped in ``state_dict`` or ``module``. This loader:

    * Unwraps ``module.`` prefixes.
    * Only loads keys whose shapes match the torchvision ResNet-50 backbone
      (i.e. keys NOT in ``head.*`` or ``fc.*``).
    * Reports loaded / skipped / mismatched counts; never silently claims
      full loading when it did not.

    Args:
        model: a GeoClassifier instance (backbone + head already constructed).
        checkpoint_path: path to ``resnet50_places365.pth.tar``.
        strict_backbone: if True, raise on any backbone key missing from ckpt
                         (default False — partial load is acceptable).

    Returns:
        dict with keys ``loaded``, ``skipped``, ``mismatched``, ``total_ckpt``,
        ``total_backbone`` for caller logging.

    Raises:
        FileNotFoundError: if checkpoint_path does not exist.
        RuntimeError: if the file is not a valid PyTorch checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Places365 checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Unwrap: official checkpoint may store under 'state_dict', 'model',
    # or be a raw state dict itself.
    if isinstance(ckpt, dict):
        # Try common wrapper keys
        sd = None
        for key in ("state_dict", "model", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        if sd is None:
            # Might already be a flat state dict (keys look like layer names)
            first_key = next(iter(ckpt), None)
            if first_key is not None and (first_key.startswith("conv1.") or first_key.startswith("layer")):
                sd = ckpt
            else:
                raise RuntimeError(
                    f"Places365 checkpoint at {checkpoint_path} has unrecognized structure. "
                    f"Top-level keys: {list(ckpt.keys())[:10]}"
                )
    else:
        raise RuntimeError(
            f"Places365 checkpoint at {checkpoint_path} is not a dict (type={type(ckpt).__name__})."
        )

    # Strip 'module.' prefix if present (DataParallel / DDP wrapper)
    cleaned_sd: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        cleaned_sd[new_k] = v

    # Build a view of the backbone's expected keys/shapes
    backbone_state = model.backbone.state_dict()
    backbone_keys = set(backbone_state.keys())

    # Filter to backbone keys only (exclude head.*)
    loaded: list[str] = []
    skipped: list[str] = []
    mismatched: list[str] = []

    for ckpt_key in sorted(cleaned_sd.keys()):
        ckpt_tensor = cleaned_sd[ckpt_key]
        if ckpt_key.startswith("head.") or ckpt_key.startswith("fc."):
            skipped.append(ckpt_key)
            continue
        if ckpt_key not in backbone_keys:
            skipped.append(ckpt_key)
            continue
        # Shape check
        target_shape = backbone_state[ckpt_key].shape
        if ckpt_tensor.shape != target_shape:
            mismatched.append(f"{ckpt_key}: ckpt {tuple(ckpt_tensor.shape)} vs model {tuple(target_shape)}")
            continue
        # Match — assign
        backbone_state[ckpt_key] = ckpt_tensor
        loaded.append(ckpt_key)

    # Check for backbone keys missing from checkpoint
    missing_from_ckpt = backbone_keys - {k for k in cleaned_sd.keys() if k in backbone_keys}
    # Only flag missing if they weren't already in mismatched
    missing_from_ckpt_names = sorted(missing_from_ckpt - set(m.split(":")[0] for m in mismatched))

    # Apply the matched weights
    model.backbone.load_state_dict(backbone_state, strict=False)

    result = {
        "loaded": len(loaded),
        "skipped": len(skipped),
        "mismatched": len(mismatched),
        "missing_from_ckpt": len(missing_from_ckpt_names),
        "total_ckpt": len(cleaned_sd),
        "total_backbone": len(backbone_keys),
        "loaded_keys": loaded,
        "skipped_keys": skipped,
        "mismatched_keys": mismatched,
        "missing_keys": missing_from_ckpt_names,
    }

    # Report
    print(f"[places365] checkpoint keys: {len(cleaned_sd)} | backbone keys: {len(backbone_keys)}")
    print(f"[places365] loaded: {result['loaded']} | skipped: {result['skipped']} "
          f"| mismatched: {result['mismatched']} | missing-from-ckpt: {result['missing_from_ckpt']}")

    if result["loaded"] == 0:
        print("[places365] WARNING: zero backbone weights loaded. Checkpoint may be incompatible.")
    elif result["loaded"] < len(backbone_keys) * 0.5:
        print(f"[places365] WARNING: only {result['loaded']}/{len(backbone_keys)} backbone keys matched "
              "— partial load; model may not perform as expected.")

    if strict_backbone and missing_from_ckpt_names:
        raise RuntimeError(
            f"strict_backbone=True but {len(missing_from_ckpt_names)} backbone keys "
            f"are missing from checkpoint: {missing_from_ckpt_names[:10]}..."
        )

    return result
