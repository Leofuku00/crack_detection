"""Shared Torch Hub loading and patch geometry for the DINOv3 models."""

import os
from pathlib import Path

import torch

from .base import BaseModel


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_hub(entrypoint, **kwargs):
    # Use the same upstream revision for the backbone and the optional text adapter.
    repo = os.environ.get("DINOV3_REPO")
    upstream = "facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751"
    return torch.hub.load(
        str(Path(repo).expanduser()) if repo else upstream,
        entrypoint,
        source="local" if repo else "github",
        trust_repo=True,
        skip_validation=True,
        **kwargs,
    )


def checkpoint(value, variable):
    value = value or os.environ.get(variable)
    if value is None:
        raise ValueError(f"Set {variable} to the official checkpoint path or HTTPS URL.")
    if not value.startswith("https://") and not Path(value).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint for {variable} does not exist: {value}")
    return value


def load_backbone(weights):
    return load_hub("dinov3_vitb16", weights=checkpoint(weights, "DINOV3_VITB16_WEIGHTS"))


class SegmentationModel(BaseModel):
    def __init__(self, *, threshold, random_seed=42, device=None):
        self.threshold = threshold
        self.random_seed = random_seed
        self.device = torch.device(device) if device else select_device()

    def patch_grid(self, images):
        height, width = images.shape[-2:]
        patch = self.patch_size
        if height < patch or width < patch or height % patch or width % patch:
            raise ValueError(f"Image height and width must be divisible by patch size {patch}.")
        return height // patch, width // patch

    def mask_from_scores(self, scores):
        return (scores > self.threshold).to(torch.uint8)

    def predict(self, X):
        """Return a binary mask, in the input spatial resolution, for each image."""
        return self.mask_from_scores(self.score_map(X))


class FrozenDINO(SegmentationModel):
    def __init__(self, *, weights=None, backbone=None, **kwargs):
        super().__init__(**kwargs)
        if backbone is None:
            backbone = load_backbone(weights)
        self.backbone = backbone.to(self.device).requires_grad_(False).eval()
        self.patch_size = self.backbone.patch_size

    @torch.no_grad()
    def patch_features(self, images):
        """Read the normalized final-block patch tokens, excluding special tokens."""
        rows, cols = self.patch_grid(images)
        tokens = self.backbone.forward_features(images.to(self.device))["x_norm_patchtokens"]
        return tokens.transpose(1, 2).reshape(len(images), -1, rows, cols).float()
