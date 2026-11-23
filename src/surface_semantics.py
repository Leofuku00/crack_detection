"""Zero-shot segmentation with Meta's pretrained DINOv3 dino.txt adapter."""

import os

import torch
from torch.nn import functional as F

from .models.dinov3_common import SegmentationModel, checkpoint, load_hub


def load_text_adapter(weights, backbone_weights):
    options = {}
    if vocab := os.environ.get("DINOTXT_BPE"):
        options["bpe_path_or_url"] = vocab
    return load_hub(
        "dinov3_vitl16_dinotxt_tet1280d20h24l",
        weights=checkpoint(weights, "DINOTXT_WEIGHTS"),
        backbone_weights=checkpoint(backbone_weights, "DINOV3_VITL16_WEIGHTS"),
        **options,
    )


class SurfaceSemantics(SegmentationModel):
    def __init__(
        self, *, threshold=0.0, prompt="crack, scratch, industrial defect",
        background_prompt="intact industrial surface", weights=None, backbone_weights=None,
        pipe_prompt="", adapter=None, **kwargs
    ):
        super().__init__(threshold=threshold, **kwargs)
        if adapter is None:
            adapter = load_text_adapter(weights, backbone_weights)
        model, tokenizer = adapter
        self.model = model.to(self.device).requires_grad_(False).eval()
        self.patch_size = self.model.visual_model.backbone.patch_size
        classes = [part.strip() for part in prompt.split(",") if part.strip()]
        self.pipe_prompts = [part.strip() for part in pipe_prompt.split(",") if part.strip()]
        background = [background_prompt] if isinstance(background_prompt, str) else background_prompt
        background = [label.strip() for label in background if label.strip()]
        if not classes or not background:
            raise ValueError("Both foreground and background prompts are required.")
        classes.extend(self.pipe_prompts)
        self.foreground_count = len(classes)
        with torch.no_grad():
            tokens = tokenizer.tokenize([*classes, *background]).to(self.device)
            embeddings = self.model.encode_text(tokens)
            # dino.txt concatenates text features aligned to the global class
            # token and local patch tokens. The second half matches dense patches.
            self.text_features = F.normalize(embeddings.chunk(2, dim=-1)[1].float(), dim=-1)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """The pretrained adapter needs no training images or masks."""
        return self

    @torch.no_grad()
    def prompt_score_maps(self, images):
        rows, cols = self.patch_grid(images)
        # The second output is text-aligned. The third contains raw backbone patches.
        _, patches, _ = self.model.encode_image_with_patch_tokens(images.to(self.device))
        patches = F.normalize(patches.float(), dim=-1)
        similarities = (patches @ self.text_features.T).transpose(1, 2)
        grids = similarities.reshape(len(images), len(self.text_features), rows, cols)
        return F.interpolate(grids, size=images.shape[-2:], mode="bilinear", align_corners=False)

    def score_map(self, images):
        scores = self.prompt_score_maps(images)
        foreground = scores[:, :self.foreground_count].amax(dim=1)
        background = scores[:, self.foreground_count:].amax(dim=1)
        return foreground - background
