"""Whole-frame dino.txt wall scores and training-only threshold calibration."""

import torch
from torch.nn import functional as F


@torch.inference_mode()
def whole_frame_inspection_scores(model, image, long_edge, pipe_prompt_count=0):
    height, width = image.shape[-2:]
    patch = model.patch_size
    scale = min(long_edge / max(height, width), 1.0)
    size = tuple(max(1, round(side * scale / patch)) * patch for side in (height, width))
    resized = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False, antialias=True)
    if pipe_prompt_count:
        prompts = model.prompt_score_maps(resized)
        wall_end = model.foreground_count - pipe_prompt_count
        wall = prompts[:, :wall_end].amax(dim=1)
        pipe = prompts[:, wall_end:model.foreground_count].amax(dim=1)
        other = prompts[:, model.foreground_count:].amax(dim=1)
        scores = torch.stack((wall - other, pipe - other), dim=1)
    else:
        scores = model.score_map(resized)[:, None]
    native = F.interpolate(scores.cpu(), size=(height, width), mode="bilinear", align_corners=False)[0]
    if pipe_prompt_count:
        wall, pipe = native.unbind()
        return torch.maximum(wall, pipe), pipe > wall
    return native[0], None


def select_wall_threshold(samples):
    """Pick the strictest useful margin retaining 99% of each frame's crack pixels."""
    # The smallest score determines whether any training pixel is excluded.
    minimum = min(scores.min() for scores, _ in samples)
    crack_scores = [scores[target.bool()] for scores, target in samples]
    for step in range(10, -11, -1):
        threshold = step / 100
        if all(int((values > threshold).sum()) * 100 >= 99 * values.numel() for values in crack_scores):
            return threshold if minimum <= threshold else None
    return None


def wall_region(scores, threshold):
    return torch.ones_like(scores, dtype=torch.bool) if threshold is None else scores > threshold
