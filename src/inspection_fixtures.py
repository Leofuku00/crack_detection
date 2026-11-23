"""Mask identified hardware before publishing native-resolution crack predictions."""

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter

from .inspection_region import detect_components, segment_proposals


def fixture_region(size, proposals, settings):
    """Keep compact masks and reject proposals dominated by the surface around a wire."""
    width, height = size
    result = np.zeros((height, width), dtype=bool)
    for group, _, box, mask in proposals:
        limits = settings["detection_prompts"][group]
        patch = mask[box]
        if patch.size > limits["max_box_area"] * result.size:
            continue
        if patch.mean() > limits["max_box_fill"]:
            continue
        result |= mask
    return result


def pad_fixtures(mask, pixels):
    if not pixels:
        return mask
    return torch.from_numpy(maximum_filter(mask.numpy(), size=2 * pixels + 1))


@torch.inference_mode()
def fixture_masks(originals, config, device):
    settings = config["fixture_filter"]
    options = {**config["inspection"], "detection_prompts": settings["detection_prompts"]}
    frames = []
    for original in originals:
        image = original.copy()
        image.thumbnail((settings["long_edge"], settings["long_edge"]), Image.Resampling.LANCZOS)
        frames.append(image)
    detections = detect_components(frames, options, device)
    proposals = segment_proposals(frames, detections, options, device)
    masks = []
    for original, frame, objects in zip(originals, frames, proposals, strict=True):
        mask = fixture_region(frame.size, objects, settings)
        native = Image.fromarray(mask).resize(original.size, Image.Resampling.NEAREST)
        masks.append(pad_fixtures(torch.from_numpy(np.array(native)), settings["padding_pixels"]))
    return masks
