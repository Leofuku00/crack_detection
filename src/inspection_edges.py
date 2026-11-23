"""Distinguish one-sided brightness boundaries from dark fissures."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def suppress_step_edges(prediction, image, settings):
    """Remove sustained transitions while retaining nearby dark troughs."""
    rows, cols = prediction.numpy().nonzero()
    if not rows.size:
        return prediction
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    gray = gaussian_filter(gray, 0.8)
    height, width = gray.shape

    def sample(dy, dx):
        return gray[np.clip(rows + dy, 0, height - 1), np.clip(cols + dx, 0, width - 1)]

    center = gray[rows, cols]
    step = np.zeros_like(center)
    trough = np.zeros_like(center)
    for ny, nx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        for radius in settings["radii"]:
            dy, dx = ny * radius, nx * radius
            left, right = sample(-dy, -dx), sample(dy, dx)
            trough = np.maximum(trough, np.minimum(left, right) - center)
            delta = right - left
            support = np.abs(delta)
            for distance in (-settings["span"], settings["span"]):
                ty, tx = nx * distance, -ny * distance
                flank = sample(ty + dy, tx + dx) - sample(ty - dy, tx - dx)
                support = np.minimum(support, np.sign(delta) * flank)
            if settings.get("bright_side_only"):
                support = np.where(center >= (left + right) * 0.5, support, 0)
            step = np.maximum(step, support)
        for radius in settings.get("protection_radii", ()):
            dy, dx = ny * radius, nx * radius
            trough = np.maximum(trough, np.minimum(sample(-dy, -dx), sample(dy, dx)) - center)
    remove = (step >= settings["min_contrast"]) & (trough < settings["preserve_trough"])
    if "min_luminance" in settings:
        remove &= center >= settings["min_luminance"]
    filtered = prediction.clone()
    filtered[rows[remove], cols[remove]] = False
    return filtered
