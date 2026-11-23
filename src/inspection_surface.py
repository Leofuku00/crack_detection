"""Refine building masks using the surrounding painted surface color."""

import numpy as np
import torch
from PIL import Image
from scipy import ndimage


def wall_surface_mask(original, building, settings):
    preview = original.copy()
    preview.thumbnail((settings["long_edge"],) * 2, Image.Resampling.LANCZOS)
    coarse = np.asarray(Image.fromarray(building.numpy()).resize(preview.size, Image.Resampling.NEAREST))
    interior = ndimage.binary_erosion(coarse, iterations=3)
    if not interior.any():
        return building

    lab = np.asarray(preview.convert("LAB"))
    # Pillow stores the signed Lab chroma channels in unsigned byte arrays.
    chroma = lab[..., 1:].view(np.int8).astype(np.float32)
    wall_chroma = np.median(chroma[interior], axis=0)
    within_color = np.linalg.norm(chroma - wall_chroma, axis=-1) <= settings["chroma_radius"]
    samples = interior & within_color
    if not samples.any():
        return building
    lightness = lab[..., 0]
    low = np.percentile(lightness[samples], 15) - settings["lightness_margin"]
    compatible = coarse & within_color & (lightness >= low)

    # Fill narrow interruptions using the color on the surrounding surface.
    # The center of a fissure can have any color.
    radius = max(1, round(settings["bridge_radius_px"] * max(preview.size) / max(original.size)))
    dy, dx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = dx ** 2 + dy ** 2 <= radius ** 2
    padded = np.pad(compatible, radius, mode="edge")
    closed = ndimage.binary_closing(padded, structure=disk)
    bridged = compatible | closed[radius:-radius, radius:-radius]
    native = np.asarray(Image.fromarray(bridged).resize(original.size, Image.Resampling.NEAREST)).copy()
    return building & torch.from_numpy(native)
