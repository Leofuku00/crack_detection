"""Paired image/mask loading and the fixed segmentation benchmark split."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def by_stem(directory, suffixes):
    paths = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in paths:
                raise ValueError(f"Duplicate sample stem {path.stem!r} in {directory}")
            paths[path.stem] = path
    return paths


class CrackDataset(Dataset):
    def __init__(
        self, images_dir, masks_dir, image_size, patch_size, *,
        bank_masks=False, expected_count=9, jpeg_masks=False
    ):
        images = by_stem(images_dir, {".png", ".jpg", ".jpeg"})
        mask_suffixes = {".jpg", ".jpeg"} if jpeg_masks else {".png", ".bmp", ".tif", ".tiff"}
        masks = by_stem(masks_dir, mask_suffixes)
        if images.keys() != masks.keys() or (expected_count is not None and len(images) != expected_count):
            expected = f" {expected_count}" if expected_count is not None else ""
            raise ValueError(f"Expected{expected} image/mask pairs with matching stems.")
        self.size = None
        if image_size is not None:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            height, width = image_size
            if min(height, width) < 1:
                raise ValueError("Image dimensions must be positive.")
            self.size = tuple(math.ceil(side / patch_size) * patch_size for side in (height, width))
        self.samples = [(images[name], masks[name]) for name in images]
        self.bank_masks = bank_masks
        self.jpeg_masks = jpeg_masks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
            if image.size != mask_image.size:
                raise ValueError(f"Image and mask dimensions differ for {image_path.stem}")
            raw = image.convert("RGB")
            if self.size is not None:
                height, width = self.size
                raw = raw.resize((width, height), Image.Resampling.BILINEAR)
            raw_array = np.array(raw, dtype=np.uint8)
            if self.jpeg_masks:
                # CCSS stores masks as lossy JPEGs. Use the same cutoff as SAC.
                foreground = np.asarray(mask_image.convert("L")) >= 128
            else:
                mask_array = np.asarray(mask_image)
                if mask_array.ndim != 2 or not np.isin(mask_array, (0, 1, 255)).all():
                    raise ValueError(f"Expected a single-channel 0/1 or 0/255 mask: {mask_path}")
                foreground = mask_array != 0
        pixels = torch.from_numpy(raw_array).permute(2, 0, 1).float() / 255
        mean = pixels.new_tensor(IMAGENET_MEAN)[:, None, None]
        std = pixels.new_tensor(IMAGENET_STD)[:, None, None]

        if self.size is None:
            return (pixels - mean) / std, torch.from_numpy(foreground).byte(), image_path.stem

        # Max-pool only when reducing a dimension so thin lines survive downsampling.
        mask = torch.from_numpy(foreground)[None, None].float()
        reduced_size = (min(mask.shape[-2], height), min(mask.shape[-1], width))
        if mask.shape[-2:] != reduced_size:
            mask = F.adaptive_max_pool2d(mask, reduced_size)
        if mask.shape[-2:] != self.size:
            mask = F.interpolate(mask, size=self.size, mode="nearest-exact")
        mask = mask[0, 0].bool()
        if self.bank_masks:
            # Bilinear resizing can carry a defect pixel into a neighboring patch.
            # Apply the same kernel to the foreground indicator in float mode so
            # even a contribution smaller than one uint8 level is detected.
            coverage = Image.fromarray(foreground.astype(np.float32)).resize(
                (width, height), Image.Resampling.BILINEAR
            )
            mask |= torch.from_numpy(np.asarray(coverage) > 0)
        return (pixels - mean) / std, mask.to(torch.uint8), image_path.stem


def segmentation_split(dataset):
    train, val = train_test_split(range(len(dataset)), test_size=3, random_state=42)
    return Subset(dataset, train), Subset(dataset, val)
