"""Fit on wall or CCSS data and visualize the three held-out wall images."""

import re
from itertools import batched, chain
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .segmentation_data import CrackDataset, IMAGENET_MEAN, IMAGENET_STD, segmentation_split


def segmentation_metrics(prediction, target):
    """Compute per-image foreground IoU and Dice, with 1 for two empty masks."""
    prediction, target = prediction.bool(), target.bool()
    intersection = (prediction & target).sum(dim=(-2, -1))
    union = (prediction | target).sum(dim=(-2, -1))
    total = prediction.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return {
        "iou": torch.where(union == 0, 1.0, intersection / union),
        "dice": torch.where(total == 0, 1.0, 2 * intersection / total),
    }


def source_image(normalized):
    mean = normalized.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = normalized.new_tensor(IMAGENET_STD)[:, None, None]
    rgb = ((normalized * std + mean).clamp(0, 1) * 255).round().byte()
    return Image.fromarray(rgb.permute(1, 2, 0).numpy())


def save_comparison(path, normalized, target, scores, prediction, *, ignored=None):
    raw = source_image(normalized)
    displayed_target = target.byte() * 255
    if ignored is not None:
        displayed_target[ignored] = 64
    gt = Image.fromarray(displayed_target.numpy())
    mask = Image.fromarray(prediction.byte().numpy() * 255)
    # Rescale for display only. Predictions use the original score units.
    score_range = (scores.max() - scores.min()).clamp_min(1e-8)
    heat = ((scores - scores.min()) / score_range).numpy()
    colors = np.stack((heat, np.sqrt(heat) * 0.6, 1 - heat), axis=-1)
    score = Image.fromarray((colors * 255).astype(np.uint8))

    width, height = raw.size
    canvas = Image.new("RGB", (4 * width, height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ("Raw", "Extracted Line GT", "Score Map", "Binary Prediction")
    if ignored is not None:
        titles = ("Raw", "Reviewed centerline (gray = unscored)", "Score Map", "Binary Prediction")
    for index, (title, panel) in enumerate(zip(titles, (raw, gt, score, mask), strict=True)):
        x = index * width
        draw.text((x + 4, 5), title, fill="black")
        canvas.paste(panel, (x, 24))
    canvas.save(path)


def tile_batches(image, target, tile_size, batch_size):
    """Extract native pixels and pad only the bottom and right edge tiles."""
    height, width = target.shape
    origins = (
        (top, left)
        for top in range(0, height, tile_size)
        for left in range(0, width, tile_size)
    )
    for batch in batched(origins, batch_size):
        images, targets, regions = [], [], []
        for top, left in batch:
            rows = slice(top, min(top + tile_size, height))
            cols = slice(left, min(left + tile_size, width))
            region = target[rows, cols]
            h, w = region.shape
            padding = (0, tile_size - w, 0, tile_size - h)
            # Zero is the ImageNet mean in normalized image space. Mask value
            # 2 marks padding so it is ignored in both the bank and the loss.
            images.append(F.pad(image[:, rows, cols], padding))
            targets.append(F.pad(region, padding, value=2))
            regions.append((rows, cols, h, w))
        yield torch.stack(images), torch.stack(targets), regions


def training_tile_batches(dataset, tile_size, batch_size):
    for image, target, _ in DataLoader(dataset, batch_size=None):
        for images, targets, _ in tile_batches(image, target, tile_size, batch_size):
            yield images, targets


def ccss_training_batches(dataset, batch_size):
    for images, targets, _ in DataLoader(dataset, batch_size=batch_size):
        yield images, targets


def source_video(stem):
    frame = re.fullmatch(r"動画像([1-9][0-9]*)-[0-9]{3}", stem)
    crop = re.fullmatch(r"v([1-9][0-9]*)-[0-9]{3}-.+", stem)
    match = frame or crop
    if match is None:
        raise ValueError(f"Cannot identify the source video for {stem!r}.")
    return int(match[1])


def select_hard_negatives(directory, validation, patch_size):
    held_out_videos = {
        source_video(validation.dataset.samples[index][0].stem) for index in validation.indices
    }
    negatives = CrackDataset(
        directory / "images", directory / "masks", None, patch_size, expected_count=None
    )
    indices = []
    for index, (image_path, mask_path) in enumerate(negatives.samples):
        if source_video(image_path.stem) in held_out_videos:
            continue
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f"Hard-negative image and mask dimensions differ for {image_path.name}.")
            pixels = np.asarray(mask)
            if pixels.ndim != 2 or pixels.any():
                raise ValueError(f"Hard-negative mask must be single-channel and all zero: {mask_path}")
        indices.append(index)
    if not indices:
        raise ValueError("No hard negatives remain after holding out the validation source videos.")
    filenames = [negatives.samples[index][0].name for index in indices]
    return Subset(negatives, indices), filenames


def tiled_scores(model, image, target, tile_size, batch_size):
    """Stitch score maps in original image coordinates, then threshold once."""
    stitched = torch.empty(target.shape, dtype=torch.float32)
    for images, _, regions in tile_batches(image, target, tile_size, batch_size):
        for scores, (rows, cols, h, w) in zip(model.score_map(images).cpu(), regions, strict=True):
            stitched[rows, cols] = scores[:h, :w]
    return stitched


def run_segmentation(config, model, results_dir, project_root):
    from .inspection_cracks import run_inspection

    evaluation_images = config.get("evaluation_images")
    expected_count = 9 if evaluation_images is None else len(evaluation_images)
    dataset = CrackDataset(
        project_root / config["images_dir"], project_root / config["masks_dir"],
        config["image_size"], model.patch_size, expected_count=expected_count,
    )
    if evaluation_images is None:
        if config["random_seed"] != 42 or config["test_size"] != 3 or config["image_size"] is not None:
            raise ValueError("Expected the native-resolution 6/3 inspection split with seed 42.")
        _, val = segmentation_split(dataset)
    else:
        indices = {path.stem: index for index, (path, _) in enumerate(dataset.samples)}
        if not evaluation_images or len(evaluation_images) != len(indices) or set(evaluation_images) != indices.keys():
            raise ValueError("evaluation_images must name each evaluation image exactly once.")
        val = Subset(dataset, [indices[name] for name in evaluation_images])

    training = {}
    batches = ()
    if not model.trained_detail_checkpoint:
        ccss = project_root / config["ccss_train_dir"]
        train = CrackDataset(ccss / "img", ccss / "lbl", None, model.patch_size,
                             expected_count=721, jpeg_masks=True)
        negatives, filenames = select_hard_negatives(
            project_root / config["hard_negative_dir"], val, model.patch_size
        )
        batches = chain(ccss_training_batches(train, config["tile_batch_size"]),
                        training_tile_batches(negatives, config["tile_size"], config["tile_batch_size"]))
        training = {"dataset": "CCSS", "split": "train/noAugmentation-noPatch",
                    "images": len(train) + len(negatives), "hard_negatives": filenames}
    start = perf_counter()
    model.fit_batches(batches)
    elapsed = perf_counter() - start
    training.update(model.training_mask_counts)
    output = results_dir / "visualizations" / config["model_name"] / f"inspection-cracks-{uuid4().hex[:8]}"
    return run_inspection(config, model, val, output, training, project_root), elapsed, output
