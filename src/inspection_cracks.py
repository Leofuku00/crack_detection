"""Native-resolution crack inference restricted to the main inspection surface."""


import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label

from .inspection_edges import suppress_step_edges
from .inspection_fixtures import fixture_masks
from .inspection_region import inspection_regions, normalize_rgb
from .segmentation_experiment import save_comparison, segmentation_metrics, tiled_scores


def masked_cracks(scores, region, threshold, min_component_pixels=0):
    prediction = region & (scores > threshold)
    if min_component_pixels > 1:
        components, count = label(prediction.numpy(), structure=np.ones((3, 3), dtype=bool))
        keep = np.bincount(components.ravel(), minlength=count + 1) >= min_component_pixels
        keep[0] = False
        prediction = torch.from_numpy(keep[components])
    return prediction


def read_region(path, size):
    with Image.open(path) as mask:
        pixels = np.array(mask)
        binary = pixels.dtype == np.bool_ or np.isin(pixels, (0, 255)).all()
        if mask.size != size or pixels.ndim != 2 or not binary:
            raise ValueError(f"Expected a matching single-channel binary region mask: {path}")
    return torch.from_numpy(pixels != 0)


def inspection_prediction(scores, region, threshold, config, original):
    prediction = masked_cracks(scores, region, threshold, config.get("inspection_min_component_pixels", 0))
    if "edge_filter" in config:
        prediction = suppress_step_edges(prediction, original, config["edge_filter"])
    return prediction


def predict_frames(model, originals, names, config, output, targets=None, regions=None):
    if regions is None:
        model.backbone.cpu()
        model.decoder.cpu()
        regions = inspection_regions(originals, config["inspection"], model.device)
        model.backbone.to(model.device)
        model.decoder.to(model.device)
    fixtures = fixture_masks(originals, config, model.device) if "fixture_filter" in config else [None] * len(names)
    for directory in ("region_masks", "region_overlays", "crack_masks", "crack_overlays"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    if targets is not None:
        (output / "unmasked_crack_masks").mkdir(exist_ok=True)
    records = []
    for index, (original, name, region, fixture) in enumerate(zip(originals, names, regions, fixtures, strict=True)):
        if fixture is not None:
            region = region & ~fixture
        image = normalize_rgb(original)
        scores = tiled_scores(model, image, region.byte(), config["tile_size"], config["tile_batch_size"])
        prediction = inspection_prediction(scores, region, model.threshold, config, original)
        for prefix, mask, color in (("region", region, "lime"), ("crack", prediction, "red")):
            binary = Image.fromarray(mask.byte().numpy() * 255)
            binary.save(output / f"{prefix}_masks" / f"{name}.png")
            tint = Image.blend(original, Image.new("RGB", original.size, color), 0.55)
            Image.composite(tint, original, binary).save(output / f"{prefix}_overlays" / f"{name}.png")
        if targets is not None:
            target = targets[index].bool()
            raw = scores > model.threshold
            Image.fromarray(raw.byte().numpy() * 255).save(output / "unmasked_crack_masks" / f"{name}.png")
            raw_metrics = segmentation_metrics(raw, target)
            final_metrics = segmentation_metrics(prediction, target)
            records.append({
                "image": name, **{k: v.item() for k, v in final_metrics.items()},
                "unmasked": {k: v.item() for k, v in raw_metrics.items()},
                "labeled_crack_pixels": int(target.sum()),
                "labeled_crack_pixels_excluded": int((target & ~region).sum()),
                "true_positive_pixels_removed": int((raw & target & ~region).sum()),
                "labeled_crack_pixels_missed_inside_region": int((target & region & ~prediction).sum()),
                "false_positive_pixels_removed": int((raw & ~target & ~region).sum()),
                "predicted_pixels_outside_region": int((prediction & ~region).sum()),
            })
            save_comparison(output / f"{name}.png", image, target, scores, prediction)
    return records


def run_inspection(config, model, val, output, training, project_root):
    output.mkdir(parents=True)
    torch.save({"backbone": "dinov3_vitb16", "config": config,
                "decoder": model.decoder.state_dict()}, output / "decoder.pt")
    originals, names, targets = [], [], []
    for index in val.indices:
        path, _ = val.dataset.samples[index]
        with Image.open(path) as source:
            originals.append(source.convert("RGB"))
        _, target, name = val.dataset[index]
        targets.append(target)
        names.append(name)
    regions = None
    if "reference_run" in config:
        folder = project_root / config["reference_run"] / "region_masks"
        regions = [read_region(folder / f"{name}.png", image.size)
                   for name, image in zip(names, originals, strict=True)]
    records = predict_frames(model, originals, names, config, output, targets, regions)
    metrics = {
        **{k: sum(row[k] for row in records) / len(records) for k in ("iou", "dice")},
        "unmasked": {k: sum(row["unmasked"][k] for row in records) / len(records) for k in ("iou", "dice")},
        "pixel_totals": {
            key: sum(row[key] for row in records)
            for key in ("labeled_crack_pixels", "labeled_crack_pixels_excluded", "true_positive_pixels_removed",
                        "labeled_crack_pixels_missed_inside_region", "false_positive_pixels_removed",
                        "predicted_pixels_outside_region")
        },
        "training": training, "per_image": records,
    }
    return metrics
