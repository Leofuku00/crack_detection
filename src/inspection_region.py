"""Select the main inspection surface by text, depth continuity and object boundaries."""

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from .surface_semantics import SurfaceSemantics
from .segmentation_data import IMAGENET_MEAN, IMAGENET_STD
from .wall_mask import whole_frame_inspection_scores


def normalize_rgb(image):
    pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255
    return (pixels - pixels.new_tensor(IMAGENET_MEAN)[:, None, None]) / pixels.new_tensor(IMAGENET_STD)[:, None, None]


def fuse_regions(margin, inverse_depth, surfaces, exclusions, pipes=None, *, settings=None):
    settings = settings or {}
    if pipes is not None:
        surfaces = surfaces | pipes
    seeds = ((margin > 0) | surfaces) & ~exclusions
    if not seeds.any():
        return np.zeros_like(seeds)
    near = inverse_depth >= settings.get("depth_ratio", 0.28) * np.quantile(
        inverse_depth[seeds], settings.get("depth_quantile", 0.65)
    )
    eligible = seeds & near

    # Propagate structural evidence over continuous depth surfaces. A dark crack
    # or stain can change the text score without creating a depth discontinuity.
    span = np.quantile(inverse_depth, 0.99) - np.quantile(inverse_depth, 0.01)
    variation = ndimage.maximum_filter(inverse_depth, 5) - ndimage.minimum_filter(inverse_depth, 5)
    planes, _ = ndimage.label(near & ~exclusions & (variation <= settings.get("depth_continuity", 0.06) * span))
    areas = np.bincount(planes.ravel())
    votes = np.bincount(planes[eligible], minlength=len(areas))
    accepted = votes > settings.get("surface_support", 0.2) * areas
    accepted[0] = False
    result = (eligible | accepted[planes]) & ~exclusions
    components, count = ndimage.label(ndimage.maximum_filter(result, size=9))
    if count:
        areas = np.bincount(components[result], minlength=count + 1)
        result &= components == areas[1:].argmax() + 1
    return result


def detect_components(frames, options, device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    repo, rev = options["detector_model"], options["detector_revision"]
    processor = AutoProcessor.from_pretrained(repo, revision=rev)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(repo, revision=rev).eval().to(device)
    detections = []
    for frame in frames:
        objects = []
        for group, prompts in options["detection_prompts"].items():
            labels = prompts["labels"]
            caption = ". ".join(labels) + "."
            inputs = processor(images=frame, text=caption, return_tensors="pt").to(device)
            predictions = detector(**inputs)
            result = processor.post_process_grounded_object_detection(
                predictions, inputs.input_ids, threshold=prompts["threshold"],
                text_threshold=prompts.get("text_threshold", 0.25), target_sizes=[frame.size[::-1]]
            )[0]
            for label, box, score in zip(result["text_labels"], result["boxes"].cpu().tolist(),
                                         result["scores"].cpu().tolist(), strict=True):
                # Some groups mix classes with different geometry. Require a
                # specific class before carving a hole in a surface.
                if prompts.get("exact_label") and label not in labels:
                    continue
                objects.append((group, box, score))
        detections.append(objects)
    return detections


def segment_proposals(frames, detections, options, device):
    from transformers import Sam2ImageProcessorFast, Sam2Model, Sam2Processor

    repo, rev = options["sam_model"], options["sam_revision"]
    processor = Sam2Processor(image_processor=Sam2ImageProcessorFast.from_pretrained(repo, revision=rev))
    model = Sam2Model.from_pretrained(repo, revision=rev).eval().to(device)
    for frame, objects in zip(frames, detections, strict=True):
        candidates = []
        inputs = processor(images=frame, return_tensors="pt").to(device)
        embeddings = model.get_image_embeddings(inputs.pixel_values)
        for start in range(0, len(objects), 8):
            group_boxes = objects[start:start + 8]
            boxes = [box for _, box, _ in group_boxes]
            prompts = processor(original_sizes=inputs.original_sizes, input_boxes=[boxes], return_tensors="pt")
            output = model(image_embeddings=embeddings, input_boxes=prompts.input_boxes.to(device),
                           multimask_output=False)
            masks = processor.post_process_masks(output.pred_masks, inputs.original_sizes)[0][:, 0]
            for (group, box, confidence), mask in zip(group_boxes, masks.cpu().numpy(), strict=True):
                # A box for a tiny fixture can otherwise prompt the adjacent wall.
                left, top = np.maximum(np.floor(box[:2]), 0).astype(int)
                right, bottom = np.maximum(np.ceil(box[2:]), 0).astype(int)
                area = np.s_[top:bottom, left:right]
                clipped = np.zeros_like(mask)
                clipped[area] = mask[area]
                if clipped.any():
                    candidates.append((group, confidence, area, clipped))
        yield candidates


def segment_boxes(frames, detections, options, device):
    regions = []
    proposals = segment_proposals(frames, detections, options, device)
    for frame, objects in zip(frames, proposals, strict=True):
        layers = {group: np.zeros(frame.size[::-1], dtype=bool) for group in options["detection_prompts"]}
        candidates = []
        for group, confidence, area, mask in objects:
            if group == "wires" and mask[area].mean() > options["wire_max_box_fill"]:
                continue
            priority = confidence + (options["pipe_conflict_margin"] if group == "pipes" else 0)
            candidates.append((priority, group, mask))
        resolve_proposals(candidates, layers)
        regions.append((layers["surfaces"], layers["exclusions"] | layers["vents"] | layers["wires"],
                        layers["pipes"]))
    return regions


def resolve_proposals(candidates, layers):
    foreground = {"surfaces", "pipes"}
    accepted = []
    for _, group, mask in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(
            (group in foreground) != (other_group in foreground)
            and (mask & other).sum() > 0.7 * (mask | other).sum()
            for other_group, other in accepted
        ):
            continue
        layers[group] |= mask
        accepted.append((group, mask))


@torch.inference_mode()
def inspection_regions(originals, options, device):
    from transformers import pipeline

    frames = []
    for original in originals:
        frame = original.copy()
        frame.thumbnail((options["long_edge"], options["long_edge"]), Image.Resampling.LANCZOS)
        frames.append(frame)
    depth = pipeline("depth-estimation", model=options["depth_model"], revision=options["depth_revision"],
                     use_fast=False, device=device)
    depths = [depth(frame)["predicted_depth"].squeeze().cpu().numpy() for frame in frames]
    del depth
    boxes = detect_components(frames, options, device)
    layers = segment_boxes(frames, boxes, options, device)
    semantic = SurfaceSemantics(prompt=options["prompt"], background_prompt=options["background_prompt"], device=device)
    regions = []
    for original, frame, depth_map, (surfaces, exclusions, pipes) in zip(originals, frames, depths, layers, strict=True):
        scores, _ = whole_frame_inspection_scores(semantic, normalize_rgb(frame), options["long_edge"])
        region = fuse_regions(scores.numpy(), depth_map, surfaces, exclusions, pipes, settings=options)
        native = Image.fromarray(region).resize(original.size, Image.Resampling.NEAREST)
        regions.append(torch.from_numpy(np.array(native)))
    return regions
