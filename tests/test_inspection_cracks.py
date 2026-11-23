import numpy as np
import torch

from src.inspection_cracks import masked_cracks
from src.inspection_region import fuse_regions


def test_depth_and_nested_objects_preserve_damage_and_exclude_fixture():
    margin = np.ones((20, 40))
    margin[6, 6] = -1  # dark crack inside the wall proposal
    wall = np.ones_like(margin, dtype=bool)
    fixture = np.zeros_like(wall)
    fixture[8:12, 10:14] = True
    depth = np.full_like(margin, 4.0)
    depth[:, 30:] = 0.2  # equally wall-like distant building
    region = fuse_regions(margin, depth, wall, fixture)
    assert region[6, 6]
    assert not region[fixture].any()
    assert not region[:, 30:].any()


def test_masked_cracks_requires_region_and_positive_decoder_score():
    scores = torch.tensor([[0.99, 0.51, 0.50, 0.01]])
    region = torch.tensor([[False, True, True, True]])
    assert masked_cracks(scores, region, 0.5).tolist() == [[False, True, False, False]]


def test_component_filter_keeps_thin_connected_pixels_after_masking():
    scores = torch.tensor([[0.9, 0.0, 0.0, 0.9], [0.0, 0.9, 0.0, 0.9], [0.0, 0.0, 0.9, 0.0]])
    region = torch.ones_like(scores, dtype=torch.bool)
    region[1, 3] = False
    prediction = masked_cracks(scores, region, 0.5, min_component_pixels=3)
    assert torch.equal(prediction, torch.eye(3, 4, dtype=torch.bool))


def test_detector_requires_an_identifiable_exclusion(monkeypatch):
    from types import SimpleNamespace
    import sys
    from PIL import Image
    from src.inspection_region import detect_components

    class Inputs(dict):
        input_ids = torch.tensor([[0]])

        def to(self, device):
            return self

    class Processor:
        def __call__(self, images, text, return_tensors):
            self.caption = text
            return Inputs()

        def post_process_grounded_object_detection(self, *args, **kwargs):
            labels = ["wall"] if self.caption == "wall." else ["floor electrical cabinet", "door"]
            return [{"text_labels": labels, "boxes": torch.ones(len(labels), 4),
                     "scores": torch.full((len(labels),), 0.5)}]

    class Detector(torch.nn.Module):
        def forward(self):
            return {}

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Processor()),
        AutoModelForZeroShotObjectDetection=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Detector()),
    ))
    settings = {"detector_model": "mock", "detector_revision": "mock", "detection_prompts": {
        "surfaces": {"labels": ["wall"], "threshold": 0.28},
        "exclusions": {"labels": ["floor", "electrical cabinet", "door"],
                       "threshold": 0.28, "exact_label": True},
    }}
    objects, = detect_components([Image.new("RGB", (16, 16))], settings, "cpu")
    assert [group for group, box, score in objects] == ["surfaces", "exclusions"]
