import numpy as np
import torch
from PIL import Image

from src.inspection_edges import suppress_step_edges
from src.inspection_fixtures import fixture_region


def test_sustained_step_is_removed_and_dark_fissure_is_preserved():
    settings = {"span": 18, "radii": [3, 6, 12], "min_contrast": 0.15, "preserve_trough": 0.04}
    prediction = torch.zeros((96, 128), dtype=torch.bool)
    prediction[48, 24:104] = True
    boundary = np.full(prediction.shape, 175, dtype=np.uint8)
    boundary[:48] = 60
    fissure = np.full(prediction.shape, 175, dtype=np.uint8)
    fissure[47:50] = 30

    assert not suppress_step_edges(prediction, Image.fromarray(boundary), settings).any()
    assert torch.equal(suppress_step_edges(prediction, Image.fromarray(fissure), settings), prediction)


def test_broad_wire_mask_cannot_erase_wall():
    settings = {"detection_prompts": {"wire": {"max_box_area": 0.5, "max_box_fill": 0.35}}}
    thin = np.zeros((20, 20), dtype=bool)
    thin[4:12, 9] = True
    broad = thin.copy()
    broad[2:10, 13:19] = True
    candidates = [
        ("wire", 0.5, np.s_[4:12, 7:12], thin),
        ("wire", 0.5, np.s_[2:10, 13:19], broad),
    ]
    np.testing.assert_array_equal(fixture_region((20, 20), candidates, settings), thin)


def test_protected_edge_filter_keeps_wide_dark_fissure():
    settings = {
        "span": 18, "radii": [3, 6, 12], "min_contrast": 0.15, "preserve_trough": 0.04,
        "bright_side_only": True, "protection_radii": [18, 24, 36], "min_luminance": 0.30,
    }
    prediction = torch.zeros((96, 128), dtype=torch.bool)
    prediction[48, 24:104] = True
    fissure = np.full(prediction.shape, 180, dtype=np.uint8)
    fissure[37:66] = 30
    boundary = np.full(prediction.shape, 180, dtype=np.uint8)
    boundary[:48] = 30

    assert torch.equal(suppress_step_edges(prediction, Image.fromarray(fissure), settings), prediction)
    assert not suppress_step_edges(prediction, Image.fromarray(boundary), settings).any()
