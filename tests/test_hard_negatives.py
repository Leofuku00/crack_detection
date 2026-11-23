from types import SimpleNamespace

import pytest
from PIL import Image
from torch.utils.data import Subset

from src.segmentation_experiment import select_hard_negatives


@pytest.fixture
def hard_negative_data(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    wall = SimpleNamespace(samples=[
        (tmp_path / "動画像2-000.png", None),
        (tmp_path / "動画像1-028.png", None),
        (tmp_path / "動画像3-049.png", None),
    ])
    return tmp_path, Subset(wall, [1, 2])


def save_pair(directory, stem, *, mask_value=0, mask_size=(16, 16)):
    Image.new("RGB", (16, 16)).save(directory / "images" / f"{stem}.png")
    Image.new("L", mask_size, mask_value).save(directory / "masks" / f"{stem}.png")


def test_hard_negatives_hold_out_entire_validation_videos(hard_negative_data):
    directory, validation = hard_negative_data
    # A crop from a held-out frame and a distant frame from a held-out video
    # must never reach training, even if their masks are valid.
    for stem in ("v1-028-wires", "動画像3-224", "v2-000-cable", "動画像2-017"):
        save_pair(directory, stem)

    selected, filenames = select_hard_negatives(directory, validation, 16)

    assert filenames == ["v2-000-cable.png", "動画像2-017.png"]
    loaded = [selected[index][2] for index in range(len(selected))]
    assert loaded == ["v2-000-cable", "動画像2-017"]


@pytest.mark.parametrize(
    ("stem", "mask_value", "mask_size", "error"),
    [
        ("unknown", 0, (16, 16), "source video"),
        ("v2-000-cable", 1, (16, 16), "all zero"),
        ("v2-000-cable", 0, (8, 8), "dimensions differ"),
    ],
)
def test_hard_negatives_reject_unverified_pairs(hard_negative_data, stem, mask_value, mask_size, error):
    directory, validation = hard_negative_data
    save_pair(directory, stem, mask_value=mask_value, mask_size=mask_size)

    with pytest.raises(ValueError, match=error):
        select_hard_negatives(directory, validation, 16)
