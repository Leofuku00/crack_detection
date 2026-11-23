from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image, ImageDraw

from src import inspection_cracks
from src.inspection_cracks import read_region


@pytest.mark.parametrize("use_saved_region", [False, True])
def test_multicolor_walls_survive_region_selection(tmp_path, monkeypatch, use_saved_region):
    image = Image.new("RGB", (96, 64), "#ddd1b7")
    draw = ImageDraw.Draw(image)
    for start, color in ((24, "#a92426"), (48, "#914725"), (72, "#284b6f")):
        draw.rectangle((start, 0, start + 23, 63), fill=color)
    draw.line((0, 32, 95, 32), fill="#25120f", width=2)

    building = torch.ones((64, 96), dtype=torch.bool)
    building[:8] = False
    fixture = torch.zeros_like(building)
    fixture[28:36, 52:56] = True
    expected = building & ~fixture
    config_path = Path(__file__).resolve().parents[1] / "configs" / "dinov3_detail_decoder.yaml"
    config = yaml.safe_load(config_path.read_text())
    monkeypatch.setattr(inspection_cracks, "fixture_masks", lambda *args: [fixture])
    monkeypatch.setattr(inspection_cracks, "inspection_regions", lambda *args: [building])

    class Model:
        threshold = 0.5
        device = "cpu"
        backbone = torch.nn.Identity()
        decoder = torch.nn.Identity()

        def score_map(self, tiles):
            scores = torch.zeros_like(tiles[:, 0])
            scores[:, 32:34, :96] = 1.0
            return scores

    inspection_cracks.predict_frames(
        Model(), [image], ["wall"], config, tmp_path,
        regions=[building] if use_saved_region else None,
    )

    assert torch.equal(read_region(tmp_path / "region_masks/wall.png", image.size), expected)
    cracks = read_region(tmp_path / "crack_masks/wall.png", image.size)
    assert not (cracks & ~expected).any()
    for columns in (slice(8, 16), slice(32, 40), slice(60, 68), slice(80, 88)):
        assert cracks[32:34, columns].all()


def test_saved_building_mask_can_be_loaded_as_an_input_mask(tmp_path):
    region = torch.tensor([[False, True], [True, False]])
    path = tmp_path / "region.png"
    Image.fromarray(region.numpy()).save(path)
    assert torch.equal(read_region(path, (2, 2)), region)
