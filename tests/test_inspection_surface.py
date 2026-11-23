import torch
from PIL import Image, ImageDraw

from src.inspection_cracks import inspection_input_region, read_region


def test_wall_color_uses_surrounding_surface_and_excludes_foreign_objects():
    image = Image.new("RGB", (180, 128), "#c2ad80")
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 25, 39, 100), fill="#1e1525")
    draw.rectangle((95, 17, 155, 108), fill="#7a8a9c")
    draw.rectangle((5, 48, 20, 80), fill="#bca77e")
    building = torch.ones((128, 180), dtype=torch.bool)
    building[:, :5] = False
    fixture = torch.zeros_like(building)
    fixture[48:81, 5:21] = True
    config = {"wall_surface": {
        "long_edge": 960, "chroma_radius": 16, "lightness_margin": 28, "bridge_radius_px": 10,
    }}

    region = inspection_input_region(image, building, config, fixture)

    assert region[30:95, 35:40].all()
    assert not region[28:95, 105:145].any()
    assert not region[48:81, 5:21].any()
    assert not region[:, :5].any()
    assert region[30:95, 45:85].all()


def test_saved_video_mask_can_be_loaded_as_an_input_mask(tmp_path):
    region = torch.tensor([[False, True], [True, False]])
    path = tmp_path / "region.png"
    Image.fromarray(region.numpy()).save(path)
    assert torch.equal(read_region(path, (2, 2)), region)
