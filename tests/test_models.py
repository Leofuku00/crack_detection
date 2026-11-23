from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from src.models import MODEL_REGISTRY
from src.segmentation_data import CrackDataset
from src.segmentation_experiment import tile_batches, tiled_scores

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


class MockBackbone(nn.Module):
    patch_size = 4
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward_features(self, images):
        batch, _, height, width = images.shape
        generator = torch.Generator().manual_seed(19)
        tokens = torch.rand(
            batch, (height // self.patch_size) * (width // self.patch_size),
            self.embed_dim, generator=generator
        )
        return {"x_norm_patchtokens": tokens * self.scale}


class MockOptimizer:
    """Run a gradient update without PyTorch's optimizer compiler startup."""

    def __init__(self, parameters, lr):
        self.parameters = tuple(parameters)
        self.lr = lr

    def zero_grad(self, *, set_to_none):
        assert set_to_none
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self):
        for parameter in self.parameters:
            parameter.add_(parameter.grad, alpha=-self.lr)


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_detail_decoder_fit_and_predict(monkeypatch):
    config = yaml.safe_load((CONFIGS_DIR / "dinov3_detail_decoder.yaml").read_text())
    assert list(MODEL_REGISTRY) == [config["model_name"]]
    params = dict(config["params"], epochs=1, trained_detail_checkpoint=None, device="cpu", backbone=MockBackbone())
    monkeypatch.setattr(torch.optim, "AdamW", MockOptimizer)
    model = MODEL_REGISTRY[config["model_name"]](**params)
    x = torch.zeros(2, 3, 32, 48)
    target = torch.zeros(2, 32, 48)
    target[:, 12, 10] = 1
    model.fit(x, target)
    predicted = model.predict(x)
    assert predicted.shape == target.shape
    assert set(predicted.unique().tolist()) <= {0, 1}


def test_native_tiles_keep_pixel_coordinates_and_exclude_padding():
    class IdentityScores:
        def score_map(self, images):
            return images[:, 0]

    pixels = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    image = pixels.expand(3, -1, -1)
    target = torch.zeros_like(pixels, dtype=torch.uint8)
    target[-1, -1] = 1
    _, masks, _ = list(tile_batches(image, target, tile_size=4, batch_size=2))[-1]
    assert masks[-1, 0, 2] == 1
    assert masks[-1, 1:, :].eq(2).all()
    assert masks[-1, :, 3].eq(2).all()
    torch.testing.assert_close(
        tiled_scores(IdentityScores(), image, target, tile_size=4, batch_size=2), pixels
    )


def test_ccss_jpeg_mask_threshold(tmp_path):
    images, masks = tmp_path / "img", tmp_path / "lbl"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (16, 8)).save(images / "sample.jpg")
    label = Image.new("L", (16, 8), 64)
    label.paste(192, (8, 0, 16, 8))
    label.save(masks / "sample.jpg", quality=100)

    _, mask, _ = CrackDataset(
        images, masks, None, 8, expected_count=1, jpeg_masks=True
    )[0]
    assert not mask[:, :8].any()
    assert mask[:, 8:].all()


def test_ridge_recovery_requires_two_flanks_and_nearby_crack_evidence():
    from src.models.dinov3_detail_decoder import ridge_probability

    gray = torch.ones(1, 1, 24, 48)
    gray[:, :, :, :12] = 0                         # One-sided intensity step.
    gray[:, :, :, 20:23] = 0.9                     # Faint fissure.
    gray[:, :, :, 38:41] = 0.9                     # Outside semantic support.
    prior = torch.zeros_like(gray)
    prior[:, :, :, 34:] = -10
    settings = {"radii": [2, 4], "min_contrast": 0.022, "relative_contrast": 0.35,
                "context_radius": 1, "min_prior_logit": -1.5}
    recovered = ridge_probability(gray, prior, settings) > 0.5
    assert not recovered[:, :, :, :12].any()
    assert recovered[:, :, :, 21].all()
    assert not recovered[:, :, :, 39].any()


def test_seam_detector_requires_a_persistent_transition():
    from src.models.dinov3_detail_decoder import long_joint

    gray = torch.ones(1, 1, 100, 256)
    gray[:, :, 24:27, :] = 0.7
    gray[:, :, 75:78, 110:140] = 0.7
    veto = long_joint(gray, {"sample_spacing": 32, "min_contrast": 0.085})
    assert veto[0, 0, 25, 128]
    assert not veto[0, 0, 76, 125]
