"""Refine a frozen patch-level crack prior using native-pixel luminance detail."""

import torch
from torch import nn
from torch.nn import functional as F

from .dinov3_common import FrozenDINO


class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, features, output_size):
        return F.interpolate(
            self.layers(features), size=output_size, mode="bilinear", align_corners=False
        )[:, 0]


class DetailHead(nn.Module):
    def __init__(self, coarse, channels):
        super().__init__()
        self.coarse = coarse.requires_grad_(False)
        self.detail = nn.Sequential(
            nn.Conv2d(5, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, dilation=2, padding=2), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, dilation=4, padding=4), nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, gray, prior):
        detail = [gray]
        for window in (3, 7, 15):
            radius = window // 2
            padded = F.pad(gray, (radius,) * 4, mode="replicate")
            detail.append((F.avg_pool2d(padded, window, stride=1) - gray) * 8)
        prior = prior.clamp(-12, 12)
        delta = self.detail(torch.cat([*detail, prior / 4], dim=1))
        return (0.35 * prior + delta)[:, 0]


class DINOv3DetailDecoder(FrozenDINO):
    def __init__(self, *, coarse_checkpoint=None, detail_channels=16, crop_size=128,
                 positive_crop_fraction=0.5, positive_weight=10.0, ridge_recovery=None,
                 trained_detail_checkpoint=None, threshold=0.5, hidden_dim=64, epochs=30,
                 batch_size=2, lr=0.001, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.coarse_checkpoint = coarse_checkpoint
        self.crop_size = crop_size
        self.positive_crop_fraction = positive_crop_fraction
        self.positive_weight = positive_weight
        self.ridge_recovery = ridge_recovery
        self.trained_detail_checkpoint = trained_detail_checkpoint
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.random_seed)
            coarse = SimpleDecoder(self.backbone.embed_dim, hidden_dim)
            torch.random.default_generator.manual_seed(self.random_seed)
            self.decoder = DetailHead(coarse, detail_channels).to(self.device)

    @torch.no_grad()
    def inputs(self, images):
        pixels = images.to(self.device)
        # Undo channel normalization before taking linear RGB luminance.
        gray = (pixels * pixels.new_tensor([0.229*0.299, 0.224*0.587, 0.225*0.114])[None,:,None,None]).sum(1, keepdim=True)
        gray = gray * 4
        prior = self.decoder.coarse(self.patch_features(images), images.shape[-2:])[:, None]
        return gray, prior

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return self.fit_batches(
            (image[None], mask[None]) for image, mask in zip(X_train, y_train, strict=True)
        )

    def fit_batches(self, batches):
        if self.trained_detail_checkpoint:
            state = torch.load(self.trained_detail_checkpoint, map_location="cpu", weights_only=True)
            self.decoder.load_state_dict(state["decoder"])
            self.training_mask_counts = {"reused_decoder": self.trained_detail_checkpoint}
            self.decoder.eval()
            return self
        if self.coarse_checkpoint:
            state = torch.load(self.coarse_checkpoint, map_location="cpu", weights_only=True)
            self.decoder.coarse.load_state_dict(state["decoder"])
        samples = []
        positives = negatives = 0
        for images, targets in batches:
            gray, prior = self.inputs(images)
            inputs = torch.cat((gray, prior), dim=1).half().cpu()
            for pair, target in zip(inputs, targets.cpu(), strict=True):
                coordinates = torch.nonzero(target == 1)
                positives += len(coordinates)
                negatives += int((target == 0).sum())
                samples.append((pair, target, coordinates))
        self.training_mask_counts = {"positive_pixels": positives, "negative_pixels": negatives,
                                     "pos_weight": self.positive_weight}
        rng = torch.Generator().manual_seed(self.random_seed)
        optimizer = torch.optim.AdamW(self.decoder.detail.parameters(), lr=self.lr)
        weight = torch.tensor(self.positive_weight, device=self.device)
        self.decoder.train()
        for _ in range(self.epochs):
            for indices in torch.randperm(len(samples), generator=rng).split(self.batch_size):
                crops, labels = [], []
                for index in indices:
                    pair, target, coordinates = samples[index]
                    h, w = target.shape
                    size = min(self.crop_size, h, w)
                    if len(coordinates) and torch.rand((), generator=rng) < self.positive_crop_fraction:
                        point = coordinates[torch.randint(len(coordinates), (), generator=rng)]
                        y, x = (point - torch.randint(size, (2,), generator=rng)).tolist()
                        y, x = max(0,min(y,h-size)), max(0,min(x,w-size))
                    else:
                        y = int(torch.randint(h-size+1, (), generator=rng))
                        x = int(torch.randint(w-size+1, (), generator=rng))
                    crops.append(pair[:, y:y+size, x:x+size])
                    labels.append(target[y:y+size, x:x+size])
                pair = torch.stack(crops).float().to(self.device)
                target = torch.stack(labels).to(self.device)
                logits = self.decoder(pair[:, :1], pair[:, 1:])
                valid, foreground = target != 2, (target == 1).float()
                bce = F.binary_cross_entropy_with_logits(logits, foreground, pos_weight=weight, reduction="none")
                probability = logits.sigmoid() * valid
                dice = (2 * (probability * foreground).sum() + 1) / (probability.sum() + foreground.sum() + 1)
                loss = (bce * valid).sum() / valid.sum().clamp_min(1) + 1 - dice
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.decoder.eval()
        return self

    @torch.no_grad()
    def score_map(self, images):
        gray, prior = self.inputs(images)
        learned = self.decoder(gray, prior).sigmoid()
        if not self.ridge_recovery:
            return learned
        return torch.maximum(learned, ridge_probability(gray, prior, self.ridge_recovery)[:, 0])


def ridge_probability(gray, prior, settings):
    """Require two brighter flanks and nearby semantic support to recover faint fissures."""
    smooth = F.avg_pool2d(F.pad(gray, (1,) * 4, mode="replicate"), 3, stride=1)
    radii = settings["radii"]
    pad = max(radii)
    padded = F.pad(smooth, (pad,) * 4, mode="replicate")
    h, w = gray.shape[-2:]
    strength = torch.zeros_like(gray)
    for radius in radii:
        for dy, dx in ((0, radius), (radius, 0), (radius, radius), (radius, -radius)):
            left = padded[:, :, pad-dy:pad-dy+h, pad-dx:pad-dx+w]
            right = padded[:, :, pad+dy:pad+dy+h, pad+dx:pad+dx+w]
            response = torch.minimum(left, right) - smooth
            if settings.get("coherent_flanks"):
                # Different brightness on opposite sides often indicates a
                # surface boundary. Require support along the axis of the line.
                response = response - 0.5 * (left - right).abs()
                step = min(radius, 4)
                sy = step if dx > 0 else -step if dx < 0 else 0
                sx = -step if dy else 0
                extended = F.pad(response, (step,) * 4, mode="replicate")
                before = extended[:, :, step-sy:step-sy+h, step-sx:step-sx+w]
                after = extended[:, :, step+sy:step+sy+h, step+sx:step+sx+w]
                response = torch.minimum(response, 0.5 * (before + after))
            strength = torch.maximum(strength, response)
    window = 2 * pad + 1
    # Use local RMS contrast so textured surfaces require stronger evidence.
    mean = F.avg_pool2d(padded, window, stride=1)
    variance = F.avg_pool2d(padded.square(), window, stride=1) - mean.square()
    threshold = settings["min_contrast"] + settings["relative_contrast"] * variance.clamp_min(0).sqrt()
    radius = settings["context_radius"]
    near = F.max_pool2d(prior, (1, 2*radius+1), stride=1, padding=(0, radius))
    near = F.max_pool2d(near, (2*radius+1, 1), stride=1, padding=(radius, 0))
    evidence = (strength - threshold) / threshold
    probability = torch.minimum(evidence, near - settings["min_prior_logit"]).sigmoid()
    if "seam" in settings:
        seam = settings["seam"]
        horizontal = long_joint(gray, seam)
        vertical = long_joint(gray.transpose(2, 3), seam).transpose(2, 3)
        probability = probability.masked_fill(horizontal | vertical, 0)
    return probability


def long_joint(gray, settings):
    """Locate persistent, nearly straight intensity transitions along a surface."""
    h, w = gray.shape[-2:]
    span = settings["sample_spacing"]
    if w < 4 * span or h < 17:
        return torch.zeros_like(gray, dtype=torch.bool)
    strip = F.avg_pool2d(F.pad(gray, (12, 12, 0, 0), mode="replicate"), (1, 25), stride=1)
    padded = F.pad(strip, (0, 0, 8, 8), mode="replicate")
    upper, lower = padded[:, :, :h], padded[:, :, 16:]
    step = upper - lower
    groove = torch.minimum(upper, lower) - strip
    # Keep the two step polarities and a dark groove as distinct hypotheses.
    # Combining them before measuring continuity links unrelated small edges.
    edge = torch.cat((step, -step, groove), dim=1)
    # Require evidence on both sides of the candidate over a long tangent.
    # Replication at tile borders retains junctions straddling adjacent tiles.
    tolerant = F.max_pool2d(edge, (9, 1), stride=1, padding=(4, 0))
    extended = F.pad(tolerant, (2*span, 2*span, 0, 0), mode="replicate")
    support = tolerant
    for offset in (0, span, 3*span, 4*span):
        support = torch.minimum(support, extended[:, :, :, offset:offset+w])
    support = support.amax(dim=1, keepdim=True)
    return F.max_pool2d(support, (9, 1), stride=1, padding=(4, 0)) > settings["min_contrast"]
