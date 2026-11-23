"""Run a saved inspection decoder on an unlabeled image."""
import argparse
from pathlib import Path
import sys

import yaml
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inspection_cracks import predict_frames, read_region
from src.models import MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--region-mask", type=Path, help="Reuse an existing full-resolution building mask.")
    args = parser.parse_args()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
        config = saved["config"]
    else:
        config = yaml.safe_load(Path("configs/dinov3_detail_decoder.yaml").read_text())
        saved = torch.load(config["params"]["trained_detail_checkpoint"], weights_only=True, map_location="cpu")
    model = MODEL_REGISTRY[config["model_name"]](random_seed=config["random_seed"], **config["params"])
    model.decoder.load_state_dict(saved["decoder"])
    with Image.open(args.image) as source:
        regions = [read_region(args.region_mask, source.size)] if args.region_mask else None
        predict_frames(model, [source.convert("RGB")], [args.image.stem], config, args.output, regions=regions)


if __name__ == "__main__":
    main()
