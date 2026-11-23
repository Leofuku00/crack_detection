# Painted-wall crack detection

The pipeline identifies an eligible painted-wall region, uses a DINOv3 detail decoder to score cracks in the original RGB image at native resolution, then filters detections near fixtures and surface edges. The trained detail decoder and its coarse crack prior are included in `weights/detail.pt`.

## Reproduce inference

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Run these commands from the repository root.

```bash
uv sync --locked
cp model.env.example model.env
```

Download the DINOv3 **ViT-B/16 LVD-1689M** checkpoint from [Meta's DINOv3 downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). To generate the building mask automatically, also download the **ViT-L/16 LVD-1689M** checkpoint, **dino.txt ViT-L/16** head and text encoder, and [tokenizer vocabulary](https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz). Save them at the paths in `model.env` or edit those paths. The code pins the DINOv3 source revision. It downloads the pinned Grounding DINO, SAM 2.1, and Depth Anything V2 models as needed.

```bash
uv run --env-file model.env python scripts/predict_inspection_cracks.py \
  --image /path/to/wall.png \
  --output results/example
```

Pass `--region-mask /path/to/building_mask.png` to use a same-size, single-channel binary building mask. This skips automatic region generation and requires only the ViT-B/16 checkpoint from Meta. The output directories contain full-resolution masks and overlays for the eligible region (green) and final crack predictions (red).

## CCSS training data and evaluation

Download the [CCSS training dataset on Kaggle](https://www.kaggle.com/datasets/parniashokri/ccssdata). The training split expected by the experiment runner is `CCSS-DATA-V4-withVal/benchmarkingDatasets/train/noAugmentation-noPatch`, with `img/` and `lbl/` subdirectories. Extract it under `data/ccss/` or update `ccss_train_dir` in `configs/dinov3_detail_decoder.yaml`. Inference only needs the model checkpoints and an input image.

```bash
uv run --locked pytest
uv run --env-file model.env python scripts/run_experiment.py
```

The experiment runner reuses the included decoder by default. To repeat the original wall-image evaluation, also supply the wall images and labels, reference building masks, and hard-negative pairs at the paths in the YAML. Those project-specific inputs are separate from CCSS.
