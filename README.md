# Painted-wall crack detection

The pipeline identifies an eligible painted-wall region, uses a DINOv3 detail decoder to score cracks in the original RGB image at native resolution, then filters detections near fixtures and surface edges. The region is selected using surface recognition and depth without filtering by a dominant wall color. Brick, masonry, and metal siding are included in the surface prompts. Inference uses the bundled 200-epoch checkpoint at `weights/detail-200ep.pt` by default.

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

## 200-epoch results

The detail head was fine-tuned for 200 epochs from `weights/detail.pt` on 721 CCSS images and three hard-negative images. The DINOv3 backbone and coarse decoder stayed frozen. The table reports mean pixel-level metrics on the same three held-out wall images, using their saved building masks and the same inference settings.

| Checkpoint | Dice | IoU |
| --- | ---: | ---: |
| Starting checkpoint | 0.1815 | 0.1019 |
| 200-epoch checkpoint | **0.1831** | **0.1029** |

The overlays show the eligible inspection region in green and predicted crack pixels in red. Some red marks follow surface edges or fixtures.

**Held-out wall 3-058**

![Raw wall image, green region overlay, and red crack overlay for held-out wall 3-058](docs/results/wall-3-058.jpg)

**Held-out wall 1-028, ceiling detail**

![Raw ceiling crop, green region overlay, and red crack overlay for held-out wall 1-028](docs/results/wall-1-028-detail.jpg)

## CCSS training data and evaluation

Download the [CCSS training dataset on Kaggle](https://www.kaggle.com/datasets/parniashokri/ccssdata). The training split expected by the experiment runner is `CCSS-DATA-V4-withVal/benchmarkingDatasets/train/noAugmentation-noPatch`, with `img/` and `lbl/` subdirectories. Extract it under `data/ccss/` or update `ccss_train_dir` in `configs/dinov3_detail_decoder.yaml`. Inference only needs the model checkpoints and an input image.

```bash
uv run --locked pytest
uv run --env-file model.env python scripts/run_experiment.py
```

The experiment runner evaluates the 200-epoch checkpoint by default. To repeat the fine-tuning run, copy the YAML and replace `trained_detail_checkpoint: weights/detail-200ep.pt` with `initial_detail_checkpoint: weights/detail.pt`. Pass the copied file with `scripts/run_experiment.py --config /path/to/config.yaml`. The runner saves the trained checkpoint under `results/visualizations/`.

The original wall-image evaluation also requires the wall images and labels, reference building masks, and hard-negative pairs at the paths in the YAML. Those project-specific inputs are separate from CCSS.
