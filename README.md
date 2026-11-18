# ReVA

**Learning When, Where, and How to Reobserve with a Vision Foundation Model for Medical Image Segmentation**

ReVA uses a frozen DINOv3 encoder for binary medical image segmentation, with at most one additional encoder pass per image.

- **DSR** produces the initial prediction from semantic features, RGB guidance, and patch occupancy.
- **OVP** scores candidate regions and viewing scales, then selects an observation or stops.
- **SER** uses the acquired evidence and the initial-view reference to update the selected region.

## Setup

Use Python 3.10+ and a CUDA-compatible PyTorch installation. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Download the [DINOv3 source and ViT-S/16 weights](https://github.com/facebookresearch/dinov3). Backbone weights and datasets are not included.

## Datasets

| Dataset | Download |
| --- | --- |
| Kvasir-SEG | [Simula](https://datasets.simula.no/kvasir-seg/) |
| CVC-ClinicDB / CVC-ColonDB | [CVC-Colon](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/) |
| ETIS-LaribPolypDB | [ETIS-Larib](https://polyp.grand-challenge.org/ETISLarib/) |
| BKAI-IGH NeoPolyp | [Kaggle](https://www.kaggle.com/competitions/bkai-igh-neopolyp/data) |
| BUSI | [Cairo University](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) |
| ISIC2017 | [ISIC Challenge](https://challenge.isic-archive.com/data/#2017) |
| ISIC2018 | [ISIC Challenge](https://challenge.isic-archive.com/data/#2018) |

Prepare `train`, `val`, and `test` splits, each containing `images/` and `masks/` with matching filename stems. Masks are binary; values above 127 indicate foreground. Keep validation and test masks at native resolution. Use the prescribed splits and remove duplicate images before training.

## Training and evaluation

Settings are in `configs/default.yaml`. Set these environment variables before running:

| Variable | Value |
| --- | --- |
| `REVA_DATA_ROOT` | Prepared dataset directory |
| `REVA_OUTPUT_DIR` | Training output directory |
| `DINOV3_WEIGHTS` | Pretrained ViT-S/16 weights |
| `DINOV3_REPO` | DINOv3 source directory; optional if importable |

```bash
reva train --config configs/default.yaml --seed 42
reva evaluate --config configs/default.yaml --split test
```

Training runs DSR, observation caching, SER, critic, and OVP fitting. Use `--stage dsr`, `cache`, `ser`, `critic`, `ovp`, or `freeze` to run individual stages. DSR selection uses validation Dice; downstream fitting and calibration use training-internal partitions. Use a separate output directory for each seed.

Evaluation saves native-resolution masks and per-image Dice and HD95, including initial, always-observe, and selective predictions. HD95 is reported in pixels. For empty masks, both-empty pairs receive Dice 1 and HD95 0; one-empty pairs receive the image diagonal as HD95. Evaluation includes multiple controls and is not a latency benchmark.

## Prediction

```bash
reva predict --run "$REVA_OUTPUT_DIR" \
  --encoder-weights "$DINOV3_WEIGHTS" \
  --encoder-repo "$DINOV3_REPO" \
  --input "$IMAGE" --output "$MASK"
```

Use `--device cpu` for CPU inference. Prediction requires no ground-truth mask.

## Structure

- `configs/`: training and runtime settings.
- `src/reva/`: models, observation geometry, training, and inference.
- `tests/`: geometry, model, metric, and configuration checks.

Run checks with `python -m unittest discover -s tests`.

## Acknowledgments

The backbone uses [DINOv3](https://github.com/facebookresearch/dinov3). The structure loss follows [PraNet](https://github.com/DengPingFan/PraNet). Third-party code and weights retain their respective licenses.
