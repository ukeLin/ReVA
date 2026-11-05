# ReVA

**Learning When, Where, and How to Reobserve with a Vision Foundation Model for Medical Image Segmentation**

ReVA performs medical image segmentation through selective visual evidence acquisition. A frozen DINOv3 encoder provides the initial features. The model then chooses whether to obtain another view, where to observe, and which viewing scale to use.

- **DSR** combines semantic features, RGB guidance, and patch occupancy for the initial prediction.
- **OVP** predicts the corrective value of region–scale pairs and selects a view or stops.
- **SER** compares acquired evidence with an initial-view reference and updates the selected region.

Inference uses at most one additional encoder pass. This implementation supports binary 2D segmentation.

## Installation

Use Python 3.10 or newer and install a PyTorch build compatible with your CUDA environment.

```bash
git clone https://github.com/ukeLin/ReVA.git
cd ReVA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Obtain the official [DINOv3 source and ViT-S/16 weights](https://github.com/facebookresearch/dinov3). Pass the source location through `--encoder-repo` or `DINOV3_REPO`; this can be omitted when `dinov3` is already importable. The pretrained backbone and datasets are not distributed here.

## Data

| Dataset | Download |
| --- | --- |
| Kvasir-SEG | [Simula](https://datasets.simula.no/kvasir-seg/) |
| CVC-ClinicDB | [CVC-Colon datasets](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/) |
| CVC-ColonDB | [CVC-Colon datasets](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/) |
| ETIS-LaribPolypDB | [ETIS-Larib](https://polyp.grand-challenge.org/ETISLarib/) |
| BKAI-IGH NeoPolyp | [Kaggle](https://www.kaggle.com/competitions/bkai-igh-neopolyp/data) |
| BUSI | [Cairo University](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) |
| ISIC2017 | [ISIC Challenge](https://challenge.isic-archive.com/data/#2017) |
| ISIC2018 | [ISIC Challenge](https://challenge.isic-archive.com/data/#2018) |

Point the program to a prepared dataset containing `train`, `val`, and `test` splits. Each split contains `images` and `masks` directories. Image and mask filenames must have matching stems. Masks are binary, with values above 127 indicating foreground. Validation and test masks retain their native resolution.

Use the prescribed dataset split and remove duplicate images before training.

## Training

All settings are in `configs/default.yaml`. Paths are supplied through environment variables or command-line arguments.

```bash
reva train --config configs/default.yaml \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --encoder-weights "$DINOV3_WEIGHTS" \
  --encoder-repo "$DINOV3_REPO" \
  --seed 42
```

The complete pipeline trains DSR, caches training observations, fits SER and the critic, learns OVP, and freezes the resulting components. The downstream training roles are disjoint subsets of the training split. DSR is selected on validation Dice; downstream selection and calibration use training-internal partitions.

Individual stages can be run with `--stage dsr`, `cache`, `ser`, `critic`, `ovp`, or `freeze`. Use separate output directories for different seeds or configurations. Rerunning an interrupted training command resumes saved checkpoints; completed stages are skipped.

## Evaluation

```bash
reva evaluate --config configs/default.yaml \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --encoder-weights "$DINOV3_WEIGHTS" \
  --encoder-repo "$DINOV3_REPO" \
  --split test
```

Evaluation saves native-resolution predictions and per-image Dice/HD95. The report contains the initial prediction, always-observe control, and selective ReVA output. Evaluation computes these controls together and is not a latency benchmark. HD95 is measured in native pixels. Both-empty masks receive Dice 1 and HD95 0; a single empty mask receives the image diagonal as HD95.

## Single-image prediction

```bash
reva predict --run "$OUTPUT_DIR" \
  --encoder-weights "$DINOV3_WEIGHTS" \
  --encoder-repo "$DINOV3_REPO" \
  --input "$IMAGE" --output "$MASK"
```

Use `--device cpu` for CPU inference. The run contains the DSR checkpoint and the executor, critic, and policy checkpoints. No ground-truth mask is used during prediction.

## Repository layout

```text
configs/          Training and runtime configuration
src/reva/
  models/         DINOv3 wrapper, DSR, OVP, SER
  geometry.py     View sampling and local correction mapping
  evidence.py     Observation descriptors and evidence processing
  train_dsr.py    Initial readout training
  engine.py       Downstream fitting and evaluation
  predictor.py    Single-image inference
  cli.py          Command-line interface
tests/            Geometry, model, metric, and configuration checks
```

## Tests

```bash
pip install -e '.[dev]'
python -m unittest discover -s tests
```

## Acknowledgments

The frozen backbone uses [DINOv3](https://github.com/facebookresearch/dinov3). Boundary-weighted BCE and IoU follow the structure-loss formulation used in [PraNet](https://github.com/DengPingFan/PraNet). DINOv3 code and weights remain subject to their respective licenses.
