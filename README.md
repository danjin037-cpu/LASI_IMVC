# LASI-IMVC

Official compact implementation of LASI-IMVC for incomplete multi-view clustering on Scene15.
This release contains one dataset, one aligned CLIP feature cache, one training entry point,
and one concise experiment configuration. Historical ablations, unrelated datasets, generated
results, notebooks, and duplicate loss implementations are intentionally excluded.

## What is included

```text
LASI-IMVC/
├── configs/scene15.yaml
├── data/
│   ├── Scene_15.mat
│   └── clip_cache/scene15_vitb32/clip_img_feat.pt
├── lasi_imvc/
│   ├── config.py
│   ├── data.py
│   ├── engine_train.py
│   ├── models/
│   └── trainers/
├── tests/
├── train.py
└── requirements.txt
```

The published protocol uses the first two feature views in `Scene_15.mat`. CLIP features are
appended as a third semantic view. Under the strict missing-view protocol, the CLIP view is
available only when both of its source views are observed; it is never restored for an
incomplete sample.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA training, install the PyTorch build that matches your CUDA runtime by following the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/), then install
the remaining requirements.

## Training

Run the five-seed Scene15 experiment:

```bash
python train.py --config configs/scene15.yaml
```

Useful command-line overrides:

```bash
python train.py --device cuda:0 --seeds 0
python train.py --missing-rate 0.3 --mask-seed 1
python train.py --output-dir outputs/custom_run
```

Each seed is written to a separate directory. Aggregate metrics are saved as `summary.csv` and
`summary.json` under the configured output directory.

## Configuration

`configs/scene15.yaml` exposes only settings that are commonly changed: device, output path,
random seeds, missing rate, batch size, epoch counts, and learning rates. Fixed architecture,
loss, fairness, and evaluation settings are centralized in `lasi_imvc/config.py`. Unknown YAML
keys are rejected to catch stale options and spelling errors early.

## Reproducibility and fairness

- Missingness is generated only for the two original Scene15 views.
- The missing rate denotes the fraction of incomplete samples.
- Every incomplete sample retains exactly one original view.
- The CLIP mask is the logical AND of the two source-view masks.
- Missing views are not encoded or used as reconstruction targets.
- K-means evaluation uses `n_init=10` and a fixed training seed.

## Tests

```bash
python -m pytest
```

The tests check configuration hygiene, Scene15/CLIP alignment, tensor shapes, and the strict
CLIP visibility rule.

## Data notice

The repository includes `Scene_15.mat` and its aligned cached CLIP features solely to make the
released experiment self-contained. Users remain responsible for complying with the original
dataset's terms and citing the corresponding Scene15 source in publications.

## License

The code is released under the MIT License. Dataset and derived feature rights may be governed
by the terms of their original sources.
