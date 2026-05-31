# 🐠 CNN Species Classifier — *Amphiprion ocellaris* Detection

[![CI](https://github.com/ayoubhartaoui/CNN_project/actions/workflows/ci.yml/badge.svg)](https://github.com/ayoubhartaoui/CNN_project/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)

> Binary image classifier distinguishing *Amphiprion ocellaris* (clownfish) from 4 morphologically similar marine species — built as part of the **Machine Learning for Earth Sciences** (MLEES) course, Master BEC CEE, University of Lausanne.

---

## Overview

This project compares two deep learning approaches for marine species identification from photographic occurrence records:

| Approach | Architecture | Dataset source | XAI |
|---|---|---|---|
| **Custom CNN** | 3× Conv2D + Dense head | GBIF API (scraped) | Saliency maps |
| **Transfer Learning** | MobileNetV2 + custom head | Same GBIF dataset | Saliency maps |

Images are sourced via the [GBIF Occurrence API](https://www.gbif.org/developer/occurrence), covering *A. ocellaris* (target) against *A. clarkii*, *Neoglyphidodon oxyodon*, *Neopetrolisthes maculatus*, and *Heteractis aurora* (non-target).

---

## Results

| Metric | Custom CNN | MobileNetV2 |
|--------|-----------|-------------|
| Test Accuracy | — | — |
| Precision | — | — |
| Recall | — | — |
| F1 Score | — | — |
| AUC-ROC | — | — |
| Training epochs | ≤30 (early stop) | ≤30 (early stop) |

> Fill in your actual values after training. See `results/` for confusion matrix and ROC plots.

---

## Project Structure

```
CNN_project/
├── src/
│   ├── data_scraper.py      # GBIF API image downloader
│   ├── dataset.py           # Organization, splitting, preprocessing
│   ├── train_cnn.py         # Custom CNN training
│   ├── train_mobilenet.py   # MobileNetV2 transfer learning
│   └── evaluate.py          # Metrics, ROC, confusion matrix, XAI saliency maps
│
├── CNN.ipynb                # Exploratory notebook — custom CNN
├── MobileNetV2.ipynb        # Exploratory notebook — transfer learning
│
├── .github/workflows/
│   └── ci.yml               # CI: lint → type-check → smoke tests → security
│
├── pyproject.toml           # Tool config (ruff, black, isort, mypy)
├── .pre-commit-config.yaml  # Pre-commit hooks
├── requirements.txt
└── Final_project_MLEES_Ayoub_Hartaoui.pdf
```

---

## Quickstart

```bash
git clone https://github.com/ayoubhartaoui/CNN_project.git
cd CNN_project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Step 1 — Scrape images from GBIF:**
```bash
python src/data_scraper.py
```

**Step 2 — Organize and split the dataset:**
```bash
python src/dataset.py
```

**Step 3 — Train:**
```bash
# Custom CNN
python src/train_cnn.py

# Transfer learning (MobileNetV2)
python src/train_mobilenet.py
```

**Step 4 — Evaluate + XAI:**
```bash
python src/evaluate.py
```

---

## Methodology

### Data pipeline
- Images scraped from [GBIF](https://www.gbif.org) via occurrence search API (StillImage media type)
- 500 target images (*A. ocellaris*) · ~300 per non-target species
- Split: **70% train / 15% val / 15% test** (stratified, seed = 42)
- Preprocessing: resize to 224×224, normalize to [0, 1]
- Augmentation (train only): rotation ±20°, horizontal flip, zoom, width/height shift

### Vanilla custom-made CNN 
3-block convolutional encoder (32→64→128 filters) with L2 regularization (λ=0.01), 50% dropout, binary sigmoid output. Trained with class-weight balancing, early stopping (patience=5), and learning rate reduction on plateau.

### MobileNetV2 (transfer learning)
ImageNet-pretrained base frozen, custom head: `GlobalAveragePooling2D → Dropout(0.5) → Dense(128, ReLU) → Dense(n_classes, softmax)`. Same callbacks as the custom CNN.

### Explainable AI
Both models use **gradient-based saliency maps** (vanilla gradients via `tf.GradientTape`) to visualize which image regions drive the prediction — rendered as heatmaps superimposed on the original image.

---

## Development

```bash
pip install -r requirements-dev.txt
pre-commit install          # Hooks: format, lint, strip notebook outputs

# Run checks manually
ruff check src/
black src/
nbstripout CNN.ipynb MobileNetV2.ipynb
```

---

## Report

The full methodology, results, and discussion are in [`Final_project_MLEES_Ayoub_Hartaoui.pdf`](Final_project_MLEES_Ayoub_Hartaoui.pdf).

---

## Contact

Questions? Email me or open an issue.

**Ayoub Hartaoui** — Master BEC CEE, University of Lausanne
