# LeafGauge

LeafGauge estimates the share of a visible apple leaf affected by damage. It exports a leaf mask, a lesion mask, and a color overlay. It can also record dated observations of the same leaf and compare compatible measurements in percentage points.

[Detailed project documentation](03-plant-severity-tracker/README.md) · [Validation report](03-plant-severity-tracker/VALIDATION.md) · [MIT License](LICENSE)

## Quick start

Use Python 3.10 or newer. From the repository root:

```bash
cd 03-plant-severity-tracker
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux or macOS, or `.\.venv\Scripts\Activate.ps1` in PowerShell. Then run:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -c "from pathlib import Path; from leafseverity.data import synthetic; synthetic(Path('outputs/example-data'))"
leaf-severity predict --model artifacts/synthetic_model.npz --image outputs/example-data/test/000.png --output outputs/query
```

The first inference downloads the pretrained ResNet18 weights. Open `outputs/query/overlay.png` to inspect the segmentation and `outputs/query/prediction.json` for the affected-area estimate. The quick start generates procedural test images locally.

For real apple photographs, use `artifacts/real_apple_model.npz` and supply your image with `--image`. Photograph one identifiable leaf and keep the view consistent across observations. See the [tracking commands](03-plant-severity-tracker/README.md#track-observations).

## Model and validation

The model uses frozen ResNet18 features and color and texture features. A trained pixel classifier predicts background, healthy leaf tissue, and damaged tissue. The affected-area estimate divides lesion pixels by total visible leaf pixels.

On 80 held-out ATLDSD apple images, average affected-area error was **3.01 percentage points**, compared with **4.37** for a constant zero-area estimate. Leaf Dice was **0.884** and lesion Dice was **0.436**. Fine lesion boundaries remain difficult to segment. The dated tracking demo uses synthetic images. [Evaluation protocol and limitations](03-plant-severity-tracker/VALIDATION.md).

## Repository guide

| Path | Contents |
| --- | --- |
| [`leafseverity/`](03-plant-severity-tracker/leafseverity) | Segmentation, CLI, and observation tracking |
| [`scripts/`](03-plant-severity-tracker/scripts) | Dataset preparation and tracking demo |
| [`tests/`](03-plant-severity-tracker/tests) | Behavioral tests |
| [`reports/`](03-plant-severity-tracker/reports) | Evaluation metrics, predictions, and split metadata |

From the project folder, run `python -m unittest discover -s tests -v`.

## License

The [MIT License](LICENSE) covers the original project software and supporting documentation. The ATLDSD-derived model is supplied for non-commercial research under CC BY-NC-SA 4.0. Third-party data and pretrained weights have their own terms. [Model provenance and attribution](03-plant-severity-tracker/artifacts/MODEL_NOTICE.md).
