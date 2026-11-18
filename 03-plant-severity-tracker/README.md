# 03 — Apple Plant Disease Severity Tracker

A runnable, one-crop research prototype that segments visible apple leaf tissue
and damaged regions, estimates the affected fraction of visible leaf area, and
stores comparable observations over time.

Implemented on **18 November 2026**. The project is scoped to **apple leaves**,
because its selected dataset supplies both leaf-tissue and symptom annotations.
It does not identify the crop automatically, diagnose a disease type, or predict
treatment response. Use a single identifiable apple leaf and comparable views.

## Quick start

Python 3.10+; validation uses Python 3.12.14 on Linux CPU. From this folder:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
leaf-severity --help
leaf-severity demo --output outputs/my-demo
```

Use a new output directory for each demo. The command generates **procedural
synthetic leaves/lesions**, fits a model, and reports segmentation and affected-area
errors on separate generated images. This is a smoke test, not field accuracy.

The first model load downloads official ImageNet-pretrained ResNet18 weights
(about 45 MB) through torchvision. Later inference is local/offline when weights
are cached. No API key, paid service or GPU is required. The program never falls
back to random weights if downloading fails. Pre-cache weights for offline use.

For the exact direct dependencies used during validation:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
```

This is a record of directly used versions, not a cross-platform transitive lockfile.
Allow pip to resolve compatible torch/torchvision builds for a different platform.
No Android deployment or phone-performance measurement was performed.

## Inspect a leaf

```bash
python -c "from pathlib import Path; from leafseverity.data import synthetic; synthetic(Path('outputs/example-data'))"
leaf-severity predict --model artifacts/synthetic_model.npz \
  --image outputs/example-data/test/000.png --output outputs/query
```

The generation command creates the synthetic image and mask files used in the prediction and tracking examples below. The synthetic model demonstrates the application pipeline. Use the included
`artifacts/real_apple_model.npz`, or train on the real dataset below, for
experimentation with real apple photographs. Inspect `VALIDATION.md` for its measured
quality and limitations; a synthetic result is not a field-performance claim.
The real-data head has a separate non-commercial research notice in
`artifacts/MODEL_NOTICE.md`.

Prediction exports:

- `leaf_mask.png`: all visible predicted leaf tissue, including damaged tissue.
- `lesion_mask.png`: visible predicted damage inside that leaf mask.
- `overlay.png`: green leaf region and red damaged region for visual review.
- `prediction.json`: pixel counts, affected-area percentage, thresholds, and
  image-quality flags.

Affected area is `100 × lesion_pixels / leaf_pixels`, where the denominator
**includes lesions**. If no leaf is detected, severity is undefined (`null`),
not zero. It is a two-dimensional visible-area estimate, not the diseased fraction
of occluded leaf tissue or the whole plant, and not a pathological severity grade.

Supply `--leaf-mask path/to/binary_leaf_mask.png` to use a reviewed/manual
denominator for the selected leaf. The mask must have the native image dimensions
and include damaged as well as healthy tissue. Lesions remain model predictions.
Manual-mask results should be identified separately from fully automatic results.

## Track observations

```bash
leaf-severity track --model artifacts/synthetic_model.npz \
  --image outputs/example-data/test/000.png --output outputs/observation-1 \
  --store outputs/observations.json --leaf-id apple-001 \
  --view-id front-flat --date 2026-11-18
leaf-severity history --store outputs/observations.json --leaf-id apple-001
```

Leaf IDs are supplied by the user; the app does not re-identify leaves across
photographs. Compare the same leaf, view, framing and acquisition conditions.
Differences are reported in **percentage points**, not percent biological growth.
The tracker withholds a difference when models, views, manual-mask methods, or
quality flags make the pair incomparable. Duplicate leaf/view/date observations
are rejected. Local stores are written atomically, but concurrent writers are
not supported.

For a reproducible same-leaf demonstration using **synthetic** damage progression:

```bash
python scripts/make_tracking_demo.py --model artifacts/synthetic_model.npz \
  --image outputs/example-data/test/000.png --mask outputs/example-data/test/000_mask.png \
  --output outputs/tracking-demo
```

Its dates and changing lesions are simulated. No real longitudinal crop study
was conducted, and changes in photographs are not proof of biological progression.

## Model and training

This is a transfer-learning segmentation baseline:

1. Letterbox the image to 224×224 while retaining aspect ratio.
2. Extract frozen ImageNet-pretrained ResNet18 layer1/layer2 features and upsample
   them to the pixel grid.
3. Add RGB, chromaticity, HSV, local color mean and standard deviation: 207 total
   features per pixel.
4. Sample at most 160 pixels per class per training image; learn feature scaling
   and a supervised multinomial logistic-regression pixel head.
5. Interpret the three output classes as background, healthy tissue, and lesion.
6. Select leaf/lesion probability cutoffs using the **validation split only**,
   maximizing the mean of per-image leaf Dice and lesion Dice over a declared
   3×4 cutoff grid.
7. Restore masks to native image dimensions and calculate the visible-area ratio.

Only the linear segmentation head is trained. The CNN is **not fine-tuned**;
this is neither a newly trained foundation model nor a reproduction of a cited
SAM/U-Net model. Sampling changes the pixel class prior, so raw class scores
and the low-margin fraction are **not calibrated uncertainty estimates**.
The low-margin fraction is an inspection aid, not a confidence interval.

Learned arrays and metadata are stored in a small portable `.npz` with pickle
disabled. The pretrained CNN weights are cached separately and not redistributed.
The model supports apple only by declared scope, not by verifying the species.

## Real dataset and reproducible experiment

Dataset: **Feng and Chao, Apple Tree Leaf Disease Segmentation Dataset**, Science
Data Bank, 2022, V1: https://doi.org/10.11922/sciencedb.01627

The official record declares **CC BY-NC-SA 4.0**. Obtain `ATLDSD.rar` from its
download panel. Official archive MD5: `7441604c86d65090606e40dc7d1f8540`.
Extract it with 7-Zip or another RAR-capable tool. Then:

```bash
python scripts/prepare_atldsd.py --root /datasets/ATLDSD \
  --archive /downloads/ATLDSD.rar --output data/apple --seed 18
leaf-severity evaluate --manifest data/apple/manifest.json \
  --output outputs/apple --seed 18
```

The preparer expects category folders with `image/*.jpg` and matching label
images. It converts the source indexed mask convention explicitly: 0 background,
1 healthy tissue, and 2–5 disease-specific lesion labels collapsed into class 2.
This conversion was checked against source image/mask pairs; it does not use
whole-image disease labels as pixel supervision.

It groups exact decoded duplicates and conservative 64-bit grayscale dHash
near duplicates (Hamming distance ≤3) **before** splitting. One representative
per group is selected. Cross-category groups are omitted rather than assigned
conflicting labels. The default custom subset uses 32 training, 8 validation and
16 test representatives per category, when enough groups exist. This is a custom
category-balanced subset, not an official benchmark split.

Grouping is a leakage precaution, not proof that all photographs depict independent
physical leaves. Acquisition/plant IDs are unavailable in this protocol, and
similar leaves or different viewpoints of one leaf can remain across splits.
Exact decoded-image and explicit group overlap are checked by the manifest loader.

`reports/real_manifest.json` records the actual selected IDs, grouping method,
source labels, seed, and archive checksum. `VALIDATION.md` records the actual split
sizes and measured results; do not assume the planned counts if source filtering
changed them.

## Train or evaluate your own annotated apple images

Use a manifest with `crop: "apple"` and `train`, `validation`, `test` arrays of
objects containing `image`, `mask`, and optionally an acquisition `group`.
Paths may be relative to the manifest. Internal masks use **0 background,
1 healthy tissue, 2 lesion**; never substitute leaf bounding boxes for lesion
annotations. Images and masks must have matching stored pixel orientation and
dimensions. The reader does not rotate images using EXIF metadata.

```bash
leaf-severity fit --manifest data/apple/manifest.json --output outputs/apple-model.npz
leaf-severity predict --model outputs/apple-model.npz \
  --image /path/to/apple-leaf.jpg --output outputs/apple-query
```

The supplied evaluator reports native-resolution, per-image macro leaf Dice,
lesion Dice/IoU, diseased-only lesion Dice, and affected-area mean absolute error
in percentage points. Empty prediction and empty reference masks score Dice=1;
diseased-only reporting helps expose inflation from healthy images. Undefined
severity cases are counted rather than silently treated as correct zero estimates.

It also evaluates digital brightness multipliers 0.7 and 1.3 with fixed model
parameters and thresholds. These stress tests do not simulate all field shadows,
white-balance shifts, camera movement, or new biological conditions. No test labels
are used for fitting or threshold selection.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover denominator correctness, undefined empty leaves, lesion containment,
synthetic segmentation, aspect-ratio restoration, manual denominator handling,
save/load equivalence, split leakage, duplicate observations, percentage-point
changes, incompatible models/views, quality flags and wrong-crop manifests.
They use synthetic images and the real pretrained backbone; first use needs
cached weights or network access. See the captured log in `test-results.txt`.

## Recent primary research context

- **ReLeaf-SAM**, Liu et al., July 2026, combines foundation-model shape cues
  with lesion-detail features for plant disease segmentation. Our lightweight
  frozen-feature pixel head does not reproduce that architecture or its results.
  https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1865649/full
- **PlantSeg**, Wei et al., Scientific Data, 2026, supplies disease-region masks
  from diverse imagery and motivates evaluating beyond class labels. PlantSeg
  itself was not used in this run. https://doi.org/10.1038/s41597-025-06513-4
- **ATLDSD**, Feng and Chao, is the primary dataset source for this apple-scoped
  prototype. https://doi.org/10.11922/sciencedb.01627
- Official ResNet18 weights/API:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

The prototype investigates measurable visible-area estimates and their sensitivity
to image conditions. It does not claim a new segmentation algorithm or infer a
medical/agronomic grade from an arbitrary percentage cutoff.

## Limitations and handoff

The mask may confuse veins, discoloration, shadows or background with lesions.
Small spots can be lost at the CNN/input resolution. Overlap, curling and foreshortening
change the visible area. Multi-leaf images may aggregate tissue from different
leaves; supply a reviewed single-leaf mask or retake the photograph.

Static held-out images do not validate temporal monitoring. A real follow-up study
needs repeated, consistently framed images of tagged leaves and expert annotations.
One crop, one subset and one seed do not establish general field reliability.

`START_HERE.html` opens an offline overview. `reports/` includes measured results,
`examples/` contains the simulated tracking sequence, and `artifacts/` contains fitted heads with
their provenance. Real source photographs are not redistributed in the package.
The original code is MIT-licensed; external data/weights retain their own terms.
