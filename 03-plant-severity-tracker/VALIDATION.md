# Validation — 18 November 2026

## Completed checks

- All 14 behavioral tests passed using procedural images and the actual pretrained backbone.
- Editable package installation and the `leaf-severity` entry point succeeded.
- Synthetic model fitting, threshold selection, inference, mask export and same-leaf tracking demonstration completed.
- Real-data supervised pixel-head fitting and validation-only threshold selection completed.
- Held-out real-data evaluation and fixed-threshold brightness stress tests completed.
- Original dataset archive MD5 matched the official record.

## Real-data protocol

The official archive contained 1641 candidate apple photographs. A conservative exact/dHash grouping excluded 5 cross-category groups, selected one representative per group, and formed a category-balanced custom subset of 160 training, 40 validation and 80 test images. The test set has 63 images with nonempty annotated lesions.

This is not an official benchmark split. Grouping does not establish independence of physical leaves or acquisition sites. Source leaf IDs are unavailable. Exact selected image/group IDs and original label indices are included in reports/real_manifest.json.

## Measured real-image results

| Condition | Leaf Dice | Lesion Dice | Lesion IoU | Lesion Dice, diseased only | Area MAE (percentage points) | Undefined estimates | Median CPU inference |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 0.8842 | 0.4356 | 0.3406 | 0.5373 | 3.014 | 0 | 158.4 ms |
| brightness_0.7 | 0.8793 | 0.4590 | 0.3642 | 0.5511 | 3.024 | 0 | 143.4 ms |
| brightness_1.3 | 0.8842 | 0.4220 | 0.3258 | 0.5201 | 3.394 | 0 | 151.7 ms |

A constant zero-affected-area predictor has MAE **4.372 percentage points** on the same test images. The fitted model reaches 3.014 percentage points under original conditions. This is a modest improvement over a simple baseline, not proof of accurate fine lesion boundaries.

Original-condition diseased-only area MAE is 3.504 points. On images with no annotated lesion, predicted false affected area averages 1.199%. The distinction matters because low disease area and healthy leaves can make aggregate area error look better than lesion localization.

## Training actually performed

A multinomial linear segmentation head was fitted to 68228 annotated training pixels using frozen ImageNet-pretrained ResNet18 features plus color/texture features. The optimizer completed 133 iterations; converged before the limit: True. Fitting plus validation took 35.66 seconds in this environment.

Validation selected leaf threshold 0.65 and lesion threshold 0.7 from the declared grid. Neither threshold nor any network parameter was tuned on test masks. The CNN was not fine-tuned, and no new neural backbone was trained.

## Interpretation and limits

Leaf masks are substantially stronger than fine lesion masks. Some damage is missed or confused with discoloration/background. The affected-area ratio can be approximately right even when lesion locations are wrong, so Dice/IoU are reported alongside area error.

Brightness changes were digital multiplications with clipping. The brighter condition increased area error. These tests do not cover different light spectra, actual camera motion, occlusion or unrelated crops.

All mask metrics are native-resolution per-image macro averages. Empty lesion reference and empty prediction receive Dice=1; diseased-only scores are also reported. No sample had an undefined area estimate in the measured real test conditions.

One crop, one custom subset and one seed were evaluated. No confidence interval, independent field-site study, real time-series crop study or phone-deployment benchmark was performed. The low-margin pixel fraction is not calibrated uncertainty.

## Tracking demonstration

The included synthetic same-leaf sequence keeps the geometry fixed while adding procedural lesions. True visible affected area is approximately 0%, 5.836%, and 15.785%; actual model estimates and percentage-point differences are in examples/tracking-demo/sequence_results.json. The images and dates are simulated, not real plant observations.

## Reproduce

Use README.md for dataset download, archive verification, subset preparation, fitting, inference, evaluation and history commands. The package includes fitted synthetic and real-data linear heads, with artifact notices, but does not redistribute real source photographs or the pretrained CNN weights.

## Actual test log

```text
test_detects_synthetic_leaf_and_lesions (test_leafseverity.LeafSeverityTests.test_detects_synthetic_leaf_and_lesions) ... ok
test_empty_leaf_is_undefined_not_healthy (test_leafseverity.LeafSeverityTests.test_empty_leaf_is_undefined_not_healthy) ... ok
test_lesions_outside_leaf_rejected (test_leafseverity.LeafSeverityTests.test_lesions_outside_leaf_rejected) ... ok
test_manifest_duplicate_or_group_leakage_rejected (test_leafseverity.LeafSeverityTests.test_manifest_duplicate_or_group_leakage_rejected) ... ok
test_manual_leaf_mask_controls_denominator (test_leafseverity.LeafSeverityTests.test_manual_leaf_mask_controls_denominator) ... ok
test_native_resolution_and_aspect_ratio_restored (test_leafseverity.LeafSeverityTests.test_native_resolution_and_aspect_ratio_restored) ... ok
test_save_load_preserves_masks (test_leafseverity.LeafSeverityTests.test_save_load_preserves_masks) ... ok
test_severity_denominator_includes_lesions (test_leafseverity.LeafSeverityTests.test_severity_denominator_includes_lesions) ... ok
test_tracker_computes_percentage_points_not_percent_growth (test_leafseverity.LeafSeverityTests.test_tracker_computes_percentage_points_not_percent_growth) ... ok
test_tracker_does_not_compare_changed_model_or_view (test_leafseverity.LeafSeverityTests.test_tracker_does_not_compare_changed_model_or_view) ... ok
test_tracker_refuses_duplicate_observation (test_leafseverity.LeafSeverityTests.test_tracker_refuses_duplicate_observation) ... ok
test_tracker_withholds_comparison_for_quality_warning (test_leafseverity.LeafSeverityTests.test_tracker_withholds_comparison_for_quality_warning) ... ok
test_validation_training_overlap_rejected (test_leafseverity.LeafSeverityTests.test_validation_training_overlap_rejected) ... ok
test_wrong_crop_manifest_rejected (test_leafseverity.LeafSeverityTests.test_wrong_crop_manifest_rejected) ... ok

----------------------------------------------------------------------
Ran 14 tests in 5.377s

OK
```
