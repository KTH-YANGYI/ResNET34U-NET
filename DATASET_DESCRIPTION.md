# Dataset Description

This document describes the current `dataset0505_crop640_roi` snapshot used by this project.

Statistics below are computed from:

- dataset root: `../dataset0505_crop640_roi`
- sample manifest: `manifests/samples.csv`
- default split: `--test-ratio 0.20 --test-seed 2026`
- CV fold seed: `42`

## Directory Layout

```text
dataset0505_crop640_roi/
|- camera/
|  |- crack/
|  |- normal/
|  `- broken/
`- phone/
   |- crack/
   |- normal/
   `- broken/
```

Class roles:

- `crack`: labeled crack/defect images. Same-stem LabelMe JSON files are converted to binary masks under `generated_masks/`.
- `normal`: negative samples with no crack mask. These are used in training and validation.
- `broken`: unlabeled holdout images. These are not used by Stage1, Stage2, or CV validation.

## Raw Counts

| source | crack | normal | broken | total |
| --- | ---: | ---: | ---: | ---: |
| camera | 157 | 38 | 44 | 239 |
| phone | 106 | 36 | 63 | 205 |
| total | 263 | 74 | 107 | 444 |

Image sizes:

| image group | size | count |
| --- | ---: | ---: |
| crack + normal | 640 x 640 | 337 |
| broken | 1920 x 1080 | 107 |

## Train / Holdout Split

By default, `prepare_samples.py` keeps 20% of labeled `crack` and `normal` images as inference-only holdout.

| split | crack | normal | broken | total | role |
| --- | ---: | ---: | ---: | ---: | --- |
| `trainval` | 211 | 59 | 0 | 270 | Stage1/Stage2 training and CV validation |
| `holdout` | 52 | 15 | 107 | 174 | inference only |

Holdout composition:

| holdout reason | count |
| --- | ---: |
| `test_split` from crack/normal | 67 |
| `broken_unlabeled` | 107 |
| total | 174 |

Split by source/device:

| split | camera crack | camera normal | camera broken | phone crack | phone normal | phone broken | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `trainval` | 126 | 30 | 0 | 85 | 29 | 0 | 270 |
| `holdout` | 31 | 8 | 44 | 21 | 7 | 63 | 174 |

## Cross-Validation Folds

Each fold uses one `cv_fold` as validation and the other three folds as training.

| fold | train images | train crack | train normal | val images | val crack | val normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 200 | 157 | 43 | 70 | 54 | 16 |
| 1 | 202 | 158 | 44 | 68 | 53 | 15 |
| 2 | 204 | 159 | 45 | 66 | 52 | 14 |
| 3 | 204 | 159 | 45 | 66 | 52 | 14 |

`broken` is not included in any fold.

## Stage1 Patch Index

Stage1 trains on patches generated from the original `trainval` images. The Stage1 `batch_size` is therefore a patch batch size, not an original-image batch size.

| fold | train patches | val patches | train steps at batch 128 |
| --- | ---: | ---: | ---: |
| 0 | 1173 | 425 | 10 |
| 1 | 1194 | 401 | 10 |
| 2 | 1195 | 402 | 10 |
| 3 | 1231 | 361 | 10 |

Patch type counts in the training indexes:

| fold | positive center | positive shift | positive context | positive boundary | near-miss negative | hard negative | normal negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 152 | 314 | 157 | 314 | 51 | 67 | 118 |
| 1 | 154 | 316 | 158 | 316 | 59 | 66 | 125 |
| 2 | 158 | 318 | 159 | 318 | 61 | 60 | 121 |
| 3 | 154 | 318 | 159 | 318 | 62 | 79 | 141 |

## Crack Mask Size

Mask statistics are computed on all 263 `crack` images. Pixel ratio means:

```text
positive mask pixels / total image pixels
```

For `crack` images, total image pixels are `640 * 640 = 409600`.

### Overall Mask Area

| metric | mask area px | mask area % |
| --- | ---: | ---: |
| min | 76 | 0.0186 |
| p25 | 418 | 0.1021 |
| median | 1113 | 0.2717 |
| mean | 3590.7 | 0.8766 |
| p75 | 5225.5 | 1.2758 |
| p90 | 12304.8 | 3.0041 |
| max | 22165 | 5.4114 |

Interpretation: most cracks occupy far below 1% of the crop. The distribution is long-tailed: a small number of camera cracks are much larger than the median.

### Largest Connected Component

| metric | largest component px | largest component % |
| --- | ---: | ---: |
| min | 76 | 0.0186 |
| p25 | 391.5 | 0.0956 |
| median | 867 | 0.2117 |
| mean | 2651.2 | 0.6473 |
| p75 | 3239 | 0.7908 |
| p90 | 7885.4 | 1.9251 |
| max | 18337 | 4.4768 |

Connected component count:

| metric | components per crack image |
| --- | ---: |
| min | 1 |
| p25 | 1 |
| median | 1 |
| mean | 1.65 |
| p75 | 2 |
| p90 | 3 |
| max | 5 |

Most crack masks are a single connected component. Camera cracks more often contain multiple components.

## Crack Bounding Boxes

Bounding box is measured around the union of all positive mask pixels in each crack image.

| metric | bbox width px | bbox height px | bbox area % |
| --- | ---: | ---: | ---: |
| min | 18 | 5 | 0.0256 |
| p25 | 39 | 19 | 0.1776 |
| median | 71 | 47 | 0.7715 |
| mean | 77.6 | 77.7 | 2.1272 |
| p75 | 116.5 | 119.5 | 3.2546 |
| p90 | 138 | 202.8 | 6.7544 |
| max | 179 | 269 | 10.6230 |

The bbox is much larger than the mask itself because cracks are thin and sparse inside the bounding region.

## Device Differences

Camera cracks are much larger than phone cracks.

| source | count | median mask px | mean mask px | median mask % | mean mask % | max mask % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| camera | 157 | 2911 | 5570.8 | 0.7107 | 1.3601 | 5.4114 |
| phone | 106 | 382 | 658.0 | 0.0933 | 0.1606 | 1.0332 |

Bounding boxes by source:

| source | median bbox width | median bbox height | median bbox area % | median components |
| --- | ---: | ---: | ---: | ---: |
| camera | 107 | 99 | 2.4072 | 2 |
| phone | 34.5 | 18 | 0.1533 | 1 |

This matters for training: the model sees both relatively large camera cracks and very small phone cracks. The phone subset is especially sensitive to class imbalance and thresholding.

## Trainval vs Holdout Crack Sizes

The default 20% holdout split preserves a similar crack-size distribution.

| split | crack count | median mask px | mean mask px | median mask % | mean mask % | max mask % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `trainval` | 211 | 1113 | 3608.4 | 0.2717 | 0.8809 | 5.4114 |
| `holdout` | 52 | 1082 | 3519.3 | 0.2642 | 0.8592 | 5.2041 |

## Size Distribution Bins

Mask area percentage bins:

| mask area % | crack images |
| --- | ---: |
| `< 0.10%` | 65 |
| `0.10% - 0.25%` | 59 |
| `0.25% - 0.50%` | 44 |
| `0.50% - 1.00%` | 21 |
| `>= 1.00%` | 74 |

Mask area pixel bins:

| mask area px | crack images |
| --- | ---: |
| `< 500` | 76 |
| `500 - 999` | 48 |
| `1000 - 2499` | 49 |
| `2500 - 4999` | 23 |
| `5000 - 9999` | 32 |
| `>= 10000` | 35 |

## Largest Crack Masks

| sample_id | split | mask px | mask % | bbox width | bbox height | components |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `camera_crack_camera099` | `trainval` | 22165 | 5.4114 | 160 | 269 | 2 |
| `camera_crack_camera123` | `holdout` | 21316 | 5.2041 | 168 | 259 | 2 |
| `camera_crack_camera060` | `trainval` | 19941 | 4.8684 | 159 | 244 | 2 |
| `camera_crack_camera065` | `trainval` | 18980 | 4.6338 | 151 | 258 | 5 |
| `camera_crack_camera284` | `trainval` | 18168 | 4.4355 | 149 | 253 | 3 |
| `camera_crack_camera170` | `trainval` | 17363 | 4.2390 | 173 | 230 | 3 |
| `camera_crack_camera082` | `trainval` | 17082 | 4.1704 | 139 | 247 | 2 |
| `camera_crack_camera250` | `trainval` | 16785 | 4.0979 | 159 | 211 | 3 |
| `camera_crack_camera025` | `holdout` | 16697 | 4.0764 | 135 | 225 | 2 |
| `camera_crack_camera196` | `trainval` | 16149 | 3.9426 | 133 | 253 | 3 |

## Smallest Crack Masks

| sample_id | split | mask px | mask % | bbox width | bbox height | components |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `phone_crack_phone760` | `trainval` | 76 | 0.0186 | 22 | 9 | 1 |
| `phone_crack_phone038` | `holdout` | 84 | 0.0205 | 18 | 8 | 1 |
| `phone_crack_phone552` | `trainval` | 84 | 0.0205 | 21 | 5 | 1 |
| `phone_crack_phone261` | `trainval` | 87 | 0.0212 | 22 | 7 | 1 |
| `phone_crack_phone859` | `trainval` | 88 | 0.0215 | 26 | 6 | 1 |
| `phone_crack_phone026` | `trainval` | 90 | 0.0220 | 21 | 7 | 1 |
| `phone_crack_phone827` | `trainval` | 101 | 0.0247 | 29 | 8 | 1 |
| `phone_crack_phone742` | `holdout` | 102 | 0.0249 | 26 | 8 | 1 |
| `phone_crack_phone869` | `holdout` | 103 | 0.0251 | 22 | 8 | 1 |
| `phone_crack_phone684` | `trainval` | 103 | 0.0251 | 25 | 5 | 1 |

## Training Implications

- The segmentation target is highly sparse: median crack mask area is only `0.2717%`.
- Phone cracks are particularly small: median phone mask area is `0.0933%`.
- `broken` should stay out of training because it has no mask and is currently treated as unlabeled holdout data.
- Stage1 patch mining is useful because full-image training would otherwise see very few positive pixels per image.
- Stage2 still needs normal images because false positives on sparse masks are easy to create.
- Online data augmentation does not increase the manifest count; it creates random variants when each sample is loaded during training.
