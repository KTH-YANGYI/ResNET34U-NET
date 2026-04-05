# ResNet34 U-Net baseline for railway catenary crack detection

## 1. Overall

This project achieves a two-stage crack segmentation baseline.

1. Stage 1: Initially learns defect textures and boundaries on local patches.
2. Stage 2: Based on the weights from stage1, stage2 transfers that to full-image ROI segmentation.
   

## 2. Why two stage

The defect areas in full ROI images are very small. If we directly train the segmentation model on the full ROI images from beginning, due to vast majority of pixels are background, the model struggles to learn the textures and boundaries of small defects.

### Stage1: Local Patch Training

First, if we feed the model with cropped patches, it will see the defect center, defect boundaries, hard negative samples near defects more frequently, which could help the model establish an ablity to identify the crack and non-crack.

### Stage2: Full-image ROI Fine-tuning 

The stage 2 model doesn't start from scratch, but instead enters the full-image task with pre-learned knowledge.


## 3.Overall Pipeline 

pipeline.png

## 4.Dataset

| Data subset | Count | Role |
| --- | ---: | --- |
| Labeled defect images | 81 | Used for training and validation of the segmentation model |
| Unlabeled defect holdout images | 42 | Used for final inference only |
| Normal image pool | 5414 | Used for negative sampling and false-positive control |
| Normal future holdout | 2082 | Reserved normal images from future holdout-related video domains |

1. 81 independent training images from vedios and 42 testing images from vedios 
2. Mapping from pictures to vedios(["picture name", "vedio ID","Frame ID"])
3. Binary ground truth(.json)

## 5. Processing the dataset

This project first converts the raw image and mapping files into standardized train/validation manifests that the later training scripts can use directly. Specificlly, attach `video_di` and `frame_id` to all images.

In this project, we use 4-fold cross validation, the split logic can be summarized in three steps:
1. Split the 81 labeled defect images into 4 folds by defect video.
2. Remove 2082 normal images into `nomal_holdout`, which belongs to vedios corresponding to the test images.
3. Build 4 independent video-level folds of defect and normal images.

The current 4-fold split is:

| Fold | defect train | defect val | normal train | normal val |
| --- | ---: | ---: | ---: | ---: |
| 0 | 60 | 21 | 2469 | 863 |
| 1 | 61 | 20 | 2512 | 820 |
| 2 | 61 | 20 | 2516 | 816 |
| 3 | 61 | 20 | 2499 | 833 |

This split logic can prevent same-video leakage.

## 6. Patch Construction

Stage1 patches are divided into four groups:

| Category | Concrete types | Purpose |
| --- | --- | --- |
| Positive patches | `positive_center`, `positive_shift`, `positive_context`, `positive_boundary` | Learn the crack body, boundary, and local context |
| Defect-image negative patches | `near_miss_negative`, `hard_negative` | Learn regions that look similar to cracks but are not cracks |
| Normal negative patches | `normal_negative` | Supply pure negative evidence from normal images |
| Replay patches | dynamically mined hard samples | Force the model to revisit its own hardest mistakes |


The generated patch counts are:

| Fold | Train total | Train positive | Train defect-negative | Train normal-negative | Val total | Val positive | Val defect-negative | Val normal-negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 698 | 360 | 169 | 169 | 244 | 126 | 59 | 59 |
| 1 | 690 | 364 | 163 | 163 | 254 | 120 | 67 | 67 |
| 2 | 730 | 366 | 182 | 182 | 219 | 119 | 50 | 50 |
| 3 | 701 | 363 | 169 | 169 | 236 | 120 | 58 | 58 |

## 7. Model

The same model is used in both Stage1 and Stage2:

- Encoder: ResNet34
- Decoder: U-Net style upsampling decoder
- Output: one-channel binary segmentation logits


## 8. Training Configuration and Optimization Strategy

### 8.1 Input Size and Data Augmentation

Under the default settings:

| Stage | Input size | Meaning |
| --- | --- | --- |
| Stage1 | `384 x 384` | Patches are cropped first, then resized |
| Stage2 | `640 x 640` | Full ROI images are resized and used directly |

Data augmentation is enabled in both stages, but Stage1 uses stronger local-appearance perturbation while Stage2 focuses more on full-image stability. The augmentation family includes:

- horizontal flip,
- vertical flip,
- small-angle rotation,
- brightness and contrast perturbation,
- gamma perturbation,
- blur,
- Gaussian noise.

### 8.2 Loss Function

The training loss is a weighted combination of `BCEWithLogitsLoss` and `DiceLoss`.

The reason is:

- BCE provides stable pixel-level optimization,
- Dice improves region-overlap quality,
- a higher `pos_weight` helps compensate for the severe sparsity of crack pixels.

Current default values:

| Parameter | Stage1 | Stage2 |
| --- | ---: | ---: |
| BCE weight | 0.5 | 0.5 |
| Dice weight | 0.5 | 0.5 |
| `pos_weight` | 12.0 | 12.0 |

### 8.3 Optimizer and Learning Rate Policy

The default optimizer is AdamW, with separate parameter groups for the encoder and decoder.

Default settings:

| Parameter | Stage1 | Stage2 |
| --- | ---: | ---: |
| encoder lr | `5e-5` | `2e-5` |
| decoder lr | `2e-4` | `1e-4` |
| weight decay | `1e-4` | `1e-4` |

This means:

- the encoder is updated more conservatively,
- the decoder is allowed to adapt faster to the crack segmentation task.

The learning rate scheduler is `ReduceLROnPlateau`, which lowers the learning rate when the core validation objective stops improving.