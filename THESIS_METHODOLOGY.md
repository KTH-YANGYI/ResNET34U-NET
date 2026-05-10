# 2 Methodology

This chapter describes the proposed crack-segmentation methodology used in the U-Net two-stage pipeline. The method is designed for sparse visual defects in railway contact-wire ROI images, where the positive pixels occupy only a very small part of each image and where false positives on visually normal samples must be tightly controlled. The central idea is to separate the learning problem into two complementary stages. The first stage trains a ResNet34-U-Net on informative local patches so that small crack pixels are seen frequently. The second stage transfers the learned representation to full ROI images and learns the image-level context needed to suppress false positives on normal samples. Final binary masks are produced by a single globally selected post-processing setting, chosen from pooled out-of-fold validation predictions.

The complete pipeline is:

```text
Raw ROI images and LabelMe polygons
  -> sample manifest and binary mask generation
  -> trainval / holdout split and 4-fold cross-validation
  -> Stage1 patch-index construction
  -> Stage1 patch-level U-Net training with balanced sampling and replay
  -> Stage2 full-image training initialized from Stage1
  -> hard-normal replay for normal false-positive suppression
  -> per-fold validation export
  -> pooled out-of-fold global post-processing search
  -> holdout inference and error analysis
```

The sections below describe each part of this flow in the same order in which data passes through the system.

## 2.1 Problem Definition

The task is binary semantic segmentation of crack-like defects in cropped ROI images. For each image \(I_n\), the model predicts a binary mask \(\hat{Y}_n\) indicating pixels that belong to a crack. The corresponding ground-truth mask is denoted as \(Y_n\). Let the image domain be \(\Omega = \{1,\ldots,H\}\times\{1,\ldots,W\}\). In the current ROI setting, most supervised crack and normal images have size \(640\times640\).

The supervised training set contains two labeled sample types:

1. **Defect samples.** These are crack images with polygon annotations. They provide positive segmentation masks.
2. **Normal samples.** These are visually normal images. They provide all-zero masks and are used to learn false-positive suppression.

Images from the `broken` folder are treated as unlabeled holdout samples in the current pipeline. They are not used for supervised training, fold validation, Dice computation, or IoU computation. They can be used only for qualitative inference unless pixel-level masks are added later.

The segmentation model is a function

$$
f_\theta: I_n \mapsto Z_n,
\tag{1}
$$

where \(Z_n \in \mathbb{R}^{H\times W}\) is a logit map. A probability map is obtained by applying the sigmoid function:

$$
P_n(x,y)
=
\sigma(Z_n(x,y))
=
\frac{1}{1+\exp(-Z_n(x,y))}.
\tag{2}
$$

The final binary mask is not taken directly from the raw probability map. Instead, thresholding and connected-component filtering are applied as the last step. This is important because the model is trained to produce dense probability maps, while the final task requires discrete crack regions.

**Suggested figure:** add an overview diagram after this subsection. The figure should show raw ROI images, LabelMe annotations, generated masks, Stage1 patch training, Stage2 full-image training, global post-processing search, and final predictions.

## 2.2 Dataset Organization

The raw ROI dataset is organized by acquisition device and sample category:

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

The current snapshot contains images from two sources: `camera` and `phone`. The class roles are:

| folder | role in the pipeline |
| --- | --- |
| `crack` | Supervised positive images. Same-stem LabelMe JSON files are converted to binary crack masks. |
| `normal` | Supervised negative images. They are assigned all-zero masks. |
| `broken` | Unlabeled holdout images. They are excluded from supervised training and validation metrics. |

The image counts are:

| source | crack | normal | broken | total |
| --- | ---: | ---: | ---: | ---: |
| camera | 157 | 38 | 44 | 239 |
| phone | 106 | 36 | 63 | 205 |
| total | 263 | 74 | 107 | 444 |

The supervised crack and normal images are \(640\times640\) ROI crops. The `broken` images are \(1920\times1080\) in the current snapshot, which is another reason they are handled as unlabeled holdout samples rather than directly mixed into the supervised ROI training set.

A key characteristic of the dataset is that crack pixels are sparse. Across all crack images, the median mask area is 1113 pixels, corresponding to only \(0.2717\%\) of a \(640\times640\) crop. The phone subset is even more sparse: the median phone crack occupies only \(0.0933\%\) of the image. This sparsity is the main motivation for patch-level Stage1 training, positive-pixel weighting, Dice loss, and hard-negative replay.

**Suggested figure:** show representative camera and phone crack examples with their masks. The figure should emphasize the difference between large camera cracks and much smaller phone cracks.

## 2.3 Annotation and Mask Generation

The data entry point of the pipeline is the sample-preparation script. It scans the raw dataset, creates a manifest row for each image, and converts polygon annotations into binary masks.

For a crack image \(I_n\), the LabelMe file contains one or more polygons:

$$
\mathcal{P}_n = \{P_{n,1}, P_{n,2}, \ldots, P_{n,K_n}\}.
\tag{3}
$$

The binary ground-truth mask is obtained by rasterizing the union of these polygons:

$$
Y_n(x,y)
=
\begin{cases}
1, & (x,y) \in \bigcup_{k=1}^{K_n} P_{n,k},\\
0, & \text{otherwise}.
\end{cases}
\tag{4}
$$

In the implementation, polygon rasterization is performed with PIL drawing operations and the resulting mask is stored as an 8-bit image, where positive pixels are saved as 255 and background pixels as 0. During dataset loading, this mask is converted to a float tensor with values in \(\{0,1\}\).

For a normal image, no polygon file is required. The dataset loader constructs an all-zero target mask:

$$
Y_n(x,y) = 0
\quad
\forall (x,y)\in\Omega.
\tag{5}
$$

The generated manifest stores the image path, mask path, device source, original folder class, supervised sample type, train/holdout split, and cross-validation fold. The manifest is the single source of truth used by Stage1, Stage2, validation, OOF search, and holdout inference.

**Suggested figure:** show an original ROI image, the LabelMe polygon overlay, and the generated binary mask side by side.

## 2.4 Train/Holdout Split and Cross-Validation

The dataset is split at the image level. The current default split keeps \(20\%\) of labeled crack and normal images as holdout, using a fixed split seed. All `broken` images are also assigned to holdout because they do not have supervised masks.

The resulting split is:

| split | crack | normal | broken | total | role |
| --- | ---: | ---: | ---: | ---: | --- |
| `trainval` | 211 | 59 | 0 | 270 | Stage1/Stage2 training and cross-validation |
| `holdout` | 52 | 15 | 107 | 174 | inference-only or final qualitative inspection |

The `trainval` subset is further divided into four cross-validation folds. For fold \(k\), the validation set is:

$$
\mathcal{D}^{(k)}_{\mathrm{val}}
=
\{(I_n,Y_n): \mathrm{fold}(n)=k\},
\tag{6}
$$

and the training set is:

$$
\mathcal{D}^{(k)}_{\mathrm{train}}
=
\{(I_n,Y_n): \mathrm{fold}(n)\neq k\}.
\tag{7}
$$

The fold composition is:

| fold | train images | train crack | train normal | val images | val crack | val normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 200 | 157 | 43 | 70 | 54 | 16 |
| 1 | 202 | 158 | 44 | 68 | 53 | 15 |
| 2 | 204 | 159 | 45 | 66 | 52 | 14 |
| 3 | 204 | 159 | 45 | 66 | 52 | 14 |

This design allows every trainval image to be predicted once by a model that did not train on that image. These out-of-fold predictions are later pooled for global post-processing selection.

The current fold assignment is an image-level random split. It does not enforce video-level or sequence-level grouping. If the thesis discusses strict video-level generalization, this should be reported as a limitation or future improvement.

## 2.5 Image Loading, Preprocessing, and Augmentation

Images are loaded with PIL and converted to RGB. Masks are loaded as single-channel images and binarized. Both image and mask are resized to the target training size. Image resizing uses bilinear interpolation, while mask resizing uses nearest-neighbor interpolation so that the label remains binary.

The tensor representation is:

$$
X_n \in \mathbb{R}^{3\times H\times W},
\qquad
Y_n \in \{0,1\}^{1\times H\times W}.
\tag{8}
$$

Pixel values are scaled to \([0,1]\). When ImageNet normalization is enabled, the image tensor is normalized channel-wise:

$$
\tilde{X}_{n,c}
=
\frac{X_{n,c}-\mu_c}{\sigma_c},
\tag{9}
$$

where

$$
\mu = [0.485, 0.456, 0.406],
\qquad
\sigma = [0.229, 0.224, 0.225].
\tag{10}
$$

The training pipeline uses synchronized geometric transforms for image and mask, and photometric transforms for image only. Stage1 uses stronger local augmentation because it operates on patches; Stage2 uses milder augmentation because it operates on full ROI images.

| augmentation | Stage1 default | Stage2 default | applied to |
| --- | ---: | ---: | --- |
| horizontal flip | 0.5 | 0.5 | image and mask |
| vertical flip | 0.5 | 0.5 | image and mask |
| rotation | 15 degrees | 10 degrees | image and mask |
| brightness | 0.15 | 0.10 | image only |
| contrast | 0.15 | 0.10 | image only |
| gamma | 0.15 | 0.0 | image only |
| Gaussian noise | 0.02 | 0.015 | image only |
| Gaussian blur | 0.15 | 0.0 | image only |

To reduce repeated disk I/O in Stage1, the patch dataset supports a worker-local cache. Each DataLoader worker can keep already loaded images and masks in memory. This changes only data-loading efficiency; it does not change the sampling distribution, augmentation policy, loss function, or model outputs.

## 2.6 Network Architecture

The segmentation model is a U-Net-style encoder-decoder network with a ResNet34 encoder. The encoder is based on the standard torchvision ResNet34 backbone. Stage1 attempts to initialize the encoder with ImageNet-pretrained weights. Stage2 initializes from the best Stage1 checkpoint of the same fold.

The forward path is:

```text
input image
  -> ResNet stem: conv1 + BatchNorm + ReLU
  -> maxpool + layer1
  -> layer2
  -> layer3
  -> layer4
  -> center convolution block
  -> decoder4 with skip connection from layer3
  -> decoder3 with skip connection from layer2
  -> decoder2 with skip connection from layer1
  -> decoder1 with skip connection from the stem feature map
  -> bilinear upsampling to input resolution
  -> segmentation head
  -> one-channel logit map
```

Each decoder block first upsamples the deeper feature map to the spatial size of the corresponding skip feature. The two feature maps are concatenated along the channel dimension and passed through a convolution block:

$$
D_j
=
\phi_j\left(
\mathrm{Concat}\left(
\mathrm{Upsample}(D_{j+1}), E_j
\right)
\right),
\tag{11}
$$

where \(D_j\) is a decoder feature, \(E_j\) is an encoder skip feature, and \(\phi_j\) is a two-layer convolution block. Each convolution block contains two \(3\times3\) convolutions, each followed by Batch Normalization and ReLU.

The final segmentation head maps the last decoder feature to one logit channel:

$$
Z_n = h(D_1).
\tag{12}
$$

The model outputs logits rather than probabilities. This is important because the binary cross-entropy component of the loss uses `BCEWithLogitsLoss`, which is numerically more stable than applying sigmoid before BCE.

Two optional auxiliary mechanisms were implemented and tested:

| option | mechanism | purpose |
| --- | --- | --- |
| deep supervision | auxiliary segmentation heads on decoder features \(D_4\), \(D_3\), and \(D_2\), each upsampled to input resolution | encourage intermediate decoder features to be segmentation-aware |
| boundary auxiliary loss | an auxiliary boundary head on the final decoder feature | encourage sharper crack boundaries |

When these auxiliary heads are enabled in Stage2, a Stage1 checkpoint without those heads can still be loaded with non-strict weight matching. The shared encoder, decoder, and main segmentation head are reused, while the new auxiliary heads are randomly initialized.

**Suggested figure:** add a U-Net architecture diagram showing the ResNet34 encoder blocks, skip connections, decoder blocks, main segmentation head, and optional auxiliary heads.

## 2.7 Segmentation Loss

The base training objective combines weighted binary cross-entropy and Dice loss:

$$
\mathcal{L}_{\mathrm{seg}}
=
\lambda_{\mathrm{bce}}\mathcal{L}_{\mathrm{bce}}
+
\lambda_{\mathrm{dice}}\mathcal{L}_{\mathrm{dice}}.
\tag{13}
$$

The default weights are:

$$
\lambda_{\mathrm{bce}} = 0.5,
\qquad
\lambda_{\mathrm{dice}} = 0.5.
\tag{14}
$$

### 2.7.1 Positive-Pixel Weighted BCE

Because crack pixels are sparse, positive pixels are up-weighted in the BCE term. For a probability map \(P_n\) and target mask \(Y_n\), the positive-weighted BCE is:

$$
\mathcal{L}_{\mathrm{bce}}
=
-\frac{1}{|\Omega|}
\sum_{(x,y)\in\Omega}
\left[
w_+ Y_n(x,y)\log P_n(x,y)
+
(1-Y_n(x,y))\log(1-P_n(x,y))
\right],
\tag{15}
$$

where \(w_+\) is the positive-pixel weight. The main selected configuration uses:

$$
w_+ = 12.
\tag{16}
$$

This setting increases the cost of missing sparse crack pixels while still allowing normal images to influence the loss through background pixels.

### 2.7.2 Dice Loss

Dice loss directly optimizes region overlap. For one predicted probability map and one target mask, the soft Dice coefficient is:

$$
\mathrm{Dice}(P_n,Y_n)
=
\frac{
2\sum_{(x,y)}P_n(x,y)Y_n(x,y)+\epsilon
}{
\sum_{(x,y)}P_n(x,y)+\sum_{(x,y)}Y_n(x,y)+\epsilon
}.
\tag{17}
$$

The Dice loss is:

$$
\mathcal{L}_{\mathrm{dice}}
=
1-\mathrm{Dice}(P_n,Y_n).
\tag{18}
$$

Dice loss is useful in this task because it is less dominated by the large number of background pixels than plain pixel-wise BCE.

### 2.7.3 Normal False-Positive Loss

An optional normal false-positive loss was implemented for experiments. It applies only to samples with empty masks. Let \(\mathcal{N}\) be the subset of batch samples whose target mask is all zero. For a normal sample \(i\), the model probabilities are flattened into a vector \(p_i\). The top-\(k\) probabilities are averaged:

$$
\mathcal{L}_{\mathrm{nfp}}
=
\frac{1}{|\mathcal{N}|}
\sum_{i\in\mathcal{N}}
\mathrm{mean}\left(
\mathrm{TopK}(p_i, r|\Omega|)
\right),
\tag{19}
$$

where \(r\) is the configured top-k ratio. The total loss becomes:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{seg}}
+
\lambda_{\mathrm{nfp}}\mathcal{L}_{\mathrm{nfp}}.
\tag{20}
$$

This mechanism was tested with \(\lambda_{\mathrm{nfp}}\in\{0.03,0.05\}\). It did not improve the selected baseline, so it is best reported as an ablation rather than the final method.

### 2.7.4 Deep Supervision Loss

When deep supervision is enabled, auxiliary segmentation logits are produced from intermediate decoder features. Each auxiliary logit map is upsampled to the input resolution and trained with the same base segmentation loss:

$$
\mathcal{L}_{\mathrm{aux}}
=
\frac{
\sum_{j=1}^{J}\alpha_j
\mathcal{L}_{\mathrm{seg}}(Z_{n,j}^{\mathrm{aux}},Y_n)
}{
\sum_{j=1}^{J}\alpha_j
},
\tag{21}
$$

where

$$
\alpha_j = \gamma^{j-1}.
\tag{22}
$$

The tested setting used a deep-supervision weight of 0.20 and decay \(\gamma=0.50\). The full loss is:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{seg}}
+
\lambda_{\mathrm{ds}}\mathcal{L}_{\mathrm{aux}}.
\tag{23}
$$

The auxiliary losses do not include the normal false-positive loss. This avoids over-penalizing normal images through multiple auxiliary outputs.

### 2.7.5 Boundary Auxiliary Loss

The boundary auxiliary branch predicts a binary boundary map. The boundary target is generated from the ground-truth mask using dilation and erosion:

$$
B_n
=
\mathrm{Dilate}(Y_n) - \mathrm{Erode}(Y_n).
\tag{24}
$$

The boundary loss is a BCE-with-logits loss:

$$
\mathcal{L}_{\mathrm{boundary}}
=
\mathrm{BCEWithLogits}(Z_n^{\mathrm{boundary}},B_n).
\tag{25}
$$

The total loss becomes:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{seg}}
+
\lambda_{\mathrm{boundary}}\mathcal{L}_{\mathrm{boundary}}.
\tag{26}
$$

This branch was tested because cracks are thin and boundary quality may influence segmentation quality. In the current experiments it did not outperform the baseline.

## 2.8 Stage1: Patch-Level Representation Learning

Stage1 addresses the main data imbalance problem: if the model is trained only on full \(640\times640\) images, many batches contain very few positive pixels. Stage1 therefore trains the same U-Net architecture on local patches extracted from the trainval images.

The Stage1 patch index does not store cropped patch images. Instead, it stores crop metadata:

```text
image_path, mask_path, crop_x, crop_y, crop_size, out_size, patch_type
```

At training time, the dataset dynamically crops the image and mask according to this metadata, resizes the crop to the Stage1 output size, applies augmentation, and returns the patch tensor.

The main Stage1 patch types are:

| patch type | role |
| --- | --- |
| `positive_center` | crop centered near a crack component |
| `positive_shift` | crop shifted while still containing the defect bounding box |
| `positive_context` | larger contextual crop containing the defect |
| `positive_boundary` | crop where the defect is near a patch edge |
| `near_miss_negative` | crop near a defect boundary but without positive pixels |
| `hard_negative` | background crop from a defect image, away from the defect |
| `normal_negative` | crop from a normal image |

This patch design gives the model three types of information:

1. what cracks look like at local scale,
2. what background near cracks looks like,
3. what fully normal background looks like.

The default Stage1 patch output size is \(384\times384\). Crop sizes include 320, 384, and 448 pixels, which introduces scale variation before resizing.

### 2.8.1 Balanced Patch Sampling

The raw patch index is not sampled uniformly. Stage1 uses a weighted sampler so that patch families appear approximately in configured proportions. Let \(g(i)\) be the patch family of patch \(i\), and let \(\rho_g\) be the desired sampling ratio for family \(g\). The sampling probability is proportional to:

$$
\Pr(i)
\propto
\frac{\rho_{g(i)}}{N_{g(i)}},
\tag{27}
$$

where \(N_{g(i)}\) is the number of patches in the same family. The default family ratios are:

```text
positive patches:        0.50
defect-negative patches: 0.25
normal-negative patches: 0.25
```

This sampler prevents the training batches from being dominated by easy negative patches and makes the model see crack pixels more frequently.

### 2.8.2 Encoder Warm-Up Freezing

At the beginning of Stage1, the encoder is frozen for a small number of epochs. During this period, the decoder and segmentation head adapt to the segmentation task while the pretrained encoder remains stable. When the encoder is frozen, encoder BatchNorm modules are put into evaluation mode so that their running means and variances are not updated by the patch distribution.

This detail matters because a frozen encoder with still-updating BatchNorm statistics is only partially frozen. The current implementation freezes both encoder parameters and encoder BatchNorm running statistics.

### 2.8.3 Stage1 Checkpoint Selection

Stage1 validation is computed on validation patches. The checkpoint criterion balances positive patch segmentation quality against negative patch false positives. The main metrics are:

| metric | meaning |
| --- | --- |
| `patch_dice_all` | mean Dice across all validation patches |
| `patch_dice_pos_only` | mean Dice across positive validation patches |
| `positive_patch_recall` | fraction of positive patches with any predicted positive pixel |
| `negative_patch_fpr` | fraction of negative patches with any predicted positive pixel |

The Stage1 score is:

$$
S_{\mathrm{stage1}}
=
\mathrm{Dice}_{\mathrm{pos}}
-
\lambda_{\mathrm{neg}}
\max(0,\mathrm{FPR}_{\mathrm{neg}}-\tau_{\mathrm{neg}}).
\tag{28}
$$

The default target negative FPR is 0.20, and the default penalty weight is 0.5. This prevents the Stage1 checkpoint from becoming a high-recall model that produces excessive positives on negative patches.

## 2.9 Stage1 Hard Patch Replay

Stage1 replay periodically mines patches that are difficult for the current model. The purpose is to make the model revisit false negatives and false positives rather than relying only on the static patch index.

After a warm-up period, the current Stage1 model evaluates the base patch index. The probability map is thresholded at 0.50 and compared with the patch target. Difficult samples are assigned replay scores.

For positive patches, the replay score is based on low overlap:

$$
R_i^{+}
=
1-\mathrm{Dice}_i
+
\mathbb{1}[\text{positive patch is completely missed}].
\tag{29}
$$

For negative patches, the score is based on the amount and confidence of false-positive prediction:

$$
R_i^{-}
=
\mathrm{PredictedPositiveRatio}_i
+
\max_{(x,y)\in\Omega_i}P_i(x,y).
\tag{30}
$$

Top-scoring replay rows are appended to the training data for later epochs. Replay rows are marked with their replay type, source epoch, and replay score. The replay mechanism is refreshed periodically rather than rebuilt at every iteration, which keeps the computation manageable.

The replay types are:

| replay type | meaning |
| --- | --- |
| `replay_positive_fn` | a positive patch that was completely missed |
| `replay_positive_hard` | a positive patch with imperfect segmentation |
| `replay_defect_negative_fp` | a false positive on a negative patch from a defect image |
| `replay_normal_negative_fp` | a false positive on a normal negative patch |

## 2.10 Stage2: Full-Image Fine-Tuning

Stage2 transfers the Stage1 representation back to the full ROI segmentation task. For each cross-validation fold, the Stage2 model loads the best Stage1 checkpoint from the same fold and continues training on full images.

The selected baseline Stage2 configuration uses:

```text
image size:                  640
batch size:                  48
epochs:                      50
positive pixel weight:       12
normal_fp_loss_weight:       0
random_normal_k_factor:      1.0
hard normal replay:          enabled
hard normal ratio:           0.40
hard normal max repeats:     2
post-processing search end:  threshold 0.95
```

For fold \(k\), the Stage2 training set contains all defect training images and a sampled set of normal images. Let \(\mathcal{D}_{+}^{(k)}\) be the defect training images and \(\mathcal{D}_{0}^{(k)}\) be the normal training images. At epoch \(e\), the training rows are:

$$
\mathcal{E}_e^{(k)}
=
\mathcal{D}_{+}^{(k)}
\cup
\mathcal{N}_{e,\mathrm{rand}}^{(k)}
\cup
\mathcal{N}_{e,\mathrm{hard}}^{(k)}.
\tag{31}
$$

The random-normal budget is:

$$
|\mathcal{N}_{e,\mathrm{rand}}^{(k)}|
\approx
\mathrm{round}
\left(
|\mathcal{D}_{+}^{(k)}| \cdot \kappa
\right),
\tag{32}
$$

where \(\kappa\) is `random_normal_k_factor`. The selected configuration uses \(\kappa=1.0\), so the number of normal training images per epoch is approximately matched to the number of defect images.

The reason for this design is that full-image training must learn both dense crack segmentation and image-level normal suppression. Using all defect images keeps positive supervision stable. Sampling normal images prevents the model from ignoring the normal class and reduces false positives.

## 2.11 Stage2 Hard-Normal Replay

Hard-normal replay is the Stage2 mechanism for controlling false positives on normal images. It periodically scans normal training images with the current Stage2 model. If a normal image produces positive pixels after post-processing, it is added to a hard-normal pool and can be replayed in subsequent epochs.

The mining flow is:

```text
normal training image
  -> current Stage2 model
  -> sigmoid probability map
  -> threshold and min-area post-processing
  -> false-positive connected components
  -> hard-normal score
  -> ranked hard-normal pool
```

The hard-normal score prioritizes large false-positive connected components:

$$
R_{\mathrm{hard}}
=
A_{\max}\cdot 10^6
+
N_{\mathrm{fp}}
+
P_{\max},
\tag{33}
$$

where \(A_{\max}\) is the largest false-positive component area, \(N_{\mathrm{fp}}\) is the total number of false-positive pixels, and \(P_{\max}\) is the maximum predicted probability. The large multiplier makes the largest connected component the dominant ranking term.

The hard-normal replay ratio controls the fraction of the normal budget filled by hard normals. If the hard pool is small, naive sampling can repeat the same hard normal image many times in one epoch. To prevent this, the current implementation uses a per-epoch repeat cap:

$$
M_{\mathrm{hard}}
=
\min
\left(
M_{\mathrm{requested}},
|\mathcal{H}|\cdot c_{\max}
\right),
\tag{34}
$$

where \(|\mathcal{H}|\) is the hard-normal pool size and \(c_{\max}\) is `hard_normal_max_repeats_per_epoch`. The selected setting uses \(c_{\max}=2\). This keeps the hard-normal replay useful without letting a tiny hard pool dominate the epoch.

## 2.12 Validation, Checkpoint Selection, and Metrics

Each Stage2 epoch is evaluated on the validation images of the current fold. During training, checkpoint selection uses a fixed train-time threshold and min-area setting rather than searching threshold/min-area at every epoch. This separates model selection from final post-processing selection.

The training-time validation flow is:

```text
validation images
  -> model logits
  -> sigmoid probability maps
  -> fixed threshold/min_area
  -> binary masks
  -> per-image metrics
  -> fold-level metrics
  -> checkpoint comparison
```

The main segmentation metrics are Dice and IoU. For a binary prediction \(\hat{Y}\) and target \(Y\):

$$
\mathrm{Dice}(\hat{Y},Y)
=
\frac{2|\hat{Y}\cap Y|+\epsilon}
{|\hat{Y}|+|Y|+\epsilon},
\tag{35}
$$

$$
\mathrm{IoU}(\hat{Y},Y)
=
\frac{|\hat{Y}\cap Y|+\epsilon}
{|\hat{Y}\cup Y|+\epsilon}.
\tag{36}
$$

For defect images, image-level recall is:

$$
\mathrm{Recall}_{\mathrm{defect}}
=
\frac{1}{N_{+}}
\sum_{n\in\mathcal{D}_{+}}
\mathbb{1}
\left[
|\hat{Y}_n|>0
\right].
\tag{37}
$$

For normal images, normal false-positive rate is:

$$
\mathrm{FPR}_{\mathrm{normal}}
=
\frac{1}{N_{0}}
\sum_{n\in\mathcal{D}_{0}}
\mathbb{1}
\left[
|\hat{Y}_n|>0
\right].
\tag{38}
$$

The Stage2 checkpoint score is:

$$
S_{\mathrm{stage2}}
=
\mathrm{Dice}_{\mathrm{defect}}
-
\lambda_{\mathrm{fpr}}
\max
\left(
0,
\mathrm{FPR}_{\mathrm{normal}}-\tau_{\mathrm{fpr}}
\right).
\tag{39}
$$

The default target normal FPR is:

$$
\tau_{\mathrm{fpr}}=0.10,
\tag{40}
$$

and the default FPR penalty weight is:

$$
\lambda_{\mathrm{fpr}}=2.0.
\tag{41}
$$

The checkpoint comparison rule is lexicographic:

1. Prefer checkpoints with normal FPR below or equal to the target.
2. Among acceptable checkpoints, prefer higher defect Dice.
3. If defect Dice ties, prefer lower normal FPR.
4. If both tie, prefer higher defect-image recall.
5. If no checkpoint satisfies the normal FPR target, compare the penalized Stage2 score.

This selection rule reflects the practical requirement of the task: false positives on normal images should be controlled first, and segmentation quality should then be maximized among stable checkpoints.

## 2.13 Post-Processing

The model produces probability maps, but the final output is a binary crack mask. Post-processing has two steps:

1. threshold the probability map,
2. remove connected components smaller than a minimum area.

For threshold \(\tau\), the raw binary mask is:

$$
B_{\tau,n}(x,y)
=
\mathbb{1}
\left[
P_n(x,y)\geq\tau
\right].
\tag{42}
$$

Connected components are computed with 8-neighborhood connectivity. Let \(\mathcal{C}(B_{\tau,n})\) be the connected components of the raw mask. The final mask is:

$$
\hat{Y}_{\tau,a,n}
=
\bigcup_{C\in\mathcal{C}(B_{\tau,n})}
C\cdot
\mathbb{1}
\left[
|C|\geq a
\right],
\tag{43}
$$

where \(a\) is the `min_area` parameter.

The search grid used in the selected experiments is:

```text
threshold start: 0.10
threshold end:   0.95
threshold step:  0.02
min_area grid:   [0, 8, 16, 24, 32, 48]
```

The two parameters have different roles. The threshold controls pixel-level confidence. The min-area filter controls small isolated components. A high threshold can suppress weak predictions but may remove faint cracks. A large min-area can remove noise but may also remove small real cracks, especially in the phone subset. Therefore, these parameters are selected on validation predictions rather than set manually.

**Suggested figure:** show a probability heatmap, thresholded mask before area filtering, final mask after connected-component filtering, and ground truth.

## 2.14 Per-Fold Validation Export

After Stage2 training, each fold exports validation predictions and post-processing search results. The exported artifacts include:

| file | content |
| --- | --- |
| `val_metrics.json` | best per-fold post-processing parameters and summary metrics |
| `val_per_image.csv` | per-image Dice, IoU, false-positive pixels, and largest false-positive component |
| `val_postprocess_search.csv` | metrics for all threshold/min-area combinations |

The per-image table is important because the aggregate Dice alone does not explain model behavior. For example, a model may have high defect Dice but still produce a few unacceptable normal false positives. Conversely, a model may be very conservative on normal images but miss small cracks. Per-image error rows make it possible to identify false positives, false negatives, and worst-case defect images.

## 2.15 Pooled Out-of-Fold Global Post-Processing

The final post-processing parameters are selected using pooled out-of-fold validation predictions. This is one of the most important parts of the methodology because it prevents each fold from using its own locally optimal threshold.

For each fold \(k\), the model trained without fold \(k\) predicts the validation images in fold \(k\). Thus, every trainval image receives exactly one prediction from a model that did not train on it:

$$
\mathcal{P}_{\mathrm{oof}}
=
\bigcup_{k=0}^{3}
\left\{
P_n^{(k)}
\mid
n\in\mathcal{D}^{(k)}_{\mathrm{val}}
\right\}.
\tag{44}
$$

The global post-processing search evaluates all candidate threshold/min-area pairs on this pooled set:

$$
(\tau^\star,a^\star)
=
\arg\max_{(\tau,a)\in\mathcal{G}}
S_{\mathrm{stage2}}
\left(
\hat{Y}_{\tau,a},
Y
\right),
\tag{45}
$$

where \(\mathcal{G}\) is the threshold/min-area grid and \(S_{\mathrm{stage2}}\) is computed from pooled defect Dice and normal FPR. The same lexicographic comparison rule used for Stage2 checkpoint selection is applied: acceptable normal FPR is prioritized, then defect Dice, then lower normal FPR, then defect-image recall.

The pooled OOF output files are:

| file | content |
| --- | --- |
| `oof_global_postprocess.json` | selected global threshold/min-area and pooled metrics |
| `oof_global_postprocess_search.csv` | all grid-search rows |
| `oof_per_image.csv` | per-image OOF prediction metrics |

For thesis reporting, pooled OOF is preferred over a single fold result because it summarizes all trainval images without training on the evaluated image. It also avoids selecting a favorable fold by chance.

## 2.16 Holdout Inference

After Stage2 models and global post-processing settings have been selected, holdout inference can be run on the held-out images. The holdout set contains:

1. held-out crack images,
2. held-out normal images,
3. unlabeled broken images.

For held-out crack and normal images, supervised evaluation is possible if their labels are available in the manifest. For broken images, the current pipeline can only generate qualitative predictions because no ground-truth segmentation masks are available.

The inference output may include:

| output | purpose |
| --- | --- |
| probability maps | inspect raw model confidence |
| binary masks | final predicted crack regions |
| overlays | qualitative visualization on the original image |
| summary CSV/JSON | track image-level prediction behavior |

It is important not to mix unlabeled broken predictions into Dice or IoU tables. They should be described as qualitative inference results unless pixel-level annotations are created.

## 2.17 Error Analysis Visualization

A dedicated error-analysis script renders validation examples into interpretable panels. Each panel can contain:

1. the original ROI image,
2. the ground-truth mask,
3. the predicted mask,
4. prediction/ground-truth overlay,
5. probability heatmap.

The visualization can group images into false positives, false negatives, and low-Dice defect samples. This is useful for explaining failure modes in the thesis. For this task, expected failure modes include:

| failure mode | likely cause |
| --- | --- |
| missed very small phone cracks | extremely low positive-pixel ratio |
| fragmented crack predictions | thin structures and imperfect boundary confidence |
| isolated normal false positives | local background texture similar to crack pixels |
| threshold-sensitive predictions | probability values near the selected threshold |

Error analysis is not only a presentation step. It also motivates future improvements such as additional labeling, stricter split design, targeted augmentation for small cracks, and improved boundary annotation quality.

## 2.18 Experimental Design and Ablations

The experiments are organized around the main components of the proposed method. The baseline is the two-stage ResNet34-U-Net with Stage1 patch training, Stage2 full-image fine-tuning, hard-normal replay, and global OOF post-processing.

The main ablation groups are:

| ablation | purpose |
| --- | --- |
| positive-pixel weight | test how strongly sparse crack pixels should be weighted in BCE |
| normal false-positive loss | test whether an explicit loss on normal-image probabilities improves false-positive suppression |
| deep supervision | test whether auxiliary decoder heads improve segmentation learning |
| boundary auxiliary loss | test whether explicit boundary supervision helps thin crack structures |
| PatchDataset cache | test runtime/data-loading optimization without changing model behavior |
| threshold/min-area range | confirm whether the global post-processing optimum remains stable when the threshold search is extended |

The best balanced U-Net setting in the completed experiments is the Stage2 `pos_weight=12` baseline. It achieved strong pooled OOF segmentation quality with near-perfect defect-image recall and zero normal false-positive rate. Later auxiliary mechanisms did not consistently improve the balanced objective. Therefore, the recommended thesis story is not that every added mechanism improved the result. The stronger and more credible story is:

1. the two-stage design is motivated by sparse positive pixels,
2. hard-normal replay and global OOF post-processing stabilize false-positive control,
3. `pos_weight=12` is the best selected loss weighting,
4. additional auxiliary losses were tested but did not outperform the simpler baseline.

This is a rigorous ablation narrative because it reports both positive and negative findings.

## 2.19 Reproducibility and Implementation Notes

The pipeline uses fixed seeds for the dataset split and cross-validation fold assignment. Stage1 augmentation uses a different DataLoader seed each epoch:

$$
s_e = s_0 + 1009e,
\tag{46}
$$

where \(s_0\) is the base seed and \(e\) is the epoch index. This avoids repeating the exact same augmentation sequence across epochs.

The main implementation choices that affect reproducibility are:

| mechanism | reproducibility role |
| --- | --- |
| fixed train/holdout seed | makes the data split repeatable |
| fixed fold assignment | makes cross-validation repeatable |
| per-epoch Stage1 DataLoader seed | avoids repeated augmentation patterns |
| fixed train-time Stage2 threshold/min-area | makes checkpoint selection stable |
| global OOF threshold/min-area search | avoids fold-specific post-processing bias |
| hard-normal replay cap | prevents unstable over-repetition when the hard pool is small |
| frozen encoder BatchNorm stats | makes encoder-freeze behavior well-defined |

The core implementation files are:

| file | role |
| --- | --- |
| `scripts/prepare_samples.py` | build manifest and generated masks |
| `scripts/build_patch_index.py` | construct Stage1 patch rows |
| `scripts/train_stage1.py` | Stage1 patch training |
| `scripts/train_stage2.py` | Stage2 full-image training |
| `scripts/evaluate_val.py` | per-fold validation export |
| `scripts/search_oof_postprocess.py` | pooled OOF global post-processing search |
| `scripts/infer_holdout.py` | holdout inference |
| `scripts/visualize_error_analysis.py` | qualitative validation error visualization |
| `src/datasets.py` | dataset loading, transforms, patch cache |
| `src/model.py` | ResNet34-U-Net model and optional heads |
| `src/losses.py` | BCE-Dice, normal FP, deep supervision, boundary loss |
| `src/metrics.py` | Dice, IoU, connected components, post-processing search |

## Suggested Figures for the Thesis

1. **Overall U-Net pipeline.** Raw ROI images and annotations flow into manifest/masks, Stage1 patch training, Stage2 full-image training, OOF post-processing search, and final predictions.
2. **Dataset and mask examples.** Camera and phone crack examples with ground-truth masks, showing the difference in crack size and sparsity.
3. **Stage1 patch construction.** Examples of positive center, positive shift, positive context, positive boundary, near-miss negative, hard negative, and normal negative patches.
4. **Network architecture.** ResNet34 encoder, U-Net decoder, skip connections, segmentation head, and optional auxiliary heads.
5. **Hard-normal replay.** Normal image false-positive prediction, connected-component scoring, hard-normal pool, and replay into later Stage2 epochs.
6. **Post-processing search.** Probability map, thresholded mask, connected-component filtering, and selected global threshold/min-area.
7. **Pooled OOF validation.** Four fold models predicting their held-out folds, followed by one global post-processing search.
8. **Error analysis panel.** Original image, ground truth, prediction, overlay, and probability heatmap for representative true positives, false positives, and false negatives.

## Thesis-Ready Method Summary

The proposed method uses a two-stage ResNet34-U-Net pipeline for sparse crack segmentation in ROI images. First, LabelMe polygon annotations are converted into binary masks and indexed in a sample manifest. Labeled crack and normal images are split into trainval and holdout subsets, while unlabeled broken images are kept for qualitative holdout inference. Within trainval, four-fold cross-validation is used.

Stage1 trains the U-Net on dynamically cropped local patches. The patch index includes positive crack patches, near-crack negatives, background negatives from defect images, and normal-image negatives. A balanced sampler increases the frequency of positive and difficult negative patches. Stage1 also uses hard patch replay, where missed positive patches and false-positive negative patches are mined and replayed in later epochs.

Stage2 initializes from the best Stage1 checkpoint of the same fold and fine-tunes the model on full \(640\times640\) ROI images. Each epoch uses all defect training images and a sampled set of normal images. During training, hard-normal replay periodically mines normal images that produce false positives and reuses them in subsequent epochs, with a repeat cap to avoid over-sampling a small hard pool.

The model is trained with a BCE-Dice loss. Positive pixels are weighted in BCE to counter the extreme sparsity of crack masks. Validation computes defect Dice, defect IoU, defect-image recall, and normal-image false-positive rate. Stage2 checkpoint selection uses a fixed train-time threshold and min-area filter, so that model selection is separated from final post-processing tuning.

After all folds are trained, out-of-fold validation predictions are pooled. A single global threshold and connected-component minimum-area setting is selected on this pooled set. This produces a unified post-processing configuration for reporting and inference, and avoids using a different locally optimized threshold for each fold.
