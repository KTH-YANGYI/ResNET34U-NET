# Two-Page Methodology Version

This document is a condensed thesis-ready version of the full U-Net two-stage methodology. It is intended to fit as one subsection or a short part of the full Methodology chapter. The detailed implementation, ablations, and artifact descriptions remain in `THESIS_METHODOLOGY.md`.

## Sparse Crack Segmentation Methodology

This work addresses binary semantic segmentation of crack-like defects in cropped railway contact-wire ROI images. Given an input image \(I_n\), the model predicts a pixel-wise crack probability map \(P_n\), which is later converted into a binary mask \(\hat{Y}_n\). The task is challenging because the crack class is extremely sparse: in the current dataset, positive pixels occupy only a small fraction of each \(640\times640\) ROI, and some phone images contain very thin defects. At the same time, false positives on normal samples are costly, because visually normal contact-wire regions should produce empty masks. The proposed method therefore combines local patch-level representation learning, full-image fine-tuning, hard-negative replay, and pooled out-of-fold post-processing selection.

The dataset contains two acquisition sources, `camera` and `phone`, and three folders: `crack`, `normal`, and `broken`. Images in `crack` have LabelMe polygon annotations that are rasterized into binary masks; images in `normal` are assigned all-zero masks; images in `broken` are treated as unlabeled holdout images and are not used for supervised training or validation metrics. The labeled crack and normal images are split into a trainval subset and a holdout subset. The trainval subset is further divided into four cross-validation folds, so that each trainval image can later receive exactly one out-of-fold prediction from a model that did not train on that image.

The segmentation network is a U-Net-style encoder-decoder model with a ResNet34 encoder. U-Net is used because its skip-connected decoder preserves localization information for dense segmentation, while the ResNet encoder provides a strong residual feature extractor initialized from ImageNet-pretrained weights. For an input image, the model outputs a one-channel logit map

$$
Z_n = f_\theta(I_n),
\qquad
P_n(x,y)=\sigma(Z_n(x,y)).
$$

The final binary mask is not taken directly from the probability map. Instead, thresholding and connected-component filtering are applied after validation-based post-processing selection. This separation is important: the network learns dense probabilities, while the final system must output discrete crack regions with small isolated false positives removed.

Training is divided into two stages. Stage1 is patch-level representation learning. Full \(640\times640\) images often contain very few positive pixels, so a model trained only on full images may see insufficient crack signal in each batch. To address this, Stage1 constructs a patch index from the training images. The index stores crop metadata rather than physical cropped files. At training time, patches are dynamically cropped, resized, augmented, and loaded. Patch types include crack-centered positive patches, shifted positive patches, positive context patches, boundary patches, near-miss negatives, background negatives from crack images, and normal-image negatives. A balanced sampler gives positive and difficult negative patches higher exposure than they would receive under uniform sampling.

Stage1 also includes hard patch replay. After a warm-up period, the current model evaluates the patch index. Positive patches with low Dice or complete misses are mined as hard positive examples, while negative patches with false-positive predictions are mined as hard negatives. These replay rows are appended to later epochs, forcing the model to revisit mistakes instead of repeatedly training only on the static patch distribution.

Stage2 transfers the Stage1 representation to full-image training. For each fold, Stage2 initializes from the best Stage1 checkpoint of the same fold and fine-tunes on full \(640\times640\) ROI images. Each epoch includes all defect training images and a sampled set of normal images. This design preserves positive supervision while ensuring that the model learns image-level normal suppression. During Stage2, hard-normal replay periodically scans normal training images. If a normal image produces connected false-positive components after thresholding and area filtering, it is added to a hard-normal pool. The pool is sampled in later epochs, but a per-epoch repeat cap prevents a small hard-normal pool from dominating training.

The main loss combines positive-weighted binary cross-entropy and Dice loss:

$$
\mathcal{L}_{\mathrm{seg}}
=
\lambda_{\mathrm{bce}}\mathcal{L}_{\mathrm{bce}}
+
\lambda_{\mathrm{dice}}\mathcal{L}_{\mathrm{dice}},
\qquad
\lambda_{\mathrm{bce}}=\lambda_{\mathrm{dice}}=0.5.
$$

Positive pixels are up-weighted in the BCE term to compensate for the sparse crack class. In the selected configuration, the positive-pixel weight is \(w_+=12\). Dice loss is included because it directly optimizes foreground-background overlap and is less dominated by the large number of background pixels. Several auxiliary mechanisms were also tested, including normal false-positive loss, deep supervision, and boundary auxiliary loss. These were useful ablations, but the best balanced setting remained the simpler two-stage baseline with \(w_+=12\), hard-normal replay, and global post-processing selection.

Validation is performed at each Stage2 epoch on the held-out fold. Checkpoint selection uses a fixed training-time threshold and minimum-area setting rather than searching post-processing parameters at every epoch. This avoids mixing model selection with post-processing tuning. The fold-level validation metrics include defect Dice, defect IoU, defect-image recall, and normal false-positive rate. The selection rule first prefers checkpoints that satisfy the normal false-positive target; among acceptable checkpoints, it prefers higher defect Dice, then lower normal false-positive rate, and finally higher defect-image recall. This reflects the practical objective of the task: normal images should remain clean, while crack segmentation quality is maximized within that constraint.

After all four folds are trained, validation predictions are pooled into a single out-of-fold set:

$$
\mathcal{P}_{\mathrm{oof}}
=
\bigcup_{k=0}^{3}
\{P_n^{(k)} \mid n\in\mathcal{D}^{(k)}_{\mathrm{val}}\}.
$$

A global search is then performed over threshold \(\tau\) and connected-component minimum area \(a\). For a candidate threshold, the probability map is binarized as

$$
B_{\tau,n}(x,y)=\mathbb{1}[P_n(x,y)\geq\tau],
$$

and connected components smaller than \(a\) pixels are removed:

$$
\hat{Y}_{\tau,a,n}
=
\bigcup_{C\in\mathcal{C}(B_{\tau,n})}
C\cdot\mathbb{1}[|C|\geq a].
$$

The selected pair \((\tau^\star,a^\star)\) is the one that gives the best pooled OOF validation score under the same false-positive-aware selection rule used during checkpoint selection. This global OOF strategy is central to the methodology: every image is evaluated by a model that did not train on it, and the final post-processing setting is shared across folds rather than tuned separately for each fold. As a result, the reported validation performance better reflects the expected behavior of one unified system.

The final pipeline can therefore be summarized as follows. LabelMe polygons are converted to binary masks, labeled crack and normal images are split into trainval and holdout sets, and four-fold cross-validation is used within trainval. Stage1 trains a ResNet34-U-Net on balanced local patches to expose the model to sparse crack pixels and difficult local backgrounds. Stage2 fine-tunes the same model on full ROI images and uses hard-normal replay to suppress normal false positives. Finally, pooled out-of-fold predictions are used to select one global threshold and connected-component minimum area. This produces a segmentation system that is designed not only for high overlap on crack images, but also for stable behavior on normal images.

## Citation Pointers

Use the full bibliography in `THESIS_METHODOLOGY.md`. The most important citations for this shortened section are: crack segmentation review, U-Net, ResNet, LabelMe, ImageNet, BatchNorm, BCEWithLogitsLoss, Dice, hard example mining, focal loss, Jaccard/IoU, and connected-component image processing.
