import numpy as np
import torch


def to_numpy_array(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()

    return np.asarray(data)


def logits_to_probs(logits):
    """
    把 logits 转成 0~1 概率图。
    """

    if torch.is_tensor(logits):
        return torch.sigmoid(logits).detach().cpu().numpy()

    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits))


def probs_to_binary_mask(probs, threshold=0.5):
    """
    固定阈值二值化。
    这里不做任何连通域过滤。
    """

    probs = to_numpy_array(probs).astype(np.float32)

    while probs.ndim > 2:
        probs = probs.squeeze(0)

    return (probs >= float(threshold)).astype(np.uint8)


def _to_binary_mask(mask):
    mask = np.asarray(mask).astype(np.uint8)

    while mask.ndim > 2:
        mask = mask.squeeze(0)

    return (mask > 0).astype(np.uint8)


def dice_score(pred, target, eps=1e-6):
    pred = _to_binary_mask(pred)
    target = _to_binary_mask(target)

    intersection = int((pred * target).sum())
    pred_sum = int(pred.sum())
    target_sum = int(target.sum())

    if pred_sum == 0 and target_sum == 0:
        return 1.0

    return float((2.0 * intersection + eps) / (pred_sum + target_sum + eps))


def iou_score(pred, target, eps=1e-6):
    pred = _to_binary_mask(pred)
    target = _to_binary_mask(target)

    intersection = int((pred * target).sum())
    union = int(((pred + target) > 0).sum())

    if union == 0:
        return 1.0

    return float((intersection + eps) / (union + eps))


def compute_defect_seg_metrics(pred_masks, gt_masks):
    dice_values = []
    iou_values = []

    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        dice_values.append(dice_score(pred_mask, gt_mask))
        iou_values.append(iou_score(pred_mask, gt_mask))

    if len(dice_values) == 0:
        return {
            "defect_dice": 0.0,
            "defect_iou": 0.0,
            "defect_count": 0,
        }

    return {
        "defect_dice": float(np.mean(dice_values)),
        "defect_iou": float(np.mean(iou_values)),
        "defect_count": len(dice_values),
    }


def compute_normal_fpr(pred_masks):
    if len(pred_masks) == 0:
        return 0.0

    false_positive_count = 0

    for pred_mask in pred_masks:
        pred_mask = _to_binary_mask(pred_mask)
        if int(pred_mask.sum()) > 0:
            false_positive_count += 1

    return float(false_positive_count / len(pred_masks))


def compute_defect_image_recall(pred_masks):
    if len(pred_masks) == 0:
        return 0.0

    recalled_count = 0

    for pred_mask in pred_masks:
        pred_mask = _to_binary_mask(pred_mask)
        if int(pred_mask.sum()) > 0:
            recalled_count += 1

    return float(recalled_count / len(pred_masks))


def summarize_metrics(per_image_rows):
    defect_dice_values = []
    defect_iou_values = []
    defect_recall_hits = []
    normal_false_positive_hits = []

    for row in per_image_rows:
        if row["is_defect_image"]:
            defect_dice_values.append(float(row["dice"]))
            defect_iou_values.append(float(row["iou"]))
            defect_recall_hits.append(1.0 if row["pred_has_positive"] else 0.0)
        else:
            normal_false_positive_hits.append(1.0 if row["pred_has_positive"] else 0.0)

    if len(defect_dice_values) > 0:
        defect_dice = float(np.mean(defect_dice_values))
        defect_iou = float(np.mean(defect_iou_values))
        defect_image_recall = float(np.mean(defect_recall_hits))
    else:
        defect_dice = 0.0
        defect_iou = 0.0
        defect_image_recall = 0.0

    if len(normal_false_positive_hits) > 0:
        normal_fpr = float(np.mean(normal_false_positive_hits))
    else:
        normal_fpr = 0.0

    return {
        "defect_dice": defect_dice,
        "defect_iou": defect_iou,
        "defect_image_recall": defect_image_recall,
        "normal_fpr": normal_fpr,
        "defect_count": len(defect_dice_values),
        "normal_count": len(normal_false_positive_hits),
    }


def _is_defect_sample(sample_type, gt_mask):
    sample_type = str(sample_type)
    gt_mask = _to_binary_mask(gt_mask)

    if sample_type.startswith("normal"):
        return False

    if int(gt_mask.sum()) > 0:
        return True

    return not sample_type.startswith("normal")


def evaluate_prob_maps(prob_maps, gt_masks, sample_types, threshold=0.5, image_names=None):
    """
    用固定阈值把概率图转成二值 mask，然后直接计算指标。
    """

    if image_names is None:
        image_names = [f"sample_{index:05d}" for index in range(len(prob_maps))]

    pred_masks = []
    per_image_rows = []

    for prob_map, gt_mask, sample_type, image_name in zip(
        prob_maps,
        gt_masks,
        sample_types,
        image_names,
    ):
        pred_mask = probs_to_binary_mask(prob_map, threshold)
        gt_mask = _to_binary_mask(gt_mask)

        is_defect_image = _is_defect_sample(sample_type, gt_mask)
        pred_has_positive = int(pred_mask.sum()) > 0

        if is_defect_image:
            dice_value = dice_score(pred_mask, gt_mask)
            iou_value = iou_score(pred_mask, gt_mask)
        else:
            dice_value = 0.0
            iou_value = 0.0

        per_image_rows.append(
            {
                "image_name": image_name,
                "sample_type": sample_type,
                "is_defect_image": is_defect_image,
                "pred_has_positive": pred_has_positive,
                "dice": dice_value,
                "iou": iou_value,
                "threshold": float(threshold),
            }
        )

        pred_masks.append(pred_mask)

    metrics = summarize_metrics(per_image_rows)

    return {
        **metrics,
        "threshold": float(threshold),
        "per_image_rows": per_image_rows,
        "pred_masks": pred_masks,
    }
