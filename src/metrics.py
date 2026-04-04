import cv2
import numpy as np
import torch


def to_numpy_array(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()

    return np.asarray(data)


def logits_to_probs(logits):
    if torch.is_tensor(logits):
        return torch.sigmoid(logits).detach().cpu().numpy()

    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits))


def _squeeze_to_hw(data):
    data = to_numpy_array(data)

    while data.ndim > 2:
        data = data.squeeze(0)

    return data


def _to_binary_mask(mask):
    return (_squeeze_to_hw(mask) > 0).astype(np.uint8)


def connected_component_stats(binary_mask):
    binary_mask = _to_binary_mask(binary_mask)

    if int(binary_mask.sum()) == 0:
        return []

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    results = []

    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        results.append({"area": area})

    return results


def largest_component_area(binary_mask):
    stats = connected_component_stats(binary_mask)
    if len(stats) == 0:
        return 0
    return int(max(item["area"] for item in stats))


def filter_small_components(binary_mask, min_area=0):
    binary_mask = _to_binary_mask(binary_mask)
    min_area = int(min_area)

    if min_area <= 0:
        return binary_mask

    if int(binary_mask.sum()) == 0:
        return binary_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered_mask[labels == label_index] = 1

    return filtered_mask


def probs_to_binary_mask(probs, threshold=0.5, min_area=0):
    probs = _squeeze_to_hw(probs).astype(np.float32)
    binary_mask = (probs >= float(threshold)).astype(np.uint8)
    return filter_small_components(binary_mask, min_area=min_area)


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


def compute_stage2_score(metrics, target_normal_fpr=0.10, lambda_fpr_penalty=2.0):
    defect_dice = float(metrics.get("defect_dice", 0.0))
    normal_fpr = float(metrics.get("normal_fpr", 0.0))
    penalty = float(lambda_fpr_penalty) * max(0.0, normal_fpr - float(target_normal_fpr))
    return defect_dice - penalty


def compare_stage2_results(current_result, best_result, target_normal_fpr=0.10):
    if best_result is None:
        return True

    current_ok = float(current_result["normal_fpr"]) <= float(target_normal_fpr)
    best_ok = float(best_result["normal_fpr"]) <= float(target_normal_fpr)

    if current_ok and not best_ok:
        return True

    if current_ok and best_ok:
        if float(current_result["defect_dice"]) > float(best_result["defect_dice"]):
            return True

        if (
            float(current_result["defect_dice"]) == float(best_result["defect_dice"])
            and float(current_result["normal_fpr"]) < float(best_result["normal_fpr"])
        ):
            return True

        if (
            float(current_result["defect_dice"]) == float(best_result["defect_dice"])
            and float(current_result["normal_fpr"]) == float(best_result["normal_fpr"])
            and float(current_result.get("defect_image_recall", 0.0)) > float(best_result.get("defect_image_recall", 0.0))
        ):
            return True

        return False

    if not current_ok and not best_ok:
        if float(current_result.get("stage2_score", -1e9)) > float(best_result.get("stage2_score", -1e9)):
            return True

        if (
            float(current_result.get("stage2_score", -1e9)) == float(best_result.get("stage2_score", -1e9))
            and float(current_result["normal_fpr"]) < float(best_result["normal_fpr"])
        ):
            return True

        if (
            float(current_result.get("stage2_score", -1e9)) == float(best_result.get("stage2_score", -1e9))
            and float(current_result["normal_fpr"]) == float(best_result["normal_fpr"])
            and float(current_result["defect_dice"]) > float(best_result["defect_dice"])
        ):
            return True

    return False


def summarize_metrics(per_image_rows):
    defect_dice_values = []
    defect_iou_values = []
    defect_recall_hits = []
    normal_false_positive_hits = []
    normal_fp_pixels = []
    normal_largest_fp_areas = []

    for row in per_image_rows:
        if row["is_defect_image"]:
            defect_dice_values.append(float(row["dice"]))
            defect_iou_values.append(float(row["iou"]))
            defect_recall_hits.append(1.0 if row["pred_has_positive"] else 0.0)
        else:
            normal_false_positive_hits.append(1.0 if row["pred_has_positive"] else 0.0)
            normal_fp_pixels.append(float(row["fp_pixel_count"]))
            normal_largest_fp_areas.append(float(row["largest_fp_component_area"]))

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
        normal_fp_pixel_mean = float(np.mean(normal_fp_pixels))
        normal_largest_fp_area_mean = float(np.mean(normal_largest_fp_areas))
    else:
        normal_fpr = 0.0
        normal_fp_pixel_mean = 0.0
        normal_largest_fp_area_mean = 0.0

    return {
        "defect_dice": defect_dice,
        "defect_iou": defect_iou,
        "defect_image_recall": defect_image_recall,
        "normal_fpr": normal_fpr,
        "normal_fp_pixel_mean": normal_fp_pixel_mean,
        "normal_largest_fp_area_mean": normal_largest_fp_area_mean,
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


def evaluate_prob_maps(prob_maps, gt_masks, sample_types, threshold=0.5, min_area=0, image_names=None):
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
        pred_mask = probs_to_binary_mask(prob_map, threshold=threshold, min_area=min_area)
        gt_mask = _to_binary_mask(gt_mask)

        is_defect_image = _is_defect_sample(sample_type, gt_mask)
        pred_has_positive = int(pred_mask.sum()) > 0
        fp_pixel_count = int(pred_mask.sum()) if not is_defect_image else 0
        largest_fp_component_area = largest_component_area(pred_mask) if not is_defect_image else 0

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
                "min_area": int(min_area),
                "fp_pixel_count": fp_pixel_count,
                "largest_fp_component_area": largest_fp_component_area,
            }
        )

        pred_masks.append(pred_mask)

    metrics = summarize_metrics(per_image_rows)
    return {
        **metrics,
        "threshold": float(threshold),
        "min_area": int(min_area),
        "per_image_rows": per_image_rows,
        "pred_masks": pred_masks,
    }


def search_postprocess_params(
    prob_maps,
    gt_masks,
    sample_types,
    threshold_values,
    min_area_values,
    image_names=None,
    target_normal_fpr=0.10,
    lambda_fpr_penalty=2.0,
):
    if image_names is None:
        image_names = [f"sample_{index:05d}" for index in range(len(prob_maps))]

    best_result = None
    search_rows = []

    for threshold in threshold_values:
        for min_area in min_area_values:
            result = evaluate_prob_maps(
                prob_maps=prob_maps,
                gt_masks=gt_masks,
                sample_types=sample_types,
                image_names=image_names,
                threshold=threshold,
                min_area=min_area,
            )
            result["stage2_score"] = compute_stage2_score(
                result,
                target_normal_fpr=target_normal_fpr,
                lambda_fpr_penalty=lambda_fpr_penalty,
            )

            search_rows.append(
                {
                    "threshold": float(result["threshold"]),
                    "min_area": int(result["min_area"]),
                    "defect_dice": float(result["defect_dice"]),
                    "defect_iou": float(result["defect_iou"]),
                    "defect_image_recall": float(result["defect_image_recall"]),
                    "normal_fpr": float(result["normal_fpr"]),
                    "normal_fp_pixel_mean": float(result["normal_fp_pixel_mean"]),
                    "normal_largest_fp_area_mean": float(result["normal_largest_fp_area_mean"]),
                    "stage2_score": float(result["stage2_score"]),
                }
            )

            if compare_stage2_results(result, best_result, target_normal_fpr=target_normal_fpr):
                best_result = result

    return {
        "best_result": best_result,
        "search_rows": search_rows,
    }
