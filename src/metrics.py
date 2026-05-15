import numpy as np
import torch

try:
    from scipy import ndimage as scipy_ndimage
except Exception:
    scipy_ndimage = None


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


def _label_components_numpy(binary_mask):
    binary_mask = _to_binary_mask(binary_mask)
    height, width = binary_mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    areas = []
    current_label = 0
    neighbor_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    start_ys, start_xs = np.where(binary_mask > 0)

    for start_y, start_x in zip(start_ys, start_xs):
        if labels[start_y, start_x] != 0:
            continue

        current_label += 1
        stack = [(int(start_y), int(start_x))]
        labels[start_y, start_x] = current_label
        area = 0

        while len(stack) > 0:
            y, x = stack.pop()
            area += 1

            for dy, dx in neighbor_offsets:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if binary_mask[ny, nx] == 0 or labels[ny, nx] != 0:
                    continue
                labels[ny, nx] = current_label
                stack.append((ny, nx))

        areas.append(area)

    return labels, np.asarray(areas, dtype=np.int64)


def label_components(binary_mask):
    binary_mask = _to_binary_mask(binary_mask)

    if int(binary_mask.sum()) == 0:
        return np.zeros_like(binary_mask, dtype=np.int32), np.asarray([], dtype=np.int64)

    if scipy_ndimage is not None:
        structure = np.ones((3, 3), dtype=np.uint8)
        labels, component_count = scipy_ndimage.label(binary_mask, structure=structure)
        if int(component_count) == 0:
            return labels.astype(np.int32), np.asarray([], dtype=np.int64)
        areas = np.bincount(labels.ravel())[1:]
        return labels.astype(np.int32), areas.astype(np.int64)

    return _label_components_numpy(binary_mask)


def connected_component_stats(binary_mask):
    binary_mask = _to_binary_mask(binary_mask)

    if int(binary_mask.sum()) == 0:
        return []

    _, areas = label_components(binary_mask)
    return [{"area": int(area)} for area in areas if int(area) > 0]


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

    labels, areas = label_components(binary_mask)
    filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for label_index, area in enumerate(areas, start=1):
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


def _safe_divide(numerator, denominator, empty_value=0.0):
    denominator = float(denominator)
    if denominator == 0.0:
        return float(empty_value)
    return float(numerator) / denominator


def pixel_confusion(pred, target):
    pred = _to_binary_mask(pred).astype(bool)
    target = _to_binary_mask(target).astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(pred), target).sum())
    return tp, fp, fn


def precision_recall_f1_from_counts(tp, fp, fn):
    precision = _safe_divide(tp, tp + fp, empty_value=1.0)
    recall = _safe_divide(tp, tp + fn, empty_value=1.0)
    f1 = _safe_divide(2.0 * tp, 2.0 * tp + fp + fn, empty_value=1.0)
    return precision, recall, f1


def average_precision_from_prob_maps(prob_maps, gt_masks):
    labels = []
    scores = []
    for prob_map, gt_mask in zip(prob_maps, gt_masks):
        labels.append(_to_binary_mask(gt_mask).reshape(-1).astype(np.uint8))
        scores.append(_squeeze_to_hw(prob_map).reshape(-1).astype(np.float32))

    if len(labels) == 0:
        return 0.0

    y_true = np.concatenate(labels)
    y_score = np.concatenate(scores)
    positive_count = int(y_true.sum())
    if positive_count == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_true_sorted)
    ranks = np.arange(1, len(y_true_sorted) + 1, dtype=np.float64)
    precision_at_rank = tp_cumsum / ranks
    return float(precision_at_rank[y_true_sorted.astype(bool)].sum() / positive_count)


def _dilate_binary(mask, iterations=1):
    mask = _to_binary_mask(mask).astype(bool)
    iterations = max(0, int(iterations))
    if iterations == 0:
        return mask

    if scipy_ndimage is not None:
        structure = np.ones((3, 3), dtype=np.uint8)
        return scipy_ndimage.binary_dilation(mask, structure=structure, iterations=iterations)

    output = mask
    for _ in range(iterations):
        padded = np.pad(output, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(output, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy:dy + output.shape[0], dx:dx + output.shape[1]]
        output = expanded
    return output


def _erode_binary(mask, iterations=1):
    mask = _to_binary_mask(mask).astype(bool)
    iterations = max(0, int(iterations))
    if iterations == 0:
        return mask

    if scipy_ndimage is not None:
        structure = np.ones((3, 3), dtype=np.uint8)
        return scipy_ndimage.binary_erosion(mask, structure=structure, iterations=iterations)

    output = mask
    for _ in range(iterations):
        padded = np.pad(output, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(output, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy:dy + output.shape[0], dx:dx + output.shape[1]]
        output = eroded
    return output


def component_metrics(pred, target, tolerance_px=3):
    pred = _to_binary_mask(pred)
    target = _to_binary_mask(target)
    pred_labels, pred_areas = label_components(pred)
    target_labels, target_areas = label_components(target)

    gt_count = len(target_areas)
    pred_count = len(pred_areas)
    if gt_count == 0 and pred_count == 0:
        return 1.0, 1.0, 1.0
    if gt_count == 0:
        return 0.0, 0.0, 0.0

    pred_dilated = _dilate_binary(pred, iterations=tolerance_px)
    target_dilated = _dilate_binary(target, iterations=tolerance_px)
    detected_gt = 0
    detected_pred = 0

    for label_index in range(1, gt_count + 1):
        gt_component = target_labels == label_index
        if bool(np.logical_and(pred_dilated, gt_component).any()):
            detected_gt += 1

    for label_index in range(1, pred_count + 1):
        pred_component = pred_labels == label_index
        if bool(np.logical_and(target_dilated, pred_component).any()):
            detected_pred += 1

    recall = _safe_divide(detected_gt, gt_count, empty_value=1.0)
    precision = _safe_divide(detected_pred, pred_count, empty_value=1.0)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall, empty_value=0.0)
    return recall, precision, f1


def boundary_f1_score(pred, target, tolerance_px=3):
    pred = _to_binary_mask(pred)
    target = _to_binary_mask(target)
    if int(pred.sum()) == 0 and int(target.sum()) == 0:
        return 1.0
    if int(pred.sum()) == 0 or int(target.sum()) == 0:
        return 0.0

    pred_boundary = pred.astype(bool) ^ _erode_binary(pred, iterations=1)
    target_boundary = target.astype(bool) ^ _erode_binary(target, iterations=1)
    if int(pred_boundary.sum()) == 0 and int(target_boundary.sum()) == 0:
        return 1.0
    if int(pred_boundary.sum()) == 0 or int(target_boundary.sum()) == 0:
        return 0.0

    pred_match = np.logical_and(pred_boundary, _dilate_binary(target_boundary, iterations=tolerance_px)).sum()
    target_match = np.logical_and(target_boundary, _dilate_binary(pred_boundary, iterations=tolerance_px)).sum()
    precision = _safe_divide(pred_match, pred_boundary.sum(), empty_value=0.0)
    recall = _safe_divide(target_match, target_boundary.sum(), empty_value=0.0)
    return _safe_divide(2.0 * precision * recall, precision + recall, empty_value=0.0)


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


def _mean_or_zero(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _median_or_zero(values):
    if len(values) == 0:
        return 0.0
    return float(np.median(values))


def _p95_or_zero(values):
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, 95))


def _max_or_zero(values):
    if len(values) == 0:
        return 0.0
    return float(np.max(values))


def summarize_metrics(per_image_rows):
    defect_dice_values = []
    defect_iou_values = []
    defect_recall_hits = []
    defect_pixel_precision_values = []
    defect_pixel_recall_values = []
    defect_pixel_f1_values = []
    defect_component_recall_values = []
    defect_component_precision_values = []
    defect_component_f1_values = []
    defect_boundary_f1_values = []
    normal_false_positive_hits = []
    normal_fp_pixels = []
    normal_largest_fp_areas = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for row in per_image_rows:
        total_tp += int(row.get("tp_pixel_count", 0))
        total_fp += int(row.get("fp_pixel_count_for_metric", 0))
        total_fn += int(row.get("fn_pixel_count", 0))
        if row["is_defect_image"]:
            defect_dice_values.append(float(row["dice"]))
            defect_iou_values.append(float(row["iou"]))
            defect_recall_hits.append(1.0 if row["pred_has_positive"] else 0.0)
            defect_pixel_precision_values.append(float(row["pixel_precision"]))
            defect_pixel_recall_values.append(float(row["pixel_recall"]))
            defect_pixel_f1_values.append(float(row["pixel_f1"]))
            defect_component_recall_values.append(float(row["component_recall_3px"]))
            defect_component_precision_values.append(float(row["component_precision_3px"]))
            defect_component_f1_values.append(float(row["component_f1_3px"]))
            defect_boundary_f1_values.append(float(row["boundary_f1_3px"]))
        else:
            normal_false_positive_hits.append(1.0 if row["pred_has_positive"] else 0.0)
            normal_fp_pixels.append(float(row["fp_pixel_count"]))
            normal_largest_fp_areas.append(float(row["largest_fp_component_area"]))

    if len(defect_dice_values) > 0:
        defect_dice = _mean_or_zero(defect_dice_values)
        defect_iou = _mean_or_zero(defect_iou_values)
        defect_image_recall = _mean_or_zero(defect_recall_hits)
        pixel_precision_defect_macro = _mean_or_zero(defect_pixel_precision_values)
        pixel_recall_defect_macro = _mean_or_zero(defect_pixel_recall_values)
        pixel_f1_defect_macro = _mean_or_zero(defect_pixel_f1_values)
        component_recall_3px = _mean_or_zero(defect_component_recall_values)
        component_precision_3px = _mean_or_zero(defect_component_precision_values)
        component_f1_3px = _mean_or_zero(defect_component_f1_values)
        boundary_f1_3px = _mean_or_zero(defect_boundary_f1_values)
    else:
        defect_dice = 0.0
        defect_iou = 0.0
        defect_image_recall = 0.0
        pixel_precision_defect_macro = 0.0
        pixel_recall_defect_macro = 0.0
        pixel_f1_defect_macro = 0.0
        component_recall_3px = 0.0
        component_precision_3px = 0.0
        component_f1_3px = 0.0
        boundary_f1_3px = 0.0

    if len(normal_false_positive_hits) > 0:
        normal_fpr = _mean_or_zero(normal_false_positive_hits)
        normal_fp_count = int(sum(1 for value in normal_false_positive_hits if float(value) > 0.0))
        normal_fp_pixel_sum = float(np.sum(normal_fp_pixels))
        normal_fp_pixel_mean = _mean_or_zero(normal_fp_pixels)
        normal_fp_pixel_median = _median_or_zero(normal_fp_pixels)
        normal_fp_pixel_p95 = _p95_or_zero(normal_fp_pixels)
        normal_largest_fp_area_mean = _mean_or_zero(normal_largest_fp_areas)
        normal_largest_fp_area_median = _median_or_zero(normal_largest_fp_areas)
        normal_largest_fp_area_p95 = _p95_or_zero(normal_largest_fp_areas)
        normal_largest_fp_area_max = _max_or_zero(normal_largest_fp_areas)
    else:
        normal_fpr = 0.0
        normal_fp_count = 0
        normal_fp_pixel_sum = 0.0
        normal_fp_pixel_mean = 0.0
        normal_fp_pixel_median = 0.0
        normal_fp_pixel_p95 = 0.0
        normal_largest_fp_area_mean = 0.0
        normal_largest_fp_area_median = 0.0
        normal_largest_fp_area_p95 = 0.0
        normal_largest_fp_area_max = 0.0

    pixel_precision_labeled_micro, pixel_recall_labeled_micro, pixel_f1_labeled_micro = precision_recall_f1_from_counts(
        total_tp,
        total_fp,
        total_fn,
    )

    return {
        "defect_dice": defect_dice,
        "defect_iou": defect_iou,
        "defect_image_recall": defect_image_recall,
        "normal_fpr": normal_fpr,
        "normal_fp_count": normal_fp_count,
        "normal_fp_pixel_sum": normal_fp_pixel_sum,
        "normal_fp_pixel_mean": normal_fp_pixel_mean,
        "normal_fp_pixel_median": normal_fp_pixel_median,
        "normal_fp_pixel_p95": normal_fp_pixel_p95,
        "normal_largest_fp_area_mean": normal_largest_fp_area_mean,
        "normal_largest_fp_area_median": normal_largest_fp_area_median,
        "normal_largest_fp_area_p95": normal_largest_fp_area_p95,
        "normal_largest_fp_area_max": normal_largest_fp_area_max,
        "pixel_precision_defect_macro": pixel_precision_defect_macro,
        "pixel_recall_defect_macro": pixel_recall_defect_macro,
        "pixel_f1_defect_macro": pixel_f1_defect_macro,
        "pixel_precision_labeled_micro": pixel_precision_labeled_micro,
        "pixel_recall_labeled_micro": pixel_recall_labeled_micro,
        "pixel_f1_labeled_micro": pixel_f1_labeled_micro,
        "component_recall_3px": component_recall_3px,
        "component_precision_3px": component_precision_3px,
        "component_f1_3px": component_f1_3px,
        "boundary_f1_3px": boundary_f1_3px,
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


def evaluate_prob_maps(
    prob_maps,
    gt_masks,
    sample_types,
    threshold=0.5,
    min_area=0,
    image_names=None,
    include_auprc=False,
    include_expensive_metrics=True,
):
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
        tp_pixel_count, fp_pixel_count_for_metric, fn_pixel_count = pixel_confusion(pred_mask, gt_mask)
        pixel_precision, pixel_recall, pixel_f1 = precision_recall_f1_from_counts(
            tp_pixel_count,
            fp_pixel_count_for_metric,
            fn_pixel_count,
        )
        fp_pixel_count = int(pred_mask.sum()) if not is_defect_image else 0
        if include_expensive_metrics and not is_defect_image:
            largest_fp_component_area = largest_component_area(pred_mask)
        else:
            largest_fp_component_area = 0

        if is_defect_image:
            dice_value = dice_score(pred_mask, gt_mask)
            iou_value = iou_score(pred_mask, gt_mask)
            if include_expensive_metrics:
                component_recall, component_precision, component_f1 = component_metrics(pred_mask, gt_mask, tolerance_px=3)
                boundary_f1 = boundary_f1_score(pred_mask, gt_mask, tolerance_px=3)
            else:
                component_recall = 0.0
                component_precision = 0.0
                component_f1 = 0.0
                boundary_f1 = 0.0
        else:
            dice_value = 0.0
            iou_value = 0.0
            component_recall = 0.0
            component_precision = 0.0
            component_f1 = 0.0
            boundary_f1 = 0.0

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
                "tp_pixel_count": tp_pixel_count,
                "fp_pixel_count_for_metric": fp_pixel_count_for_metric,
                "fn_pixel_count": fn_pixel_count,
                "pred_pixel_count": int(pred_mask.sum()),
                "target_pixel_count": int(gt_mask.sum()),
                "pixel_precision": pixel_precision,
                "pixel_recall": pixel_recall,
                "pixel_f1": pixel_f1,
                "largest_fp_component_area": largest_fp_component_area,
                "component_recall_3px": component_recall,
                "component_precision_3px": component_precision,
                "component_f1_3px": component_f1,
                "boundary_f1_3px": boundary_f1,
            }
        )

        pred_masks.append(pred_mask)

    metrics = summarize_metrics(per_image_rows)
    metrics["pixel_auprc_all_labeled"] = average_precision_from_prob_maps(prob_maps, gt_masks) if include_auprc else 0.0
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
    include_auprc=False,
):
    if image_names is None:
        image_names = [f"sample_{index:05d}" for index in range(len(prob_maps))]

    best_result = None
    search_rows = []
    pixel_auprc_all_labeled = average_precision_from_prob_maps(prob_maps, gt_masks) if include_auprc else 0.0

    for threshold in threshold_values:
        for min_area in min_area_values:
            result = evaluate_prob_maps(
                prob_maps=prob_maps,
                gt_masks=gt_masks,
                sample_types=sample_types,
                image_names=image_names,
                threshold=threshold,
                min_area=min_area,
                include_auprc=False,
                include_expensive_metrics=False,
            )
            result["pixel_auprc_all_labeled"] = pixel_auprc_all_labeled
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
                    "normal_count": int(result["normal_count"]),
                    "normal_fp_count": int(result["normal_fp_count"]),
                    "normal_fp_pixel_sum": float(result["normal_fp_pixel_sum"]),
                    "normal_fp_pixel_mean": float(result["normal_fp_pixel_mean"]),
                    "normal_fp_pixel_median": float(result["normal_fp_pixel_median"]),
                    "normal_fp_pixel_p95": float(result["normal_fp_pixel_p95"]),
                    "normal_largest_fp_area_mean": float(result["normal_largest_fp_area_mean"]),
                    "normal_largest_fp_area_median": float(result["normal_largest_fp_area_median"]),
                    "normal_largest_fp_area_p95": float(result["normal_largest_fp_area_p95"]),
                    "normal_largest_fp_area_max": float(result["normal_largest_fp_area_max"]),
                    "pixel_precision_defect_macro": float(result["pixel_precision_defect_macro"]),
                    "pixel_recall_defect_macro": float(result["pixel_recall_defect_macro"]),
                    "pixel_f1_defect_macro": float(result["pixel_f1_defect_macro"]),
                    "pixel_precision_labeled_micro": float(result["pixel_precision_labeled_micro"]),
                    "pixel_recall_labeled_micro": float(result["pixel_recall_labeled_micro"]),
                    "pixel_f1_labeled_micro": float(result["pixel_f1_labeled_micro"]),
                    "pixel_auprc_all_labeled": float(result["pixel_auprc_all_labeled"]),
                    "component_recall_3px": float(result["component_recall_3px"]),
                    "component_precision_3px": float(result["component_precision_3px"]),
                    "component_f1_3px": float(result["component_f1_3px"]),
                    "boundary_f1_3px": float(result["boundary_f1_3px"]),
                    "stage2_score": float(result["stage2_score"]),
                }
            )

            if compare_stage2_results(result, best_result, target_normal_fpr=target_normal_fpr):
                best_result = result

    if best_result is not None:
        best_result = evaluate_prob_maps(
            prob_maps=prob_maps,
            gt_masks=gt_masks,
            sample_types=sample_types,
            image_names=image_names,
            threshold=float(best_result["threshold"]),
            min_area=int(best_result["min_area"]),
            include_auprc=False,
            include_expensive_metrics=True,
        )
        best_result["pixel_auprc_all_labeled"] = pixel_auprc_all_labeled
        best_result["stage2_score"] = compute_stage2_score(
            best_result,
            target_normal_fpr=target_normal_fpr,
            lambda_fpr_penalty=lambda_fpr_penalty,
        )

    return {
        "best_result": best_result,
        "search_rows": search_rows,
    }
