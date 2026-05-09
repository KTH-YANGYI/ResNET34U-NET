# YOLO 实验说明 - 2026-05-09

这份说明整理的是 `/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/yolo` 里这轮 YOLO 相关输出。核心目标是把 U-Net 的 mask 数据转换成检测框，看看 YOLO detection 能不能作为另一条路线，尤其是和 U-Net 的分割结果相比是否更稳。

## 数据集

- 数据来自 U-Net 项目的 `manifests/samples.csv`。
- 当前 YOLO 数据集目录：`yolo_data_set`。
- `train`: 200 张图，157 个 crack bbox，43 张 normal 空标签。
- `val`: 70 张图，54 个 crack bbox，16 张 normal 空标签。
- `test`: 67 张图，52 个 crack bbox，15 张 normal 空标签。
- bbox 由 mask 转换得到，当前主要 baseline 参数是 `min_area=8`、`bbox_padding=2`。
- class 定义：`0=crack`，`1=broken`；这轮 train/val/test 实际只有 crack bbox，unlabeled broken 被单独放在 `images/predict_broken` 供后续预测检查。

## 我做过的尝试

### 4GPU baseline attempt

- run: `yolo11n_crack_broken_e80_4gpu_20260509`
- 配置: model=yolo11n, epochs=80, imgsz=640, device=4GPU
- 目的: DDP 卡住，已停止并保留日志，没有有效 results.csv
- 状态: failed/no_results

### e80 baseline

- run: `yolo11n_crack_broken_e80_2gpu_20260509`
- 配置: model=yolo11n, epochs=80, imgsz=640, device=2GPU
- 目的: 第一条可用 baseline，bbox_padding=2/min_area=8
- 状态: completed
- true best mAP50-95: 0.59920 @ epoch 79，对应 precision=0.86776, recall=0.85185, mAP50=0.91277
- best mAP50: 0.92936 @ epoch 68，对应 recall=0.87037
- best recall: 0.92593 @ epoch 5，对应 mAP50=0.25185, mAP50-95=0.09679
- last epoch 80: precision=0.86724, recall=0.84687, mAP50=0.91589, mAP50-95=0.59855

### pad0/min8 bbox variant

- run: `yolo11n_crack_broken_e80_2gpu_pad0_min8_20260509`
- 配置: model=yolo11n, epochs=80, imgsz=640, device=2GPU
- 目的: 去掉 bbox padding，测试框更紧是否改善定位
- 状态: completed
- true best mAP50-95: 0.56471 @ epoch 73，对应 precision=0.88051, recall=0.85185, mAP50=0.92418
- best mAP50: 0.92418 @ epoch 73，对应 recall=0.85185
- best recall: 0.92593 @ epoch 62，对应 mAP50=0.88216, mAP50-95=0.49940
- last epoch 80: precision=0.84532, recall=0.83333, mAP50=0.86518, mAP50-95=0.54069

### larger model

- run: `yolo26n_crack_broken_e80_2gpu_baseline_20260509`
- 配置: model=yolo26n, epochs=80, imgsz=640, device=2GPU
- 目的: 测试更大模型是否提升
- 状态: completed
- true best mAP50-95: 0.55857 @ epoch 61，对应 precision=0.89103, recall=0.75717, mAP50=0.89434
- best mAP50: 0.89434 @ epoch 61，对应 recall=0.75717
- best recall: 0.83582 @ epoch 63，对应 mAP50=0.85543, mAP50-95=0.52680
- last epoch 80: precision=0.77582, recall=0.79630, mAP50=0.86532, mAP50-95=0.53592

### longer training e140

- run: `yolo11n_crack_broken_e140_2gpu_baseline_20260509`
- 配置: model=yolo11n, epochs=140, imgsz=640, device=2GPU
- 目的: 延长训练，当前首选方向
- 状态: completed
- true best mAP50-95: 0.60735 @ epoch 121，对应 precision=0.90093, recall=0.84212, mAP50=0.91386
- best mAP50: 0.93251 @ epoch 135，对应 recall=0.82953
- best recall: 0.96296 @ epoch 7，对应 mAP50=0.28859, mAP50-95=0.12236
- last epoch 140: precision=0.90232, recall=0.85535, mAP50=0.90232, mAP50-95=0.56540

### longer training e200

- run: `yolo11n_crack_broken_e200_2gpu_baseline_20260509`
- 配置: model=yolo11n, epochs=200, imgsz=640, device=2GPU
- 目的: 继续延长训练，检查 e140 后是否还能涨
- 状态: completed
- true best mAP50-95: 0.60550 @ epoch 152，对应 precision=0.83135, recall=0.87037, mAP50=0.90769
- best mAP50: 0.93687 @ epoch 137，对应 recall=0.81481
- best recall: 0.92593 @ epoch 141，对应 mAP50=0.89431, mAP50-95=0.54369
- last epoch 200: precision=0.85400, recall=0.87037, mAP50=0.90374, mAP50-95=0.57979

### higher resolution img832

- run: `yolo11n_crack_broken_e140_img832_2gpu_20260509`
- 配置: model=yolo11n, epochs=140, imgsz=832, device=2GPU
- 目的: 高分辨率输入，测试定位是否改善
- 状态: completed
- true best mAP50-95: 0.60104 @ epoch 108，对应 precision=0.88468, recall=0.79630, mAP50=0.91369
- best mAP50: 0.94454 @ epoch 117，对应 recall=0.88889
- best recall: 0.94444 @ epoch 100，对应 mAP50=0.89544, mAP50-95=0.56338
- last epoch 140: precision=0.90526, recall=0.81481, mAP50=0.88263, mAP50-95=0.58702

## 当前最好的结果

- 如果按 detection 定位质量主指标 `mAP50-95` 看，最好的是 `yolo11n_crack_broken_e140_2gpu_baseline_20260509`。
- 它的 true best mAP50-95 = 0.60735，在 epoch 121；当时 precision=0.90093, recall=0.84212, mAP50=0.91386。
- 如果只看较宽松的 `mAP50`，最高的是 `yolo11n_crack_broken_e140_img832_2gpu_20260509`，mAP50=0.94454，但它的 mAP50-95 不如上面的首选，说明框更容易粗略命中，但精确定位没有更好。

我的建议：当前首选保留 `yolo11n_crack_broken_e140_2gpu_baseline_20260509`，不要选 img832 或 e200 作为主线。原因是 e140 baseline 的 mAP50-95 最高，整体定位质量最好；img832 虽然 mAP50 更高，但 recall 和 mAP50-95 没赢，说明高分辨率只是更容易粗命中，不一定更准。

## 额外检查

- confidence/normal FP 检查：`runs/yolo11n_crack_broken_e80_2gpu_20260509/conf_sweep_20260509`。
  - val 推荐点 normal_fp_images=0，recall_iou50=0.90741。
  - test 推荐点 normal_fp_images=0，recall_iou50=0.94231。
  - 结论：normal/background 误检不是主要瓶颈。
- label/重复框检查：`runs/yolo11n_crack_broken_e140_2gpu_baseline_20260509/error_inspection_20260509`。
  - 标签检查：val invalid_boxes=0, tiny_boxes=0, duplicate_label_pairs_iou80=0。
  - 预测检查：val conf=0.25 时 duplicate_pred_pairs_iou50=6，images_with_duplicate_predictions=5。
  - 结论：标签本身比较干净，问题更多来自预测框重复/定位不够稳定。
- NMS sweep：`runs/yolo11n_crack_broken_e140_2gpu_baseline_20260509/nms_sweep_20260509`。
  - val 上较好的 NMS IoU 约为 0.55，mAP50-95=0.60730。
  - 结论：推理时可以优先试 NMS IoU around 0.55，帮助减少重复框，但这不是训练本身的根本提升。

## 当前瓶颈

1. 数据量偏小：train 只有 200 张，其中 crack bbox 157 个，YOLO 很容易在不同 epoch 间波动。
2. 检测框来自 mask 的外接框，裂纹/接触线这类细长目标用 bbox 表达会比较粗，mAP50 容易高，但 mAP50-95 对精确定位要求更高，所以提升困难。
3. 主要错误不是 normal 图误报，而是 defect 图上的定位不够准、重复预测、以及不同阈值下 precision/recall 的取舍。
4. 更大模型 yolo26n 没带来提升，说明瓶颈不像是模型容量不足；更高分辨率 img832 也没带来 mAP50-95 提升，说明仅靠加分辨率也不够。

## 后续建议

- 短期使用：用 `yolo11n_crack_broken_e140_2gpu_baseline_20260509/weights/best.pt` 作为当前 YOLO 首选 checkpoint。
- 推理参数：优先测试 NMS IoU around 0.55；confidence 可以按需求在 recall/precision 之间调。
- 如果继续优化 YOLO，优先方向不是继续盲目加 epoch 或换大模型，而是做更有针对性的 label/box 策略：例如用更合理的 bbox 生成方式、分裂过长框、或者改成 YOLO segmentation/keypoint/line-like 表达。
- 和 U-Net 的关系：U-Net 仍然更适合像素级裂纹区域；YOLO 可以作为快速 coarse detector 或候选框生成器，但当前不是比 U-Net 更稳的最终主模型。

## 相关输出位置

- 本说明：`runs/yolo_strategy_review_20260509/YOLO_EXPERIMENTS_EXPLAINED.md`
- 指标表：`runs/yolo_strategy_review_20260509/yolo_experiment_explanation_metrics.csv`
- bbox 可视化：`runs/yolo_strategy_review_20260509/bbox_visualizations/index.html`
- label dump：`runs/yolo_strategy_review_20260509/crack_bbox_labels_dump.txt`