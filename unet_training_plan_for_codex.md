# 金属工件缺陷分割：U-Net 基线训练实施计划（给 Codex 直接落地）

## 1. 文档目的

这份计划只解决一件事：

**把你现在的 ROI 图和像素级缺陷标注，训练成一版可运行、可验证、可复现的 U-Net 分割基线。**

这版基线的目标不是“一步到位做最终模型”，而是：

1. 先在 ROI 上把缺陷分出来；
2. 再从分割结果派生检测框；
3. 再从分割结果派生像素级几何量化结果；
4. 为后面的扩散模型增强、相机域扩展和论文实验打基础。

---

## 2. 本计划的默认前提

- 已经完成 ROI 裁剪；
- ROI 图片尺寸默认是 `640 x 640`；
- 已有 100 多张**独立缺陷样本**；
- 正常 ROI 图很多，大约 1 万张；
- 每张缺陷图都有对应的二值掩码；
- 缺陷在 ROI 中很小，约占 `0.4%` 左右；
- 当前只做**单类缺陷分割**，不做多类别分割；
- 当前不做真实物理尺寸测量，只做像素级几何量化。

---

## 3. 成功标准

Codex 实现完成后，至少要达到下面这些结果：

1. 能稳定读取 ROI 图和对应掩码；
2. 能按“固定测试集 + 训练/验证”流程训练；
3. 能先做局部块训练，再做整图微调；
4. 能输出验证集和测试集的：
   - 缺陷图 Dice
   - 缺陷图 IoU
   - 图像级召回
   - 正常图误报率
5. 能输出最终二值掩码；
6. 能从掩码自动派生：
   - 检测框
   - 缺陷面积
   - 周长
   - 外接框宽高
7. 能保存最优模型与完整日志；
8. 代码可以重复运行，固定随机种子后结果基本稳定。

---

## 4. 总体训练路线

本项目采用两阶段训练：

### 阶段 1：局部块训练
目标：先让模型学会“缺陷长什么样、边界怎么分、正常背景是什么样”。

### 阶段 2：整张 ROI 微调
目标：再让模型学会“在整张 640×640 的 ROI 中找到缺陷并输出完整掩码”。

这条路线的核心原因是：你的缺陷在整图里太小，直接整图训练很容易学成全背景。U-Net 原始论文就强调，它依赖强数据增强来更高效地利用少量标注图像；nnU-Net 的经验也说明，数据组织、补丁训练、训练与后处理配置往往比盲目堆复杂网络更重要。[1][2]

---

## 5. 推荐目录结构

```text
project/
  data/
    raw/
      roi_images/
      roi_masks/
    splits/
      test_ids.txt
      fold_0_train_ids.txt
      fold_0_val_ids.txt
      fold_1_train_ids.txt
      fold_1_val_ids.txt
      ...
    patch_cache/
      # 可选：如果想把局部块提前生成到磁盘
  src/
    datasets/
      roi_dataset.py
      patch_dataset.py
    models/
      unet_resnet34.py
    losses/
      dice_loss.py
      combo_loss.py
    engine/
      trainer.py
      evaluator.py
      infer.py
    utils/
      metrics.py
      postprocess.py
      geometry.py
      seed.py
      io.py
  scripts/
    make_splits.py
    build_patch_index.py
    train_stage1_patch.py
    train_stage2_full.py
    validate.py
    test.py
    export_predictions.py
  outputs/
    fold_0/
      stage1/
      stage2/
    fold_1/
      ...
  configs/
    stage1_patch.yaml
    stage2_full.yaml
  metadata.csv
```

---

## 6. 数据划分规则

### 6.1 固定测试集
先从全部**缺陷事件 / 工件实例**里划出固定测试集，建议占缺陷样本总数的 15% 到 20%。

例子：如果有 120 张独立缺陷图，可先固定 20 到 24 张作为测试集。

### 6.2 训练和验证
剩余缺陷图用于训练和验证。推荐：

- 做 **5 折交叉验证**；
- 或至少保留单独验证集。

### 6.3 正常图划分原则
正常图也必须跟着同样的“视频 / 工件 / 拍摄批次”逻辑划分，避免：

- 训练集中出现某段视频；
- 测试集中出现同一段视频的相似帧。

这会造成验证和测试结果虚高。

### 6.4 正常图使用策略
不要每个 epoch 都把 1 万张正常图全部喂进去。建议：

- 每个 epoch 使用全部训练缺陷图；
- 每个 epoch 只随机抽取和缺陷图数量接近的一部分正常图；
- 下一轮再换不同的正常图。

这样既保留正常背景多样性，又不会把训练“淹没”。

---

## 7. 局部块数据集设计（阶段 1 关键）

## 7.1 为什么一定要做局部块
因为你的缺陷在整张 ROI 中只占很小比例。直接整图训练，模型会更倾向于全部预测为背景。

局部块的目标是：

- 把缺陷在输入中的相对占比提高；
- 让模型先学会边界；
- 让模型先学会背景与缺陷的区别。

## 7.2 正样本局部块生成规则

对每一张缺陷图，根据掩码计算：

- 缺陷最小外接矩形；
- 缺陷中心点；
- 缺陷面积。

然后在线生成 3 类正样本局部块：

### 正样本块 A：中心标准块
- 以缺陷中心点为中心裁块；
- 裁剪边长从 `320 / 384 / 448` 中随机选一个；
- 再统一缩放到 `384 x 384`。

### 正样本块 B：轻微平移块
- 在缺陷中心附近随机平移；
- 平移范围建议不超过 24 像素；
- 要保证缺陷完整保留在块内；
- 同样缩放到 `384 x 384`。

### 正样本块 C：上下文块
- 缺陷不一定处于正中心；
- 但缺陷必须完整保留；
- 目的：让模型学习缺陷在局部结构中的相对位置和背景关系。

## 7.3 负样本局部块生成规则

负样本要分两种来源：

### 负样本块 A：正常图局部块
从正常 ROI 中，在**缺陷常出现区域**裁块，而不是随机从无关背景处裁。

### 负样本块 B：缺陷图中的非缺陷区域块
从缺陷图中避开缺陷 mask，在相近结构位置裁块。  
这种块很重要，因为它最接近真实“困难负样本”。

## 7.4 一个 batch 的推荐构成（阶段 1）

以 `batch_size = 8` 为例：

- 4 张正样本局部块
- 2 张正常图负样本块
- 2 张缺陷图非缺陷区域负样本块

这样能保证：

- 正负样本平衡；
- 背景不会压倒缺陷；
- 困难负样本被反复看到。

---

## 8. 模型结构

## 8.1 主模型
使用 **U-Net**。

## 8.2 编码器
使用 **ResNet34 预训练权重**作为编码器。

原因：

- 你数据少，随机初始化不划算；
- 预训练特征提取器能更快学到边缘、纹理和局部形状；
- ResNet34 的规模相对适中，不像更大骨干那样容易过拟合。

TorchVision 官方提供预训练权重接口，可直接加载。[3]

## 8.3 输出层
输出一个单通道 logits 图，大小与输入相同。

- 训练时不先手动做 Sigmoid；
- 损失函数内部处理 logits；
- 推理时再对 logits 做 Sigmoid，得到概率图。

## 8.4 实现建议
Codex 可以优先采用两种实现方式之一：

### 方案 A：优先推荐
手写一个标准 U-Net 解码器，编码器调用 torchvision 的 ResNet34 预训练权重。

### 方案 B：如果环境允许
也可以先用成熟库快速搭建，但必须保留相同训练流程和配置。  
无论用哪种方式，外部行为要一致。

---

## 9. 损失函数设计

## 9.1 推荐组合
使用：

- **带对数几率的二元交叉熵**
- **Dice 损失**

组合为：

```text
总损失 = 0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss
```

### 为什么这么配
- 前者负责逐像素分类；
- 后者负责区域重叠；
- 两者一起用，比单独只用一种更稳。

PyTorch 官方文档说明，`BCEWithLogitsLoss` 将 Sigmoid 和二元交叉熵合在一起，数值更稳定，并且支持 `pos_weight` 调节正类权重。[4]

## 9.2 正类权重
由于前景很少，建议给正类更高权重。

起步试 3 组：

- `pos_weight = 8`
- `pos_weight = 12`
- `pos_weight = 16`

在验证集上比较：

- 缺陷召回
- 正常图误报率
- Dice

### 经验判断
- 如果模型总是漏掉小缺陷，说明正类权重可能偏小；
- 如果正常图噪点特别多，说明正类权重可能偏大。

## 9.3 Dice 损失实现
可自行实现，也可以参考成熟实现。MONAI 文档中提供了 Dice 及 Dice+交叉熵类损失的实现方式，可作为核对依据。[5]

---

## 10. 优化器、学习率和调度器

## 10.1 优化器
使用 **AdamW**。

PyTorch 官方文档说明，AdamW 采用解耦权重衰减的实现方式。[6]

默认参数建议：

```text
weight_decay = 1e-4
betas = (0.9, 0.999)
```

## 10.2 学习率

### 阶段 1：局部块训练
建议分组学习率：

- 编码器：`5e-5`
- 解码器和输出头：`2e-4`

如果 Codex 先想做简单版本，也可以统一为：

- 全模型：`1e-4`

### 阶段 2：整图微调
继续分组学习率：

- 编码器：`2e-5`
- 解码器和输出头：`1e-4`

## 10.3 学习率调度器
使用 **ReduceLROnPlateau**，监控验证集缺陷图 Dice。

推荐参数：

```text
factor = 0.5
patience = 5
min_lr = 1e-6
mode = "max"
```

PyTorch 官方文档说明，ReduceLROnPlateau 会在指标停止改进时自动降低学习率。[7]

## 10.4 自动混合精度
训练时开启：

- `torch.amp.autocast("cuda")`
- `torch.amp.GradScaler("cuda")`

PyTorch 官方文档建议在混合精度训练中搭配使用两者，以减少显存并提升速度。[8][9]

---

## 11. 阶段 1：局部块训练详细计划

## 11.1 输入尺寸
统一为：

```text
384 x 384
```

## 11.2 数据增强
使用**保守增强**，只模拟真实拍摄波动：

### 几何增强
- 小角度旋转：`-5° ~ +5°`
- 轻微平移：不超过图宽高 3%
- 轻微缩放：`0.95 ~ 1.05`

### 亮度对比度增强
- 亮度变化：`±10%`
- 对比度变化：`±10%`

### 成像扰动
- 轻微高斯模糊
- 轻微噪声

### 不建议
- 大角度旋转
- 夸张色彩扰动
- 弹性形变
- 随机裁剪到破坏工件结构

## 11.3 训练轮数
先设：

```text
epochs = 30
```

## 11.4 冻结策略
建议：

- 前 3 个 epoch 冻结编码器；
- 第 4 个 epoch 开始全部解冻。

## 11.5 验证频率
每个 epoch 验证一次。

## 11.6 保存策略
按“验证集缺陷图 Dice 最高”保存 `best_stage1.pth`。

## 11.7 阶段 1 结束条件
满足下面任一条件即可进入阶段 2：

- 验证集 Dice 连续 8 个 epoch 不再明显上升；
- 小缺陷已经能被初步分出；
- 局部块上的边界已经比较稳定。

---

## 12. 阶段 2：整张 ROI 微调详细计划

## 12.1 输入尺寸
直接使用完整 ROI：

```text
640 x 640
```

## 12.2 初始权重
从 `best_stage1.pth` 加载。

## 12.3 batch 组成
建议：

- `batch_size = 4`
- 2 张缺陷 ROI
- 2 张正常 ROI

如果显存紧张，可降为 2。

## 12.4 整图训练时是否完全放弃局部块
不建议完全放弃。  
推荐“局部块回放”机制：

- 每训练完 1 个整图 epoch；
- 再额外做一次小规模局部块更新（例如 100 到 200 step）。

作用：

- 防止模型在整图训练中遗忘小缺陷边界；
- 保持对局部纹理和边界的敏感性。

## 12.5 学习率
使用第 10 节中的阶段 2 配置。

## 12.6 训练轮数
先设：

```text
epochs = 20 ~ 30
```

## 12.7 保存策略
按验证集综合指标保存：

优先级建议：

1. 正常图误报率不过高；
2. 在此基础上缺陷图 Dice 最大。

---

## 13. 验证指标与模型选择

## 13.1 不要只看总体像素准确率
背景像素太多，这个指标没有参考价值。

## 13.2 必须输出的指标

### 分割类
- 验证集缺陷图 Dice
- 验证集缺陷图 IoU

### 图像级
- 缺陷图图像级召回
- 正常图误报率

### 子集分析
- 小缺陷子集 Dice
- 小缺陷子集召回

## 13.3 模型选择规则
建议这样定：

- 先筛掉“正常图误报率超过阈值”的模型；
- 在剩余模型中选择缺陷图 Dice 最高者。

例如：

```text
正常图误报率 < 10%
```

再在满足条件的 checkpoint 中选最佳 Dice。

---

## 14. 推理后处理

U-Net 输出的是概率图，不是最终掩码。  
因此必须做后处理。

## 14.1 阈值搜索
不要固定写死 0.5。  
在验证集上搜索：

```text
0.30, 0.35, 0.40, 0.45, 0.50, 0.55
```

选综合结果最好的阈值。

## 14.2 连通域过滤
搜索最小保留面积：

```text
5, 10, 20 像素
```

注意：你的缺陷本来就小，这个阈值不能设大。

## 14.3 检查区域掩码
如果你已经知道缺陷只可能出现在 ROI 的某个子区域，  
可以额外准备一个“检查区域掩码”，把明显不可能的位置清掉。

---

## 15. 从分割结果派生检测与像素量化

训练主模型时仍然只做分割。  
检测和量化作为后处理派生，不单独再训练一个大模型。

## 15.1 检测框
从最终二值掩码中：

- 找连通域；
- 对每个连通域取外接矩形框。

输出：
- `x_min, y_min, x_max, y_max`

## 15.2 像素级几何量
对每个连通域计算：

- 面积（像素）
- 周长（像素）
- 外接框宽
- 外接框高
- 长宽比
- 等效直径（可选）

这些值写入 `json` 或 `csv`。

---

## 16. 早停规则

建议设：

```text
early_stop_patience = 12
```

如果验证集主指标连续 12 个 epoch 没提升，就停止当前阶段训练。

---

## 17. 随机种子与复现

Codex 实现时必须统一随机种子：

- Python
- NumPy
- PyTorch
- DataLoader worker

每一轮训练都记录：

- 随机种子
- 配置文件
- 提交版本
- 数据划分文件
- 模型权重路径

---

## 18. Codex 具体开发任务拆解

下面这部分是直接给 Codex 的待办清单。

## 任务 1：数据划分脚本
文件：`scripts/make_splits.py`

要做的事：

1. 读取 `metadata.csv`；
2. 按 `event_id / part_id / video_id` 划分；
3. 固定测试集；
4. 对剩余样本生成 5 折 train/val 列表；
5. 输出 `txt` 或 `csv`。

验收标准：
- 同一事件不会同时出现在 train 和 val/test；
- 输出文件可复用。

---

## 任务 2：局部块索引构建
文件：`scripts/build_patch_index.py`

要做的事：

1. 读取缺陷图和 mask；
2. 计算缺陷中心与外接框；
3. 为每张缺陷图生成正样本裁块参数；
4. 为正常图和缺陷图非缺陷区域生成负样本裁块参数；
5. 输出 patch 索引文件。

验收标准：
- 每条索引都能唯一定位到一张原图和一个裁块参数；
- 正样本块内缺陷不被裁掉；
- 负样本块不包含缺陷 mask。

---

## 任务 3：PatchDataset
文件：`src/datasets/patch_dataset.py`

要做的事：

1. 根据 patch 索引在线裁图；
2. 执行同步增强；
3. 返回：
   - image tensor
   - mask tensor
   - meta 信息

验收标准：
- image 和 mask 尺寸一致；
- 掩码在增强后不漂移；
- 正负样本标签正确。

---

## 任务 4：ROIDataset
文件：`src/datasets/roi_dataset.py`

要做的事：

1. 读取整张 ROI 图与 mask；
2. 支持 train / val / test 模式；
3. 支持正常图没有缺陷时自动生成空 mask；
4. 支持保守增强。

验收标准：
- 缺陷 ROI 与空 mask 正常 ROI 都能读取；
- batch 中维度一致；
- 正常图 mask 全零。

---

## 任务 5：模型实现
文件：`src/models/unet_resnet34.py`

要做的事：

1. 用 torchvision 的 ResNet34 预训练权重做编码器；
2. 实现标准 U-Net 解码器；
3. 输出单通道 logits。

验收标准：
- 输入 `B x 3 x H x W`；
- 输出 `B x 1 x H x W`；
- 参数可正常保存/加载。

---

## 任务 6：损失函数
文件：`src/losses/dice_loss.py`、`src/losses/combo_loss.py`

要做的事：

1. 实现 DiceLoss；
2. 实现 `BCEWithLogitsLoss + DiceLoss` 组合；
3. 支持 `pos_weight`。

验收标准：
- 前向和反向都正常；
- 对全背景 batch 不出现 NaN；
- loss 值合理。

---

## 任务 7：训练器
文件：`src/engine/trainer.py`

要做的事：

1. 支持 AMP；
2. 支持 AdamW；
3. 支持 ReduceLROnPlateau；
4. 支持 early stopping；
5. 支持按验证集主指标保存最优模型；
6. 记录日志到 `csv` 和终端。

验收标准：
- 能完整跑完一个 epoch；
- 能自动保存 best checkpoint；
- 能恢复中断训练。

---

## 任务 8：评估器
文件：`src/engine/evaluator.py`

要做的事：

1. 计算 Dice、IoU；
2. 计算图像级召回；
3. 计算正常图误报率；
4. 计算小缺陷子集结果；
5. 支持阈值搜索与最小连通域搜索。

验收标准：
- 指标结果可复现；
- 输出到 `json` 或 `csv`；
- 能按不同阈值批量评估。


---

## 任务 10：训练脚本
文件：
- `scripts/train_stage1_patch.py`
- `scripts/train_stage2_full.py`

要做的事：

### `train_stage1_patch.py`
- 读取阶段 1 配置；
- 训练局部块模型；
- 保存 `best_stage1.pth`。

### `train_stage2_full.py`
- 读取阶段 2 配置；
- 加载 `best_stage1.pth`；
- 训练整图模型；
- 保存 `best_stage2.pth`。

验收标准：
- 两个阶段都能独立运行；
- 第二阶段可直接加载第一阶段权重。

---

## 19. 第一版默认配置（可以直接做 YAML）

## `configs/stage1_patch.yaml`

```yaml
seed: 42
image_size: 384
batch_size: 8
epochs: 30
optimizer: adamw
encoder_lr: 5.0e-5
decoder_lr: 2.0e-4
weight_decay: 1.0e-4
scheduler: reduce_on_plateau
scheduler_factor: 0.5
scheduler_patience: 5
min_lr: 1.0e-6
loss:
  bce_weight: 0.5
  dice_weight: 0.5
  pos_weight: 12.0
amp: true
freeze_encoder_epochs: 3
early_stop_patience: 12
val_metric: defect_dice
```

## `configs/stage2_full.yaml`

```yaml
seed: 42
image_size: 640
batch_size: 4
epochs: 25
optimizer: adamw
encoder_lr: 2.0e-5
decoder_lr: 1.0e-4
weight_decay: 1.0e-4
scheduler: reduce_on_plateau
scheduler_factor: 0.5
scheduler_patience: 5
min_lr: 1.0e-6
loss:
  bce_weight: 0.5
  dice_weight: 0.5
  pos_weight: 12.0
amp: true
early_stop_patience: 12
val_metric: defect_dice
threshold_candidates: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
min_component_area_candidates: [5, 10, 20]
```

---

## 20. 推荐运行顺序

### 第一步
执行数据划分：

```bash
python scripts/make_splits.py
```

### 第二步
生成局部块索引：

```bash
python scripts/build_patch_index.py
```

### 第三步
训练阶段 1：

```bash
python scripts/train_stage1_patch.py --config configs/stage1_patch.yaml --fold 0
```

### 第四步
训练阶段 2：

```bash
python scripts/train_stage2_full.py --config configs/stage2_full.yaml --fold 0
```

### 第五步
在验证集搜索阈值和连通域参数：

```bash
python scripts/validate.py --config configs/stage2_full.yaml --fold 0
```

### 第六步
固定阈值后跑测试：

```bash
python scripts/test.py --config configs/stage2_full.yaml --fold 0
```

---

## 21. 第一版不要做的事

1. 不要从零初始化整个 U-Net；
2. 不要一开始就直接只用整张 640×640 ROI 训练；
3. 不要每个 epoch 都把 1 万张正常图全塞进去；
4. 不要只看总体像素准确率；
5. 不要一上来做多类别分割；
6. 不要先上扩散模型，再回头补基线；
7. 不要把验证集和测试集用于阈值之外的训练。

---

## 22. 预计最常见的失败现象与处理方式

## 现象 1：模型几乎全部预测背景
处理顺序：

1. 提高 `pos_weight`；
2. 增加正样本局部块比例；
3. 先检查标注是否过小或错位；
4. 检查阈值是不是过高。

## 现象 2：正常图到处飘噪点
处理顺序：

1. 降低 `pos_weight`；
2. 增加困难负样本；
3. 提高最小连通域阈值；
4. 检查增强是否过强。

## 现象 3：整图阶段效果不如局部块阶段
处理顺序：

1. 加入局部块回放；
2. 降低整图微调学习率；
3. 检查整图正常样本是否太多。

## 现象 4：训练集好，验证集差
处理顺序：

1. 减少模型自由度；
2. 增加数据增强；
3. 检查训练/验证是否存在分布偏差；
4. 回到事件级划分确认无泄漏。

---

## 23. 计划收尾：交付物清单

Codex 实现完成后，最终应交付：

1. 训练/验证/测试划分文件；
2. 局部块索引文件；
3. 阶段 1 最优模型；
4. 阶段 2 最优模型；
5. 验证集指标报告；
6. 测试集指标报告；
7. 每张测试图的预测掩码；
8. 每张测试图的几何量化结果；
9. 一份可重复运行的 README。

---

## 24. 参考依据

[1] Ronneberger O, Fischer P, Brox T. **U-Net: Convolutional Networks for Biomedical Image Segmentation**. arXiv:1505.04597.  
https://arxiv.org/abs/1505.04597

[2] Isensee F, Jaeger PF, Kohl SAA, et al. **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation**. Nature Methods, 2021.  
https://www.nature.com/articles/s41592-020-01008-z

[3] TorchVision 官方文档：**Models and pre-trained weights**  
https://docs.pytorch.org/vision/main/models.html

[4] PyTorch 官方文档：**BCEWithLogitsLoss**  
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

[5] MONAI 官方文档：**Loss functions / Dice 系列损失**  
https://monai.readthedocs.io/en/1.4.0/losses.html

[6] PyTorch 官方文档：**AdamW**  
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html

[7] PyTorch 官方文档：**ReduceLROnPlateau**  
https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

[8] PyTorch 官方文档：**torch.amp**  
https://docs.pytorch.org/docs/stable/amp.html

[9] PyTorch 官方文档：**Automatic Mixed Precision examples**  
https://docs.pytorch.org/docs/stable/notes/amp_examples.html
