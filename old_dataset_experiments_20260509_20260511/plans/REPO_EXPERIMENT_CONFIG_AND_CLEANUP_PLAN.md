# ResNET34U-NET：正式实验配置与工程整理建议

> Superseded note: this was the 2026-05-11 cleanup plan. The current repository
> has been collapsed to one active config, `configs/canonical_baseline.yaml`.
> Historical four-experiment outputs are frozen under
> `old_dataset_experiments_20260509_20260511/`.

生成日期：2026-05-11  
适用仓库：`KTH-YANGYI/ResNET34U-NET` 当前 `main` 分支  
目标：确定 baseline 与 3 个创新实验的正式配置，并给出代码/目录整理方案，方便 Codex 执行。

---

## 0. 结论先行

正式实验不要重新发明一套 baseline。应当以之前消融实验中最强的 U-Net 设置作为唯一正式 baseline：

```text
Stage1: configs/stage1_p0_a40.yaml 对应的 Stage1 checkpoints
Stage2 baseline: configs/stage2_p0_a40_e50_bs48_pw12.yaml
```

原因：之前消融表中，`pos_weight=12` 的 U-Net 在 OOF 上取得了最平衡的结果：

```text
Dice = 0.756876
IoU = 0.633878
defect recall = 0.995261
normal FPR = 0.000000
normal FP = 0
threshold / min_area = 0.80 / 24
```

所以正式的 4 个实验应该是：

| ID | 实验名 | 方法 | 对照关系 |
|---|---|---|---|
| 00 | `baseline_resnet34_unet_pw12` | 原 ResNet34-U-Net，`pos_weight=12`，无新模块 | 论文主 baseline |
| 01 | `tbn_d1` | baseline + 1 层 Transformer bottleneck | 测试全局上下文是否有用 |
| 02 | `tbn_d1_hnproto` | baseline + Transformer bottleneck + hard-negative prototype attention | 测试 hard-negative prototype 是否比普通 TBN 进一步有效 |
| 03 | `skipgate_d4d3` | baseline + decoder skip attention gate at `d4,d3` | 测试 skip filtering 是否能减少误检/改善边界 |

`deep supervision`、`boundary auxiliary`、`normal_fp_loss_weight` 不建议进入正式新方法配置，因为之前消融里它们没有超过 `pw12` baseline，甚至可能带来 normal FP 或 recall 下降。

---

## 1. 为什么选 `pw12` 作为正式 baseline

### 1.1 不能用默认 `stage2.yaml` 当正式 baseline

默认 `configs/stage2.yaml` 适合开发和基本训练，不适合作为论文主 baseline。正式实验应该使用已经稳定过的 P0 A40/A100 训练设置：

```text
image_size = 640
batch_size = 48
epochs = 50
encoder_lr = 2.0e-5
decoder_lr = 1.0e-4
bce_weight = 0.5
dice_weight = 0.5
pos_weight = 12.0
use_hard_normal_replay = true
stage2_hard_normal_ratio = 0.40
threshold_grid_end = 0.95
min_area_grid = [0, 8, 16, 24, 32, 48]
```

### 1.2 不建议用 `pw6` 或 `pw8`

之前结果：

| variant | Dice | IoU | defect recall | normal FPR | normal FP |
|---|---:|---:|---:|---:|---:|
| `pw6` | 0.755910 | 0.635336 | 0.990521 | 0.016949 | 1 |
| `pw8` | 0.751339 | 0.631859 | 0.981043 | 0.016949 | 1 |
| `pw12` | 0.756876 | 0.633878 | 0.995261 | 0.000000 | 0 |

`pw6` 的 IoU 略高于 `pw12`，但有 1 个 normal FP，且 defect recall 低于 `pw12`。你的任务里 normal 图误检是非常关键的风险，所以 `pw12` 是更合理的正式 baseline。

### 1.3 不建议用 `stage1_p0_a40_s1e50.yaml` 做正式主线

仓库里有一个 50 epoch 的 Stage1 配置：

```text
configs/stage1_p0_a40_s1e50.yaml
```

但之前最强 Stage2 baseline 的 checkpoint 路径指向的是：

```text
outputs/experiments/p0_a40_20260508/stage1/fold{fold}/best_stage1.pt
```

对应的是 `configs/stage1_p0_a40.yaml` 那条主线。正式实验为了和历史最优 baseline 一致，应优先使用同一套 Stage1 checkpoint。除非你愿意重新跑：

```text
Stage1 s1e50 + Stage2 baseline + 3 个新方法
```

否则不要混用 Stage1 checkpoint。最安全方案是：

```text
所有 4 个正式实验共享同一套 Stage1 checkpoint。
baseline 和三个新方法只改变 Stage2 模型结构。
```

---

## 2. 推荐的正式配置集合

建议不要继续叫 `transformer_a40_20260510`，因为你现在有 4 张 A100。更推荐把硬件从方法配置里拿掉：

```text
configs/experiments/attention_20260511/
  00_baseline_resnet34_unet_pw12.yaml
  01_tbn_d1.yaml
  02_tbn_d1_hnproto.yaml
  03_skipgate_d4d3.yaml
```

如果短期不想重构 config 系统，可以先直接复制现有 YAML 并改路径。长期再做 `base + override`。

---

## 3. 00 baseline 正式配置

基于：

```text
configs/stage2_p0_a40_e50_bs48_pw12.yaml
```

新建：

```text
configs/experiments/attention_20260511/00_baseline_resnet34_unet_pw12.yaml
```

关键配置：

```yaml
seed: 42
fold: 0
model_name: resnet34_unet_pw12

image_size: 640
samples_path: manifests/samples.csv
batch_size: 48
epochs: 50

encoder_lr: 2.0e-5
decoder_lr: 1.0e-4
weight_decay: 1.0e-4

bce_weight: 0.5
dice_weight: 0.5
pos_weight: 12.0
normal_fp_loss_weight: 0.0
normal_fp_topk_ratio: 0.10

amp: true
early_stop_patience: 16
early_stop_min_delta: 0.0
lr_factor: 0.5
lr_patience: 4
min_lr: 1.0e-7

num_workers: 8
pretrained: false
device: cuda
auto_evaluate_after_train: true

threshold: 0.50
train_eval_threshold: 0.50
train_eval_min_area: 0

use_imagenet_normalize: true
augment_enable: true
augment_hflip_p: 0.5
augment_vflip_p: 0.5
augment_rotate_deg: 10
augment_brightness: 0.10
augment_contrast: 0.10
augment_gamma: 0.0
augment_noise_std: 0.015
augment_blur_p: 0.0

target_normal_fpr: 0.10
lambda_fpr_penalty: 2.0
threshold_grid_start: 0.10
threshold_grid_end: 0.95
threshold_grid_step: 0.02
min_area_grid: [0, 8, 16, 24, 32, 48]

random_normal_k_factor: 1.0
use_hard_normal_replay: true
stage2_hard_normal_ratio: 0.40
hard_normal_max_repeats_per_epoch: 2
hard_normal_warmup_epochs: 2
hard_normal_refresh_every: 2
hard_normal_pool_factor: 3.0

# model switches: all off
transformer_bottleneck_enable: false
prototype_attention_enable: false
skip_attention_enable: false
deep_supervision_enable: false
boundary_aux_enable: false

stage1_checkpoint_template: outputs/experiments/p0_a40_20260508/stage1/fold{fold}/best_stage1.pt
save_dir_template: outputs/experiments/attention_20260511/00_baseline_resnet34_unet_pw12/results/stage2/fold{fold}
global_postprocess_path: outputs/experiments/attention_20260511/00_baseline_resnet34_unet_pw12/results/stage2/oof_global_postprocess.json
```

注意：如果你在 Alvis/A100 上用绝对路径，就把最后三个路径改成绝对路径。但建议后续支持 `${PROJECT_ROOT}` 和 `${EXPERIMENT_ROOT}`，不要把 `/mimer/.../Yi Yang/...` 这种机器路径提交到主配置。

---

## 4. 01 Transformer bottleneck 配置

新建：

```text
configs/experiments/attention_20260511/01_tbn_d1.yaml
```

继承 baseline 的全部训练/数据/后处理设置，只改模型开关和输出路径：

```yaml
model_name: resnet34_unet_tbn_d1

transformer_bottleneck_enable: true
transformer_bottleneck_layers: 1
transformer_bottleneck_heads: 8
transformer_bottleneck_dropout: 0.1

prototype_attention_enable: false
skip_attention_enable: false
deep_supervision_enable: false
boundary_aux_enable: false

# 新模块没有 Stage1 权重，所以需要非严格加载
stage1_load_strict: false

stage1_checkpoint_template: outputs/experiments/p0_a40_20260508/stage1/fold{fold}/best_stage1.pt
save_dir_template: outputs/experiments/attention_20260511/01_tbn_d1/results/stage2/fold{fold}
global_postprocess_path: outputs/experiments/attention_20260511/01_tbn_d1/results/stage2/oof_global_postprocess.json
```

为什么用 `layers=1`：

```text
1. bottleneck feature 大约是 20x20 token，attention 成本可控；
2. 数据量小，先用浅层 Transformer 降低过拟合风险；
3. 这是最干净的第一增量实验。
```

后续如果 01 有提升，可以在附录或后续 ablation 增加：

```text
01b_tbn_d2: transformer_bottleneck_layers = 2
```

但不要把 `d2` 当第一轮主实验，否则计算变多且结论不一定更清楚。

---

## 5. 02 Hard-negative prototype attention 配置

新建：

```text
configs/experiments/attention_20260511/02_tbn_d1_hnproto.yaml
```

这个实验建议建立在 `01_tbn_d1` 上，而不是直接 baseline + prototype。原因是 prototype attention 是 bottleneck token 到 patch-level prototype memory 的 cross-attention，本质上和 TBN 属于同一层 feature refinement。

关键配置：

```yaml
model_name: resnet34_unet_tbn_d1_hnproto

transformer_bottleneck_enable: true
transformer_bottleneck_layers: 1
transformer_bottleneck_heads: 8
transformer_bottleneck_dropout: 0.1

prototype_attention_enable: true
prototype_attention_heads: 8
prototype_attention_dropout: 0.1
prototype_pos_max: 128
prototype_neg_max: 128
prototype_l2_normalize: true
prototype_batch_size: 64

skip_attention_enable: false
deep_supervision_enable: false
boundary_aux_enable: false

stage1_load_strict: false
stage1_checkpoint_template: outputs/experiments/p0_a40_20260508/stage1/fold{fold}/best_stage1.pt
prototype_bank_path_template: outputs/experiments/attention_20260511/02_tbn_d1_hnproto/results/prototype_banks/fold{fold}/prototype_bank.pt
save_dir_template: outputs/experiments/attention_20260511/02_tbn_d1_hnproto/results/stage2/fold{fold}
global_postprocess_path: outputs/experiments/attention_20260511/02_tbn_d1_hnproto/results/stage2/oof_global_postprocess.json
```

正式报告中要比较两种 delta：

```text
Delta-1: 02_tbn_d1_hnproto - 00_baseline_resnet34_unet_pw12
Delta-2: 02_tbn_d1_hnproto - 01_tbn_d1
```

`Delta-2` 更重要，因为它回答的是：hard-negative prototypes 是否在普通 Transformer bottleneck 之上有额外贡献。

建议 Codex 给 prototype bank 输出一个 summary：

```text
prototype_bank_summary.json
  fold
  num_pos_rows_before_limit
  num_neg_rows_before_limit
  num_pos_prototypes
  num_neg_prototypes
  pos_patch_type_counts
  neg_patch_type_counts
  feature_dim
  source_stage1_checkpoint
```

否则审稿人可能问：你的 hard negative memory 里到底放了什么。

---

## 6. 03 Decoder skip attention gate 配置

新建：

```text
configs/experiments/attention_20260511/03_skipgate_d4d3.yaml
```

关键配置：

```yaml
model_name: resnet34_unet_skipgate_d4d3

transformer_bottleneck_enable: false
prototype_attention_enable: false

skip_attention_enable: true
skip_attention_levels: ["d4", "d3"]

# 建议先改代码支持这个参数。理由见第 10.3 节。
skip_attention_gamma_init: 0.0

deep_supervision_enable: false
boundary_aux_enable: false

stage1_load_strict: false
stage1_checkpoint_template: outputs/experiments/p0_a40_20260508/stage1/fold{fold}/best_stage1.pt
save_dir_template: outputs/experiments/attention_20260511/03_skipgate_d4d3/results/stage2/fold{fold}
global_postprocess_path: outputs/experiments/attention_20260511/03_skipgate_d4d3/results/stage2/oof_global_postprocess.json
```

如果暂时不改代码，当前 `SkipAttentionGate.gamma` 初始化为 `1.0`，实际效果约等于一开始就用 sigmoid attention 缩放 skip feature。它不一定错，但它不是 identity initialization；为了公平和稳定，建议把 `gamma` 初始值做成 config，并在正式实验中用 `0.0`。

---

## 7. A100 运行方式

你有 4 张 A100，建议仍然保持：

```text
一张 GPU 跑一个 fold
batch_size = 48 不变
folds = 0,1,2,3 并行
```

不要因为 A100 显存更大就把 `batch_size` 从 48 提到更高。否则 baseline 和新方法虽然都可以重跑，但它们不再和历史最优消融表完全一致，论文解释会更麻烦。

建议把 runner 改名：

```text
scripts/run_attention_experiment_set_4gpu.sh
scripts/cluster/alvis_attention_4a100.sbatch
```

不要继续使用：

```text
run_transformer_attention_experiment_set_4a40.sh
alvis_transformer_attention_4a40.sbatch
configs/transformer_a40_20260510/
```

因为正式实验记录里出现 A40/A100 混杂，会降低复现可信度。

---

## 8. 正式实验流程

### 8.1 准备数据与 Stage1

如果服务器上已有这套 Stage1 checkpoint：

```text
outputs/experiments/p0_a40_20260508/stage1/fold0/best_stage1.pt
outputs/experiments/p0_a40_20260508/stage1/fold1/best_stage1.pt
outputs/experiments/p0_a40_20260508/stage1/fold2/best_stage1.pt
outputs/experiments/p0_a40_20260508/stage1/fold3/best_stage1.pt
```

则可以直接复用。否则先跑：

```bash
python scripts/prepare_samples.py --test-ratio 0.20 --test-seed 2026
python scripts/build_patch_index.py --config configs/stage1_p0_a40.yaml
bash scripts/train_all_folds.sh --gpus 0,1,2,3 --stage1-only --skip-prepare
```

注意：如果 `train_all_folds.sh` 默认使用 `configs/stage1.yaml`，Codex 需要确认它是否支持传入 `configs/stage1_p0_a40.yaml`。不能误跑默认 Stage1。

### 8.2 跑四个 Stage2 实验

建议新 runner 顺序：

```text
00_baseline_resnet34_unet_pw12
01_tbn_d1
02_tbn_d1_hnproto
03_skipgate_d4d3
```

每个实验内部：

```bash
# 1. 如需要，构建 prototype bank；仅 02 需要
python scripts/build_stage1_prototype_bank.py --config CONFIG --fold FOLD

# 2. 训练四折 Stage2
python scripts/train_stage2.py --config CONFIG --fold FOLD

# 3. OOF pooled threshold/min_area search
python scripts/search_oof_postprocess.py --config CONFIG --folds 0,1,2,3

# 4. frozen holdout ensemble
python scripts/infer_holdout_ensemble.py --config CONFIG --folds 0,1,2,3
```

### 8.3 汇总方式

汇总脚本必须包含 baseline，而不是只汇总 3 个新方法：

```text
00_baseline_resnet34_unet_pw12
01_tbn_d1
02_tbn_d1_hnproto
03_skipgate_d4d3
```

最终报告至少包含：

```text
OOF table
Frozen labeled holdout table
Delta vs baseline table
Delta vs TBN table for HNProto
Paired bootstrap CI
Model complexity table
```

---

## 9. 如何判断“有提升”

正式标准建议提前写死：

```text
Clear improvement:
1. OOF defect Dice 比 baseline 至少 +0.010；
2. paired bootstrap 95% CI 的 DeltaDice lower bound > 0；
3. normal_fp_count 不高于 baseline；如多 1 个 FP，需要 Dice/Boundary/Component 明显提升并在论文解释；
4. defect_image_recall >= 0.99；
5. pixel AUPRC 或 boundary F1 至少一个同步提升；
6. frozen labeled holdout 不明显退化；
7. threshold/min_area 必须由 OOF 冻结，不能在 holdout 上搜索。
```

如果只有平均 Dice 提升，但 CI 包含 0，论文里只能写：

```text
promising but not statistically conclusive
```

不能写 strong improvement。

### 9.1 指标体系是否需要大改

不需要大改，但要继续坚持“多层级指标”：

| 指标层级 | 建议指标 | 用途 |
|---|---|---|
| pixel overlap | Dice, IoU | 主分割质量 |
| false alarm | normal FP count, normal FPR | normal 图误检控制 |
| image-level safety | defect image recall | 是否漏掉缺陷图 |
| imbalanced pixel classification | pixel precision/recall/F1, pixel AUPRC | 类别极不平衡下的概率质量 |
| object/component | component recall at tolerance | 细小 crack 是否按连通结构被检出 |
| boundary | boundary F1 at tolerance | 边界/细线质量 |
| selection-only | stage2_score | 只用于 checkpoint / post-processing selection |

权威依据：

- Metrics Reloaded 强调 metric selection 应该根据 problem fingerprint，而不是只报一个常规指标。
- Taha & Hanbury 讨论了分割指标的选择问题，说明不同指标适合不同误差类型。
- Saito & Rehmsmeier 指出在类别不平衡问题里，PR/precision-recall 信息通常比 ROC 更能反映正类预测质量。

---

## 10. 当前代码优化建议

### 10.1 P0：先修正式实验可信度相关问题

#### 10.1.1 汇总脚本必须加入 baseline

当前 `scripts/summarize_transformer_experiments.py` 只汇总：

```text
01_transformer_bottleneck
02_hard_negative_prototype_attention
03_decoder_skip_attention_gate
```

正式版本应改为：

```python
EXPERIMENTS = [
    ("00_baseline_resnet34_unet_pw12", "Baseline ResNet34-U-Net pw12"),
    ("01_tbn_d1", "Transformer bottleneck d1"),
    ("02_tbn_d1_hnproto", "TBN d1 + hard-negative prototype attention"),
    ("03_skipgate_d4d3", "Decoder skip attention gate d4+d3"),
]
```

并增加：

```text
DeltaDice_vs_baseline
DeltaIoU_vs_baseline
DeltaNormalFP_vs_baseline
DeltaBoundaryF1_vs_baseline
DeltaComponentRecall_vs_baseline
```

#### 10.1.2 增加 paired significance 脚本

新增：

```text
scripts/analysis/paired_oof_significance.py
```

输入：

```text
baseline oof_per_image.csv
variant  oof_per_image.csv
```

输出：

```text
significance_report.json
significance_report.md
```

至少计算：

```text
Delta defect Dice mean
Delta pixel F1 mean
Delta boundary F1 mean
Delta component recall mean
Delta normal FP count
95% paired bootstrap CI
paired permutation p-value，可选
```

#### 10.1.3 runner 不要写死 fold -> GPU

当前 `run_stage2_variant_4gpu.sh` 使用：

```bash
local gpu="${fold}"
```

如果将来只跑 fold `2,3` 或使用非连续 GPU，会出问题。建议改成：

```bash
CUDA_FOLD_GPUS="${CUDA_FOLD_GPUS:-0,1,2,3}"
IFS=',' read -r -a gpu_list <<< "${CUDA_FOLD_GPUS}"

for idx in "${!folds[@]}"; do
  fold="${folds[$idx]}"
  gpu="${gpu_list[$idx]}"
  ...
done
```

#### 10.1.4 不要用 `eval` 执行训练命令

当前 runner 中有：

```bash
eval "${command_template}"
```

项目路径里有空格，例如 `Yi Yang`，`eval` 更容易引入 quoting 问题。建议拆成显式函数或数组命令：

```bash
python scripts/train_stage2.py --config "${config}" --fold "${fold}"
```

对于 prototype 阶段也直接写函数，不要构造字符串再 eval。

#### 10.1.5 Slurm 与脚本命名改成 A100 或 4GPU

把：

```text
alvis_transformer_attention_4a40.sbatch
run_transformer_attention_experiment_set_4a40.sh
configs/transformer_a40_20260510/
```

改成：

```text
scripts/cluster/alvis_attention_4a100.sbatch
scripts/experiments/run_attention_experiment_set_4gpu.sh
configs/experiments/attention_20260511/
```

Slurm：

```bash
#SBATCH --gpus-per-node=A100:4
```

---

### 10.2 P1：模型代码优化

#### 10.2.1 减少 `build_model` 的重复代码

当前 `build_model` 里 `UNetResNet34(...)` 参数重复了三遍：

```text
pretrained=false 分支
pretrained=true 成功分支
pretrained=true 失败 fallback 分支
```

建议改成：

```python
def _make_unet(encoder_weights, **kwargs):
    return UNetResNet34(encoder_weights=encoder_weights, **kwargs)
```

或者用 dataclass：

```python
@dataclass
class ModelConfig:
    pretrained: bool = False
    deep_supervision: bool = False
    boundary_aux: bool = False
    transformer_bottleneck_enable: bool = False
    transformer_bottleneck_layers: int = 0
    transformer_bottleneck_heads: int = 8
    transformer_bottleneck_dropout: float = 0.1
    prototype_attention_enable: bool = False
    prototype_bank_path: str = ""
    prototype_attention_heads: int = 8
    prototype_attention_dropout: float = 0.1
    skip_attention_enable: bool = False
    skip_attention_levels: tuple[str, ...] = ("d4", "d3")
```

这样以后加模块不会继续让构造函数膨胀。

#### 10.2.2 预训练权重失败不要静默变成随机初始化

当前逻辑是：如果 ImageNet 权重加载失败，就 warning 然后随机初始化。这在开发时友好，但在正式实验中危险，因为 Stage1 可能悄悄从 pretrained 变成 random。

建议：

```yaml
pretrained: true
pretrained_strict: true
```

当 `pretrained_strict=true` 且权重加载失败时，直接报错。正式实验应避免 silent fallback。

#### 10.2.3 Skip gate 建议 identity initialization

当前：

```python
self.gamma = nn.Parameter(torch.ones(1))
return skip * (1.0 + self.gamma * (attention - 1.0))
```

当 `gamma=1` 时，初始输出约等于：

```text
skip * attention
```

这会从训练一开始就强行抑制 skip feature。建议改成：

```python
class SkipAttentionGate(nn.Module):
    def __init__(..., gamma_init=0.0):
        ...
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
```

配置里加：

```yaml
skip_attention_gamma_init: 0.0
```

这样初始时是 identity：

```text
skip * 1.0
```

训练会自己学是否需要 suppression。这个和 Transformer/prototype 的 `gamma=0` residual gate 思路一致，更公平。

#### 10.2.4 记录新模块 gamma

训练结束时，在 checkpoint summary 或 metrics JSON 里记录：

```text
transformer_bottleneck.gamma
prototype_attention.gamma
skip_gate_d4.gamma
skip_gate_d3.gamma
```

如果 gamma 训练后仍接近 0，说明新模块实际贡献很小；如果变大，可以作为解释性分析。

#### 10.2.5 Prototype attention 增加可解释性输出

建议 `PrototypeCrossAttention` 支持可选返回 attention summary：

```text
mean_pos_attention
mean_neg_attention
max_pos_attention
max_neg_attention
```

用于分析：

```text
defect images
normal images
false-positive images
false-negative images
```

这对论文很有帮助，尤其是证明 hard-negative prototypes 不是装饰。

---

### 10.3 P1：训练与数据加载优化

#### 10.3.1 DataLoader 参数做成 config

建议在 Stage1/Stage2 config 中支持：

```yaml
pin_memory: true
persistent_workers: true
prefetch_factor: 2
```

A100 很快，训练可能更容易被 CPU/data loading 卡住。`pin_memory` 和 `persistent_workers` 一般能减少 host-to-device 和 worker 重启开销。

#### 10.3.2 可复现性与速度分开

当前 `set_seed` 会设置：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

这对可复现有利，但可能牺牲 A100 速度。建议加 config：

```yaml
deterministic: true
cudnn_benchmark: false
```

正式指标实验用 deterministic；大规模调参或 sanity check 可以开 benchmark。

#### 10.3.3 增加模型复杂度脚本

新增：

```text
scripts/analysis/model_complexity.py
```

输出：

```text
params_total
params_trainable
forward_time_ms at 640x640
peak_cuda_memory_mb
```

这样论文可以说明新方法的代价。审稿人常会问：提升一点 Dice，计算成本是多少？

---

## 11. 配置系统整理建议

当前问题：

```text
1. configs/ 里混有默认配置、P0 实验、A40 实验、消融实验、Transformer 实验；
2. 新配置中出现绝对服务器路径；
3. 文件名里混有硬件名 A40，但实际可能在 A100 上运行；
4. 大量配置重复 90% 以上内容，后续容易改漏。
```

### 11.1 推荐配置目录

```text
configs/
  base/
    data_roi640.yaml
    stage1_patch.yaml
    stage2_pw12.yaml
    metrics_oof.yaml
    augmentation_stage1.yaml
    augmentation_stage2.yaml

  experiments/
    attention_20260511/
      00_baseline_resnet34_unet_pw12.yaml
      01_tbn_d1.yaml
      02_tbn_d1_hnproto.yaml
      03_skipgate_d4d3.yaml
      README.md

  archive/
    20260508_p0/
      stage1_p0_a40.yaml
      stage2_p0_a40.yaml
    20260509_ablation/
      stage2_p0_a40_e50_bs48_pw6.yaml
      stage2_p0_a40_e50_bs48_pw8.yaml
      stage2_p0_a40_e50_bs48_pw12.yaml
      stage2_p0_a40_e50_bs48_pw12_fp003.yaml
      stage2_p0_a40_e50_bs48_pw12_fp005.yaml
      stage2_p0_a40_e50_bs48_pw12_deepsup.yaml
      stage2_p0_a40_e50_bs48_pw12_boundary.yaml
```

### 11.2 支持 `extends`

建议升级 `load_yaml`，支持：

```yaml
extends:
  - configs/base/data_roi640.yaml
  - configs/base/stage2_pw12.yaml
  - configs/base/metrics_oof.yaml

model_name: resnet34_unet_tbn_d1
transformer_bottleneck_enable: true
transformer_bottleneck_layers: 1
save_dir_template: ${EXPERIMENT_ROOT}/01_tbn_d1/results/stage2/fold{fold}
```

加载逻辑：

```python
base_cfg = {}
for base_path in cfg.pop("extends", []):
    base_cfg = deep_merge(base_cfg, load_yaml(base_path))
cfg = deep_merge(base_cfg, cfg)
```

注意：如果暂时不想引入 Hydra/OmegaConf，可以自己实现简单 deep merge。当前项目依赖很少，这是优点，不一定要为了 config 继承引入大框架。

### 11.3 支持环境变量展开

正式配置里不要写：

```text
/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/UNET_two_stage/...
```

建议写：

```yaml
project_root: ${PROJECT_ROOT}
experiment_root: ${EXPERIMENT_ROOT}
samples_path: ${PROJECT_ROOT}/manifests/samples.csv
stage1_checkpoint_template: ${PROJECT_ROOT}/outputs/experiments/p0_a40_20260508/stage1/fold{fold}/best_stage1.pt
save_dir_template: ${EXPERIMENT_ROOT}/01_tbn_d1/results/stage2/fold{fold}
```

在 Python 里做：

```python
import os
value = os.path.expandvars(value)
```

这样本地、Alvis、不同用户路径都能复用。

---

## 12. 目录结构整理建议

当前根目录里有：

```text
configs/
experiment_logs/
outputs/
scripts/
src/
DATASET_DESCRIPTION.md
EXPERIMENTS_20260509.md
README.md
THESIS_METHODOLOGY.md
UNET_DATA_FLOW.md
requirements.txt
```

这对实验推进可以，但论文/长期维护会显得散。建议整理成：

```text
ResNET34U-NET/
  README.md
  requirements.txt
  pyproject.toml                 # 可选，但建议加

  configs/
    base/
    experiments/
    archive/

  src/
    unetcw/                       # 建议换成项目包名
      data/
        datasets.py
        samples.py
        transforms.py
      models/
        resnet34_unet.py
        transformer_blocks.py
        prototype_memory.py
      training/
        trainer.py
        losses.py
        optim.py
      evaluation/
        metrics.py
        postprocess.py
      mining/
        patch_mining.py
        hard_normal.py
      utils/
        config.py
        io.py
        seed.py

  scripts/
    data/
      prepare_samples.py
      build_patch_index.py
    train/
      train_stage1.py
      train_stage2.py
      train_all_folds.sh
    eval/
      evaluate_val.py
      search_oof_postprocess.py
      infer_holdout.py
      infer_holdout_ensemble.py
    analysis/
      summarize_transformer_experiments.py
      paired_oof_significance.py
      model_complexity.py
      visualize_error_analysis.py
    experiments/
      run_attention_experiment_set_4gpu.sh
      setup_attention_experiment_root.py
    cluster/
      alvis_train.sbatch
      alvis_train_a100.sbatch
      alvis_attention_4a100.sbatch

  docs/
    methodology/
      THESIS_METHODOLOGY.md
      UNET_DATA_FLOW.md
      DATASET_DESCRIPTION.md
    experiments/
      EXPERIMENTS_20260509.md
      ATTENTION_EXPERIMENTS_20260511.md

  tests/
    test_config_loading.py
    test_model_forward.py
    test_postprocess_metrics.py
    test_no_holdout_leakage.py

  outputs/                         # gitignored, local/server only
  experiment_logs/                  # gitignored or only summary docs committed
```

### 12.1 `outputs/` 不建议长期放进 GitHub

当前 `.gitignore` 已经忽略 `outputs/*`，但又允许 CSV/JSON/log/out/err 被追踪。这对短期复盘有用，但长期会让仓库越来越乱。

建议规则：

```text
outputs/ 完全不提交
experiment_logs/ 只提交轻量 summary，而不是每个 run 的散乱日志
正式结果放 docs/experiments/*.md + 少量表格 CSV
```

例如：

```text
docs/experiments/20260509_unet_ablation/summary.md
docs/experiments/20260509_unet_ablation/unet_oof_comparison.csv
docs/experiments/20260511_attention/summary.md
docs/experiments/20260511_attention/oof_comparison.csv
```

---

## 13. 给 Codex 的执行任务清单

### P0：正式实验前必须完成

```text
1. 新建 configs/experiments/attention_20260511/：
   - 00_baseline_resnet34_unet_pw12.yaml
   - 01_tbn_d1.yaml
   - 02_tbn_d1_hnproto.yaml
   - 03_skipgate_d4d3.yaml

2. 新建 scripts/experiments/run_attention_experiment_set_4gpu.sh：
   - 运行 00/01/02/03 四个实验；
   - 02 自动先 build prototype bank；
   - 一折一张 GPU；
   - 支持 CUDA_FOLD_GPUS=0,1,2,3；
   - 不使用 eval；
   - 状态文件写 4 x A100 或 4GPU，不写 A40。

3. 新建 scripts/cluster/alvis_attention_4a100.sbatch：
   - #SBATCH --gpus-per-node=A100:4
   - 支持 PROJECT_ROOT 和 EXPERIMENT_ROOT；
   - 支持 FOLDS 和 CUDA_FOLD_GPUS。

4. 修改 summarize_transformer_experiments.py：
   - 加入 baseline row；
   - 输出 delta vs baseline；
   - 对 02 额外输出 delta vs 01；
   - stage2_score 只标注为 selection-only。

5. 新增 scripts/analysis/paired_oof_significance.py：
   - baseline vs each variant；
   - paired bootstrap 95% CI；
   - 输出 markdown + json。

6. 修改 SkipAttentionGate：
   - 支持 gamma_init；
   - config 中 03 设置 skip_attention_gamma_init: 0.0；
   - 训练后记录 gamma。
```

### P1：正式实验后，论文前完成

```text
1. 新增 scripts/analysis/model_complexity.py。
2. prototype bank 输出 summary JSON。
3. HNProto 输出 attention summary 或至少保存若干解释性统计。
4. DataLoader 支持 pin_memory / persistent_workers / prefetch_factor。
5. build_model 用 dataclass 或 helper 函数减少重复。
6. pretrained 权重加载支持 pretrained_strict。
```

### P2：仓库长期整理

```text
1. configs/base + configs/experiments + configs/archive。
2. src/ 拆成 data/models/training/evaluation/mining/utils。
3. scripts/ 拆成 data/train/eval/analysis/experiments/cluster。
4. docs/experiments 存实验报告，outputs 不再作为 GitHub 结果目录。
5. 加 tests/：config loading、model forward、metrics、no leakage。
```

---

## 14. 最终建议

正式实验现在最合理的配置路线是：

```text
Baseline = stage2_p0_a40_e50_bs48_pw12 的同环境重跑
01 = Baseline + Transformer bottleneck d1
02 = 01 + hard-negative prototype attention
03 = Baseline + skip attention gate d4,d3
```

不要把 deep supervision、boundary aux、normal FP loss 放入这次正式新方法主线；它们适合作为负消融出现在论文里。

工程上，最重要的不是立刻大拆目录，而是先完成：

```text
baseline row + A100 runner + paired CI + config 去 A40/绝对路径化
```

这些会直接影响实验是否可信。目录大整理可以分阶段做，不要在正式实验开跑前做过大的 src 包迁移，以免引入不必要 bug。

---

## 15. 参考依据

仓库依据：

- Current repository: https://github.com/KTH-YANGYI/ResNET34U-NET
- README pipeline and dataset split: https://github.com/KTH-YANGYI/ResNET34U-NET/blob/main/README.md
- Previous ablation summary: https://github.com/KTH-YANGYI/ResNET34U-NET/blob/main/EXPERIMENTS_20260509.md
- Model implementation: https://github.com/KTH-YANGYI/ResNET34U-NET/blob/main/src/model.py
- Transformer/attention modules: https://github.com/KTH-YANGYI/ResNET34U-NET/blob/main/src/transformer_blocks.py
- Current transformer configs: https://github.com/KTH-YANGYI/ResNET34U-NET/tree/main/configs/transformer_a40_20260510

指标依据：

- Maier-Hein et al., Metrics Reloaded: recommendations for image analysis validation, Nature Methods, 2024.
- Taha and Hanbury, Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool, BMC Medical Imaging, 2015.
- Saito and Rehmsmeier, The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets, PLOS ONE, 2015.
