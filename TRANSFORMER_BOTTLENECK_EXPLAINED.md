# Transformer Bottleneck in the Two-Stage ResNet34-U-Net

本文档解释旧实验 `01_tbn_d1` 中使用的 Transformer bottleneck。它不是把整个 U-Net 替换成 Transformer，而是在 ResNet34-U-Net 的最深层 bottleneck 位置插入一个轻量级 Transformer encoder module，用于增强全局上下文建模能力。

相关代码：

- `src/model.py`: 在 U-Net bottleneck 处创建并调用 `TransformerBottleneck`
- `src/transformer_blocks.py`: 定义 `TransformerBottleneck`
- `old_dataset_experiments_20260509_20260511/transformer_attention/configs/transformer_a40_20260510/stage2_tbn_d1_a40.yaml`: `01_tbn_d1` 实验配置

## 1. 加在模型的什么位置

原始 ResNet34-U-Net 的主干可以概括为：

```text
Input ROI
-> ResNet34 Encoder
-> Conv Bottleneck
-> U-Net Decoder
-> Segmentation Mask
```

加入 Transformer bottleneck 后变成：

```text
Input ROI
-> ResNet34 Encoder
-> Conv Bottleneck
-> Transformer Bottleneck
-> U-Net Decoder
-> Segmentation Mask
```

代码中的 forward 逻辑是：

```python
x4 = self.encoder_layer4(x3)
x4 = self.center(x4)
if self.transformer_bottleneck is not None:
    x4 = self.transformer_bottleneck(x4)
```

也就是说，Transformer 处理的是 ResNet34 encoder 最深层的特征图，而不是原始图像。

如果输入 ROI 图像大小为 `640 x 640`，经过 ResNet34 encoder 后，最深层特征通常可以近似看作：

$$
\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}
$$

其中：

$$
C = 512, \quad H \approx 20, \quad W \approx 20
$$

因此 bottleneck feature map 大约是：

$$
\mathbf{X} \in \mathbb{R}^{B \times 512 \times 20 \times 20}
$$

这里的每一个空间位置都不再是原始像素，而是 encoder 提取出的高层语义表示。

## 2. 为什么放在 bottleneck

Self-attention 的计算复杂度与 token 数量的平方相关：

$$
O(N^2)
$$

其中：

$$
N = H \times W
$$

如果直接在原始图像上做 attention：

$$
N = 640 \times 640 = 409600
$$

计算量会非常大。

但在 U-Net bottleneck 上：

$$
N = 20 \times 20 = 400
$$

这时做 self-attention 就比较轻量。因此 bottleneck 是一个合适的折中位置：

- 空间分辨率低，attention 计算量可控；
- 语义层级高，适合建模全局上下文；
- decoder 和 skip connections 仍然保留 U-Net 的细节恢复能力。

## 3. TransformerBottleneck 的代码结构

`TransformerBottleneck` 的核心代码如下：

```python
class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        channels=512,
        num_layers=1,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.pos = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_pos = x + self.pos(x)
        tokens = x_pos.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        y = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * y
```

旧实验 `01_tbn_d1` 使用的配置为：

```yaml
transformer_bottleneck_enable: true
transformer_bottleneck_layers: 1
transformer_bottleneck_heads: 8
transformer_bottleneck_dropout: 0.1
stage1_load_strict: false
```

也就是说，该实验只加了一个 1-layer Transformer encoder bottleneck，没有 prototype attention，也没有 skip attention gate。

## 4. Step 1: 卷积式位置编码

Transformer self-attention 本身只处理 token 序列，不天然知道二维空间位置。对于图像特征，如果不加入位置信息，模型很难区分某个 token 来自左上角还是右下角。

代码中使用 depthwise convolution 作为位置编码：

```python
self.pos = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
x_pos = x + self.pos(x)
```

数学上可以写作：

$$
\mathbf{X}_{pos} = \mathbf{X} + \operatorname{DWConv}(\mathbf{X})
$$

其中：

- $\mathbf{X}$ 是输入 bottleneck feature map；
- $\operatorname{DWConv}$ 是 depthwise convolution；
- depthwise convolution 只在每个通道内部做空间卷积，不混合不同通道，因此参数量较小；
- 这个操作为每个空间位置注入局部空间结构信息。

这和原始 Transformer 论文中的 sinusoidal positional encoding 不完全一样。这里使用的是更适合图像特征的卷积式位置编码。

## 5. Step 2: 将特征图展平成 token 序列

经过位置编码后，代码执行：

```python
tokens = x_pos.flatten(2).transpose(1, 2)
```

形状变化为：

$$
\mathbf{X}_{pos} \in \mathbb{R}^{B \times C \times H \times W}
$$

先展平空间维度：

$$
\mathbf{X}_{flat} \in \mathbb{R}^{B \times C \times N}
$$

其中：

$$
N = H \times W
$$

然后转置为 Transformer 需要的 token 格式：

$$
\mathbf{Z} \in \mathbb{R}^{B \times N \times C}
$$

在 `640 x 640` 输入的典型情况下：

$$
B \times 512 \times 20 \times 20
\rightarrow
B \times 400 \times 512
$$

这里：

- $N=400$ 表示有 400 个空间 token；
- 每个 token 是 512 维特征；
- 每个 token 对应 bottleneck feature map 上的一个空间位置。

## 6. Step 3: Transformer Encoder Layer 的计算

你的实现使用 PyTorch 的 `nn.TransformerEncoderLayer`，参数为：

```python
d_model = 512
nhead = 8
dim_feedforward = 2048
activation = "gelu"
norm_first = True
```

因此它是一个 pre-LayerNorm Transformer encoder layer。

设输入 token 为：

$$
\mathbf{Z} \in \mathbb{R}^{B \times N \times C}
$$

其中：

$$
C = 512, \quad N = H \times W
$$

### 6.1 LayerNorm

因为 `norm_first=True`，attention 之前先做 LayerNorm：

$$
\hat{\mathbf{Z}} = \operatorname{LN}(\mathbf{Z})
$$

LayerNorm 对每个 token 的通道维进行归一化。对单个 token $\mathbf{z} \in \mathbb{R}^{C}$，LayerNorm 可以写为：

$$
\operatorname{LN}(\mathbf{z})
=
\boldsymbol{\alpha}
\odot
\frac{\mathbf{z} - \mu}{\sqrt{\sigma^2 + \epsilon}}
+
\boldsymbol{\beta}
$$

其中：

$$
\mu = \frac{1}{C}\sum_{j=1}^{C} z_j
$$

$$
\sigma^2 = \frac{1}{C}\sum_{j=1}^{C}(z_j - \mu)^2
$$

$\boldsymbol{\alpha}$ 和 $\boldsymbol{\beta}$ 是可学习参数，$\epsilon$ 是数值稳定项。

LayerNorm 的作用是稳定 token 特征分布，使 attention 和 FFN 的训练更稳定。

### 6.2 Query, Key, Value

Self-attention 会从同一个 token 序列中生成 Query、Key、Value：

$$
\mathbf{Q} = \hat{\mathbf{Z}}\mathbf{W}^{Q}
$$

$$
\mathbf{K} = \hat{\mathbf{Z}}\mathbf{W}^{K}
$$

$$
\mathbf{V} = \hat{\mathbf{Z}}\mathbf{W}^{V}
$$

其中：

$$
\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times N \times C}
$$

直觉上：

- Query 表示当前 token 想找什么信息；
- Key 表示每个 token 可以被匹配的特征；
- Value 表示被聚合的信息内容。

### 6.3 Multi-Head Self-Attention

你的配置使用 8 个 attention heads：

$$
h = 8
$$

每个 head 的维度为：

$$
d_h = \frac{C}{h} = \frac{512}{8} = 64
$$

对第 $i$ 个 head：

$$
\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i
\in
\mathbb{R}^{B \times N \times d_h}
$$

Scaled dot-product attention 为：

$$
\operatorname{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i)
=
\operatorname{softmax}
\left(
\frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d_h}}
\right)
\mathbf{V}_i
$$

其中：

$$
\mathbf{Q}_i \mathbf{K}_i^\top
\in
\mathbb{R}^{B \times N \times N}
$$

如果 $N=400$，那么每个 head 会形成一个：

$$
400 \times 400
$$

的 attention matrix。这个矩阵表示：

```text
每个 bottleneck 空间位置，对其它所有空间位置的关注权重。
```

每个 head 得到：

$$
\mathbf{H}_i
=
\operatorname{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i)
\in
\mathbb{R}^{B \times N \times d_h}
$$

然后将所有 heads 拼接：

$$
\mathbf{H}
=
\operatorname{Concat}(\mathbf{H}_1, \ldots, \mathbf{H}_h)
\in
\mathbb{R}^{B \times N \times C}
$$

最后经过输出投影：

$$
\operatorname{MHA}(\hat{\mathbf{Z}})
=
\mathbf{H}\mathbf{W}^{O}
$$

### 6.4 第一个残差连接

Transformer 不直接替换原 token，而是做残差更新：

$$
\mathbf{Z}_1
=
\mathbf{Z}
+
\operatorname{Dropout}
\left(
\operatorname{MHA}(\operatorname{LN}(\mathbf{Z}))
\right)
$$

这一步的作用是：

- 保留原始 token 表示；
- 加入 self-attention 聚合到的全局上下文；
- 让训练更稳定。

### 6.5 Feed-Forward Network

Transformer encoder layer 中还有一个位置前馈网络，也就是 FFN。它对每个 token 独立作用，不在 token 之间交换信息。

你的配置中：

$$
C = 512
$$

$$
\text{hidden dimension} = 4C = 2048
$$

FFN 可以写为：

$$
\operatorname{FFN}(\mathbf{u})
=
\mathbf{W}_2
\operatorname{GELU}
(\mathbf{W}_1\mathbf{u} + \mathbf{b}_1)
+
\mathbf{b}_2
$$

其中：

$$
\mathbf{W}_1: 512 \rightarrow 2048
$$

$$
\mathbf{W}_2: 2048 \rightarrow 512
$$

对整个 token 序列：

$$
\operatorname{FFN}(\operatorname{LN}(\mathbf{Z}_1))
\in
\mathbb{R}^{B \times N \times C}
$$

直觉上：

- Multi-head self-attention 负责不同空间 token 之间的信息交流；
- FFN 负责每个 token 内部的非线性特征变换。

### 6.6 第二个残差连接

FFN 后再次做残差：

$$
\mathbf{Z}_2
=
\mathbf{Z}_1
+
\operatorname{Dropout}
\left(
\operatorname{FFN}(\operatorname{LN}(\mathbf{Z}_1))
\right)
$$

因此，一个 pre-LN Transformer encoder layer 可以概括为：

$$
\mathbf{Z}_1
=
\mathbf{Z}
+
\operatorname{Dropout}
\left(
\operatorname{MHA}(\operatorname{LN}(\mathbf{Z}))
\right)
$$

$$
\mathbf{Z}_2
=
\mathbf{Z}_1
+
\operatorname{Dropout}
\left(
\operatorname{FFN}(\operatorname{LN}(\mathbf{Z}_1))
\right)
$$

在旧实验 `01_tbn_d1` 中，`num_layers=1`，所以只堆叠一层这样的 encoder layer。

## 7. Step 4: 额外 LayerNorm 与 reshape

Transformer encoder 输出后，代码又做了一次 LayerNorm：

```python
tokens = self.norm(tokens)
```

记为：

$$
\mathbf{Z}_{out} = \operatorname{LN}_{out}(\mathbf{Z}_2)
$$

然后 reshape 回图像特征图：

```python
y = tokens.transpose(1, 2).reshape(b, c, h, w)
```

数学上：

$$
\mathbf{Z}_{out}
\in
\mathbb{R}^{B \times N \times C}
$$

恢复为：

$$
\mathbf{Y}
\in
\mathbb{R}^{B \times C \times H \times W}
$$

也就是：

$$
B \times 400 \times 512
\rightarrow
B \times 512 \times 20 \times 20
$$

这样输出可以继续送入原来的 U-Net decoder。

## 8. Step 5: 可学习残差缩放

最后一步是：

```python
return x + self.gamma * y
```

数学上：

$$
\mathbf{X}_{out}
=
\mathbf{X}
+
\gamma \mathbf{Y}
$$

其中：

$$
\gamma
$$

是一个可学习标量参数，并且初始化为：

$$
\gamma = 0
$$

这点非常重要。因为 Stage1 训练得到的是普通 ResNet34-U-Net checkpoint，并不包含 Transformer 参数。如果在 Stage2 直接插入一个随机初始化的 Transformer，可能会破坏已有表示。

但当 $\gamma=0$ 时：

$$
\mathbf{X}_{out} = \mathbf{X}
$$

也就是说，刚开始训练时，整个模型等价于原始 U-Net。随着训练进行，如果 Transformer 信息有用，模型会逐渐学习增大或调整 $\gamma$，让 Transformer 分支逐步参与预测。

这是一种稳定的 residual fine-tuning 设计。

## 9. 完整张量流

将上述步骤串起来，Transformer bottleneck 的完整计算为：

$$
\mathbf{X}
\in
\mathbb{R}^{B \times C \times H \times W}
$$

### 位置编码

$$
\mathbf{X}_{pos}
=
\mathbf{X}
+
\operatorname{DWConv}(\mathbf{X})
$$

### 展平为空间 token

$$
\mathbf{Z}
=
\operatorname{Flatten}_{HW}(\mathbf{X}_{pos})
\in
\mathbb{R}^{B \times N \times C}
$$

其中：

$$
N = H \times W
$$

### Transformer encoder

$$
\mathbf{Z}_1
=
\mathbf{Z}
+
\operatorname{Dropout}
\left(
\operatorname{MHA}(\operatorname{LN}(\mathbf{Z}))
\right)
$$

$$
\mathbf{Z}_2
=
\mathbf{Z}_1
+
\operatorname{Dropout}
\left(
\operatorname{FFN}(\operatorname{LN}(\mathbf{Z}_1))
\right)
$$

### 输出归一化并恢复为 feature map

$$
\mathbf{Y}
=
\operatorname{Reshape}
\left(
\operatorname{LN}_{out}(\mathbf{Z}_2)
\right)
\in
\mathbb{R}^{B \times C \times H \times W}
$$

### 残差缩放输出

$$
\mathbf{X}_{out}
=
\mathbf{X}
+
\gamma \mathbf{Y}
$$

这就是代码中 `TransformerBottleneck.forward()` 的完整数学表达。

## 10. 和 Attention Is All You Need 的关系

这个模块借用了原始 Transformer 的核心思想：

- scaled dot-product attention；
- multi-head self-attention；
- feed-forward network；
- residual connection；
- LayerNorm；
- dropout。

但它不是完整的 encoder-decoder Transformer。

不同点主要有：

1. 只使用 Transformer encoder layer，没有 decoder。
2. 输入不是文本 token，而是 CNN bottleneck feature map 展平后的空间 token。
3. 位置编码不是 sinusoidal positional encoding，而是 depthwise convolutional positional encoding。
4. 使用 `norm_first=True`，即 pre-LayerNorm 结构。
5. FFN 激活函数是 GELU，而原始论文中常见的是 ReLU。
6. 模块外部增加了一个初始化为 0 的 residual scaling 参数 $\gamma$。
7. 最终输出仍然进入 U-Net decoder，而不是直接生成序列输出。

因此，更准确的说法是：

```text
This is a lightweight Transformer encoder block inserted at the CNN bottleneck of a ResNet34-U-Net.
```

中文可以表述为：

```text
这是一个插入 ResNet34-U-Net bottleneck 位置的轻量级 Transformer encoder 模块，而不是完整的原版 Transformer 架构。
```

## 11. 为什么它可能帮助裂纹分割

裂纹分割的问题有几个特点：

- 裂纹像素很少；
- 裂纹通常细长；
- 局部纹理容易和背景噪声混淆；
- normal 图像上的小误检会造成 false positive；
- 单纯局部卷积可能难以利用整张 ROI 的结构上下文。

卷积擅长局部模式建模，但 self-attention 可以让每个 bottleneck token 直接和所有其它空间 token 交互。因此，Transformer bottleneck 提供的是全局上下文补充。

它可能帮助模型回答：

```text
这个局部响应是真裂纹，还是背景纹理？
这个位置和其它位置的结构关系是否支持它是 crack？
远处是否存在连续或相似的 defect pattern？
```

整体分工可以理解为：

```text
ResNet34 encoder:
    提取局部到高层的卷积特征。

Transformer bottleneck:
    在低分辨率高语义空间上建模全局 token 关系。

U-Net decoder + skip connections:
    恢复空间分辨率和细节，输出 segmentation mask。
```

## 12. 为什么 `stage1_load_strict: false`

`01_tbn_d1` 是 Stage2 结构改动。Stage1 checkpoint 是普通 U-Net 训练得到的，它不包含以下新参数：

- Transformer positional convolution；
- Transformer attention weights；
- Transformer FFN weights；
- Transformer LayerNorm parameters；
- residual scaling parameter $\gamma$。

如果 strict loading，checkpoint 中的参数名和当前模型结构不完全匹配，会报错。因此配置中使用：

```yaml
stage1_load_strict: false
```

含义是：

- 能匹配上的原 U-Net 权重正常加载；
- 新增 Transformer 参数随机初始化；
- 因为 $\gamma=0$，新增模块初始时不会强行破坏原模型表示。

## 13. 可以放进论文的方法描述

英文版本：

```text
To enhance global contextual reasoning while preserving the convolutional U-Net structure, a lightweight Transformer encoder was inserted at the bottleneck of the ResNet34-U-Net. Given a bottleneck feature map X in R^{B x C x H x W}, a depthwise convolutional positional encoding was first added to inject local spatial information. The feature map was then flattened into N=H x W spatial tokens and processed by a one-layer pre-normalized multi-head self-attention encoder. The output tokens were reshaped back to the original feature-map layout and added to the input through a learnable residual scaling parameter initialized to zero. This design allows the model to gradually incorporate global attention-based context during Stage2 fine-tuning without disrupting the Stage1 U-Net initialization.
```

中文版本：

```text
为了在保留卷积式 U-Net 结构的同时增强全局上下文建模能力，本文在 ResNet34-U-Net 的 bottleneck 位置插入一个轻量级 Transformer encoder。给定 bottleneck 特征图 X in R^{B x C x H x W}，首先通过 depthwise convolutional positional encoding 注入局部空间位置信息，然后将特征图展平成 N=H x W 个空间 token，并输入一层 pre-normalized multi-head self-attention encoder。Transformer 输出再恢复为原始特征图形状，并通过一个初始化为 0 的可学习残差缩放参数加回输入特征。该设计使模型能够在 Stage2 fine-tuning 过程中逐渐引入基于 attention 的全局上下文，而不会破坏 Stage1 U-Net checkpoint 的初始化表示。
```

## 14. 一句话总结

`TransformerBottleneck` 的作用是：

```text
在 U-Net 最深层、低分辨率、高语义的 bottleneck feature map 上，把每个空间位置当作 token，通过 multi-head self-attention 建模全局空间关系，再用 residual scaling 稳定地加回原特征。
```

它的核心不是替代 U-Net，而是给原本以卷积为主的 two-stage U-Net 增加一个可控、轻量、稳定的全局上下文建模模块。
