import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SpatialReductionSelfAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=4,
        dropout=0.1,
        sr_ratio=1,
        gamma_init=0.0,
    ):
        super().__init__()
        channels = int(channels)
        num_heads = int(num_heads)
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.sr_ratio = max(1, int(sr_ratio))
        self.pos = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.q_norm = nn.LayerNorm(channels)
        self.kv_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        b, c, h, w = x.shape
        x_pos = x + self.pos(x)

        q = x_pos.flatten(2).transpose(1, 2)
        q = self.q_norm(q)

        if self.sr_ratio > 1:
            kv_map = F.avg_pool2d(
                x_pos,
                kernel_size=self.sr_ratio,
                stride=self.sr_ratio,
                ceil_mode=True,
                count_include_pad=False,
            )
        else:
            kv_map = x_pos

        kv = kv_map.flatten(2).transpose(1, 2)
        kv = self.kv_norm(kv)

        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = self.out_norm(out)
        y = out.transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * y


class SkipAttentionGate(nn.Module):
    def __init__(self, skip_channels, gate_channels, inter_channels=None, gamma_init=0.0):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(16, min(skip_channels, gate_channels) // 2)

        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, skip, gate):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * (1.0 + self.gamma * (attention - 1.0))


class PrototypeCrossAttention(nn.Module):
    def __init__(self, pos_prototypes, neg_prototypes, channels=512, num_heads=8, dropout=0.1):
        super().__init__()
        pos_prototypes = torch.as_tensor(pos_prototypes, dtype=torch.float32)
        neg_prototypes = torch.as_tensor(neg_prototypes, dtype=torch.float32)

        if pos_prototypes.ndim != 2 or pos_prototypes.shape[1] != channels:
            raise ValueError(f"pos_prototypes must have shape [N, {channels}]")
        if neg_prototypes.ndim != 2 or neg_prototypes.shape[1] != channels:
            raise ValueError(f"neg_prototypes must have shape [N, {channels}]")
        if pos_prototypes.shape[0] == 0 or neg_prototypes.shape[0] == 0:
            raise ValueError("prototype attention requires at least one positive and one negative prototype")

        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(channels)
        self.norm_out = nn.LayerNorm(channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.type_embed = nn.Parameter(torch.zeros(2, channels))
        self.register_buffer("pos_prototypes", F.normalize(pos_prototypes, dim=1))
        self.register_buffer("neg_prototypes", F.normalize(neg_prototypes, dim=1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)
        q = self.norm_q(q)

        prototypes = torch.cat(
            [
                self.pos_prototypes + self.type_embed[0],
                self.neg_prototypes + self.type_embed[1],
            ],
            dim=0,
        )
        prototypes = prototypes.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.mha(q, prototypes, prototypes, need_weights=False)
        out = self.norm_out(out)
        y = out.transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * y
