import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EncoderConfig:
    channels: int
    hidden_dim: int = 64
    embedding_dim: int = 128
    transformer_layers: int = 1
    attention_heads: int = 4
    dropout: float = 0.1
    attention_scope: str = "axial_patches"
    token_stride: int = 10
    position_encoding: str = "sinusoidal"
    max_position_tokens: int = 64
    patch_samples: int = 125
    # Legacy metadata fields retained solely to load the three original checkpoints.
    spectral_bins: int = 24
    spectral_segment_samples: int = 250
    sample_rate: float = 250.0
    spectral_min_hz: float = 0.5
    spectral_max_hz: float = 45.0
    quality_aware: bool = True
    quality_soft_weights: bool = True

    def __post_init__(self):
        if min(self.channels, self.hidden_dim, self.embedding_dim, self.attention_heads) < 1:
            raise ValueError("Encoder dimensions must be positive")
        if self.hidden_dim % 8 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by 8 and attention_heads")
        if self.attention_scope != "axial_patches":
            raise ValueError("Unknown attention_scope")
        if not isinstance(self.token_stride, int) or isinstance(self.token_stride, bool) or self.token_stride < 1:
            raise ValueError("token_stride must be a positive integer")
        if self.transformer_layers < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid Transformer depth or dropout")
        if self.position_encoding != "sinusoidal":
            raise ValueError("Only sinusoidal positions are supported")
        if not isinstance(self.quality_aware, bool):
            raise ValueError("quality_aware must be boolean")
        if not isinstance(self.quality_soft_weights, bool):
            raise ValueError("quality_soft_weights must be boolean")
        if self.quality_aware and (self.attention_scope != "axial_patches" or self.transformer_layers < 1):
            raise ValueError("Quality attention requires axial_patches with at least one layer")
        if not isinstance(self.patch_samples, int) or isinstance(self.patch_samples, bool) or self.patch_samples < 2:
            raise ValueError("patch_samples must be an integer >=2")
        if self.attention_scope == "axial_patches" and (not self.quality_aware or not self.quality_soft_weights):
            raise ValueError("Axial Transformer requires hard masks and soft quality weights")


def sinusoidal_positions(count, hidden, device):
    positions = torch.arange(count, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, hidden, 2, device=device, dtype=torch.float32) *
                            (-math.log(10000.0) / hidden))
    angles = positions * frequencies
    encoding = torch.zeros(count, hidden, device=device)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles.cos()
    return encoding


def make_transformer(hidden, heads, layers, dropout):
    if not layers:
        return None
    layer = nn.TransformerEncoderLayer(
        d_model=hidden, nhead=heads, dim_feedforward=2 * hidden, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True,
    )
    transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
    for module in transformer.modules():
        if isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return transformer


def quality_attention(transformer, tokens, quality, heads):
    """Hard-invalid keys never transmit; log quality biases softmax key weights.

    Skip all-masked sequences instead of softmax(-inf,...,-inf). Invalid query
    outputs are removed after each layer, including residual and FF paths.
    """
    active = quality.sum(-1) > 0
    result = torch.zeros_like(tokens)
    if not active.any():
        return result
    x = tokens[active]
    q = quality[active]
    bias = torch.where(q > 0, q.clamp_min(torch.finfo(q.dtype).tiny).log(), -torch.inf)
    mask = bias[:, None, None, :].expand(-1, heads, x.shape[1], -1).reshape(-1, x.shape[1], x.shape[1])
    for layer in transformer.layers:
        # Fused eval EncoderLayer kernels can treat floating biases as boolean
        # masks. Public MHA's float-mask path preserves log(q) in train and eval.
        attention_input = layer.norm1(x) if layer.norm_first else x
        attended = layer.self_attn(attention_input, attention_input, attention_input,
                                   attn_mask=mask, need_weights=False)[0]
        x = x + layer.dropout1(attended)
        if not layer.norm_first:
            x = layer.norm1(x)
        ff_input = layer.norm2(x) if layer.norm_first else x
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_input))))
        x = x + layer.dropout2(ff)
        if not layer.norm_first:
            x = layer.norm2(x)
        x = x.masked_fill((q == 0).unsqueeze(-1), 0)
    if transformer.norm is not None:
        x = transformer.norm(x)
    result[active] = x.masked_fill((q == 0).unsqueeze(-1), 0)
    return result


class ElectrodeTokenizer(nn.Module):
    """Project per-electrode waveform patches, preserving electrode/time axes."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.patch_samples
        self.input_projection = nn.Sequential(nn.Linear(width, config.hidden_dim), nn.LayerNorm(config.hidden_dim))
        self.channel_embeddings = nn.Parameter(torch.empty(config.channels, config.hidden_dim))
        nn.init.normal_(self.channel_embeddings, std=0.02)

    def forward(self, x):
        if x.shape[-1] % self.config.patch_samples:
            raise ValueError("Window samples must be divisible by patch_samples; no silent tail truncation")
        patches = x.unfold(-1, self.config.patch_samples, self.config.patch_samples)
        tokens = self.input_projection(patches)
        positions = sinusoidal_positions(tokens.shape[2], self.config.hidden_dim, x.device)
        return tokens + self.channel_embeddings[None, :, None, :] + positions.to(tokens.dtype)[None, None, :, :]


class EEGEncoder(nn.Module):
    """Quality-aware axial encoder with hard masks and soft attention/pooling.

    Input [batch, windows, channels, samples]; output normalized identity vectors.
    Invalid windows never enter feature extraction or the pooling denominator.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.tokenizer = ElectrodeTokenizer(config)
        self.temporal_transformer = nn.ModuleList([
            make_transformer(h, config.attention_heads, 1, config.dropout)
            for _ in range(config.transformer_layers)
        ])
        self.transformer = nn.ModuleList([
            make_transformer(h, config.attention_heads, 1, config.dropout)
            for _ in range(config.transformer_layers)
        ])
        self.token_norm = nn.LayerNorm(h)
        self.projection = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, config.embedding_dim))
        return

    def forward(self, x: torch.Tensor, valid: torch.Tensor, quality: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or valid.shape != x.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Expected x [B,N,C,T] and a boolean valid mask [B,N]")
        if x.shape[2] != self.config.channels or x.shape[3] < 4:
            raise ValueError("Unexpected channel count or fewer than four samples")
        if self.config.quality_aware:
            p = self.config.patch_samples
            if x.shape[-1] % p or quality is None or quality.shape != (*x.shape[:3], x.shape[-1] // p):
                raise ValueError("Quality encoder requires electrode-time patch weights [B,N,C,P]")
            if not quality.is_floating_point() or quality.device != x.device or not torch.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
                raise ValueError("Quality must be finite floating weights in [0,1] on the input device")
            valid = valid & (quality.sum(dim=(2,3)) > 0)
            # Remove lost cells before linear projection, normalization, or attention.
            x = torch.where((quality > 0).repeat_interleave(p, dim=-1), x, 0)
        elif quality is not None:
            raise ValueError("Quality metadata supplied to a quality-unaware encoder")
        if not valid.any(dim=1).all():
            raise ValueError("Every sample must contain at least one valid window")
        b, n, c, t = x.shape
        flat_valid = valid.reshape(-1)
        selected = x.reshape(b * n, c, t)[flat_valid]
        if not torch.isfinite(selected).all():
            raise ValueError("Non-finite values in a valid window")
        tokens = self.tokenizer(selected)
        m, channels, patches, hidden = tokens.shape
        q = quality.reshape(b * n, channels, patches)[flat_valid].to(tokens.dtype)
        tokens = tokens.masked_fill((q == 0).unsqueeze(-1), 0)
        for temporal, spatial in zip(self.temporal_transformer, self.transformer):
            tokens = quality_attention(temporal, tokens.reshape(m * channels, patches, hidden),
                                       q.reshape(m * channels, patches), self.config.attention_heads)
            tokens = tokens.reshape(m, channels, patches, hidden).transpose(1, 2)
            tokens = quality_attention(spatial, tokens.reshape(m * patches, channels, hidden),
                                       q.transpose(1, 2).reshape(m * patches, channels), self.config.attention_heads)
            tokens = tokens.reshape(m, patches, channels, hidden).transpose(1, 2).contiguous()
        normalized = self.token_norm(tokens).masked_fill((q == 0).unsqueeze(-1), 0)
        encoded = (normalized * q.unsqueeze(-1)).sum(dim=(1, 2)) / q.sum(dim=(1, 2)).unsqueeze(-1)
        features = encoded.new_zeros((b * n, self.config.hidden_dim))
        features[flat_valid] = encoded
        features = features.reshape(b, n, -1)
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        if self.config.quality_aware:
            window_weights = quality.sum(dim=(2,3)).to(features.dtype) * valid
            pooled = (features * window_weights.unsqueeze(-1)).sum(dim=1) / window_weights.sum(dim=1, keepdim=True)
        else:
            pooled = features.sum(dim=1) / valid.sum(dim=1, keepdim=True)
        return F.normalize(self.projection(pooled), dim=-1)

    def forward_pair(self, left, left_valid, right, right_valid, left_quality=None, right_quality=None):
        """Two branches use this exact encoder and the same parameters."""
        z_left = self(left, left_valid, left_quality)
        z_right = self(right, right_valid, right_quality)
        return z_left, z_right, (z_left * z_right).sum(dim=-1)

    def config_dict(self):
        return asdict(self.config)

