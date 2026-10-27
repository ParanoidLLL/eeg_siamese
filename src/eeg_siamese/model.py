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
    channel_gate: str = "none"
    task_names: tuple = ()
    gate_hidden: int = 32
    gate_power: float = 1.0
    selection_mode: str = 'none'
    selected_channels: int = 16
    selection_random_seed: int = 12345
    selection_temperature_start: float = 2.0
    selection_temperature_end: float = .25
    selection_anneal_epochs: int = 30
    selection_noise_start: float = 1.0
    selection_noise_end: float = 1.0
    selection_noise_anneal_epochs: int = 30
    selection_fixed_scores: tuple = ()
    window_pooling: str = 'quality'
    window_scorer_hidden: int = 32
    window_residual_mix: float = .5
    window_multiplier_bound: float = .5
    spectral_fusion: str = 'none'
    spectral_feature_mean: tuple = ()
    spectral_feature_scale: tuple = ()
    spectral_input_scale_uv: float = 100.
    spectral_window_seconds: float = 4.
    record_fusion: str = 'none'
    spectral_modulation: str = 'none'
    spectral_modulation_groups: int = 4
    spectral_modulation_bound: float = .5

    @property
    def requires_task(self):
        return self.channel_gate in ("task_fixed", "task_dynamic") or self.selection_mode == 'learned_task'

    def __post_init__(self):
        if self.spectral_modulation not in ('none','global','conditional'):
            raise ValueError('Unknown spectral_modulation')
        if self.spectral_modulation!='none':
            if self.record_fusion!='none' or self.spectral_fusion!='log':
                raise ValueError('Spectral modulation requires standalone log encoder')
            if not isinstance(self.spectral_modulation_groups,int) or isinstance(self.spectral_modulation_groups,bool) or self.spectral_modulation_groups<1 or self.hidden_dim%self.spectral_modulation_groups:
                raise ValueError('Modulation groups must divide hidden dimensions')
            if not math.isfinite(self.spectral_modulation_bound) or not 0<self.spectral_modulation_bound<1:
                raise ValueError('Modulation bound must be in (0,1)')
        if self.record_fusion not in ('none','fixed','adaptive','single'):
            raise ValueError('Unknown record_fusion')
        if self.record_fusion!='none':
            if self.spectral_fusion not in ('none','log'):
                raise ValueError('Record fusion requires waveform or log spectral configuration')
            if self.record_fusion=='adaptive' and self.spectral_fusion!='log':
                raise ValueError('Adaptive record context requires log spectral features')
            if self.channel_gate!='none' or self.selection_mode!='none' or self.window_pooling!='quality':
                raise ValueError('Record fusion requires full-channel quality pooling')
        if self.spectral_fusion not in ('none','absolute','log'):
            raise ValueError('Unknown spectral_fusion')
        object.__setattr__(self,'spectral_feature_mean',tuple(self.spectral_feature_mean))
        object.__setattr__(self,'spectral_feature_scale',tuple(self.spectral_feature_scale))
        if self.spectral_fusion!='none':
            if self.channel_gate!='none' or self.selection_mode!='none' or self.window_pooling!='quality':
                raise ValueError('Spectral residual study requires original full-channel quality pooling')
            if len(self.spectral_feature_mean)!=self.channels*5 or len(self.spectral_feature_scale)!=self.channels*5:
                raise ValueError('Need train-fitted spectral constants for every channel/band')
            if not all(math.isfinite(v) for v in self.spectral_feature_mean) or not all(math.isfinite(v) and v>0 for v in self.spectral_feature_scale):
                raise ValueError('Invalid spectral constants')
            if not math.isfinite(self.sample_rate) or self.sample_rate<80 or not math.isfinite(self.spectral_input_scale_uv) or self.spectral_input_scale_uv<=0:
                raise ValueError('Invalid spectral sample rate or physical units')
            if not math.isfinite(self.spectral_window_seconds) or self.spectral_window_seconds not in (4.,5.):
                raise ValueError('Spectral windows must use the explicit four/five-second protocol')
        if self.window_pooling not in ('quality','learned','reliable'):
            raise ValueError('Unknown window_pooling')
        if self.window_pooling!='quality' and (self.channel_gate!='none' or self.selection_mode!='none'):
            raise ValueError('Window pooling study requires full channels without channel gates')
        if not isinstance(self.window_scorer_hidden,int) or isinstance(self.window_scorer_hidden,bool) or self.window_scorer_hidden<1:
            raise ValueError('Invalid window scorer dimension')
        if not math.isfinite(self.window_residual_mix) or not 0<self.window_residual_mix<=.5:
            raise ValueError('Window residual mix must preserve at least half the baseline')
        if not math.isfinite(self.window_multiplier_bound) or not 0<self.window_multiplier_bound<1:
            raise ValueError('Window multiplier bound must be in (0,1)')
        object.__setattr__(self,'selection_fixed_scores',tuple(self.selection_fixed_scores))
        if self.selection_mode not in ('none','learned_global','learned_task','random_fixed','fixed_scores'):
            raise ValueError('Unknown selection_mode')
        if self.selection_mode != 'none' and self.channel_gate != 'none':
            raise ValueError('Subset selection must be evaluated without additional channel gates')
        if not math.isfinite(self.gate_power) or self.gate_power not in (1.0, 2.0):
            raise ValueError('gate_power must be 1 or 2')
        object.__setattr__(self, 'task_names', tuple(self.task_names))
        if self.channel_gate not in ('none', 'task_fixed', 'dynamic', 'task_dynamic'):
            raise ValueError('Unknown channel_gate')
        if self.gate_hidden < 1 or len(set(self.task_names)) != len(self.task_names):
            raise ValueError('Invalid gate dimensions/task vocabulary')
        if self.requires_task and not self.task_names:
            raise ValueError('Task-conditioned gates require a frozen task vocabulary')
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

    def forward(self, x, channel_indices=None):
        if x.shape[-1] % self.config.patch_samples:
            raise ValueError("Window samples must be divisible by patch_samples; no silent tail truncation")
        patches = x.unfold(-1, self.config.patch_samples, self.config.patch_samples)
        tokens = self.input_projection(patches)
        positions = sinusoidal_positions(tokens.shape[2], self.config.hidden_dim, x.device)
        electrodes = (self.channel_embeddings[None, :, None, :] if channel_indices is None
                      else self.channel_embeddings[channel_indices][:, :, None, :])
        return tokens + electrodes + positions.to(tokens.dtype)[None, None, :, :]


class EEGEncoder(nn.Module):
    """Quality-aware axial encoder with hard masks and soft attention/pooling.

    Input [batch, windows, channels, samples]; output normalized identity vectors.
    Invalid windows never enter feature extraction or the pooling denominator.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        if config.record_fusion!='none':
            from .record_fusion import RecordFusion
            self.record_fuser=RecordFusion(config)
            return
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
        if config.spectral_fusion!='none':
            from .spectral_residual import SpectralResidual
            # Do not change the original backbone/classifier/sampling RNG start.
            with torch.random.fork_rng(devices=[]):
                self.spectral_branch=SpectralResidual(config)
        if config.spectral_modulation!='none':
            from .spectral_modulation import SpectralModulation
            with torch.random.fork_rng(devices=[]):
                self.spectral_modulator=SpectralModulation(config)
        if config.window_pooling!='quality':
            from .window_pooling import ResidualWindowPooling
            self.window_aggregator=ResidualWindowPooling(h,config.window_scorer_hidden,
                config.window_residual_mix,config.window_multiplier_bound)
        if config.selection_mode != 'none':
            from .channel_selection import ElectrodeSubsetSelector
            self.selector = ElectrodeSubsetSelector(config.channels, config.selected_channels,
                config.selection_mode, len(config.task_names), config.selection_random_seed,
                config.selection_temperature_start, config.selection_temperature_end,
                config.selection_anneal_epochs, config.selection_noise_start, config.selection_noise_end,
                config.selection_noise_anneal_epochs, config.selection_fixed_scores)
        if config.channel_gate == 'task_fixed':
            self.channel_gate = nn.Embedding(len(config.task_names), config.channels)
            nn.init.zeros_(self.channel_gate.weight)
        elif config.channel_gate in ('dynamic', 'task_dynamic'):
            extra = 8 if config.requires_task else 0
            if config.requires_task:
                self.task_embedding = nn.Embedding(len(config.task_names), extra)
            self.channel_gate = nn.Sequential(nn.Linear(2 * h + extra, config.gate_hidden),
                                              nn.GELU(), nn.Linear(config.gate_hidden, 1))
            nn.init.zeros_(self.channel_gate[-1].weight)
            nn.init.zeros_(self.channel_gate[-1].bias)
        return

    def task_index(self, name):
        if name not in self.config.task_names:
            raise ValueError(f'Missing or unknown task_id: {name!r}')
        return self.config.task_names.index(name)

    def _contributions(self, tokens, quality, tasks):
        if self.config.channel_gate == 'none':
            return torch.ones_like(quality[:, :, 0])
        if self.config.channel_gate == 'task_fixed':
            logits = self.channel_gate(tasks)
        else:
            summary = (tokens * quality.unsqueeze(-1)).sum(2) / quality.sum(2).clamp_min(1e-8).unsqueeze(-1)
            summary = summary.masked_fill((quality.sum(2) == 0).unsqueeze(-1), 0)
            global_summary = (summary * quality.sum(2).unsqueeze(-1)).sum(1) / quality.sum((1, 2)).clamp_min(1e-8).unsqueeze(-1)
            parts = [summary, global_summary[:, None].expand_as(summary)]
            if self.config.requires_task:
                parts.append(self.task_embedding(tasks)[:, None].expand(-1, tokens.shape[1], -1))
            logits = self.channel_gate(torch.cat(parts, -1)).squeeze(-1)
        return 1 + 0.5 * logits.tanh()

    def forward(self, x: torch.Tensor, valid: torch.Tensor, quality: torch.Tensor | None = None,
                tasks: torch.Tensor | None = None, return_gate=False, return_selection=False,
                return_pooling=False, return_branches=False, return_modulation=False) -> torch.Tensor:
        if self.config.record_fusion!='none':
            if return_gate or return_selection or return_pooling or return_modulation or tasks is not None:
                raise ValueError('Record fusion supports branch diagnostics only')
            result,branches=self.record_fuser(x,valid,quality)
            return (result,branches) if return_branches else result
        if return_branches:
            raise ValueError('Branch diagnostics require record fusion')
        if sum((return_gate,return_selection,return_pooling,return_modulation))>1:
            raise ValueError('Choose one diagnostic return type')
        if return_selection and self.config.selection_mode == 'none':
            raise ValueError('Selection diagnostics require a subset model')
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
        if self.config.requires_task:
            if tasks is None or tasks.shape != (b,) or tasks.dtype != torch.long or tasks.device != x.device:
                raise ValueError('Task gate requires integer task indices [B] on input device')
            if ((tasks < 0) | (tasks >= len(self.config.task_names))).any():
                raise ValueError('Task index outside frozen vocabulary')
        flat_valid = valid.reshape(-1)
        selected_tasks = tasks[:, None].expand(b, n).reshape(-1)[flat_valid] if self.config.requires_task else None
        selected = x.reshape(b * n, c, t)[flat_valid]
        if not torch.isfinite(selected).all():
            raise ValueError("Non-finite values in a valid window")
        q = quality.reshape(b * n, c, t // self.config.patch_samples)[flat_valid].to(selected.dtype)
        selection = None
        if self.config.selection_mode != 'none':
            selected, q, selection = self.selector(selected, q, selected_tasks)
        tokens = self.tokenizer(selected, None if selection is None else selection['indices'])
        m, channels, patches, hidden = tokens.shape
        q = q.to(tokens.dtype)
        tokens = tokens.masked_fill((q == 0).unsqueeze(-1), 0)
        contributions = None
        for temporal, spatial in zip(self.temporal_transformer, self.transformer):
            tokens = quality_attention(temporal, tokens.reshape(m * channels, patches, hidden),
                                       q.reshape(m * channels, patches), self.config.attention_heads)
            tokens = tokens.reshape(m, channels, patches, hidden)
            if contributions is None:
                contributions = self._contributions(tokens, q, selected_tasks)
            spatial_q = q * contributions.pow(self.config.gate_power).unsqueeze(-1)
            tokens = tokens.transpose(1, 2)
            tokens = quality_attention(spatial, tokens.reshape(m * patches, channels, hidden),
                                       spatial_q.transpose(1, 2).reshape(m * patches, channels), self.config.attention_heads)
            tokens = tokens.reshape(m, patches, channels, hidden).transpose(1, 2).contiguous()
        normalized = self.token_norm(tokens).masked_fill((q == 0).unsqueeze(-1), 0)
        pool_q = q * contributions.pow(self.config.gate_power).unsqueeze(-1)
        encoded = (normalized * pool_q.unsqueeze(-1)).sum(dim=(1, 2)) / pool_q.sum(dim=(1, 2)).unsqueeze(-1)
        modulation=None
        if self.config.spectral_fusion!='none':
            if self.config.spectral_modulation!='none':
                residual,f,active=self.spectral_branch(selected,q,return_features=True)
                residual,groups=self.spectral_modulator(encoded,residual,f,q,flat_valid,b,n)
                modulation=dict(groups=groups)
                encoded=encoded+residual
            else:
                encoded=encoded+self.spectral_branch(selected,q)
        features = encoded.new_zeros((b * n, self.config.hidden_dim))
        features[flat_valid] = encoded
        features = features.reshape(b, n, -1)
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        if self.config.quality_aware:
            if selection is None:
                window_weights = quality.sum(dim=(2,3)).to(features.dtype) * valid
            else:
                # Unselected quality also cannot influence window fusion.
                window_weights = features.new_zeros(b * n)
                window_weights[flat_valid] = q.sum((1,2))
                window_weights = window_weights.reshape(b,n)
            pooled = (features * window_weights.unsqueeze(-1)).sum(dim=1) / window_weights.sum(dim=1, keepdim=True)
        else:
            pooled = features.sum(dim=1) / valid.sum(dim=1, keepdim=True)
        pooling=None
        if self.config.window_pooling!='quality':
            pooled,pooling=self.window_aggregator(features,window_weights)
        elif return_pooling:
            base=window_weights/window_weights.sum(1,keepdim=True)
            pooling=dict(base_weights=base,weights=base,multipliers=torch.ones_like(base),
                window_features=features,valid=base>0)
        result = F.normalize(self.projection(pooled), dim=-1)
        if return_modulation:
            if modulation is None:
                modulation=dict(groups=result.new_ones(b,self.config.spectral_modulation_groups))
            return result,modulation
        if return_pooling:
            return result,pooling
        if return_selection:
            selection.update(flat_valid=flat_valid)
            return result, selection
        if return_gate:
            # Invalid channels do not count toward regularization or summaries.
            active = q.sum(-1) > 0
            regularizer = (contributions[active] - 1).square().mean()
            return result, dict(weights=contributions, effective_weights=contributions.pow(self.config.gate_power),
                                active=active, regularizer=regularizer)
        return result

    def forward_pair(self, left, left_valid, right, right_valid, left_quality=None, right_quality=None,
                     left_tasks=None, right_tasks=None):
        """Two branches use this exact encoder and the same parameters."""
        z_left = self(left, left_valid, left_quality, tasks=left_tasks)
        z_right = self(right, right_valid, right_quality, tasks=right_tasks)
        return z_left, z_right, (z_left * z_right).sum(dim=-1)

    def config_dict(self):
        return asdict(self.config)

