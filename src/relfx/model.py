"""RelFx dual-branch cross-attention encoder.

The Cnn14-style frontend and convolutional blocks trace to PANNs and were
adapted for RelFx through FxEncoder++.
See THIRD_PARTY_NOTICES.md and the retained third-party licenses.

核心改进（对比 V3）:
    V3: 4 通道拼接 → 单一 CNN → 严重依赖逐帧差分
    V4: 双分支 CNN + 交叉注意力 → 各自编码频谱特征 → 差分提取 FX signature

架构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   干声 A (2ch)  ──→  SharedEncoder ──→ feat_dry                │
    │                           ↕ CrossAttn (层间交互)                │
    │   湿声 FX(A') (2ch) ──→ SharedEncoder ──→ feat_wet             │
    │                                                                 │
    │   分支投影: feat → fc_embed → emb_dry, emb_wet                 │
    │   融合: e_fx = Fuse(emb_dry, emb_wet)                          │
    │       Base: concatenation gate + differential/absolute terms   │
    │       Bidirectional: symmetric gate + differential term only   │
    │                                                                 │
    │   e_fx (2048-d) → projection → z (128-d)                       │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

交叉注意力在 CNN 中间层注入 (Stage 3, 5 后):
    - 每个交叉枝用轻量 cross-attention: Q 来自一个分支, K/V 来自另一个
    - 保持主干参数共享, 只有 CrossAttn 模块是额外参数
    - 让两个分支"知道"对方的特征, 便于后续差分

公开表示接口:
    - forward(original, processed)["embedding"] → normalized z (B, 128)
    - forward(original, processed)["fusion"] → e_fx (B, 2048)
    - get_embedding(original, processed) → normalized z (B, 128)

模型变体:
    - "base": paper Eq. (4), standard Diff-Gate + ReLU projection
    - "bidirectional": paper Eqs. (6)-(7), antisymmetric fusion + Tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


MODEL_VARIANTS = ("base", "bidirectional")


def validate_model_variant(model_variant: str) -> None:
    if model_variant not in MODEL_VARIANTS:
        choices = ", ".join(MODEL_VARIANTS)
        raise ValueError(
            f"Unknown model_variant: {model_variant!r}. Expected one of: {choices}"
        )


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        return x


# ===================== 交叉注意力模块 =====================

class CrossAttentionBlock(nn.Module):
    """
    轻量级交叉注意力: 让两个分支的特征图互相 attend。

    输入: feat_a (B, C, T, F), feat_b (B, C, T, F)
    输出: feat_a' (B, C, T, F), feat_b' (B, C, T, F)

    双向交叉:
        feat_a' = feat_a + α * CrossAttn(Q=feat_a, K=feat_b, V=feat_b)
        feat_b' = feat_b + α * CrossAttn(Q=feat_b, K=feat_a, V=feat_a)

    使用空间池化将 (T, F) 压缩到 token 序列，降低计算量。
    """

    def __init__(self, channels, num_heads=4, pool_size=4, residual_scale=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.residual_scale = residual_scale

        # 将 CNN 特征图压缩为 token 序列
        # 对 (T, F) 做 adaptive pool → (T//pool, F//pool)，然后 flatten 成 seq
        self.norm_a = nn.LayerNorm(channels)
        self.norm_b = nn.LayerNorm(channels)

        # Q/K/V 投影 (共享维度 = channels)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

        self.scale = (channels // num_heads) ** -0.5

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)

    def _feat_to_tokens(self, x):
        """(B, C, T, freq) → (B, S, C) where S = (T//pool) * (freq//pool)"""
        B, C, T, freq = x.shape
        # 先做 adaptive avg pool 降低序列长度
        t_out = max(T // self.pool_size, 1)
        f_out = max(freq // self.pool_size, 1)
        x_pooled = F.adaptive_avg_pool2d(x, (t_out, f_out))  # (B, C, t_out, f_out)
        # reshape 成 (B, S, C)
        x_tokens = x_pooled.permute(0, 2, 3, 1).reshape(B, t_out * f_out, C)
        return x_tokens, (t_out, f_out)

    def _cross_attend(self, q_tokens, kv_tokens, norm_q, B):
        """单向交叉注意力: Q attend to KV"""
        S_q = q_tokens.shape[1]
        S_kv = kv_tokens.shape[1]
        H = self.num_heads
        D = self.channels // H

        q = self.q_proj(norm_q)    # (B, S_q, C)
        k = self.k_proj(kv_tokens)  # (B, S_kv, C)
        v = self.v_proj(kv_tokens)  # (B, S_kv, C)

        # Multi-head
        q = q.reshape(B, S_q, H, D).permute(0, 2, 1, 3)   # (B, H, S_q, D)
        k = k.reshape(B, S_kv, H, D).permute(0, 2, 1, 3)  # (B, H, S_kv, D)
        v = v.reshape(B, S_kv, H, D).permute(0, 2, 1, 3)  # (B, H, S_kv, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, S_q, S_kv)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v)  # (B, H, S_q, D)
        out = out.permute(0, 2, 1, 3).reshape(B, S_q, self.channels)  # (B, S_q, C)
        out = self.out_proj(out)
        return out

    def forward(self, feat_a, feat_b):
        """
        双向交叉注意力。

        Args:
            feat_a: (B, C, T, F) — 分支 A 的特征图
            feat_b: (B, C, T, F) — 分支 B 的特征图
        Returns:
            feat_a_out: (B, C, T, F) — 增强后的分支 A
            feat_b_out: (B, C, T, F) — 增强后的分支 B
        """
        B, C, T, F_dim = feat_a.shape

        # 转换为 token 序列
        tokens_a, (t_out, f_out) = self._feat_to_tokens(feat_a)  # (B, S, C)
        tokens_b, _ = self._feat_to_tokens(feat_b)                # (B, S, C)

        # LayerNorm
        norm_tokens_a = self.norm_a(tokens_a)
        norm_tokens_b = self.norm_b(tokens_b)

        # 双向交叉注意力
        # A attend to B
        delta_a = self._cross_attend(tokens_a, tokens_b, norm_tokens_a, B)  # (B, S, C)
        # B attend to A
        delta_b = self._cross_attend(tokens_b, tokens_a, norm_tokens_b, B)  # (B, S, C)

        # 将 delta 上采样回原始空间尺寸，然后做残差加
        # (B, S, C) → (B, C, t_out, f_out) → 上采样 → (B, C, T, F)
        delta_a = delta_a.permute(0, 2, 1).reshape(B, C, t_out, f_out)
        delta_b = delta_b.permute(0, 2, 1).reshape(B, C, t_out, f_out)

        delta_a = F.interpolate(delta_a, size=(T, F_dim), mode='bilinear', align_corners=False)
        delta_b = F.interpolate(delta_b, size=(T, F_dim), mode='bilinear', align_corners=False)

        feat_a_out = feat_a + self.residual_scale * delta_a
        feat_b_out = feat_b + self.residual_scale * delta_b

        return feat_a_out, feat_b_out


# ===================== 融合模块 =====================

class FusionModule(nn.Module):
    """
    融合两个分支的 embedding 为最终的 FX embedding。

    策略:
        "diff":       FX_emb = emb_wet - emb_dry  (最简洁，内容信息被抵消)
        "concat_mlp": FX_emb = MLP(cat(emb_dry, emb_wet))
        "gate":       FX_emb = gate * emb_wet + (1-gate) * emb_dry
                      gate = sigmoid(W * cat(emb_dry, emb_wet))
        Base "diff_gate":
            FX_emb = gate(cat(dry, wet)) * (wet - dry) + (1-gate) * wet
        Bidirectional "diff_gate":
            FX_emb = gate(dry + wet) * (wet - dry)
            The symmetric gate makes the fusion strictly antisymmetric.
    """

    def __init__(
        self, embed_dim, fusion_type="diff_gate", model_variant="base"
    ):
        super().__init__()
        validate_model_variant(model_variant)
        if model_variant == "bidirectional" and fusion_type != "diff_gate":
            raise ValueError(
                "The bidirectional variant requires fusion_type='diff_gate'"
            )
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.model_variant = model_variant

        if fusion_type == "concat_mlp":
            self.fuse_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(inplace=False),
                nn.Linear(embed_dim, embed_dim),
            )
            self._init_mlp(self.fuse_mlp)

        elif fusion_type == "gate":
            self.gate_proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            self._init_mlp(self.gate_proj)

        elif fusion_type == "diff_gate":
            gate_input_dim = (
                embed_dim if model_variant == "bidirectional" else embed_dim * 2
            )
            self.gate_proj = nn.Sequential(
                nn.Linear(gate_input_dim, embed_dim),
                nn.Sigmoid(),
            )
            self._init_mlp(self.gate_proj)

        elif fusion_type == "diff":
            pass  # 无额外参数

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def _init_mlp(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init_layer(m)

    def forward(self, emb_dry, emb_wet):
        """
        Args:
            emb_dry: (B, D) — 干声分支的 embedding
            emb_wet: (B, D) — 湿声分支的 embedding
        Returns:
            fx_emb: (B, D) — 融合后的 FX embedding
        """
        if self.fusion_type == "diff":
            return emb_wet - emb_dry

        elif self.fusion_type == "concat_mlp":
            return self.fuse_mlp(torch.cat([emb_dry, emb_wet], dim=-1))

        elif self.fusion_type == "gate":
            gate = self.gate_proj(torch.cat([emb_dry, emb_wet], dim=-1))
            return gate * emb_wet + (1 - gate) * emb_dry

        elif self.fusion_type == "diff_gate":
            diff = emb_wet - emb_dry
            if self.model_variant == "bidirectional":
                gate = self.gate_proj(emb_dry + emb_wet)
                return gate * diff
            gate = self.gate_proj(torch.cat([emb_dry, emb_wet], dim=-1))
            return gate * diff + (1 - gate) * emb_wet


class ProjectionHead(nn.Sequential):
    """Projection used by one of the two paper model variants.

    The bidirectional variant uses Tanh to promote sign preservation. Its
    Linear layers retain learned biases, so strict antisymmetry is guaranteed
    only at the fusion output, not at the final normalized projection.
    """

    def __init__(self, embed_dim, proj_dim, model_variant="base"):
        validate_model_variant(model_variant)
        activation = (
            nn.Tanh()
            if model_variant == "bidirectional"
            else nn.ReLU(inplace=False)
        )
        super().__init__(
            nn.Linear(embed_dim, embed_dim),
            activation,
            nn.Linear(embed_dim, proj_dim),
        )
        self.model_variant = model_variant


# ===================== V4 主模型 =====================

class DualBranchFxEncoder(nn.Module):
    """
    V4 双分支 FX Encoder — 交叉注意力 + 差分融合。

    两个分支共享 CNN 权重 (Siamese)，在中间层插入交叉注意力模块让分支互相感知。
    最后通过融合模块提取 FX signature。

    输入:
        original:  (B, 2, T) 干声 stereo
        processed: (B, 2, T) 湿声 stereo
    输出:
        {'embedding': (B, proj_dim), 'z': (B, proj_dim),
         'fusion': (B, embed_dim)}
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        window_size: int = 2048,
        hop_size: int = 512,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 18000,
        embed_dim: int = 2048,
        proj_dim: int = 128,
        fusion_type: str = "diff_gate",
        cross_attn_stages: list = None,
        cross_attn_heads: int = 4,
        cross_attn_pool: int = 4,
        cross_attn_scale: float = 0.1,
        model_variant: str = "base",
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.model_variant = model_variant

        # 默认在 Stage 3 和 Stage 5 后插入交叉注意力
        if cross_attn_stages is None:
            cross_attn_stages = [3, 5]
        self.cross_attn_stages = set(cross_attn_stages)

        # ============ Spectrogram Extraction (共享) ============
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

        # ============ 共享 CNN Backbone (Cnn14 style, 2ch 输入) ============
        # 注意: V3 是 4ch (拼接), V4 是 2ch (每个分支独立)
        stage_channels = [64, 128, 256, 512, 1024, 2048]
        self.conv_block1 = ConvBlock(in_channels=2, out_channels=stage_channels[0])
        self.conv_block2 = ConvBlock(in_channels=stage_channels[0], out_channels=stage_channels[1])
        self.conv_block3 = ConvBlock(in_channels=stage_channels[1], out_channels=stage_channels[2])
        self.conv_block4 = ConvBlock(in_channels=stage_channels[2], out_channels=stage_channels[3])
        self.conv_block5 = ConvBlock(in_channels=stage_channels[3], out_channels=stage_channels[4])
        self.conv_block6 = ConvBlock(in_channels=stage_channels[4], out_channels=stage_channels[5])

        self.conv_blocks = [
            self.conv_block1, self.conv_block2, self.conv_block3,
            self.conv_block4, self.conv_block5, self.conv_block6,
        ]

        # ============ 交叉注意力模块 (在指定 stage 后插入) ============
        self.cross_attns = nn.ModuleDict()
        for stage_idx in self.cross_attn_stages:
            if 1 <= stage_idx <= 6:
                ch = stage_channels[stage_idx - 1]
                self.cross_attns[str(stage_idx)] = CrossAttentionBlock(
                    channels=ch,
                    num_heads=cross_attn_heads,
                    pool_size=cross_attn_pool,
                    residual_scale=cross_attn_scale,
                )

        # ============ 分支独立的 fc ============
        self.fc_embed = nn.Linear(2048, embed_dim, bias=True)

        # ============ 融合模块 ============
        self.fusion = FusionModule(
            embed_dim,
            fusion_type=fusion_type,
            model_variant=model_variant,
        )

        # ============ Projection Head ============
        self.projection = ProjectionHead(
            embed_dim,
            proj_dim,
            model_variant=model_variant,
        )

        self._init_weights()

    def _init_weights(self):
        init_layer(self.fc_embed)
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                init_layer(m)

    def extract_features(self, x):
        """提取 logmel 特征。x: (B, 2, T) → (B, 2, time_steps, mel_bins)"""
        batch_size, chs, seq_len = x.size()
        x = x.reshape(batch_size * chs, seq_len)
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = x.clamp(-80, 40.0)
        x = (x + 80) / 120
        x = (x * 2) - 1
        x = x.reshape(batch_size, chs, x.size(-2), x.size(-1))
        return x

    def forward_dual_backbone(self, x_dry, x_wet):
        """
        双分支共享骨干网络 + 交叉注意力。

        Args:
            x_dry: (B, 2, T) 干声
            x_wet: (B, 2, T) 湿声
        Returns:
            emb_dry: (B, embed_dim)
            emb_wet: (B, embed_dim)
        """
        # 提取频谱特征
        feat_dry = self.extract_features(x_dry)  # (B, 2, time, mel)
        feat_wet = self.extract_features(x_wet)  # (B, 2, time, mel)

        # 共享 CNN + 交叉注意力
        pool_sizes = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (1, 1)]

        for stage_idx, (conv_block, pool_size) in enumerate(
            zip(self.conv_blocks, pool_sizes), start=1
        ):
            feat_dry = conv_block(feat_dry, pool_size=pool_size, pool_type='avg')
            feat_dry = F.dropout(feat_dry, p=0.2, training=self.training)

            feat_wet = conv_block(feat_wet, pool_size=pool_size, pool_type='avg')
            feat_wet = F.dropout(feat_wet, p=0.2, training=self.training)

            # 交叉注意力 (如果该 stage 启用)
            if str(stage_idx) in self.cross_attns:
                feat_dry, feat_wet = self.cross_attns[str(stage_idx)](feat_dry, feat_wet)

        # 全局池化: (B, 2048, T', F') → (B, 2048)
        def global_pool(x):
            x = torch.mean(x, dim=3)       # 频率维度平均
            x1, _ = torch.max(x, dim=2)    # 时间维度最大
            x2 = torch.mean(x, dim=2)      # 时间维度平均
            return x1 + x2

        emb_dry = F.relu(self.fc_embed(global_pool(feat_dry)))
        emb_wet = F.relu(self.fc_embed(global_pool(feat_wet)))

        return emb_dry, emb_wet

    def forward_projection(self, fusion):
        """投影头: e_fx → z (L2 normalized)"""
        z = self.projection(fusion)
        z = F.normalize(z, dim=-1)
        return z

    def forward(self, original, processed):
        """
        Complete forward pass following the paper's representation hierarchy.

        Args:
            original:  (B, 2, T) 干声
            processed: (B, 2, T) 湿声
        Returns:
            dict: {'embedding': (B, proj_dim), 'z': (B, proj_dim),
                   'fusion': (B, embed_dim), 'emb_dry': (B, embed_dim),
                   'emb_wet': (B, embed_dim)}
        """
        emb_dry, emb_wet = self.forward_dual_backbone(original, processed)

        # 融合
        fx_embedding = self.fusion(emb_dry, emb_wet)

        # 投影
        z = self.forward_projection(fx_embedding)

        return {
            'embedding': z,
            'z': z,
            'fusion': fx_embedding,
            'emb_dry': emb_dry,
            'emb_wet': emb_wet,
        }

    def get_embedding(self, original, processed):
        """Return the final normalized 128-d embedding used in the paper."""
        return self.forward(original, processed)['z']

    def get_fusion_representation(self, original, processed, normalized=False):
        """Return the 2048-d e_fx representation before projection."""
        emb_dry, emb_wet = self.forward_dual_backbone(original, processed)
        fx_embedding = self.fusion(emb_dry, emb_wet)
        if normalized:
            fx_embedding = F.normalize(fx_embedding, dim=-1)
        return fx_embedding


def create_model(
    sample_rate=44100,
    embed_dim=2048,
    proj_dim=128,
    fusion_type="diff_gate",
    cross_attn_stages=None,
    cross_attn_heads=4,
    cross_attn_pool=4,
    cross_attn_scale=0.1,
    model_variant="base",
):
    """Create the RelFx paper model."""
    model = DualBranchFxEncoder(
        sample_rate=sample_rate,
        embed_dim=embed_dim,
        proj_dim=proj_dim,
        fusion_type=fusion_type,
        model_variant=model_variant,
        cross_attn_stages=cross_attn_stages,
        cross_attn_heads=cross_attn_heads,
        cross_attn_pool=cross_attn_pool,
        cross_attn_scale=cross_attn_scale,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[RelFx] DualBranchFxEncoder | variant={model_variant} | "
        f"fusion={fusion_type}"
    )
    print(f"[RelFx] Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # 分解参数量
    backbone_params = sum(
        p.numel() for n, p in model.named_parameters()
        if 'cross_attn' not in n and 'fusion' not in n and 'projection' not in n
    )
    cross_attn_params = sum(
        p.numel() for n, p in model.named_parameters() if 'cross_attn' in n
    )
    fusion_params = sum(
        p.numel() for n, p in model.named_parameters() if 'fusion' in n
    )
    print(
        f"[RelFx] Backbone: {backbone_params:,} | "
        f"CrossAttn: {cross_attn_params:,} | Fusion: {fusion_params:,}"
    )

    return model
