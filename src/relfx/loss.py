"""
Cascaded FX Contrastive Learning V3 — 损失函数

所有改进方法都作为独立模块，由 CombinedLoss 统一管理开关。

架构:
    CombinedLoss
    ├── NTXentLoss             (baseline)          — 标准 SimCLR InfoNCE
    ├── ParamRegressionLoss    (方法 1 开关)        — 参数回归辅助 loss
    ├── SoftContrastiveLoss    (方法 2 开关)        — 连续权重 InfoNCE
    ├── DistanceMarginTriplet  (方法 3 开关)        — 参数距离感知 Margin
    └── (Hard Negative Mining)  (方法 4)            — 在采样层实现，见 dataset.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 工具函数 =====================

def compute_param_distance(
    params_i, params_j,
    activate_i=None, activate_j=None,
    distance_type="hybrid",
    fx_param_ranges=None,
):
    """
    计算两组 FX 参数之间的距离。

    Args:
        params_i, params_j: (B, total_params) 归一化参数 [0,1]
        activate_i, activate_j: (B, num_fx) 激活掩码 {0,1}
        distance_type:
            - "param_only": 只看参数值的 L2 距离 (忽略 activate)
            - "activate_aware": 只看两者都激活的效果器的参数距离
            - "hybrid": activate 不同 → 最大距离; 相同 → 参数距离
        fx_param_ranges: list of (start, end) — 每个 FX 在 param 向量中的范围

    Returns:
        distances: (B,) 归一化到 [0, 1] 的距离
    """
    B = params_i.shape[0]

    if distance_type == "param_only":
        # 简单 L2 距离，归一化
        diff = params_i - params_j
        dist = torch.norm(diff, dim=-1)  # (B,)
        # 归一化: 最大可能距离 = sqrt(num_params)
        max_dist = (params_i.shape[-1]) ** 0.5
        return (dist / max_dist).clamp(0, 1)

    elif distance_type == "activate_aware":
        assert activate_i is not None and activate_j is not None
        assert fx_param_ranges is not None

        distances = torch.zeros(B, device=params_i.device)
        num_fx = activate_i.shape[1]
        active_count = torch.zeros(B, device=params_i.device)

        for fx_idx in range(num_fx):
            both_active = activate_i[:, fx_idx] * activate_j[:, fx_idx]  # (B,)
            if fx_param_ranges is not None and fx_idx < len(fx_param_ranges):
                s, e = fx_param_ranges[fx_idx]
                fx_diff = params_i[:, s:e] - params_j[:, s:e]
                fx_dist = torch.norm(fx_diff, dim=-1)  # (B,)
                n_params = e - s
                fx_dist = fx_dist / max(n_params ** 0.5, 1e-6)  # 归一化到 ~[0,1]
                distances += both_active * fx_dist
                active_count += both_active

        # 平均每个效果器的距离
        distances = distances / active_count.clamp(min=1)
        return distances.clamp(0, 1)

    elif distance_type == "hybrid":
        assert activate_i is not None and activate_j is not None

        # activate 不同的 → 视为距离 1.0
        activate_same = (activate_i == activate_j).all(dim=-1).float()  # (B,)

        # activate 相同的 → 用参数距离
        diff = params_i - params_j
        param_dist = torch.norm(diff, dim=-1)
        max_dist = (params_i.shape[-1]) ** 0.5
        param_dist_norm = (param_dist / max_dist).clamp(0, 1)

        # 混合: activate 不同 = 1.0, activate 相同 = param_dist
        distances = activate_same * param_dist_norm + (1 - activate_same) * 1.0
        return distances

    else:
        raise ValueError(f"Unknown distance_type: {distance_type}")


# ===================== Baseline: historical contrastive loss =====================

class NTXentLoss(nn.Module):
    """Contrastive objective used to train the released paper checkpoint.

    This historical implementation prepends the positive logit while the
    off-diagonal comparison matrix still contains that same positive. The
    positive therefore appears twice in the denominator. It is preserved here
    for checkpoint reproducibility and must not be silently replaced by the
    canonical SimCLR NT-Xent objective.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        B = z_i.shape[0]
        N = 2 * B

        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        sim = sim / self.temperature

        sim_ij = torch.diag(sim, B)
        sim_ji = torch.diag(sim, -B)
        positive_samples = torch.cat([sim_ij, sim_ji], dim=0)

        mask = torch.ones((N, N), dtype=torch.bool, device=z.device)
        mask.fill_diagonal_(False)
        negative_samples = sim[mask].reshape(N, N - 1)

        logits = torch.cat([positive_samples.unsqueeze(1), negative_samples], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=z.device)

        loss = F.cross_entropy(logits, labels)
        return loss


# ===================== 方法 1: 参数回归 Loss =====================

class ParamRegressionHead(nn.Module):
    """
    参数回归 Head — 从 2048-d fusion representation 预测 FX 参数值。

    强制 fusion representation 编码参数的连续值，否则回归 loss 不会下降。
    可选择同时预测 activate（开关），用 BCE loss。
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        total_fx_params: int = 57,
        num_fx: int = 8,
        hidden_dim: int = 512,
        predict_activate: bool = True,
        per_fx_head: bool = False,
        fx_param_ranges: list = None,
    ):
        super().__init__()
        self.total_fx_params = total_fx_params
        self.num_fx = num_fx
        self.predict_activate = predict_activate
        self.per_fx_head = per_fx_head
        self.fx_param_ranges = fx_param_ranges

        if per_fx_head and fx_param_ranges is not None:
            # 每个效果器独立的回归 head
            self.param_heads = nn.ModuleList()
            for fx_idx, (s, e) in enumerate(fx_param_ranges):
                n_params = e - s
                if n_params > 0:
                    self.param_heads.append(nn.Sequential(
                        nn.Linear(embed_dim, hidden_dim),
                        nn.ReLU(inplace=False),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, n_params),
                        nn.Sigmoid(),  # 参数在 [0, 1]
                    ))
                else:
                    self.param_heads.append(None)
        else:
            # 共享回归 head
            self.param_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=False),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dim, total_fx_params),
                nn.Sigmoid(),  # 参数在 [0, 1]
            )

        if predict_activate:
            self.activate_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dim, num_fx),
            )

    def forward(self, fusion):
        """
        Args:
            fusion: (B, embed_dim)
        Returns:
            param_pred: (B, total_fx_params)
            activate_pred: (B, num_fx) logits — 如果 predict_activate
        """
        if self.per_fx_head and self.fx_param_ranges is not None:
            parts = []
            for fx_idx, (s, e) in enumerate(self.fx_param_ranges):
                if e > s and self.param_heads[fx_idx] is not None:
                    parts.append(self.param_heads[fx_idx](fusion))
                else:
                    parts.append(torch.zeros(
                        fusion.shape[0], 0, device=fusion.device
                    ))
            param_pred = torch.cat(parts, dim=-1)
        else:
            param_pred = self.param_head(fusion)

        activate_pred = None
        if self.predict_activate:
            activate_pred = self.activate_head(fusion)

        return param_pred, activate_pred


class ParamRegressionLoss(nn.Module):
    """
    方法 1: 参数回归损失

    L_param = MSE(pred_params, gt_params) + λ_act * BCE(pred_activate, gt_activate)

    只对"激活的"效果器的参数计算 MSE (未激活的参数是随机的，不应该学)。
    """

    def __init__(
        self,
        weight: float = 0.5,
        activate_weight: float = 0.3,
        predict_activate: bool = True,
        fx_param_ranges: list = None,
    ):
        super().__init__()
        self.weight = weight
        self.activate_weight = activate_weight
        self.predict_activate = predict_activate
        self.fx_param_ranges = fx_param_ranges

    def forward(self, param_pred, param_gt, activate_pred=None, activate_gt=None):
        """
        Args:
            param_pred: (B, total_params) — 模型预测
            param_gt: (B, total_params) — 真实参数
            activate_pred: (B, num_fx) logits
            activate_gt: (B, num_fx) {0, 1}
        Returns:
            loss: scalar
            loss_dict: dict
        """
        loss_dict = {}

        # 参数 MSE — 只算激活的效果器
        if activate_gt is not None and self.fx_param_ranges is not None:
            masked_mse = 0.0
            count = 0
            for fx_idx, (s, e) in enumerate(self.fx_param_ranges):
                if e <= s:
                    continue
                # (B,) → (B, 1) mask
                mask = activate_gt[:, fx_idx].unsqueeze(-1)  # (B, 1)
                fx_pred = param_pred[:, s:e]  # (B, n_params)
                fx_gt = param_gt[:, s:e]
                # 只对激活的样本算 MSE
                diff = (fx_pred - fx_gt) ** 2 * mask  # (B, n_params)
                masked_mse += diff.sum()
                count += mask.sum() * (e - s)

            param_loss = masked_mse / count.clamp(min=1)
        else:
            param_loss = F.mse_loss(param_pred, param_gt)

        loss_dict['param_mse'] = param_loss.item()
        total = self.weight * param_loss

        # Activate BCE
        if self.predict_activate and activate_pred is not None and activate_gt is not None:
            act_loss = F.binary_cross_entropy_with_logits(activate_pred, activate_gt)
            loss_dict['activate_bce'] = act_loss.item()
            total = total + self.weight * self.activate_weight * act_loss

        loss_dict['param_regression_total'] = total.item()
        return total, loss_dict


# ===================== 方法 2: Soft Contrastive Loss =====================

class SoftContrastiveLoss(nn.Module):
    """
    方法 2: Soft InfoNCE — 用参数距离给 batch 内所有对赋连续权重

    标准 InfoNCE: target = one-hot [1, 0, 0, ...]
    Soft InfoNCE:  target = softmax(exp(-d(θ_i, θ_j) / σ))

    参数相近的样本不再被当作完全负样本推开。
    """

    def __init__(
        self,
        temperature: float = 0.1,
        sigma: float = 0.2,
        distance_type: str = "hybrid",
        replace_infonce: bool = False,
        extra_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma
        self.distance_type = distance_type
        self.replace_infonce = replace_infonce
        self.extra_weight = extra_weight

    def forward(
        self,
        z_a, z_b,
        params_a, params_b,
        activate_a=None, activate_b=None,
        params_diff=None, activate_diff=None,
        fx_param_ranges=None,
    ):
        """
        Args:
            z_a, z_b: (B, proj_dim) — 正样本对的 embeddings
            params_a, params_b: 这里实际上 params_a == params_b (正样本共享参数)
                                但 batch 内不同样本有不同参数
            activate_a: (B, num_fx) — 共享的 activate
            params_diff: (B, total_params) — 不同 FX 参数 (triplet 负样本)
            activate_diff: (B, num_fx)
            fx_param_ranges: 每个 FX 的参数范围
        """
        B = z_a.shape[0]

        # 构建 batch 内所有样本的参数和 z
        # z: (2B, dim)  — [z_a_0, ..., z_a_B-1, z_b_0, ..., z_b_B-1]
        z = torch.cat([z_a, z_b], dim=0)
        N = z.shape[0]

        # 参数: 前 B 个是 params_a (== params_b), 后 B 个也是 params_a
        params_all = torch.cat([params_a, params_a], dim=0)       # (2B, total_params)

        if activate_a is not None:
            activate_all = torch.cat([activate_a, activate_a], dim=0)  # (2B, num_fx)
        else:
            activate_all = None

        # 计算 embedding 相似度矩阵: (N, N)
        sim_matrix = F.cosine_similarity(
            z.unsqueeze(1), z.unsqueeze(0), dim=-1
        ) / self.temperature

        # 计算参数距离矩阵: (N, N) — 向量化计算避免 for 循环
        # 对于 param_only 和 hybrid 模式可以向量化
        # 展开成 (N, N, D) 然后算距离
        params_i = params_all.unsqueeze(1).expand(N, N, -1)  # (N, N, D)
        params_j = params_all.unsqueeze(0).expand(N, N, -1)  # (N, N, D)
        param_diff = params_i - params_j
        param_dist_flat = torch.norm(param_diff, dim=-1)  # (N, N)
        max_dist = (params_all.shape[-1]) ** 0.5
        param_dist_norm = (param_dist_flat / max_dist).clamp(0, 1)

        if self.distance_type == "hybrid" and activate_all is not None:
            act_i = activate_all.unsqueeze(1).expand(N, N, -1)
            act_j = activate_all.unsqueeze(0).expand(N, N, -1)
            activate_same = (act_i == act_j).all(dim=-1).float()  # (N, N)
            param_dist_matrix = activate_same * param_dist_norm + (1 - activate_same) * 1.0
        else:
            param_dist_matrix = param_dist_norm

        # Soft targets: 参数距离越小 → 权重越高
        soft_targets = torch.exp(-param_dist_matrix / self.sigma)  # (N, N)

        # 去掉自身对角线
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
        soft_targets = soft_targets * diag_mask.float()

        # 归一化成概率分布 (每行)
        row_sums = soft_targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        soft_targets = soft_targets / row_sums

        # 相似度矩阵: 去掉对角线后 reshape 成 (N, N-1)
        # 这样避免 -inf 导致的 NaN
        sim_off_diag = sim_matrix[diag_mask].reshape(N, N - 1)
        soft_off_diag = soft_targets[diag_mask].reshape(N, N - 1)

        # log-softmax (在 N-1 维上)
        log_probs = F.log_softmax(sim_off_diag, dim=-1)  # (N, N-1)

        # Cross-entropy with soft targets
        loss = -(soft_off_diag * log_probs).sum(dim=-1).mean()

        return loss, {'soft_contrastive_loss': loss.item()}


# ===================== 方法 3: 距离感知 Margin Triplet =====================

class DistanceMarginTripletLoss(nn.Module):
    """
    方法 3: 参数距离感知的 Margin Triplet Loss

    标准 triplet: margin = 固定值 (0.3)
    改进: margin(Δθ) = base_margin + margin_scale * param_dist(θ_pos, θ_neg)

    参数差异大的负样本 → 更大的 margin → 推得更远
    参数差异小的负样本 → 更小的 margin → 允许更近
    """

    def __init__(
        self,
        weight: float = 0.5,
        base_margin: float = 0.1,
        margin_scale: float = 0.5,
        max_margin: float = 1.0,
        distance_type: str = "hybrid",
    ):
        super().__init__()
        self.weight = weight
        self.base_margin = base_margin
        self.margin_scale = margin_scale
        self.max_margin = max_margin
        self.distance_type = distance_type

    def forward(
        self,
        z_anchor, z_pos, z_neg,
        params_pos, params_neg,
        activate_pos=None, activate_neg=None,
        fx_param_ranges=None,
    ):
        """
        Args:
            z_anchor: (B, dim) — anchor embedding (audio_a with shared FX)
            z_pos: (B, dim) — positive (audio_b with shared FX)
            z_neg: (B, dim) — negative (audio_a with different FX)
            params_pos: (B, total_params) — 正样本 FX 参数
            params_neg: (B, total_params) — 负样本 FX 参数
            activate_pos, activate_neg: (B, num_fx)
            fx_param_ranges: list of (start, end)
        """
        # 计算正负样本的参数距离
        param_dist = compute_param_distance(
            params_pos, params_neg,
            activate_pos, activate_neg,
            distance_type=self.distance_type,
            fx_param_ranges=fx_param_ranges,
        )  # (B,)

        # 动态 margin
        margin = self.base_margin + self.margin_scale * param_dist
        margin = margin.clamp(max=self.max_margin)  # (B,)

        # 计算 embedding 距离
        d_pos = 1.0 - F.cosine_similarity(z_anchor, z_pos, dim=-1)  # (B,)
        d_neg = 1.0 - F.cosine_similarity(z_anchor, z_neg, dim=-1)  # (B,)

        # Triplet loss with dynamic margin
        loss = F.relu(d_pos - d_neg + margin).mean()

        return self.weight * loss, {
            'dist_margin_triplet': loss.item(),
            'mean_margin': margin.mean().item(),
            'mean_param_dist': param_dist.mean().item(),
        }


# ===================== 组合 Loss =====================

class CombinedLoss(nn.Module):
    """
    统一管理所有 loss 的组合。

    通过 config 中的 LOSS_SWITCHES 控制开关。
    每个方法完全独立，可以任意叠加。

    Usage:
        criterion = CombinedLoss(switches, configs, fx_param_ranges)
        loss, loss_dict = criterion(
            z_a, z_b, z_diff,
            fusion_a,
            params_shared, activate_shared,
            params_diff, activate_diff,
        )
    """

    def __init__(
        self,
        switches: dict,
        temperature: float = 0.15,
        use_baseline_triplet: bool = True,
        baseline_triplet_margin: float = 0.3,
        baseline_triplet_weight: float = 0.5,
        param_regression_config: dict = None,
        soft_contrastive_config: dict = None,
        distance_margin_config: dict = None,
        total_fx_params: int = 57,
        num_fx: int = 8,
        embed_dim: int = 2048,
        fx_param_ranges: list = None,
    ):
        super().__init__()

        self.switches = switches
        self.fx_param_ranges = fx_param_ranges

        # ---- Baseline ----
        self.nt_xent = NTXentLoss(temperature)
        self.use_baseline_triplet = use_baseline_triplet
        self.baseline_triplet_margin = baseline_triplet_margin
        self.baseline_triplet_weight = baseline_triplet_weight

        # 如果方法 3 开启，替代 baseline triplet
        if switches.get("distance_margin", False):
            self.use_baseline_triplet = False

        # ---- 方法 1: 参数回归 ----
        self.param_regression_loss = None
        self.param_regression_head = None
        if switches.get("param_regression", False):
            cfg = param_regression_config or {}
            self.param_regression_head = ParamRegressionHead(
                embed_dim=embed_dim,
                total_fx_params=total_fx_params,
                num_fx=num_fx,
                hidden_dim=cfg.get("hidden_dim", 512),
                predict_activate=cfg.get("predict_activate", True),
                per_fx_head=cfg.get("per_fx_head", False),
                fx_param_ranges=fx_param_ranges,
            )
            self.param_regression_loss = ParamRegressionLoss(
                weight=cfg.get("weight", 0.5),
                activate_weight=cfg.get("activate_weight", 0.3),
                predict_activate=cfg.get("predict_activate", True),
                fx_param_ranges=fx_param_ranges,
            )

        # ---- 方法 2: Soft Contrastive ----
        self.soft_contrastive = None
        if switches.get("soft_contrastive", False):
            cfg = soft_contrastive_config or {}
            self.soft_contrastive = SoftContrastiveLoss(
                temperature=temperature,
                sigma=cfg.get("sigma", 0.2),
                distance_type=cfg.get("distance_type", "hybrid"),
                replace_infonce=cfg.get("replace_infonce", False),
                extra_weight=cfg.get("extra_weight", 0.5),
            )

        # ---- 方法 3: 距离感知 Margin Triplet ----
        self.distance_margin_triplet = None
        if switches.get("distance_margin", False):
            cfg = distance_margin_config or {}
            self.distance_margin_triplet = DistanceMarginTripletLoss(
                weight=cfg.get("weight", 0.5),
                base_margin=cfg.get("base_margin", 0.1),
                margin_scale=cfg.get("margin_scale", 0.5),
                max_margin=cfg.get("max_margin", 1.0),
                distance_type=cfg.get("distance_type", "hybrid"),
            )

    def forward(
        self,
        z_a, z_b, z_diff=None,
        fusion_a=None,
        params_shared=None, activate_shared=None,
        params_diff=None, activate_diff=None,
    ):
        """
        统一前向传播。

        Args:
            z_a, z_b: (B, proj_dim) — 正样本对的 projection
            z_diff: (B, proj_dim) — 负样本 (不同 FX) 的 projection
            fusion_a: (B, embed_dim) — anchor 的 e_fx (参数回归用)
            params_shared: (B, total_params) — 正样本共享的 FX 参数
            activate_shared: (B, num_fx) — 正样本共享的 activate
            params_diff: (B, total_params) — 负样本的 FX 参数
            activate_diff: (B, num_fx) — 负样本的 activate
        """
        result = {}
        total_loss = torch.tensor(0.0, device=z_a.device)

        # ---- Baseline InfoNCE ----
        # 如果 soft_contrastive 开启且 replace_infonce=True，跳过
        skip_infonce = (
            self.soft_contrastive is not None
            and self.soft_contrastive.replace_infonce
        )

        if not skip_infonce:
            nt_xent_loss = self.nt_xent(z_a, z_b)
            total_loss = total_loss + nt_xent_loss
            result['nt_xent_loss'] = nt_xent_loss.item()

        # ---- Baseline Triplet (固定 margin) ----
        if self.use_baseline_triplet and z_diff is not None:
            d_pos = 1.0 - F.cosine_similarity(z_a, z_b, dim=-1)
            d_neg = 1.0 - F.cosine_similarity(z_a, z_diff, dim=-1)
            triplet_loss = F.relu(
                d_pos - d_neg + self.baseline_triplet_margin
            ).mean()
            total_loss = total_loss + self.baseline_triplet_weight * triplet_loss
            result['baseline_triplet'] = triplet_loss.item()

        # ---- 方法 1: 参数回归 ----
        if self.param_regression_loss is not None and fusion_a is not None:
            assert params_shared is not None, "方法1需要 params_shared"
            param_pred, activate_pred = self.param_regression_head(fusion_a)
            reg_loss, reg_dict = self.param_regression_loss(
                param_pred, params_shared, activate_pred, activate_shared,
            )
            total_loss = total_loss + reg_loss
            result.update(reg_dict)

        # ---- 方法 2: Soft Contrastive ----
        if self.soft_contrastive is not None:
            assert params_shared is not None, "方法2需要 params_shared"
            soft_loss, soft_dict = self.soft_contrastive(
                z_a, z_b,
                params_shared, params_shared,  # 正样本对共享参数
                activate_shared, activate_shared,
                params_diff, activate_diff,
                fx_param_ranges=self.fx_param_ranges,
            )
            if self.soft_contrastive.replace_infonce:
                total_loss = total_loss + soft_loss
            else:
                total_loss = total_loss + self.soft_contrastive.extra_weight * soft_loss
            result.update(soft_dict)

        # ---- 方法 3: 距离感知 Margin Triplet ----
        if self.distance_margin_triplet is not None and z_diff is not None:
            assert params_shared is not None and params_diff is not None, \
                "方法3需要 params_shared 和 params_diff"
            dm_loss, dm_dict = self.distance_margin_triplet(
                z_a, z_b, z_diff,
                params_shared, params_diff,
                activate_shared, activate_diff,
                fx_param_ranges=self.fx_param_ranges,
            )
            total_loss = total_loss + dm_loss
            result.update(dm_dict)

        result['total_loss'] = total_loss.item()
        return total_loss, result
