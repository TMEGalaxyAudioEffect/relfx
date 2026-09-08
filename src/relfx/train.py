"""RelFx V6-fix training entry point with optional distributed data parallelism.

V6 核心改进（对比 V5）:
    ★ 动态 FX 概率调度 (Dynamic FX Probability Scheduling):
      - 定期对每个效果器单独评估检索能力
      - 学得差的效果器 → 训练时出现概率提高
      - 学得好的效果器 → 概率降低
      - 带平滑系数防止概率突变

    其余与 V5 完全一致:
    ┌──────────────────────────────────────────────────────────┐
    │  模型: DualBranchFxEncoder                               │
    │    - 共享 CNN 骨干 (Siamese)                              │
    │    - 中间层交叉注意力 (互相感知)                            │
    │    - 融合模块 (diff_gate / diff / concat_mlp / gate)      │
    │                                                          │
    │  开关 1-4: 与 V3 完全一致, 复用 loss.py                    │
    │  数据: 默认 cross_segment=True                            │
    │  ★ 新增: 动态 FX 概率调度                                 │
    └──────────────────────────────────────────────────────────┘

启动方式 (4 卡 DDP):
    torchrun --nproc_per_node=4 -m relfx.train

也可命令行覆盖:
    torchrun --nproc_per_node=4 -m relfx.train \\
        --enable param_regression \\
        --fusion-type diff_gate
"""

import os
import time
import argparse
import logging
import json
from datetime import datetime

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta
from tqdm import tqdm

from . import config as cfg
from .dataset import AudioSegmentDataset, FxContrastiveCollator
from .model import MODEL_VARIANTS, create_model
from .loss import CombinedLoss
from .bidirectional import orient_fusion_for_regression, swap_ordered_pairs
from .fx_chain.fx_aug import Random_FX_Chain

SAMPLE_RATE = cfg.SAMPLE_RATE
AUDIO_DIR = cfg.AUDIO_DIR
CHECKPOINT_DIR = cfg.CHECKPOINT_DIR
LOG_DIR = cfg.LOG_DIR
FX_CHAIN_ORDER = cfg.FX_CHAIN_ORDER
FX_PROB = cfg.FX_PROB
TRAIN_CONFIG = cfg.TRAIN_CONFIG
MODEL_CONFIG = cfg.MODEL_CONFIG
LOSS_SWITCHES = cfg.LOSS_SWITCHES
CROSS_SEGMENT = cfg.CROSS_SEGMENT
CROSS_SEGMENT_POLICY = cfg.CROSS_SEGMENT_POLICY
CROSS_SEGMENT_RECIPE_VERSION = cfg.CROSS_SEGMENT_RECIPE_VERSION
BASE_CHECKPOINT_VERSION = cfg.BASE_CHECKPOINT_VERSION
STRUCTURE_SEGMENT_JSON = cfg.STRUCTURE_SEGMENT_JSON
DENSITY_FILTER_AUDIO_DIRS = cfg.DENSITY_FILTER_AUDIO_DIRS
DYNAMIC_FX_PROB_CONFIG = cfg.DYNAMIC_FX_PROB_CONFIG
BIDIRECTIONAL_CONFIG = getattr(cfg, 'BIDIRECTIONAL_CONFIG', {"enabled": False})
ensure_dirs = cfg.ensure_dirs


# ===================== DDP 工具函数 =====================

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main_process():
    return get_rank() == 0

def setup_ddp():
    if "RANK" not in os.environ:
        return
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


def setup_logging(log_dir, rank=0):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    handlers = []
    if rank == 0:
        handlers.append(logging.FileHandler(log_file))
        handlers.append(logging.StreamHandler())
    else:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers,
    )
    return log_file


def get_fx_param_ranges(fx_chain):
    """获取每个 FX 在参数向量中的 (start, end) 范围"""
    ranges = []
    start = 0
    for fx_name in fx_chain.fx_indices.keys():
        n = fx_chain.fx_processors[fx_name].num_params
        ranges.append((start, start + n))
        start += n
    return ranges


# ===================== V6 新增: 动态 FX 概率调度 =====================

@torch.no_grad()
def evaluate_per_fx(model, val_loader, fx_chain, device, cross_segment=False,
                    max_batches=8):
    """
    对每个效果器单独评估检索能力。

    对 batch 中的数据，分别只激活单个效果器，计算正样本对的 embedding 相似度。
    相似度越低 → 模型对该效果器学得越差 → 应提高该效果器的训练概率。

    Returns:
        per_fx_losses: dict {fx_name: float} — 每个效果器的平均 loss (越高越差)
    """
    model.eval()

    fx_names = list(fx_chain.fx_indices.keys())
    per_fx_losses = {fx: 0.0 for fx in fx_names}
    per_fx_counts = {fx: 0 for fx in fx_names}

    total_steps = min(max_batches, len(val_loader)) * len(fx_names)
    pbar = tqdm(
        total=total_steps,
        desc="  [DynFxProb] Evaluating per-FX",
        disable=not is_main_process(),
        leave=False,
        ncols=100,
    )

    batch_count = 0
    for batch in val_loader:
        if batch_count >= max_batches:
            break
        batch_count += 1

        audio_a = batch['audio_a'].to(device)

        if cross_segment:
            fx_source = batch['audio_a_alt'].to(device)
        else:
            fx_source = audio_a

        B = audio_a.size(0)

        for fx_name in fx_names:
            fx_idx = fx_chain.fx_indices[fx_name]

            # 只激活该效果器
            activate = torch.zeros(B, len(fx_names), device=device)
            activate[:, fx_idx] = 1.0

            # 采样两组不同的参数
            nn_param_1 = torch.rand(B, fx_chain.total_num_param, device=device)
            nn_param_2 = torch.rand(B, fx_chain.total_num_param, device=device)

            # 只处理当前效果器，跳过其他效果器的计算
            processors_order = [fx_name]

            try:
                # 用相同参数处理两个不同片段 → 正样本对
                processed_1, _, _ = fx_chain(
                    fx_source.clone(), nn_param=nn_param_1,
                    activate=activate, processors_order=processors_order,
                )

                # 用不同参数处理同一片段 → 负样本
                processed_2, _, _ = fx_chain(
                    fx_source.clone(), nn_param=nn_param_2,
                    activate=activate, processors_order=processors_order,
                )

                # 防止 clipping
                for t in [processed_1, processed_2]:
                    max_val = t.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
                    mask = max_val > 1.0
                    if mask.any():
                        t.data = torch.where(mask, t / max_val * 0.99, t)

                # 计算 embedding
                out_1 = model(audio_a, processed_1)
                out_2 = model(audio_a, processed_2)

                z_1 = out_1['z']  # (B, proj_dim)
                z_2 = out_2['z']  # (B, proj_dim)

                # 用 z_1 和 z_2 的"不相似度"衡量模型能否区分不同参数
                # 如果模型学得好，不同参数应该得到不同的 embedding → 相似度低 → loss 低
                # 如果模型学得差，不同参数的 embedding 差不多 → 相似度高 → loss 高
                sim = torch.nn.functional.cosine_similarity(z_1, z_2, dim=-1)  # (B,)
                # loss = 相似度均值 (越高说明模型越分不清不同参数 → 学得越差)
                fx_loss = sim.mean().item()

                per_fx_losses[fx_name] += fx_loss
                per_fx_counts[fx_name] += 1

            except Exception as e:
                logging.warning(f"Error evaluating FX '{fx_name}': {e}")
            finally:
                pbar.update(1)
                pbar.set_postfix_str(f"{fx_name}: {per_fx_losses[fx_name]/(per_fx_counts[fx_name] or 1):.4f}")

    pbar.close()

    # 取平均
    for fx in fx_names:
        if per_fx_counts[fx] > 0:
            per_fx_losses[fx] /= per_fx_counts[fx]
        else:
            per_fx_losses[fx] = 0.5  # fallback

    model.train()
    return per_fx_losses


def update_fx_probabilities(fx_chain, collator, per_fx_losses, old_probs,
                            dyn_config, epoch):
    """
    根据每个效果器的评估 loss 更新训练概率。

    loss 越高 (学得越差) → 概率越大 → 训练中出现更多
    loss 越低 (学得越好) → 概率越小 → 减少训练占比

    Args:
        fx_chain: Random_FX_Chain
        collator: FxContrastiveCollator (内含 fx_prob 用于采样 activate)
        per_fx_losses: dict {fx_name: float}
        old_probs: dict {fx_name: float} 当前概率
        dyn_config: DYNAMIC_FX_PROB_CONFIG
        epoch: 当前 epoch

    Returns:
        new_probs: dict {fx_name: float}
    """
    warmup = dyn_config.get("warmup_epochs", 10)
    if epoch < warmup:
        logging.info(f"  [DynFxProb] Warmup phase (epoch {epoch} < {warmup}), skipping update")
        return old_probs

    min_prob = dyn_config.get("min_prob", 0.2)
    max_prob = dyn_config.get("max_prob", 1.0)
    alpha = dyn_config.get("smoothing_alpha", 0.5)

    losses = per_fx_losses
    loss_values = list(losses.values())
    min_loss = min(loss_values)
    max_loss = max(loss_values)

    new_probs = {}
    if max_loss - min_loss < 1e-6:
        # 所有效果器 loss 差不多，保持原概率
        logging.info("  [DynFxProb] All FX losses similar, keeping current probs")
        return old_probs

    for fx_name, loss in losses.items():
        # 线性映射: loss 越高 → 概率越大
        raw_prob = (loss - min_loss) / (max_loss - min_loss) * (max_prob - min_prob) + min_prob

        # 平滑: new = alpha * raw + (1 - alpha) * old
        old_p = old_probs.get(fx_name, 0.6)
        smoothed_prob = alpha * raw_prob + (1 - alpha) * old_p
        new_probs[fx_name] = round(smoothed_prob, 4)

    # 更新 fx_chain 的概率
    fx_chain.update_fx_prob(new_probs)

    # 日志
    logging.info("  [DynFxProb] Updated FX probabilities:")
    for fx_name in sorted(new_probs.keys()):
        old_p = old_probs.get(fx_name, 0.6)
        new_p = new_probs[fx_name]
        loss_val = losses[fx_name]
        arrow = "↑" if new_p > old_p + 0.01 else ("↓" if new_p < old_p - 0.01 else "→")
        logging.info(f"    {fx_name:20s}: {old_p:.3f} {arrow} {new_p:.3f}  (loss={loss_val:.4f})")

    return new_probs


def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler,
    epoch, device, fx_chain, use_amp=False, grad_accum_steps=1,
    cross_segment=False, max_batches=None, bidirectional_config=None,
):
    """
    训练一个 epoch — V4 改动：
    1. 模型是双分支, forward(original, processed) 内部自动分流
    2. 训练流程与 V3 基本一致, 只是模型换了
    3. cross_segment 模式下 FX 施加在 alt 片段上
    """
    model.train()
    criterion.train()

    total_loss_sum = 0.0
    loss_sums = {}
    n_batches = 0

    epoch_start = time.time()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_start = time.time()

        audio_a = batch['audio_a'].to(device)
        audio_b = batch['audio_b'].to(device)
        nn_param_shared = batch['nn_param_shared'].to(device)
        nn_param_diff = batch['nn_param_diff'].to(device)
        activate_shared = batch['activate_shared'].to(device)
        activate_diff = batch['activate_diff'].to(device)
        processors_order = batch['processors_order']
        processors_order_diff = batch['processors_order_diff']

        # cross_segment: 所有输入的 dry 和 wet 都来自不同位置
        if cross_segment:
            fx_source_a = batch['audio_a_alt'].to(device)
            fx_source_b = batch['audio_b_alt'].to(device)
        else:
            fx_source_a = audio_a
            fx_source_b = audio_b

        # 在 GPU 上施加效果链
        with torch.no_grad():
            # 正样本: FX(alt片段, P) — dry=audio_a, wet=FX(audio_a_alt, P)
            processed_a, _, _ = fx_chain(
                fx_source_a.clone(), nn_param=nn_param_shared,
                activate=activate_shared, processors_order=processors_order,
            )
            processed_b, _, _ = fx_chain(
                fx_source_b.clone(), nn_param=nn_param_shared,
                activate=activate_shared, processors_order=processors_order,
            )
            # 负样本: 也用 cross_segment — dry=audio_a, wet=FX(audio_a_alt, P')
            # 与正样本唯一区别是 FX 参数不同 (P vs P')
            processed_a_diff, _, _ = fx_chain(
                fx_source_a.clone(), nn_param=nn_param_diff,
                activate=activate_diff, processors_order=processors_order_diff,
            )

            # 防止 clipping
            for t in [processed_a, processed_b, processed_a_diff]:
                max_val = t.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
                mask = max_val > 1.0
                if mask.any():
                    t.data = torch.where(mask, t / max_val * 0.99, t)

        # V4: 模型是双分支，分别输入 (dry, wet)
        # 需要分别对三组做 forward: (a, proc_a), (b, proc_b), (a, proc_a_diff)
        B = audio_a.size(0)

        # Bidirectional training swaps all members of a triplet with one mask.
        # Antisymmetric fusion then reverses the relation direction directly.
        bidir_cfg = bidirectional_config or {"enabled": False}
        if bidir_cfg.get("enabled", False):
            flip_ratio = bidir_cfg.get("flip_ratio", 0.5)
            flip_mask = torch.rand(B, device=audio_a.device) < flip_ratio  # (B,) bool

            if flip_mask.any():
                orig_a, wet_a = swap_ordered_pairs(
                    audio_a, processed_a, flip_mask
                )
                orig_b, wet_b = swap_ordered_pairs(
                    audio_b, processed_b, flip_mask
                )
                orig_a_diff, wet_a_diff = swap_ordered_pairs(
                    audio_a, processed_a_diff, flip_mask
                )
            else:
                orig_a, wet_a = audio_a, processed_a
                orig_b, wet_b = audio_b, processed_b
                orig_a_diff, wet_a_diff = audio_a, processed_a_diff
        else:
            orig_a, wet_a = audio_a, processed_a
            orig_b, wet_b = audio_b, processed_b
            orig_a_diff, wet_a_diff = audio_a, processed_a_diff

        # 拼成一个大 batch 统一 forward (共享 backbone 高效)
        orig_cat = torch.cat([orig_a, orig_b, orig_a_diff], dim=0)  # (3B, 2, T)
        proc_cat = torch.cat([wet_a, wet_b, wet_a_diff], dim=0)     # (3B, 2, T)

        if use_amp:
            with autocast():
                out_all = model(orig_cat, proc_cat)
                z_a    = out_all['z'][:B]
                z_b    = out_all['z'][B:2*B]
                z_diff = out_all['z'][2*B:]

                fusion_a = out_all['fusion'][:B]
                fusion_for_regression = fusion_a
                if bidir_cfg.get("enabled", False) and flip_mask.any():
                    fusion_for_regression = orient_fusion_for_regression(
                        fusion_a, flip_mask
                    )

                loss, loss_dict = criterion(
                    z_a, z_b, z_diff,
                    fusion_a=fusion_for_regression,
                    params_shared=nn_param_shared,
                    activate_shared=activate_shared,
                    params_diff=nn_param_diff,
                    activate_diff=activate_diff,
                )
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
        else:
            out_all = model(orig_cat, proc_cat)
            z_a    = out_all['z'][:B]
            z_b    = out_all['z'][B:2*B]
            z_diff = out_all['z'][2*B:]

            fusion_a = out_all['fusion'][:B]
            fusion_for_regression = fusion_a
            if bidir_cfg.get("enabled", False) and flip_mask.any():
                fusion_for_regression = orient_fusion_for_regression(
                    fusion_a, flip_mask
                )

            loss, loss_dict = criterion(
                z_a, z_b, z_diff,
                fusion_a=fusion_for_regression,
                params_shared=nn_param_shared,
                activate_shared=activate_shared,
                params_diff=nn_param_diff,
                activate_diff=activate_diff,
            )
            loss = loss / grad_accum_steps

            loss.backward()

        # 梯度累积
        reached_batch_limit = (
            max_batches is not None and (batch_idx + 1) >= max_batches
        )
        if (
            (batch_idx + 1) % grad_accum_steps == 0
            or (batch_idx + 1) == len(dataloader)
            or reached_batch_limit
        ):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss_sum += loss_dict['total_loss']
        for k, v in loss_dict.items():
            if k != 'total_loss':
                loss_sums[k] = loss_sums.get(k, 0.0) + v
        n_batches += 1

        batch_time = time.time() - batch_start

        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            parts = [f"Loss: {loss_dict['total_loss']:.4f}"]
            for k, v in loss_dict.items():
                if k != 'total_loss':
                    parts.append(f"{k}: {v:.4f}")
            detail = " | ".join(parts)
            logging.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"{detail} | Time: {batch_time:.2f}s"
            )

    epoch_time = time.time() - epoch_start

    metrics = {
        'epoch': epoch,
        'total_loss': total_loss_sum / max(n_batches, 1),
        'epoch_time': epoch_time,
        'n_batches': n_batches,
    }
    for k, v in loss_sums.items():
        metrics[k] = v / max(n_batches, 1)

    parts = [f"Avg Loss: {metrics['total_loss']:.4f}"]
    for k, v in loss_sums.items():
        parts.append(f"{k}: {v/max(n_batches,1):.4f}")
    detail = " | ".join(parts)
    logging.info(f"Epoch {epoch} Summary | {detail} | Time: {epoch_time:.1f}s")

    return metrics


@torch.no_grad()
def validate(
    model, dataloader, criterion, device, fx_chain, cross_segment=False,
    max_batches=None,
):
    """验证 — V4 版本"""
    model.eval()
    criterion.eval()

    total_loss_sum = 0.0
    n_batches = 0
    embeddings_a = []
    embeddings_b = []

    pbar = tqdm(
        dataloader,
        desc="  Validating",
        disable=not is_main_process(),
        leave=False,
        ncols=100,
    )
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        audio_a = batch['audio_a'].to(device)
        audio_b = batch['audio_b'].to(device)
        nn_param_shared = batch['nn_param_shared'].to(device)
        nn_param_diff = batch['nn_param_diff'].to(device)
        activate_shared = batch['activate_shared'].to(device)
        activate_diff = batch['activate_diff'].to(device)
        processors_order = batch['processors_order']
        processors_order_diff = batch['processors_order_diff']

        if cross_segment:
            fx_source_a = batch['audio_a_alt'].to(device)
            fx_source_b = batch['audio_b_alt'].to(device)
        else:
            fx_source_a = audio_a
            fx_source_b = audio_b

        # 正样本: FX(alt片段, P)
        processed_a, _, _ = fx_chain(
            fx_source_a.clone(), nn_param=nn_param_shared,
            activate=activate_shared, processors_order=processors_order,
        )
        processed_b, _, _ = fx_chain(
            fx_source_b.clone(), nn_param=nn_param_shared,
            activate=activate_shared, processors_order=processors_order,
        )
        # 负样本: 也用 cross_segment — FX(alt片段, P')
        processed_a_diff, _, _ = fx_chain(
            fx_source_a.clone(), nn_param=nn_param_diff,
            activate=activate_diff, processors_order=processors_order_diff,
        )

        B = audio_a.size(0)
        orig_cat = torch.cat([audio_a, audio_b, audio_a], dim=0)
        proc_cat = torch.cat([processed_a, processed_b, processed_a_diff], dim=0)

        out_all = model(orig_cat, proc_cat)
        z_a    = out_all['z'][:B]
        z_b    = out_all['z'][B:2*B]
        z_diff = out_all['z'][2*B:]
        fusion_a = out_all['fusion'][:B]

        loss, loss_dict = criterion(
            z_a, z_b, z_diff,
            fusion_a=fusion_a,
            params_shared=nn_param_shared,
            activate_shared=activate_shared,
            params_diff=nn_param_diff,
            activate_diff=activate_diff,
        )

        total_loss_sum += loss_dict['total_loss']
        n_batches += 1
        embeddings_a.append(z_a.cpu())
        embeddings_b.append(z_b.cpu())

    # Recall@K
    all_a = torch.cat(embeddings_a, dim=0)
    all_b = torch.cat(embeddings_b, dim=0)
    sim = all_a @ all_b.T
    ranking = torch.argsort(sim, descending=True, dim=1)
    gt = torch.arange(len(all_a)).unsqueeze(1)
    ranks = (ranking == gt).float().argmax(dim=1)

    metrics = {
        'val_loss': total_loss_sum / max(n_batches, 1),
        'R@1': (ranks < 1).float().mean().item(),
        'R@5': (ranks < 5).float().mean().item(),
        'R@10': (ranks < 10).float().mean().item(),
        'mean_rank': ranks.float().mean().item() + 1,
        'median_rank': ranks.float().median().item() + 1,
    }

    logging.info(
        f"Validation | Loss: {metrics['val_loss']:.4f} | "
        f"R@1: {metrics['R@1']:.4f} | R@5: {metrics['R@5']:.4f} | R@10: {metrics['R@10']:.4f} | "
        f"MeanRank: {metrics['mean_rank']:.1f} | MedianRank: {metrics['median_rank']:.1f}"
    )
    return metrics


@torch.no_grad()
def evaluate_ld_regression(model, criterion, fx_chain, device, triplets_dir, dataset="musdb18"):
    """
    用三元组数据做完整 Ld 评估：
      model(clean, reference) → fusion → param_head → params → fx_chain(clean, params)
      Ld = MR_STFT(output, target)

    Args:
        triplets_dir: 三元组数据根目录 (包含 {dataset}/{inst}/{sample_id}/)
        dataset: 数据集名 (musdb18 等)
    """
    import soundfile as sf
    import numpy as np
    import auraloss.freq
    from .fx_chain.constants import ALL_PROCESSORS

    model.eval()
    criterion.eval()

    # 获取回归头
    reg_head = criterion.module.param_regression_head if hasattr(criterion, 'module') else criterion.param_regression_head
    if reg_head is None:
        logging.info("  [Ld] 跳过: 没有 param_regression_head")
        return {}

    mrstft = auraloss.freq.MultiResolutionSTFTLoss().to(device)

    # reverb 关闭 (三元组数据不含 reverb)
    reverb_idx = ALL_PROCESSORS.index('reverb')
    activate_template = torch.ones(1, len(ALL_PROCESSORS), device=device)
    activate_template[0, reverb_idx] = 0.0

    # FX 参数索引: 72 维 → 47 维 (去 reverb)
    # 需要和 test_v6_regression.py 一致的映射
    base_dir = os.path.join(triplets_dir, dataset)
    if not os.path.exists(base_dir):
        logging.warning(f"  [Ld] 三元组目录不存在: {base_dir}")
        return {}

    instruments = ['drums', 'bass', 'vocals', 'other']
    all_results = {}
    overall_lds = []

    for inst in instruments:
        inst_dir = os.path.join(base_dir, inst)
        if not os.path.isdir(inst_dir):
            continue

        samples = sorted([d for d in os.listdir(inst_dir) if os.path.isdir(os.path.join(inst_dir, d))])
        lds = []
        baselines = []

        for sid in samples:
            d = os.path.join(inst_dir, sid)
            try:
                clean = sf.read(os.path.join(d, 'clean.wav'))[0].T       # (C, T)
                target = sf.read(os.path.join(d, 'target.wav'))[0].T
                ref = sf.read(os.path.join(d, 'reference.wav'))[0].T
            except Exception:
                continue

            clean_t = torch.from_numpy(clean).unsqueeze(0).float().to(device)
            target_t = torch.from_numpy(target).unsqueeze(0).float().to(device)
            ref_t = torch.from_numpy(ref).unsqueeze(0).float().to(device)

            # model(clean, reference) → fusion → param_head → params (72 维)
            out = model(clean_t, ref_t)
            fusion = out['fusion']  # (1, 2048)
            param_pred, _ = reg_head(fusion)  # (1, 72)

            # 过效果链 (训练用的 Random_FX_Chain 接收 72 维参数)
            # 三元组数据不含 reverb，所以 reverb activate=0
            pred_output, _, _ = fx_chain(
                clean_t.clone(), nn_param=param_pred,
                activate=activate_template,
                processors_order=ALL_PROCESSORS,
            )

            # clip
            peak = pred_output.abs().max()
            if peak > 1.0:
                pred_output = pred_output / peak * 0.99

            ld = mrstft(pred_output, target_t).item()
            bl = mrstft(clean_t, target_t).item()
            lds.append(ld)
            baselines.append(bl)

        if lds:
            ld_arr = np.array(lds)
            bl_arr = np.array(baselines)
            all_results[inst] = {
                'ld_mean': float(ld_arr.mean()),
                'ld_std': float(ld_arr.std()),
                'baseline': float(bl_arr.mean()),
                'n': len(lds),
            }
            overall_lds.extend(lds)

    # 打印汇总
    if all_results:
        parts = []
        for inst in instruments:
            if inst in all_results:
                r = all_results[inst]
                parts.append(f"{inst}={r['ld_mean']:.4f}")
        avg_ld = float(np.mean(overall_lds)) if overall_lds else 0.0
        all_results['average'] = avg_ld
        logging.info(f"  [Ld] {' | '.join(parts)} | AVG={avg_ld:.4f}")

    return all_results


def save_checkpoint(
    model, optimizer, criterion, epoch, metrics, save_path, switches,
    cross_segment=False, model_config=None, fx_probs=None,
    bidirectional_config=None, cross_segment_policy=None,
    sampling_config=None,
):
    """保存 checkpoint"""
    if not is_main_process():
        return

    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    criterion_state = criterion.module.state_dict() if hasattr(criterion, 'module') else criterion.state_dict()

    checkpoint_version = (
        'v8'
        if (model_config or {}).get('model_variant') == 'bidirectional'
        else BASE_CHECKPOINT_VERSION
    )
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'criterion_state_dict': criterion_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'switches': switches,
        'cross_segment': cross_segment,
        'cross_segment_policy': (
            cross_segment_policy if cross_segment else None
        ),
        'cross_segment_recipe_version': (
            CROSS_SEGMENT_RECIPE_VERSION if cross_segment else None
        ),
        'sampling_config': sampling_config,
        'model_config': model_config,
        'bidirectional_config': bidirectional_config,
        'fx_probs': fx_probs,
        'version': checkpoint_version,
    }, save_path)
    logging.info(f"Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the ISMIR 2026 RelFx model"
    )

    # 数据
    parser.add_argument(
        "--audio-dir",
        action="append",
        default=None,
        help="Audio root; repeat for multiple sources. Overrides RELFX_AUDIO_DIRS.",
    )
    parser.add_argument(
        "--structure-segments",
        default=STRUCTURE_SEGMENT_JSON,
        help="JSON manifest defining ranges for adjacent-pair sampling.",
    )
    parser.add_argument(
        "--density-filter-audio-dir",
        action="append",
        default=None,
        help=(
            "Audio root requiring the 70%% non-silent filter; repeat for "
            "multiple roots. Overrides RELFX_DENSITY_FILTER_AUDIO_DIRS."
        ),
    )
    # 训练
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=TRAIN_CONFIG["weight_decay"])
    parser.add_argument("--num-workers", type=int, default=TRAIN_CONFIG["num_workers"])

    # 对比学习
    parser.add_argument("--temperature", type=float, default=TRAIN_CONFIG["temperature"])
    parser.add_argument("--proj-dim", type=int, default=TRAIN_CONFIG["proj_dim"])

    # 训练优化
    parser.add_argument("--grad-accum-steps", type=int,
                        default=TRAIN_CONFIG.get("grad_accum_steps", 1))
    parser.add_argument("--warmup-epochs", type=int,
                        default=TRAIN_CONFIG.get("warmup_epochs", 0))
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit each epoch to N training batches for release validation.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit validation to N batches for release validation.",
    )

    # 设备
    parser.add_argument("--device", type=str, default=TRAIN_CONFIG["device"])
    parser.add_argument("--amp", action="store_true", help="混合精度训练")

    # 保存
    parser.add_argument("--save-every", type=int, default=TRAIN_CONFIG["save_every"])
    parser.add_argument("--val-every", type=int, default=TRAIN_CONFIG["val_every"])
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing checkpoints or training-history files.",
    )

    # 恢复训练
    parser.add_argument("--resume", type=str, default=None)

    # ======== V4: 模型配置 ========
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default=MODEL_CONFIG.get("model_variant", "base"),
        help=(
            "Paper model variant. 'bidirectional' enables antisymmetric "
            "Diff-Gate, Tanh projection, and paired input-order swaps."
        ),
    )
    parser.add_argument(
        "--fusion-type", type=str, default=MODEL_CONFIG["fusion_type"],
        choices=["diff", "concat_mlp", "gate", "diff_gate"],
        help="融合策略",
    )
    parser.add_argument(
        "--bidirectional-flip-ratio",
        type=float,
        default=BIDIRECTIONAL_CONFIG.get("flip_ratio", 0.5),
        help=(
            "Fraction of each batch whose ordered pairs are swapped when "
            "--model-variant bidirectional is selected."
        ),
    )
    parser.add_argument(
        "--cross-attn-stages", nargs="*", type=int,
        default=MODEL_CONFIG["cross_attn_stages"],
        help="交叉注意力插入位置 (1~6)",
    )
    parser.add_argument(
        "--no-cross-attn", action="store_true",
        help="禁用交叉注意力 (纯 Siamese 双分支)",
    )

    # ======== V3 兼容: 开关控制 ========
    parser.add_argument(
        "--enable", nargs="*", default=None,
        help="启用的改进方法: param_regression soft_contrastive distance_margin hard_negative",
    )
    parser.add_argument(
        "--disable", nargs="*", default=None,
        help="禁用的改进方法",
    )
    parser.add_argument(
        "--cross-segment", action="store_true", default=None,
        help="启用 cross_segment 模式 (V4 默认开启)",
    )
    parser.add_argument(
        "--no-cross-segment", action="store_true", default=None,
        help="禁用 cross_segment 模式",
    )

    args = parser.parse_args()
    for name in ("max_train_batches", "max_val_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if not 0.0 <= args.bidirectional_flip_ratio <= 1.0:
        parser.error("--bidirectional-flip-ratio must be between 0 and 1")
    if args.model_variant == "bidirectional" and args.fusion_type != "diff_gate":
        parser.error("--model-variant bidirectional requires --fusion-type diff_gate")
    audio_dirs = args.audio_dir or cfg.AUDIO_DIRS
    density_filter_audio_dirs = (
        set(args.density_filter_audio_dir)
        if args.density_filter_audio_dir is not None
        else DENSITY_FILTER_AUDIO_DIRS
    )
    if not audio_dirs:
        parser.error(
            "provide at least one --audio-dir or set RELFX_AUDIO_DIRS"
        )

    # 合并开关
    switches = LOSS_SWITCHES.copy()
    if args.enable is not None:
        for method in args.enable:
            if method in switches:
                switches[method] = True
            else:
                print(f"[Warning] Unknown method: {method}")
    if args.disable is not None:
        for method in args.disable:
            if method in switches:
                switches[method] = False

    # cross_segment
    cross_segment = CROSS_SEGMENT
    if args.cross_segment:
        cross_segment = True
    elif args.no_cross_segment:
        cross_segment = False

    # 模型配置
    model_config = MODEL_CONFIG.copy()
    model_config["model_variant"] = args.model_variant
    model_config["fusion_type"] = args.fusion_type
    if args.no_cross_attn:
        model_config["cross_attn_stages"] = []
    else:
        model_config["cross_attn_stages"] = args.cross_attn_stages

    bidirectional_config = BIDIRECTIONAL_CONFIG.copy()
    bidirectional_config["enabled"] = args.model_variant == "bidirectional"
    bidirectional_config["flip_ratio"] = args.bidirectional_flip_ratio

    # ===================== DDP 初始化 =====================
    setup_ddp()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if is_dist():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ===================== 初始化 =====================
    ensure_dirs()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    setup_logging(LOG_DIR, rank=rank)

    effective_batch = args.batch_size * world_size * args.grad_accum_steps

    logging.info("=" * 70)
    logging.info("Cascaded FX Contrastive Learning V6-fix — Dual-Branch + DynFxProb")
    logging.info("=" * 70)
    logging.info(f"Audio dirs: {audio_dirs}")
    logging.info(f"World size: {world_size} GPUs, Device: {device}")
    logging.info(f"Epochs: {args.epochs}, Batch/GPU: {args.batch_size}")
    logging.info(f"Grad accum: {args.grad_accum_steps}, Effective batch: {effective_batch}")
    logging.info(f"LR: {args.lr}, Temperature: {args.temperature}")
    logging.info(f"Warmup epochs: {args.warmup_epochs}")
    logging.info(f"Cross-segment: {'✅ ON' if cross_segment else '❌ OFF'}")
    if cross_segment:
        logging.info(f"Cross-segment policy: {CROSS_SEGMENT_POLICY}")
    logging.info("Model: DualBranchFxEncoder")
    logging.info(f"  Variant: {model_config['model_variant']}")
    logging.info(f"  Fusion: {model_config['fusion_type']}")
    logging.info(f"  CrossAttn stages: {model_config['cross_attn_stages']}")

    # V6: 动态 FX 概率调度配置
    dyn_fx_config = DYNAMIC_FX_PROB_CONFIG
    dyn_fx_enabled = dyn_fx_config.get("enabled", False)
    logging.info(f"Dynamic FX Prob: {'✅ ON' if dyn_fx_enabled else '❌ OFF'}")
    if dyn_fx_enabled:
        logging.info(f"  Update every: {dyn_fx_config['update_every_epochs']} epochs")
        logging.info(f"  Prob range: [{dyn_fx_config['min_prob']}, {dyn_fx_config['max_prob']}]")
        logging.info(f"  Smoothing alpha: {dyn_fx_config['smoothing_alpha']}")
        logging.info(f"  Warmup: {dyn_fx_config['warmup_epochs']} epochs")

    # V6: 双向训练配置
    bidir_enabled = bidirectional_config.get("enabled", False)
    logging.info(f"Bidirectional: {'✅ ON' if bidir_enabled else '❌ OFF'}")
    if bidir_enabled:
        logging.info(f"  Flip ratio: {bidirectional_config['flip_ratio']}")

    # 打印开关状态
    logging.info("\n" + "=" * 70)
    logging.info("IMPROVEMENT SWITCHES:")
    logging.info("=" * 70)
    for method, enabled in switches.items():
        status = "✅ ON" if enabled else "❌ OFF"
        logging.info(f"  {method:25s} : {status}")
    logging.info(f"  {'dynamic_fx_prob':25s} : {'✅ ON' if dyn_fx_enabled else '❌ OFF'}")
    logging.info(f"  Active combination: {cfg.get_active_methods()}")
    logging.info("=" * 70)

    # ===================== 效果链 =====================
    logging.info("\nInitializing FX chain...")
    fx_chain = Random_FX_Chain(sample_rate=SAMPLE_RATE, device=device)
    fx_param_ranges = get_fx_param_ranges(fx_chain)
    logging.info(f"FX chain: {fx_chain.total_num_param} total params")
    for fx_name in FX_CHAIN_ORDER:
        if fx_name in fx_chain.fx_processors:
            n = fx_chain.fx_processors[fx_name].num_params
            logging.info(f"  {fx_name}: {n} params")

    # ===================== 数据 =====================
    logging.info("\nCreating dataloader...")

    train_dataset = AudioSegmentDataset(
        audio_dirs=audio_dirs,
        cross_segment=cross_segment,
        cross_segment_policy=CROSS_SEGMENT_POLICY,
        structure_segment_json=args.structure_segments,
        density_filter_audio_dirs=density_filter_audio_dirs,
        split="train",
    )
    sampling_config = train_dataset.sampling_metadata()
    logging.info(
        "Sampling provenance: "
        f"{json.dumps(sampling_config, sort_keys=True)}"
    )

    hn_enabled = switches.get("hard_negative", False)
    hn_config = cfg.HARD_NEGATIVE_CONFIG if hn_enabled else None

    train_collator = FxContrastiveCollator(
        fx_chain=fx_chain,
        shuffle_order=True,
        hard_negative_enabled=hn_enabled,
        hard_negative_config=hn_config,
        cross_segment=cross_segment,
        fx_sample_mode=TRAIN_CONFIG.get("fx_sample_mode", "uniform"),
        conservative_ratio=TRAIN_CONFIG.get("conservative_ratio", 0.4),
        conservative_range=TRAIN_CONFIG.get("conservative_range", (0.3, 0.7)),
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
    ) if is_dist() else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collator,
    )

    # 验证集 — 所有 rank 都创建，避免评估时 NCCL 超时
    val_loader = None
    val_dataset = AudioSegmentDataset(
        audio_dirs=audio_dirs,
        cross_segment=cross_segment,
        cross_segment_policy=CROSS_SEGMENT_POLICY,
        structure_segment_json=args.structure_segments,
        density_filter_audio_dirs=density_filter_audio_dirs,
        split="val",
    )
    val_collator = FxContrastiveCollator(
        fx_chain=fx_chain, shuffle_order=False,
        hard_negative_enabled=False,
        cross_segment=cross_segment,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=val_collator,
    )

    # ===================== V4 模型 =====================
    logging.info("\nCreating model (DualBranchFxEncoder)...")
    model = create_model(
        sample_rate=SAMPLE_RATE,
        embed_dim=TRAIN_CONFIG["embed_dim"],
        proj_dim=args.proj_dim,
        fusion_type=model_config["fusion_type"],
        model_variant=model_config["model_variant"],
        cross_attn_stages=model_config["cross_attn_stages"],
        cross_attn_heads=model_config.get("cross_attn_heads", 4),
        cross_attn_pool=model_config.get("cross_attn_pool", 4),
        cross_attn_scale=model_config.get("cross_attn_scale", 0.1),
    )
    model = model.to(device)

    # ===================== 损失函数 (复用 V3 CombinedLoss) =====================
    logging.info("\nCreating CombinedLoss...")
    criterion = CombinedLoss(
        switches=switches,
        temperature=args.temperature,
        use_baseline_triplet=TRAIN_CONFIG.get("use_triplet", True),
        baseline_triplet_margin=TRAIN_CONFIG.get("triplet_margin", 0.3),
        baseline_triplet_weight=TRAIN_CONFIG.get("triplet_weight", 0.5),
        param_regression_config=cfg.PARAM_REGRESSION_CONFIG,
        soft_contrastive_config=cfg.SOFT_CONTRASTIVE_CONFIG,
        distance_margin_config=cfg.DISTANCE_MARGIN_CONFIG,
        total_fx_params=fx_chain.total_num_param,
        num_fx=len(fx_chain.fx_indices),
        embed_dim=TRAIN_CONFIG["embed_dim"],
        fx_param_ranges=fx_param_ranges,
    ).to(device)

    criterion_params = sum(p.numel() for p in criterion.parameters() if p.requires_grad)
    logging.info(f"CombinedLoss trainable params: {criterion_params:,}")

    # DDP
    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if criterion_params > 0:
            criterion = DDP(criterion, device_ids=[local_rank], output_device=local_rank)
        logging.info(f"Model + Criterion wrapped with DDP on cuda:{local_rank}")

    # ===================== 优化器 =====================
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(
        all_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6,
    )

    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0/20, end_factor=1.0,
            total_iters=args.warmup_epochs,
        )

    scaler = GradScaler() if args.amp else None

    # 恢复训练
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)

            if cross_segment:
                saved_policy = checkpoint.get('cross_segment_policy')
                if saved_policy != CROSS_SEGMENT_POLICY:
                    raise RuntimeError(
                        "Refusing to resume a checkpoint with sampling policy "
                        f"{saved_policy!r}; expected {CROSS_SEGMENT_POLICY!r}. "
                        "Start paper-aligned training from scratch."
                    )

            model_state = checkpoint['model_state_dict']
            if is_dist():
                model.module.load_state_dict(model_state)
            else:
                model.load_state_dict(model_state)

            if 'criterion_state_dict' in checkpoint:
                criterion_module = criterion.module if hasattr(criterion, 'module') else criterion
                try:
                    criterion_module.load_state_dict(checkpoint['criterion_state_dict'])
                    logging.info("Criterion state loaded")
                except Exception as e:
                    logging.warning(f"Could not load criterion state: {e}")

            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    logging.warning(f"Could not load optimizer state: {e}")

            start_epoch = checkpoint.get('epoch', -1) + 1
            logging.info(f"Resumed from epoch {start_epoch}")

            if 'switches' in checkpoint:
                prev = checkpoint['switches']
                logging.info(f"Previous switches: {prev}")
                if prev != switches:
                    logging.warning("⚠️  Switches changed since checkpoint!")
        else:
            logging.warning(f"Checkpoint not found: {args.resume}")

    # ===================== 训练循环 =====================
    logging.info("\n" + "=" * 70)
    logging.info("Starting V6-fix training...")
    logging.info("=" * 70)

    best_val_loss = float('inf')
    history = []

    # V6: 初始化动态 FX 概率跟踪
    current_fx_probs = FX_PROB.copy()
    fx_prob_history = []  # 记录概率变化历史

    criterion_module = criterion.module if hasattr(criterion, 'module') else criterion

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if hn_enabled:
            train_collator.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, device, fx_chain, use_amp=args.amp,
            grad_accum_steps=args.grad_accum_steps,
            cross_segment=cross_segment,
            max_batches=args.max_train_batches,
            bidirectional_config=bidirectional_config,
        )

        # 学习率调度
        if warmup_scheduler is not None and epoch < args.warmup_epochs:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"  [Warmup] LR = {current_lr:.6f}")
        else:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"  [Cosine] LR = {current_lr:.6f}")

        history.append(train_metrics)

        # 验证 — 所有 rank 都参与，避免 NCCL 超时
        if val_loader is not None:
            if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
                val_model = model.module if hasattr(model, 'module') else model
                val_criterion = criterion_module
                val_metrics = validate(
                    val_model, val_loader, val_criterion, device, fx_chain,
                    cross_segment=cross_segment,
                    max_batches=args.max_val_batches,
                )
                train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                is_best = val_metrics['val_loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                if not args.no_save and is_main_process() and is_best:
                    save_checkpoint(
                        model, optimizer, criterion, epoch, val_metrics,
                        os.path.join(args.checkpoint_dir, "best.pt"),
                        switches, cross_segment=cross_segment,
                        model_config=model_config,
                        fx_probs=current_fx_probs,
                        bidirectional_config=bidirectional_config,
                        cross_segment_policy=CROSS_SEGMENT_POLICY,
                        sampling_config=sampling_config,
                    )

        # ===================== Ld 回归评估 (三元组数据) =====================
        ld_eval_cfg = getattr(cfg, 'LD_EVAL_CONFIG', {})
        if (ld_eval_cfg.get('enabled', False)
            and is_main_process()
            and (epoch + 1) % ld_eval_cfg.get('every_epochs', 10) == 0):

            triplets_path = os.path.join(
                str(cfg.REPOSITORY_ROOT), ld_eval_cfg.get("triplets_dir", "")
            )
            ld_dataset = ld_eval_cfg.get('dataset', 'musdb18')

            val_model = model.module if hasattr(model, 'module') else model
            val_criterion = criterion_module

            logging.info(f"\n  [Ld] Evaluating regression Ld at epoch {epoch+1}...")
            evaluate_ld_regression(
                val_model, val_criterion, fx_chain, device,
                triplets_dir=triplets_path, dataset=ld_dataset,
            )

        # ===================== V6: 动态 FX 概率调度 =====================
        # 所有 rank 都参与评估，避免 NCCL 超时
        if dyn_fx_enabled and val_loader is not None:
            update_every = dyn_fx_config.get("update_every_epochs", 5)
            prob_warmup = dyn_fx_config.get("warmup_epochs", 10)
            if (epoch + 1) % update_every == 0 and epoch >= prob_warmup:
                logging.info(f"\n  [DynFxProb] Evaluating per-FX performance at epoch {epoch}...")

                val_model = model.module if hasattr(model, 'module') else model
                per_fx_losses = evaluate_per_fx(
                    val_model, val_loader, fx_chain, device,
                    cross_segment=cross_segment,
                    max_batches=max(dyn_fx_config.get("eval_samples_per_fx", 64) // max(args.batch_size, 1), 2),
                )

                if is_main_process():
                    train_metrics['per_fx_losses'] = per_fx_losses

                current_fx_probs = update_fx_probabilities(
                    fx_chain, train_collator, per_fx_losses,
                    current_fx_probs, dyn_fx_config, epoch,
                )

                # 同步更新 collator 中的 FX 概率
                train_collator.update_fx_prob(current_fx_probs)

                fx_prob_history.append({
                    'epoch': epoch,
                    'probs': current_fx_probs.copy(),
                    'losses': per_fx_losses.copy(),
                })

        # 广播更新后的概率到所有进程
        if dyn_fx_enabled and is_dist():
            dist.barrier()

        # 定期保存
        if not args.no_save and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, criterion, epoch, train_metrics,
                os.path.join(args.checkpoint_dir, f"epoch_{epoch:04d}.pt"),
                switches, cross_segment=cross_segment,
                model_config=model_config,
                fx_probs=current_fx_probs,
                bidirectional_config=bidirectional_config,
                cross_segment_policy=CROSS_SEGMENT_POLICY,
                sampling_config=sampling_config,
            )

        if is_dist():
            dist.barrier()

    # 保存最终模型
    if not args.no_save:
        save_checkpoint(
            model, optimizer, criterion, args.epochs - 1, train_metrics,
            os.path.join(args.checkpoint_dir, "final.pt"),
            switches, cross_segment=cross_segment,
            model_config=model_config,
            fx_probs=current_fx_probs,
            bidirectional_config=bidirectional_config,
            cross_segment_policy=CROSS_SEGMENT_POLICY,
            sampling_config=sampling_config,
        )

    if is_main_process():
        history_path = None
        if not args.no_save:
            history_path = os.path.join(LOG_DIR, "training_history.json")
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

        logging.info("\n" + "=" * 70)
        logging.info("V6-fix training completed!")
        logging.info(f"Best val loss: {best_val_loss:.4f}")
        logging.info(f"Model: DualBranchFxEncoder (fusion={model_config['fusion_type']})")
        logging.info(f"Active methods: {cfg.get_active_methods()}")
        logging.info(f"Dynamic FX Prob: {'✅ ON' if dyn_fx_enabled else '❌ OFF'}")
        if dyn_fx_enabled and fx_prob_history:
            logging.info(f"Final FX probs: {current_fx_probs}")
        if args.no_save:
            logging.info("Checkpoint and history output disabled")
        else:
            logging.info(f"Checkpoints: {args.checkpoint_dir}")
            logging.info(f"History: {history_path}")
        logging.info("=" * 70)

        # V6: 保存概率变化历史
        if not args.no_save and dyn_fx_enabled and fx_prob_history:
            prob_history_path = os.path.join(LOG_DIR, "fx_prob_history.json")
            with open(prob_history_path, "w") as f:
                json.dump(fx_prob_history, f, indent=2)
            logging.info(f"FX prob history saved: {prob_history_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
