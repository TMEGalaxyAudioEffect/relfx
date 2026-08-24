"""
08 Parameter Matching — 参数匹配优化器

实现 FX-Encoder++ 论文 Section 4.2 的参数匹配流程 (可微分梯度优化)。

论文原理:
  使用 dasp-pytorch + torchcomp 构建可微分效果链，
  通过 embedding cosine distance 作为梯度引导信号，
  用 Adam 优化器迭代更新效果链参数。

评估流程:
  给定 (clean, reference, target, clean_ref) 四元组:

  1. 提取 reference 的 embedding (风格模板，不参与梯度计算)
  2. 随机初始化效果链参数 → sigmoid → [0,1]
  3. 迭代优化 (梯度反向传播):
     - forward: fx_chain(clean, params) → output
     - loss: cos_dist(embed(output), embed(reference))
     - backward: loss.backward() → 更新 params
  4. 收敛后评估: Ld = STFT_loss(fx_chain(clean, best_params), target)

  → 不同嵌入模型质量不同 → 引导到不同参数 → 导致不同 Ld

特殊模式:
  "stft" loss — 直接优化 STFT loss (oracle baseline，直接看 target):
    min_params STFT_loss(fx_chain(clean, params), target)
"""

import torch
import torch.nn.functional as F
import numpy as np
import auraloss

from .config import ITO_CONFIG, SAMPLE_RATE


class ParameterMatcher:
    """
    参数匹配优化器。

    使用 Inference-Time Optimization (ITO) 来优化效果链参数:
      - embedding 模式: 用 embedding cosine distance 引导，STFT loss 评估
      - stft 模式: 直接用 STFT loss 优化 (oracle baseline)

    支持 batch 并行: 多个 sample × 多个 restart 同时在 GPU 上优化。
    """

    def __init__(
        self,
        fx_chain,
        sample_rate: int = SAMPLE_RATE,
        loss_type: str = "embedding",
        lr: float = 0.01,
        n_iters: int = 200,
        num_restarts: int = 3,
        sample_batch_size: int = 1,
        es_patience: int = 50,
        device: str = "cpu",
        embed_mode: str = "wet_wet",
    ):
        """
        Args:
            embed_mode: 双分支模型 (V4/V6) 的评估方案:
                "wet_wet" (方案A, 默认):
                    ref_embed  = E(ref, ref)        — identity embedding
                    out_embed  = E(ref, output)     — ref→output 变换
                    目标: 让 output 趋近 ref (output→ref 变换为零)
                "dry_wet" (方案B):
                    ref_embed  = E(clean, ref)      — clean→ref 变换
                    out_embed  = E(clean, output)   — clean→output 变换
                    目标: 让 clean→output 变换匹配 clean→ref 变换
                "cross_dry_wet" (方案C):
                    ref_embed  = E(clean_ref, ref)  — B→refB 变换 (完整参考)
                    out_embed  = E(clean, output)   — A→learnedA 变换
                    目标: 让 A→learnedA 变换匹配 B→refB 变换
                    需要 clean_ref (seg_B 干声) 数据
                对单分支模型 (AFx-Rep, CLAP, FxEnc++) 无影响。
        """
        self.fx_chain = fx_chain
        self.sample_rate = sample_rate
        self.loss_type = loss_type
        self.lr = lr
        self.n_iters = n_iters
        self.num_restarts = num_restarts
        self.sample_batch_size = sample_batch_size  # 同时优化几个 sample
        self.es_patience = es_patience  # early stopping patience (0=禁用)
        self.device = device
        self.embed_mode = embed_mode
        self.total_params = fx_chain.get_total_params()

        # Multi-resolution STFT loss (论文用的 Ld 评估)
        self.mrstft_loss = auraloss.freq.MultiResolutionSTFTLoss().to(device)

    @staticmethod
    def _safe_clip(audio: torch.Tensor) -> torch.Tensor:
        """
        防止 clipping，与 prepare_data.apply_fx 对齐:
            if max > 1.0: audio = audio / max * 0.99
            else: audio 不变
        可微分实现: audio * min(1.0, 0.99 / max)
        当 max <= 0.99 时 scale=1.0 (不变); max > 1.0 时 scale=0.99/max (缩放)
        0.99 < max <= 1.0 时轻微缩放 (和原始行为一致: 不触发)
        """
        max_val = audio.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        scale = (0.99 / max_val).clamp(max=1.0)
        return audio * scale

    def _optimize_batch(
        self,
        cleans: list,
        targets: list,
        references: list = None,
        model=None,
        embed_func=None,
        clean_refs: list = None,
        sample_indices: list = None,
        verbose: bool = False,
        log_prefix: str = "",
        disabled_fx: list = None,
    ) -> list:
        """
        Batch 并行参数优化: S 个 sample × R 个 restart 同时跑。

        总 batch 维度 = S * R，一次前向/反向覆盖所有。

        Args:
            cleans: list of (1, channels, samples) tensors — 每个 sample 的干声
            targets: list of (1, channels, samples) tensors — 每个 sample 的目标音频
            references: list of (1, channels, samples) tensors — 参考音频 (embedding 用)
            model: 嵌入模型
            embed_func: 嵌入函数
            clean_refs: list of (1, channels, samples) tensors — reference 的干声 (V2 用)
            sample_indices: 各 sample 的编号 (日志用)
            verbose: 是否打印详细优化日志
            disabled_fx: 被禁用的效果器名称列表

        Returns:
            list of (best_ld, best_params, best_output) — 每个 sample 一个结果
        """
        S = len(cleans)  # sample 数
        R = self.num_restarts  # restart 数
        B = S * R  # 总 batch 维度

        if sample_indices is None:
            sample_indices = list(range(S))

        # === 获取被禁用效果器的参数索引 ===
        frozen_indices = []
        if disabled_fx:
            frozen_indices = self.fx_chain.get_fx_param_indices(disabled_fx)

        # === 对齐音频长度 (pad to max length) ===
        max_len = max(c.shape[-1] for c in cleans)

        def pad_to(t, length):
            """pad (1, C, T) → (1, C, length)"""
            t = t.to(self.device)
            if t.shape[-1] < length:
                pad_size = length - t.shape[-1]
                t = torch.nn.functional.pad(t, (0, pad_size))
            return t

        # 构建 (S, C, T) 然后 repeat_interleave R 次 → (B, C, T)
        cleans_padded = torch.cat([pad_to(c, max_len) for c in cleans], dim=0)  # (S, C, T)
        targets_padded = torch.cat([pad_to(t, max_len) for t in targets], dim=0)  # (S, C, T)

        # 每个 sample 复制 R 次: (S, C, T) → (S*R, C, T)
        clean_batch = cleans_padded.repeat_interleave(R, dim=0)  # (B, C, T)
        target_batch = targets_padded.repeat_interleave(R, dim=0)  # (B, C, T)

        # sample_map[b] = sample 编号 (0..S-1)
        sample_map = torch.arange(S).repeat_interleave(R)  # (B,)

        pfx = log_prefix if log_prefix else ""

        if verbose:
            for s_idx, si in enumerate(sample_indices):
                with torch.no_grad():
                    baseline_ld = self.mrstft_loss(
                        cleans_padded[s_idx:s_idx+1], targets_padded[s_idx:s_idx+1]
                    ).item()
                print(f"  {pfx}[{si}] Baseline Ld: {baseline_ld:.4f}")
            if frozen_indices:
                n_active = self.total_params - len(frozen_indices)
                print(f"  {pfx} disabled_fx={disabled_fx} → frozen {len(frozen_indices)}, active {n_active}/{self.total_params}")
            print(f"  {pfx} Batch parallel: {S} samples × {R} restarts = {B} total")

        # === 预计算 reference embeddings (不参与梯度) ===
        ref_embed_cached = None
        ref_batch = None
        clean_batch_for_embed = None  # 用于 dry_wet / cross_dry_wet 模式
        if self.loss_type == "embedding" and model is not None and references is not None:
            refs_padded = torch.cat([pad_to(r, max_len) for r in references], dim=0)  # (S, C, T)
            ref_batch_full = refs_padded.repeat_interleave(R, dim=0)  # (B, C, T)
            ref_batch = ref_batch_full  # 保存用于双分支模型

            # clean_refs 预处理 (方案C需要)
            clean_refs_padded = None
            if clean_refs is not None:
                clean_refs_padded = torch.cat([pad_to(cr, max_len) for cr in clean_refs], dim=0)  # (S, C, T)

            model_type = self._detect_model_type(model)
            with torch.no_grad():
                if model_type in ("cascaded_v2", "cascaded_v4"):
                    if self.embed_mode == "cross_dry_wet":
                        # 方案C: ref_embed = E(clean_ref, ref) — B→refB 变换
                        if clean_refs_padded is None:
                            raise ValueError("cross_dry_wet 模式需要 clean_ref 数据")
                        ref_embed_cached = self._compute_embed(
                            refs_padded.to(self.device), model, embed_func,
                            original=clean_refs_padded.to(self.device)
                        )  # (S, embed_dim)
                        # out_embed 用 clean 作为 original
                        clean_batch_for_embed = clean_batch.clone()
                    elif self.embed_mode == "dry_wet":
                        # 方案B: ref_embed = E(clean, ref) — clean→ref 变换
                        ref_embed_cached = self._compute_embed(
                            refs_padded.to(self.device), model, embed_func,
                            original=cleans_padded.to(self.device)
                        )  # (S, embed_dim)
                        # 保存 clean_batch 用于后续 output embed 计算
                        clean_batch_for_embed = clean_batch.clone()
                    else:
                        # 方案A (默认): ref_embed = E(ref, ref) — identity embedding
                        ref_embed_cached = self._compute_embed(
                            refs_padded.to(self.device), model, embed_func,
                            original=refs_padded.to(self.device)
                        )  # (S, embed_dim)
                else:
                    ref_embed_cached = self._compute_embed(
                        refs_padded.to(self.device), model, embed_func
                    )  # (S, embed_dim)
                # expand 到 (B, embed_dim)
                ref_embed_cached = ref_embed_cached.repeat_interleave(R, dim=0)

        # === 初始化参数: 均匀分布 [-2, 2]，sigmoid 后分散在 [0.12, 0.88] ===
        w = (torch.rand(B, self.total_params, device=self.device) * 4.0 - 2.0)
        w.requires_grad_(True)

        optimizer = torch.optim.Adam([w], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_iters, eta_min=self.lr * 0.01
        )

        # 每个 batch 元素各自跟踪最优 optimization loss
        best_opt_losses = torch.full((B,), float("inf"), device=self.device)
        best_ws = w.data.clone()  # (B, total_params)

        # Early stopping: 改为 per-sample 判据 —— 每个 sample 独立维护
        # best-so-far 与 stagnation counter，只有当【所有 S 个 sample】都连续
        # patience 步不再下降时才停。这样可避免 batch 里最容易收敛的 sample
        # 提前锁死 es，导致同批其他难 sample 的优化尚未启动就被一起终止。
        use_es = self.es_patience > 0
        es_best_per_sample = [float("inf")] * S   # 每个 sample 的历史最优 (取该 sample 下 R 个 restart 的 min)
        es_counter_per_sample = [0] * S
        es_best_global = float("inf")             # 仅用于进度条显示/调试

        from tqdm import tqdm
        pbar = tqdm(range(self.n_iters), desc=f"  {pfx}Batch({S}×{R})", leave=False) if verbose else range(self.n_iters)

        for step in pbar:
            optimizer.zero_grad()

            # sigmoid 映射
            params = torch.sigmoid(w)  # (B, total_params)

            # 冻结被禁用效果器参数
            if frozen_indices:
                params_frozen = params.clone()
                params_frozen[:, frozen_indices] = params[:, frozen_indices].detach() * 0.0
                params = params_frozen

            # 效果链处理
            output = self.fx_chain(clean_batch.clone(), params, disabled_fx=disabled_fx)  # (B, C, T)
            output = self._safe_clip(output)

            if self.loss_type == "stft":
                # Oracle: 直接 STFT loss vs target (逐 sample 计算)
                per_item_loss = torch.stack([
                    self.mrstft_loss(output[b:b+1], target_batch[b:b+1])
                    for b in range(B)
                ])  # (B,)
                loss = per_item_loss.mean()

            elif self.loss_type == "embedding":
                if model is None:
                    raise ValueError("embedding loss requires model")

                model_type = self._detect_model_type(model)
                if model_type in ("cascaded_v2", "cascaded_v4"):
                    if self.embed_mode in ("dry_wet", "cross_dry_wet"):
                        # 方案B/C: out_embed = E(clean, output) — clean→output 变换
                        output_embed = self._compute_embed(
                            output, model, embed_func,
                            original=clean_batch_for_embed.to(self.device)
                        )
                    else:
                        # 方案A: out_embed = E(ref, output) — ref→output 变换
                        output_embed = self._compute_embed(
                            ref_batch.to(self.device), model, embed_func,
                            original=output
                        )
                else:
                    output_embed = self._compute_embed(output, model, embed_func)

                per_item_sim = F.cosine_similarity(output_embed, ref_embed_cached, dim=-1)  # (B,)
                per_item_loss = 1.0 - per_item_sim
                loss = per_item_loss.mean()
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            if torch.isnan(loss):
                if verbose:
                    print(f"\n  {pfx} NaN at step {step+1}, stopping")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # 更新每个 batch 元素的最优
            with torch.no_grad():
                improved = per_item_loss < best_opt_losses
                best_opt_losses = torch.where(improved, per_item_loss, best_opt_losses)
                if improved.any():
                    best_ws[improved] = w.data[improved].clone()

            # Early stopping (per-sample): 每个 sample 独立维护 best 与 counter，
            # 只有当全部 S 个 sample 都连续 patience 步不降时才终止。
            if use_es:
                with torch.no_grad():
                    # 每个 sample 的当前 best = 该 sample 下 R 个 restart 的 min
                    per_sample_best = best_opt_losses.view(S, R).min(dim=1).values  # (S,)
                    per_sample_best_list = per_sample_best.tolist()

                all_stagnant = True
                for s in range(S):
                    cur = per_sample_best_list[s]
                    if cur < es_best_per_sample[s] - 1e-6:
                        es_best_per_sample[s] = cur
                        es_counter_per_sample[s] = 0
                    else:
                        es_counter_per_sample[s] += 1
                    if es_counter_per_sample[s] < self.es_patience:
                        all_stagnant = False

                # 仅用于显示: 全局最优的最小 counter 值
                es_best_global = min(es_best_per_sample)

                if all_stagnant:
                    if verbose:
                        max_counter = max(es_counter_per_sample)
                        print(f"\n  {pfx} Early stopping at step {step+1} "
                              f"(all {S} samples stagnant for ≥{self.es_patience} steps, "
                              f"max counter={max_counter})")
                    break

            # 日志: 进度条展示 loss / sim / lr / es
            if verbose and hasattr(pbar, 'set_postfix'):
                cur_loss_val = loss.item()
                cur_lr = scheduler.get_last_lr()[0]

                # 当前 sim (本步所有 restart 中最大)
                if self.loss_type == "embedding":
                    cur_sim_max = per_item_sim.max().item()
                    cur_sim_str = f"{cur_sim_max:.4f}"
                else:
                    cur_sim_str = "n/a"

                # 每个 sample 的历史最优 sim
                sample_bests = []
                for s in range(S):
                    s_losses = best_opt_losses[s*R:(s+1)*R]
                    sample_bests.append(1.0 - s_losses.min().item())
                best_str = "/".join(f"{v:.4f}" for v in sample_bests)

                postfix = {
                    "loss": f"{cur_loss_val:.4f}",
                    "sim": cur_sim_str,
                    "best": best_str,
                    "lr": f"{cur_lr:.1e}",
                }
                if use_es:
                    es_cmin = min(es_counter_per_sample)
                    es_cmax = max(es_counter_per_sample)
                    postfix["es"] = f"{es_cmin}~{es_cmax}/{self.es_patience}"
                pbar.set_postfix(postfix)

            if verbose and not hasattr(pbar, 'set_postfix') and ((step + 1) % 50 == 0 or step == 0):
                cur_lr = scheduler.get_last_lr()[0]
                cur_loss_val = loss.item()
                for s in range(S):
                    s_losses = best_opt_losses[s*R:(s+1)*R]
                    best_sim = 1.0 - s_losses.min().item()
                    si = sample_indices[s]
                    print(f"  {pfx}[{si}] step={step+1:3d} loss={cur_loss_val:.4f} best_sim={best_sim:.4f} lr={cur_lr:.6f}")

        # === 评估: 用 STFT loss 从每个 sample 的 R 个 restart 中选最优 ===
        results = []
        with torch.no_grad():
            final_params_all = torch.sigmoid(best_ws)  # (B, total_params)
            if frozen_indices:
                final_params_all[:, frozen_indices] = 0.0

            # 逐 sample 评估 Ld (避免一次性 forward 太大的 batch)
            for s in range(S):
                si = sample_indices[s]
                s_start = s * R
                s_end = s_start + R
                s_params = final_params_all[s_start:s_end]  # (R, total_params)
                s_clean = cleans_padded[s:s+1].expand(R, -1, -1)  # (R, C, T)
                s_target = targets_padded[s:s+1].expand(R, -1, -1)  # (R, C, T)

                # forward R 个 restart
                s_output = self.fx_chain(s_clean.clone(), s_params, disabled_fx=disabled_fx)
                s_output = self._safe_clip(s_output)

                # 逐 restart 计算 Ld
                s_lds = []
                for r in range(R):
                    ld = self.mrstft_loss(s_output[r:r+1], s_target[r:r+1]).item()
                    s_lds.append(ld)

                # 选 Ld 最低的 restart
                best_r = int(np.argmin(s_lds))
                best_ld = s_lds[best_r]
                best_params = final_params_all[s_start + best_r:s_start + best_r + 1].cpu()
                best_output = s_output[best_r:best_r+1].cpu()

                if verbose:
                    for r in range(R):
                        opt_loss = best_opt_losses[s_start + r].item()
                        mark = " ★" if r == best_r else ""
                        print(f"  {pfx}[{si}] R{r+1} opt_loss={opt_loss:.4f} → Ld={s_lds[r]:.4f}{mark}")

                    baseline_ld = self.mrstft_loss(
                        cleans_padded[s:s+1], targets_padded[s:s+1]
                    ).item()
                    print(f"  {pfx}[{si}] BEST Ld={best_ld:.4f} (baseline={baseline_ld:.4f} Δ={baseline_ld - best_ld:+.4f})")

                results.append((best_ld, best_params, best_output))

        return results

    def _optimize_single(
        self,
        clean: torch.Tensor,
        target: torch.Tensor,
        reference: torch.Tensor = None,
        model=None,
        embed_func=None,
        clean_ref: torch.Tensor = None,
        sample_idx: int = -1,
        verbose: bool = False,
        log_prefix: str = "",
        disabled_fx: list = None,
    ) -> tuple:
        """
        单次参数优化 (兼容旧接口，内部转发到 _optimize_batch)。
        """
        refs = [reference] if reference is not None else None
        clean_refs = [clean_ref] if clean_ref is not None else None

        results = self._optimize_batch(
            cleans=[clean],
            targets=[target],
            references=refs,
            model=model,
            embed_func=embed_func,
            clean_refs=clean_refs,
            sample_indices=[sample_idx],
            verbose=verbose,
            log_prefix=log_prefix,
            disabled_fx=disabled_fx,
        )
        return results[0]  # (best_ld, best_params, best_output)

    def _compute_embed(self, audio, model, embed_func, original=None):
        """
        计算音频嵌入 (保持梯度)。

        Args:
            audio: (1, channels, samples) — 处理后音频
            model: 嵌入模型
            embed_func: 嵌入函数
            original: (1, channels, samples) — 原始音频 (V2/V4 模型需要)

        Returns:
            embed: (1, embed_dim) tensor
        """
        model_type = self._detect_model_type(model)

        if model_type == "clap":
            return self._compute_embed_clap(audio, model)
        elif model_type == "vggish":
            return self._compute_embed_vggish(audio, model)
        elif model_type == "fxencpp":
            return self._compute_embed_fxencpp(audio, model)
        elif model_type == "cascaded_v2":
            return self._compute_embed_cascaded_v2(audio, model, original)
        elif model_type == "cascaded_v4":
            return self._compute_embed_cascaded_v4(audio, model, original)
        else:
            return self._compute_embed_afx_rep(audio, model)

    def _detect_model_type(self, model):
        """检测模型类型"""
        # 优先使用模型自带的类型标记 (我们在加载时设置)
        if hasattr(model, '_model_type'):
            return model._model_type

        # 回退到特征检测
        if hasattr(model, "get_audio_embedding_from_data"):
            return "clap"
        # FX-Encoder++ 特征: 有 fx_encoder 和 get_fx_embedding
        if hasattr(model, "fx_encoder") and hasattr(model, "get_fx_embedding"):
            return "fxencpp"
        # V4 特征: 有 fusion 模块
        if hasattr(model, "fusion") and hasattr(model, "forward_dual_backbone"):
            return "cascaded_v4"
        # V2 特征: 有 forward_backbone 和 forward_projection
        if hasattr(model, "forward_backbone") and hasattr(model, "forward_projection"):
            return "cascaded_v2"
        return "afx_rep"

    def _compute_embed_clap(self, audio, model):
        """CLAP 嵌入 (保持梯度)"""
        if audio.shape[1] == 2:
            x = audio.mean(dim=1, keepdim=True).squeeze(1)  # (batch, samples)
        else:
            x = audio.squeeze(1)

        embed = model.get_audio_embedding_from_data(x, use_tensor=True)
        return embed

    def _compute_embed_vggish(self, audio, model):
        """
        VGGish 嵌入 (torchaudio 实现，全 tensor 操作，保持梯度)。

        torchaudio VGGish:
            - 期望 16kHz 单声道输入
            - input_processor: waveform (T,) → (n_example, 1, n_frame, 64)
            - model: (n_example, 1, n_frame, 64) → (n_example, 128)
            - 对时间维度取 mean → (128,)

        输入: (B, C, T) 音频 tensor
        输出: (B, 128) embedding tensor
        """
        import torchaudio
        from torchaudio.prototype.pipelines._vggish._vggish_impl import _waveform_to_examples

        bs = audio.shape[0]
        # 立体声 → 单声道
        if audio.shape[1] == 2:
            x = audio.mean(dim=1)  # (B, T)
        else:
            x = audio.squeeze(1)   # (B, T)

        # 重采样到 16kHz (可微分)
        vggish_sr = getattr(model, '_vggish_sr', 16000)
        if self.sample_rate != vggish_sr:
            x = torchaudio.functional.resample(x, self.sample_rate, vggish_sr)

        embeddings = []
        for i in range(bs):
            # _waveform_to_examples 是纯 tensor 操作 (可微分)
            examples = _waveform_to_examples(x[i])  # (n_example, 1, n_frame, 64)
            emb = model(examples)  # (n_example, 128)
            emb = emb.mean(dim=0, keepdim=True)  # (1, 128)
            embeddings.append(emb)

        embeddings = torch.cat(embeddings, dim=0)  # (B, 128)
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _compute_embed_fxencpp(self, audio, model):
        """
        FX-Encoder++ (MixtureFxEncoder) 嵌入 (保持梯度)。

        使用 get_fx_embedding() 获取 mixture-level fx embedding (128-d, L2 normalized)。
        输入: (B, 2, T) 立体声音频。

        FX-Encoder++ 的 MixtureFxEncoder 是 Cnn14 风格 backbone:
            Audio → Spectrogram → LogMel → 6×ConvBlock → GlobalPool → FC → 2048-d
            → fx_encoder_projection (2048→128) → L2 normalize → 128-d embedding
        """
        import torchaudio

        # FX-Encoder++ 需要 44100Hz 立体声输入
        model_sr = getattr(model, '_eval_sample_rate', 44100)
        if self.sample_rate != model_sr:
            audio = torchaudio.functional.resample(audio, self.sample_rate, model_sr)

        # 确保是立体声 (B, 2, T)
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)

        # get_fx_embedding: (B, 2, T) → (B, 128) L2-normalized
        embed = model.get_fx_embedding(audio)
        return embed

    def _compute_embed_afx_rep(self, audio, model):
        """
        AFx-Rep (Cnn14) 嵌入 (保持梯度)。

        AFx-Rep 模型输入: (batch, chs, seq_len) 原始音频
        模型内部自己做: Spectrogram → LogMel → ConvBlocks → embedding
        需要重采样到 48000Hz (ST-ITO 训练采样率)。

        返回: mid embedding (L2-normalized)
        """
        import torchaudio

        # 重采样到 48000Hz (AFx-Rep / ST-ITO 训练时的采样率)
        if self.sample_rate != 48000:
            audio = torchaudio.functional.resample(audio, self.sample_rate, 48000)

        # peak normalize each batch item (non in-place to preserve gradient graph)
        peaks = audio.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        audio = audio / peaks

        # model forward: (batch, chs, seq_len) → (mid_embeddings, side_embeddings)
        mid_embeddings, side_embeddings = model(audio)

        # L2 normalize
        mid_embeddings = F.normalize(mid_embeddings, p=2, dim=-1)

        return mid_embeddings

    def _compute_embed_cascaded_v2(self, processed, model, original=None):
        """
        Cascaded FX CL V2 (DiffFxEncoder) 嵌入 (保持梯度)。

        V2 模型需要同时输入原始音频和处理后音频。
        """
        import torchaudio

        if original is None:
            raise ValueError("cascaded-v2 模型需要 original 音频输入")

        # V2 模型训练时用 44100Hz
        model_sr = getattr(model, '_eval_sample_rate', 44100)
        if self.sample_rate != model_sr:
            processed_rs = torchaudio.functional.resample(processed, self.sample_rate, model_sr)
            original_rs = torchaudio.functional.resample(original, self.sample_rate, model_sr)
        else:
            processed_rs = processed
            original_rs = original

        # V2 模型使用最终投影表示 z 进行 ITO。
        out = model(original_rs, processed_rs)
        z = out['z']  # (B, proj_dim), 已经 L2 normalized

        return z

    def _compute_embed_cascaded_v4(self, processed, model, original=None):
        """
        Cascaded FX CL V4 (DualBranchFxEncoder) 嵌入 (保持梯度)。

        V4 模型是双分支架构，需要同时输入干声和湿声:
            - original (干声): 未经效果处理的音频
            - processed (湿声): 经过效果链处理后的音频

        ITO 使用最终投影表示 z，它由 fusion representation e_fx 经投影头
        和 L2 normalization 得到，代表输入音频之间的效果变换。
        """
        import torchaudio

        if original is None:
            raise ValueError("cascaded-v4 模型需要 original (干声) 音频输入")

        # V4 模型训练时用 44100Hz
        model_sr = getattr(model, '_eval_sample_rate', 44100)
        if self.sample_rate != model_sr:
            processed_rs = torchaudio.functional.resample(processed, self.sample_rate, model_sr)
            original_rs = torchaudio.functional.resample(original, self.sample_rate, model_sr)
        else:
            processed_rs = processed
            original_rs = original

        # RelFx ITO 与论文一致，使用最终 128-d projected representation z。
        out = model(original_rs, processed_rs)
        z = out['z']  # (B, proj_dim), 已经 L2 normalized

        return z

    def match_single(
        self,
        clean: np.ndarray,
        target: np.ndarray,
        reference: np.ndarray = None,
        model=None,
        embed_func=None,
        clean_ref: np.ndarray = None,
        sample_idx: int = -1,
        verbose: bool = False,
        log_prefix: str = "",
        disabled_fx: list = None,
    ) -> dict:
        """
        对单个三元组进行参数匹配。

        Args:
            clean: (channels, samples) numpy — seg_a 的干声
            target: (channels, samples) numpy — fx(seg_a, params)
            reference: (channels, samples) numpy — fx(seg_b, params)
            model: 嵌入模型 (optional)
            embed_func: 嵌入函数 (optional)
            clean_ref: (channels, samples) numpy — seg_b 的干声 (V2 模型用)
            sample_idx: 当前样本编号 (用于日志)
            verbose: 是否打印详细优化日志
            log_prefix: 日志前缀 (如 "[drums]")
            disabled_fx: 被禁用的效果器名称列表

        Returns:
            dict with "ld" (STFT loss), "params", etc.
        """
        # 转为 tensor
        clean_t = torch.from_numpy(clean).unsqueeze(0).float()
        target_t = torch.from_numpy(target).unsqueeze(0).float()
        reference_t = None
        if reference is not None:
            reference_t = torch.from_numpy(reference).unsqueeze(0).float()
        clean_ref_t = None
        if clean_ref is not None:
            clean_ref_t = torch.from_numpy(clean_ref).unsqueeze(0).float()

        ld, params, output = self._optimize_single(
            clean_t, target_t, reference_t, model, embed_func, clean_ref=clean_ref_t,
            sample_idx=sample_idx, verbose=verbose, log_prefix=log_prefix,
            disabled_fx=disabled_fx,
        )

        return {
            "ld": ld,
            "params": params.squeeze(0).numpy().tolist(),
            "output": output.squeeze(0).numpy(),
        }

    def match_batch_parallel(
        self,
        items: list,
        model=None,
        embed_func=None,
        verbose: bool = True,
        log_prefix: str = "",
        disabled_fx_list: list = None,
    ) -> list:
        """
        对多个三元组进行批量并行参数匹配 (多 sample × 多 restart GPU 并行)。

        Args:
            items: list of dicts, 每个包含 'clean', 'target', 'reference', 'clean_ref' (numpy)
            model: 嵌入模型
            embed_func: 嵌入函数
            verbose: 是否打印详细日志
            log_prefix: 日志前缀
            disabled_fx_list: list of (disabled_fx or None) 每个 sample 各自的 disabled_fx
                             如果为 None，所有 sample 共享 disabled_fx=None

        Returns:
            list of result dicts (与 match_single 格式一致)
        """
        S = len(items)
        if S == 0:
            return []

        # 统一 disabled_fx — 当前假设所有 sample 共享同一设置 (简化)
        disabled_fx = None
        if disabled_fx_list and disabled_fx_list[0] is not None:
            disabled_fx = disabled_fx_list[0]

        # 转为 tensor
        cleans = [torch.from_numpy(it['clean']).unsqueeze(0).float() for it in items]
        targets = [torch.from_numpy(it['target']).unsqueeze(0).float() for it in items]
        refs = None
        if items[0].get('reference') is not None:
            refs = [torch.from_numpy(it['reference']).unsqueeze(0).float() for it in items]
        clean_refs = None
        if items[0].get('clean_ref') is not None:
            clean_refs = [torch.from_numpy(it['clean_ref']).unsqueeze(0).float() for it in items]

        sample_indices = [it.get('sample_idx', i+1) for i, it in enumerate(items)]

        # batch 并行优化
        batch_results = self._optimize_batch(
            cleans=cleans,
            targets=targets,
            references=refs,
            model=model,
            embed_func=embed_func,
            clean_refs=clean_refs,
            sample_indices=sample_indices,
            verbose=verbose,
            log_prefix=log_prefix,
            disabled_fx=disabled_fx,
        )

        # 转为标准 dict 格式
        results = []
        for ld, params, output in batch_results:
            results.append({
                "ld": ld,
                "params": params.squeeze(0).numpy().tolist(),
                "output": output.squeeze(0).numpy(),
            })

        return results

    def match_batch(
        self,
        triplets: list,
        model=None,
        embed_func=None,
        verbose: bool = True,
    ) -> list:
        """
        对一批三元组进行参数匹配 (兼容旧接口)。

        自动根据 sample_batch_size 决定是逐个跑还是批量并行。

        Args:
            triplets: list of (clean, target, reference, clean_ref) numpy arrays
            model: 嵌入模型
            embed_func: 嵌入函数
            verbose: 是否打印进度和详细优化日志

        Returns:
            list of result dicts
        """
        from tqdm import tqdm

        results = []
        batch_size = self.sample_batch_size

        if batch_size <= 1:
            # 逐个跑 (兼容旧行为，但内部 restart 已并行)
            iterator = tqdm(triplets, desc="Parameter matching") if verbose else triplets

            for idx, item in enumerate(iterator):
                if len(item) == 4:
                    clean, target, reference, clean_ref = item
                else:
                    clean, target, reference = item
                    clean_ref = None

                if verbose:
                    print(f"\n  === Sample {idx+1}/{len(triplets)} ===")

                result = self.match_single(
                    clean, target, reference, model, embed_func, clean_ref=clean_ref,
                    sample_idx=idx+1, verbose=verbose,
                )
                results.append(result)

                if verbose:
                    iterator.set_postfix(ld=f"{result['ld']:.4f}")
        else:
            # 批量并行
            total = len(triplets)
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch_items = []
                for idx in range(batch_start, batch_end):
                    item = triplets[idx]
                    if len(item) == 4:
                        clean, target, reference, clean_ref = item
                    else:
                        clean, target, reference = item
                        clean_ref = None
                    batch_items.append({
                        'clean': clean,
                        'target': target,
                        'reference': reference,
                        'clean_ref': clean_ref,
                        'sample_idx': idx + 1,
                    })

                if verbose:
                    print(f"\n  === Batch {batch_start+1}~{batch_end}/{total} "
                          f"({len(batch_items)} samples × {self.num_restarts} restarts) ===")

                batch_results = self.match_batch_parallel(
                    batch_items, model, embed_func, verbose=verbose,
                )
                results.extend(batch_results)

        return results
