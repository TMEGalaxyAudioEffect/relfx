"""
08 Parameter Matching — 可微分效果链

封装 FX-Encoder++ 的 7 种效果器为可微分参数匹配链 (论文 Section 3.3 + 4.2):
  EQ → Multiband Compressor → Stereo Imager → Gain → Distortion → Delay → Limiter

论文依据 (可微分特性):
  - Section 3.3: "We implement audio processors using dasp-pytorch [41] and torchcomp [42]"
    → dasp-pytorch 是专门的可微分音频信号处理库，torchcomp 是 PyTorch 可微分压缩器
    → 所有效果器支持梯度反向传播，是可微分链路的直接技术支撑
  - Section 4.2: "optimize differentiable effect chain parameters"
    → 明确点明效果链的可微分属性，参数通过梯度优化

参数数量 (已从 FX-Encoder++ 代码验证):
  eq:           18 params (dasp_pytorch.ParametricEQ, 6 bands × 3)
  multiband_comp: 21 params
  imager:        1 param
  gain:          1 param
  distortion:    1 param
  delay:         2 params
  limiter:       3 params
  -----------------------
  Total:        47 params

默认使用 "fxenc" 后端 (FX-Encoder++ 的可微分效果链)，以保证与论文一致。
"""

import torch
import torch.nn as nn
import numpy as np

from .config import FX_CHAIN_ORDER, FX_PROB, SAMPLE_RATE
from relfx.fx_chain.fx_aug import Random_FX_Chain


# ===================== FX-Encoder++ 的可微分效果链 (7 效果器, 47 params) =====================

class FxEncFxChain(nn.Module):
    """
    FX-Encoder++ 的 7 效果器可微分链。

    使用 fx_chain.fx_aug.Random_FX_Chain 的处理器，
    但以确定性方式（给定参数）应用效果。

    效果链顺序: eq → multiband_comp → imager → gain → distortion → delay → limiter
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: str = "cpu"):
        super().__init__()
        self.sample_rate = sample_rate
        self.device_str = device

        self.fx_chain = Random_FX_Chain(sample_rate=sample_rate, device=device)
        self.full_num_params = self.fx_chain.total_num_param  # 72 (含 reverb)
        self.processors_order = FX_CHAIN_ORDER

        # 计算 7 个效果器使用的参数索引 (不含 reverb)
        self.valid_param_indices = []
        self.param_info = {}
        for fx_name in FX_CHAIN_ORDER:
            idx = self.fx_chain.fx_indices[fx_name]
            start, end = self.fx_chain.param_range[idx]
            n_params = self.fx_chain.fx_processors[fx_name].num_params
            self.param_info[fx_name] = {
                "start": start,
                "end": end,
                "n_params": n_params,
            }
            self.valid_param_indices.extend(range(start, end))

        self.total_num_params = len(self.valid_param_indices)  # 47

        # 构建 47→72 的索引映射
        self._valid_indices_tensor = torch.tensor(self.valid_param_indices, dtype=torch.long)

    def forward(
        self,
        audio: torch.Tensor,
        params: torch.Tensor,
        disabled_fx: list = None,
        activate_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        应用效果链。

        Args:
            audio: (batch, channels, samples) 输入音频
            params: (batch, 47) 或 (batch, 72) 归一化参数 [0, 1]
                    如果是 47 维，自动 pad 到 72 维 (reverb 参数用 0 填充)
            disabled_fx: 要屏蔽的效果器名称列表，例如 ["distortion", "delay"]
                         被屏蔽的效果器完全旁路 (bypass)，等价于不存在
            activate_mask: 可选效果器激活掩码。支持 (batch, 7) 的 FX_CHAIN_ORDER 顺序，
                           或 (batch, len(fx_indices)) 的 FX-Encoder++ 原始顺序。

        Returns:
            processed: (batch, channels, samples) 处理后的音频
        """
        # 如果输入是 47 维，pad 到 72 维
        if params.shape[-1] == self.total_num_params:
            full_params = torch.zeros(params.shape[0], self.full_num_params,
                                     device=params.device, dtype=params.dtype)
            full_params[:, self._valid_indices_tensor.to(params.device)] = params
        elif params.shape[-1] == self.full_num_params:
            full_params = params
        else:
            raise ValueError(
                f"Expected params dim {self.total_num_params} or {self.full_num_params}, "
                f"got {params.shape[-1]}"
            )

        # 构建 activate 掩码: 默认全部激活；如传入 activate_mask，则按样本使用该掩码。
        if activate_mask is None:
            activate = torch.ones(
                audio.shape[0],
                len(self.fx_chain.fx_indices),
                device=audio.device,
                dtype=params.dtype,
            )
        else:
            activate = self._coerce_activate_mask(
                activate_mask,
                batch_size=audio.shape[0],
                device=audio.device,
                dtype=params.dtype,
            )
        if disabled_fx:
            for fx_name in disabled_fx:
                if fx_name in self.fx_chain.fx_indices:
                    fx_idx = self.fx_chain.fx_indices[fx_name]
                    activate[:, fx_idx] = 0.0
                else:
                    print(f"Warning: unknown fx '{fx_name}', available: {list(self.fx_chain.fx_indices.keys())}")

        processed, _, _ = self.fx_chain(
            audio,
            nn_param=full_params,
            activate=activate,
            processors_order=self.processors_order,
        )

        return processed

    def _coerce_activate_mask(
        self,
        activate_mask: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert a 7-FX or full FX-Encoder++ activation mask to full activate shape."""
        if not torch.is_tensor(activate_mask):
            activate_mask = torch.tensor(activate_mask)
        activate_mask = activate_mask.to(device=device, dtype=dtype)
        if activate_mask.dim() == 1:
            activate_mask = activate_mask.unsqueeze(0)
        if activate_mask.shape[0] == 1 and batch_size > 1:
            activate_mask = activate_mask.expand(batch_size, -1)
        if activate_mask.shape[0] != batch_size:
            raise ValueError(
                f"activate_mask batch {activate_mask.shape[0]} does not match audio batch {batch_size}"
            )

        full_width = len(self.fx_chain.fx_indices)
        if activate_mask.shape[1] == full_width:
            return activate_mask

        if activate_mask.shape[1] != len(FX_CHAIN_ORDER):
            raise ValueError(
                f"activate_mask width must be {len(FX_CHAIN_ORDER)} or {full_width}, "
                f"got {activate_mask.shape[1]}"
            )

        full = torch.ones(batch_size, full_width, device=device, dtype=dtype)
        for local_idx, fx_name in enumerate(FX_CHAIN_ORDER):
            fx_idx = self.fx_chain.fx_indices[fx_name]
            full[:, fx_idx] = activate_mask[:, local_idx]
        return full

    def get_param_info(self) -> dict:
        """获取参数信息"""
        return self.param_info

    def get_total_params(self) -> int:
        """获取总参数数"""
        return self.total_num_params


# ===================== 统一接口 =====================

class DifferentiableFxChain:
    """
    可微分效果链 (FX-Encoder++ 论文 Section 4.2)。

    7 效果器链: EQ → Multiband Compressor → Stereo Imager → Gain → Distortion → Delay → Limiter
    共 47 个可微分参数。
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device = device

        self.chain = FxEncFxChain(sample_rate=sample_rate, device=device)
        self.total_params = self.chain.total_num_params  # 47 (不含 reverb)

        print(f"DifferentiableFxChain initialized: total_params={self.total_params}, sample_rate={sample_rate}")

    def __call__(self, audio: torch.Tensor, params: torch.Tensor,
                 disabled_fx: list = None,
                 activate_mask: torch.Tensor = None) -> torch.Tensor:
        """
        应用效果链。

        Args:
            audio: (batch, channels, samples)
            params: (batch, 47) 归一化参数 [0, 1]
            disabled_fx: 要屏蔽的效果器名称列表，例如 ["distortion", "delay"]
                         被屏蔽的效果器完全旁路 (bypass)，音频不经过该效果器
            activate_mask: 可选效果器激活掩码，用于复现 FX-Encoder++ 原始 Bernoulli 开启概率

        Returns:
            processed: (batch, channels, samples)
        """
        return self.chain(audio, params, disabled_fx=disabled_fx, activate_mask=activate_mask)

    def sample_random_params(self, batch_size: int = 1) -> torch.Tensor:
        """采样随机参数"""
        return torch.rand(batch_size, self.total_params)

    def sample_activation_mask(
        self,
        batch_size: int = 1,
        mode: str = "all",
        disabled_fx: list = None,
    ) -> torch.Tensor:
        """
        Sample an activation mask for the 7-FX chain.

        mode="all" keeps the historical demo behavior.
        mode="original-prob" uses FX_PROB Bernoulli probabilities from FX-Encoder++.
        The returned mask is in the full FX-Encoder++ activate width.
        """
        full_width = len(self.chain.fx_chain.fx_indices)
        mask = torch.ones(batch_size, full_width)

        if mode == "all":
            pass
        elif mode == "original-prob":
            for fx_name in FX_CHAIN_ORDER:
                fx_idx = self.chain.fx_chain.fx_indices[fx_name]
                prob = float(FX_PROB.get(fx_name, 1.0))
                mask[:, fx_idx] = torch.bernoulli(torch.full((batch_size,), prob))
        else:
            raise ValueError(f"Unknown activation mode: {mode}")

        if disabled_fx:
            for fx_name in disabled_fx:
                if fx_name in self.chain.fx_chain.fx_indices:
                    mask[:, self.chain.fx_chain.fx_indices[fx_name]] = 0.0

        return mask

    def params_active_mask(
        self,
        activate_mask: torch.Tensor = None,
        disabled_fx: list = None,
        batch_size: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Return a (batch, 47) mask where inactive effect parameters are 0."""
        device = device or torch.device("cpu")
        param_mask = torch.ones(batch_size, self.total_params, device=device)

        if activate_mask is not None:
            full_mask = self.chain._coerce_activate_mask(
                activate_mask,
                batch_size=batch_size,
                device=device,
                dtype=param_mask.dtype,
            )
            for fx_name in FX_CHAIN_ORDER:
                fx_idx = self.chain.fx_chain.fx_indices[fx_name]
                indices = self.get_fx_param_indices([fx_name])
                if indices:
                    param_mask[:, indices] = full_mask[:, fx_idx].unsqueeze(1)

        if disabled_fx:
            for idx in self.get_fx_param_indices(disabled_fx):
                param_mask[:, idx] = 0.0

        return param_mask

    def sample_conservative_params(
        self,
        batch_size: int = 1,
        low: float = 0.42,
        high: float = 0.58,
        disabled_fx: list = None,
    ) -> torch.Tensor:
        """
        采样保守参数: 避免极端值, 考虑每个效果器的中性点。

        设计原则:
          - 大部分参数在 [low, high] 范围内均匀采样
          - 某些效果器的"无效果"状态不在 0.5, 需要偏移范围:
            * gain: 0.5 = 0dB (中性), 范围 [0.42, 0.58] 避免过大/过小增益
            * eq: 各频段 0.5 = 0dB (中性), 范围 [0.4, 0.6]
            * multiband_comp: threshold/ratio 保守取值
            * limiter: threshold 不宜太低 (过度压限)
          - disabled_fx 的参数设为 0

        Args:
            batch_size: 批大小
            low: 默认下界 (归一化空间 [0,1])
            high: 默认上界
            disabled_fx: 要禁用的效果器列表

        Returns:
            params: (batch_size, 47) 保守参数
        """
        params = torch.rand(batch_size, self.total_params) * (high - low) + low

        param_info = self.get_param_info()
        valid_indices = self.chain.valid_param_indices
        idx_72_to_47 = {v: i for i, v in enumerate(valid_indices)}

        # 每个效果器的保守范围覆盖 (基于物理意义)
        conservative_ranges = {
            "gain": (0.42, 0.58),         # 0.5 = 0dB, 微调范围
            "eq": (0.40, 0.60),           # 各频段增益 0.5=0dB
            "multiband_comp": (0.40, 0.60),
            "limiter": (0.45, 0.60),      # threshold 不宜太低
        }

        for fx_name, (fx_low, fx_high) in conservative_ranges.items():
            if fx_name not in param_info:
                continue
            info = param_info[fx_name]
            for idx_72 in range(info["start"], info["end"]):
                if idx_72 in idx_72_to_47:
                    idx_47 = idx_72_to_47[idx_72]
                    params[:, idx_47] = torch.rand(batch_size) * (fx_high - fx_low) + fx_low

        # disabled_fx 的参数设为 0
        if disabled_fx:
            frozen_indices = self.get_fx_param_indices(disabled_fx)
            for idx in frozen_indices:
                params[:, idx] = 0.0

        return params

    def get_total_params(self) -> int:
        return self.total_params

    def get_param_info(self) -> dict:
        """获取参数信息 (每个效果器的参数起止位置)"""
        return self.chain.get_param_info()

    def get_fx_param_indices(self, fx_names: list) -> list:
        """
        获取指定效果器在 47 维参数空间中的参数索引列表。

        用途: 在 ITO 优化中冻结被屏蔽效果器的参数 (不参与梯度更新)。

        Args:
            fx_names: 效果器名称列表，例如 ["distortion", "delay"]

        Returns:
            indices: 参数索引列表 (在 47 维空间中的位置)
        """
        param_info = self.chain.get_param_info()
        # 需要从 72 维索引映射回 47 维索引
        valid_indices = self.chain.valid_param_indices  # 72 维空间的有效索引列表
        # 构建 72→47 的逆映射
        idx_72_to_47 = {v: i for i, v in enumerate(valid_indices)}

        result = []
        for fx_name in fx_names:
            if fx_name not in param_info:
                print(f"Warning: unknown fx '{fx_name}', available: {list(param_info.keys())}")
                continue
            info = param_info[fx_name]
            for idx_72 in range(info["start"], info["end"]):
                if idx_72 in idx_72_to_47:
                    result.append(idx_72_to_47[idx_72])
        return sorted(result)

    @staticmethod
    def available_fx() -> list:
        """返回可用的效果器名称列表"""
        return ["eq", "multiband_comp", "imager", "gain", "distortion", "delay", "limiter"]
