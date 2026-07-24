"""Audio sampling and collation for RelFx training."""

import os
import re
import random
import numpy as np
import torch
import torchaudio
import soundfile as sf
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from . import config as cfg

SAMPLE_RATE = cfg.SAMPLE_RATE
SEGMENT_SAMPLES = cfg.SEGMENT_SAMPLES
AUDIO_DIR = cfg.AUDIO_DIR
AUDIO_DIRS = getattr(cfg, 'AUDIO_DIRS', [AUDIO_DIR])
FX_CHAIN_ORDER = cfg.FX_CHAIN_ORDER
FX_PROB = cfg.FX_PROB

STRUCTURE_SEGMENT_JSON = getattr(cfg, "STRUCTURE_SEGMENT_JSON", None)
STRUCTURED_AUDIO_DIRS = getattr(cfg, "STRUCTURED_AUDIO_DIRS", set())
DENSITY_FILTER_AUDIO_DIRS = getattr(cfg, "DENSITY_FILTER_AUDIO_DIRS", set())
VALID_STRUCTURE_LABELS = getattr(
    cfg, "VALID_STRUCTURE_LABELS", {"verse", "chorus"}
)


class AudioSegmentDataset(Dataset):
    """
    音频加载器 — V5 版本，支持多数据源 + cross_segment + train/val split。

    V5 相比 V4:
        - 支持 audio_dirs (列表)，合并多个数据源
        - 支持常见的数字 ID 与 UUID 目录布局

    split 模式 (按歌曲 ID 划分，确保 train/val 无歌曲泄漏):
        - "train": 只使用训练集歌曲 (85%)
        - "val":   只使用验证集歌曲 (15%)
        - None:    使用全部歌曲 (向后兼容)

    cross_segment=False (默认, V2 行为):
        audio_a, audio_b 来自不同歌曲的随机片段

    cross_segment=True:
        额外返回 audio_a_alt (同一首歌的不同位置)
    """

    SUPPORTED_EXT = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}

    def __init__(
        self,
        audio_dir=None,
        audio_dirs=None,
        segment_samples: int = SEGMENT_SAMPLES,
        sample_rate: int = SAMPLE_RATE,
        cross_segment: bool = False,
        split: str = None,
        val_ratio: float = 0.15,
        split_seed: int = 42,
        structure_segment_json: str = STRUCTURE_SEGMENT_JSON,
        structured_audio_dirs=None,
        density_filter_audio_dirs=None,
        valid_structure_labels=None,
    ):
        super().__init__()
        self.segment_samples = segment_samples
        self.sample_rate = sample_rate
        self.cross_segment = cross_segment

        # 支持多数据源
        if audio_dirs is not None:
            dirs_to_scan = audio_dirs
        elif audio_dir is not None:
            dirs_to_scan = [audio_dir] if isinstance(audio_dir, str) else audio_dir
        else:
            dirs_to_scan = AUDIO_DIRS

        self.audio_dirs = [os.path.abspath(d) for d in dirs_to_scan]
        structured_audio_dirs = (
            STRUCTURED_AUDIO_DIRS
            if structured_audio_dirs is None
            else structured_audio_dirs
        )
        density_filter_audio_dirs = (
            DENSITY_FILTER_AUDIO_DIRS
            if density_filter_audio_dirs is None
            else density_filter_audio_dirs
        )
        self.structured_audio_dirs = {
            os.path.abspath(d) for d in structured_audio_dirs
        }
        self.density_filter_audio_dirs = {
            os.path.abspath(d) for d in density_filter_audio_dirs
        }
        valid_structure_labels = (
            VALID_STRUCTURE_LABELS
            if valid_structure_labels is None
            else set(valid_structure_labels)
        )

        all_files = []
        self._density_filter_files = set()
        self._structured_files = set()
        for d in self.audio_dirs:
            found = self._scan_audio_files(d)
            print(f"  [Scan] {d}: {len(found)} files")
            if d in self.density_filter_audio_dirs:
                self._density_filter_files.update(found)
            if d in self.structured_audio_dirs:
                self._structured_files.update(found)
            all_files.extend(found)

        self._structure_segments = {}
        if (
            structure_segment_json
            and os.path.exists(structure_segment_json)
            and self._structured_files
        ):
            self._load_structure_segments(
                structure_segment_json, valid_structure_labels
            )

        if len(all_files) == 0:
            raise RuntimeError(f"No audio files found in {dirs_to_scan}")

        # 按歌曲 ID 做 train/val split
        if split is not None:
            self.audio_files = self._split_by_song_id(
                all_files, split=split, val_ratio=val_ratio, seed=split_seed,
            )
        else:
            self.audio_files = all_files

        if len(self.audio_files) == 0:
            raise RuntimeError(f"No audio files after split='{split}' in {audio_dir}")

        mode_str = "cross_segment" if cross_segment else "same_segment (V2)"
        split_str = f"split={split}" if split else "no split"
        print(f"[Dataset V6] {len(self.audio_files)} files ({split_str}) from {len(dirs_to_scan)} sources | mode: {mode_str}")

    @staticmethod
    def _extract_song_id(filepath: str) -> str:
        """
        从文件路径提取歌曲 ID。

        支持两种数据格式:

        1. 文件名含数字 ID:
            .../101007690_1_V1.wav           → "101007690"
            .../101007690-JGRJ7qoFyG_1_V1.wav → "101007690"

        2. UUID 目录结构:
            .../014f3712-.../vocals/f6a45ee6-....wav → "014f3712-293b-42af-9f29-0ed1785be792"
            song_id = 文件的 grandparent 目录名 (song_uuid/stem/track.wav)
        """
        basename = os.path.basename(filepath)

        # 文件名以数字 ID 开头
        match = re.match(r'^(\d+)', basename)
        if match:
            return match.group(1)

        # UUID 布局下使用 grandparent 目录作为 song_id
        # song_uuid/stem_name/track_uuid.wav
        parent = os.path.basename(os.path.dirname(filepath))      # stem_name (e.g. "vocals")
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(filepath)))  # song_uuid
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-')
        if uuid_pattern.match(grandparent):
            return grandparent

        # fallback: 上级目录名
        return parent if parent else basename

    @staticmethod
    def _split_by_song_id(all_files, split, val_ratio=0.15, seed=42):
        """
        按歌曲 ID 级别划分 train/val，保证同一首歌的所有变体
        (SQ/PA/不同增强) 不会跨 split。
        """
        # 按歌曲 ID 分组
        song_groups = defaultdict(list)
        for f in all_files:
            song_id = AudioSegmentDataset._extract_song_id(f)
            song_groups[song_id].append(f)

        # 固定种子排序后划分
        song_ids = sorted(song_groups.keys())
        rng = random.Random(seed)
        rng.shuffle(song_ids)

        n_val = max(1, int(len(song_ids) * val_ratio))
        val_ids = set(song_ids[:n_val])
        train_ids = set(song_ids[n_val:])

        if split == "val":
            selected_ids = val_ids
        elif split == "train":
            selected_ids = train_ids
        else:
            raise ValueError(f"split must be 'train', 'val', or None, got '{split}'")

        selected_files = []
        for sid in sorted(selected_ids):
            selected_files.extend(song_groups[sid])

        print(f"[Split] Total songs: {len(song_ids)} | "
              f"Train songs: {len(train_ids)} | Val songs: {len(val_ids)} | "
              f"Selected ({split}): {len(selected_files)} files")

        return sorted(selected_files)

    def _scan_audio_files(self, root_dir: str):
        """扫描音频文件，过滤掉空/过短/损坏的文件"""
        files = []
        skipped = 0
        for root, dirs, filenames in os.walk(root_dir):
            for fname in filenames:
                if Path(fname).suffix.lower() in self.SUPPORTED_EXT:
                    fpath = os.path.join(root, fname)
                    try:
                        info = sf.info(fpath)
                        # 至少 2 秒
                        if info.frames < self.sample_rate * 2:
                            skipped += 1
                            continue
                        files.append(fpath)
                    except Exception:
                        skipped += 1
                        continue
        if skipped > 0:
            print(f"  [Scan] Skipped {skipped} short/broken files in {root_dir}")
        return sorted(files)

    # ===================== Optional structural segments =====================

    def _load_structure_segments(self, json_path, valid_labels):
        """
        Load public structural-segment metadata.

        构建 {track_id: [(start_sec, end_sec), ...]} 映射。
        """
        import json as _json
        with open(json_path) as f:
            data = _json.load(f)

        n_tracks = 0
        n_segments = 0
        for track_id, info in data.items():
            segs = info.get('segments', [])
            valid = []
            for s in segs:
                if s['label'] in valid_labels and s['duration'] >= self.segment_samples / self.sample_rate:
                    valid.append((s['start'], s['end']))
            if len(valid) >= 2:  # 至少需要 2 个有效段才能做 cross-segment
                self._structure_segments[track_id] = valid
                n_tracks += 1
                n_segments += len(valid)

        print(
            "  [Structure] Loaded segment info: "
            f"{n_tracks} tracks with {n_segments} eligible segments"
        )

    def _get_structure_track_id(self, filepath):
        """Use the filename stem as the public structural-metadata key."""
        return os.path.basename(filepath).rsplit('.', 1)[0]

    def _load_two_structured_segments(self, filepath):
        """
        Load two clips from distinct eligible structural segments.

        优先从不同的段落中各取一个 10s 片段 (真正的 cross-segment)。
        如果只有同一段落可用，则在段落内取两个不重叠的位置。

        Returns:
            seg1, seg2: (2, segment_samples) numpy arrays
        """
        track_id = self._get_structure_track_id(filepath)
        segments = self._structure_segments.get(track_id, None)

        if segments is None or len(segments) < 2:
            # fallback: 没有分段信息，用原始方法
            return self._load_two_different_segments(filepath)

        try:
            info = sf.info(filepath)
            sr = info.samplerate
            needed_sec = self.segment_samples / self.sample_rate

            # 随机选两个不同的段落
            if len(segments) >= 2:
                idx1, idx2 = random.sample(range(len(segments)), 2)
            else:
                idx1 = idx2 = 0

            seg1_start, seg1_end = segments[idx1]
            seg2_start, seg2_end = segments[idx2]

            # 在段落内随机选一个起始位置
            def pick_start(seg_start, seg_end):
                available = seg_end - seg_start - needed_sec
                if available <= 0:
                    return seg_start
                offset = random.random() * available
                return seg_start + offset

            start1_sec = pick_start(seg1_start, seg1_end)
            start2_sec = pick_start(seg2_start, seg2_end)

            start1_frame = int(start1_sec * sr)
            start2_frame = int(start2_sec * sr)

            s1 = self._load_segment(filepath, start_hint=start1_frame)
            s2 = self._load_segment(filepath, start_hint=start2_frame)
            return s1, s2

        except Exception as e:
            print(f"[Warning] Structural segment load error {filepath}: {e}")
            return self._load_two_different_segments(filepath)

    # 内容密度阈值: 片段中"有音频"帧占比需 >= 此值
    MIN_CONTENT_RATIO = 0.7
    # 判定"有音频"的 RMS 阈值 (每 hop 窗口)
    SILENCE_RMS_THRESHOLD = 1e-4
    # 密度检测 hop (每 0.1 秒检测一次)
    CONTENT_HOP = 0.1

    @staticmethod
    def _compute_content_ratio(audio: np.ndarray, sr: int,
                                hop_sec: float = 0.1, rms_thresh: float = 1e-4) -> float:
        """
        计算音频片段的内容密度 (非静音帧占比)。

        将音频按 hop_sec 分帧，每帧算 RMS，RMS > rms_thresh 视为有内容。
        返回有内容帧数 / 总帧数。
        """
        hop_samples = int(sr * hop_sec)
        total_samples = audio.shape[-1]
        if total_samples == 0:
            return 0.0

        n_frames = total_samples // hop_samples
        if n_frames == 0:
            return 1.0 if np.abs(audio).max() > rms_thresh else 0.0

        # 取单声道计算 (更快)
        mono = audio[0] if audio.ndim == 2 else audio
        mono = mono[:n_frames * hop_samples].reshape(n_frames, hop_samples)
        rms_per_frame = np.sqrt(np.mean(mono ** 2, axis=1))

        content_frames = np.sum(rms_per_frame > rms_thresh)
        return content_frames / n_frames

    def _load_segment(self, filepath: str, start_hint: int = None,
                      max_retries: int = 5) -> np.ndarray:
        """
        从文件中加载一段音频。

        For configured density-filter sources, require at least 70% non-silent
        frames. Other sources are loaded without density retries.

        Args:
            filepath: 音频文件路径
            start_hint: 指定起始帧 (None = 随机)
            max_retries: density-filter retry count
        Returns:
            (2, segment_samples) stereo numpy array
        """
        need_density_check = filepath in self._density_filter_files
        retries = max_retries if need_density_check else 1
        try:
            info = sf.info(filepath)
            total_frames = info.frames
            sr = info.samplerate

            if sr != self.sample_rate:
                needed_frames = int(self.segment_samples * sr / self.sample_rate) + 1024
            else:
                needed_frames = self.segment_samples

            best_audio = None
            best_ratio = -1.0

            for attempt in range(retries):
                if attempt == 0 and start_hint is not None and total_frames > needed_frames:
                    start = min(start_hint, total_frames - needed_frames)
                elif total_frames > needed_frames:
                    start = random.randint(0, total_frames - needed_frames)
                else:
                    start = 0

                audio, file_sr = sf.read(filepath, start=start, frames=needed_frames, dtype='float32')

                if audio.ndim == 1:
                    audio = np.stack([audio, audio], axis=0)
                else:
                    audio = audio.T

                if audio.shape[0] == 1:
                    audio = np.repeat(audio, 2, axis=0)
                elif audio.shape[0] > 2:
                    audio = audio[:2]

                if file_sr != self.sample_rate:
                    audio_tensor = torch.from_numpy(audio).float()
                    audio_tensor = torchaudio.functional.resample(audio_tensor, file_sr, self.sample_rate)
                    audio = audio_tensor.numpy()

                if audio.shape[1] < self.segment_samples:
                    repeats = int(np.ceil(self.segment_samples / audio.shape[1]))
                    audio = np.tile(audio, (1, repeats))
                audio = audio[:, :self.segment_samples]

                # Optional content-density filtering
                if need_density_check:
                    ratio = self._compute_content_ratio(
                        audio, self.sample_rate,
                        self.CONTENT_HOP, self.SILENCE_RMS_THRESHOLD
                    )

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_audio = audio

                    if ratio >= self.MIN_CONTENT_RATIO:
                        break  # 满足要求，不用重试
                else:
                    best_audio = audio
                    break

            audio = best_audio

            max_val = np.abs(audio).max()
            if max_val > 1e-6:
                audio = audio / max_val * 0.5
            else:
                audio = np.random.randn(2, self.segment_samples).astype(np.float32) * 0.01

            return audio.astype(np.float32)

        except Exception as e:
            print(f"[Warning] Error loading {filepath}: {e}")
            return np.random.randn(2, self.segment_samples).astype(np.float32) * 0.01

    def _load_two_different_segments(self, filepath: str):
        """
        从同一首歌加载两个不重叠的片段。

        Returns:
            seg1: (2, segment_samples) — 第一个片段
            seg2: (2, segment_samples) — 第二个片段 (不同位置)
        """
        try:
            info = sf.info(filepath)
            total_frames = info.frames
            sr = info.samplerate

            if sr != self.sample_rate:
                needed_frames = int(self.segment_samples * sr / self.sample_rate) + 1024
            else:
                needed_frames = self.segment_samples

            # 需要至少 2 个片段长度才能不重叠
            if total_frames >= needed_frames * 2:
                # 把文件分成两半，各取一个随机位置
                half = total_frames // 2
                start1 = random.randint(0, half - needed_frames)
                start2 = random.randint(half, total_frames - needed_frames)
            elif total_frames > needed_frames:
                # 文件不够长，允许部分重叠但尽量拉开
                start1 = random.randint(0, total_frames - needed_frames)
                # 尝试取一个距离 start1 尽可能远的位置
                if start1 < total_frames // 2:
                    start2 = random.randint(
                        min(start1 + needed_frames // 2, total_frames - needed_frames),
                        total_frames - needed_frames,
                    )
                else:
                    start2 = random.randint(
                        0,
                        max(start1 - needed_frames // 2, 0),
                    )
            else:
                # 文件太短，两段只能一样了 (fallback)
                start1 = 0
                start2 = 0

            seg1 = self._load_segment(filepath, start_hint=start1)
            seg2 = self._load_segment(filepath, start_hint=start2)
            return seg1, seg2

        except Exception as e:
            print(f"[Warning] Error loading two segments from {filepath}: {e}")
            fallback = np.random.randn(2, self.segment_samples).astype(np.float32) * 0.01
            return fallback.copy(), fallback.copy()

    def __len__(self):
        return min(len(self.audio_files), 8000)

    def __getitem__(self, idx):
        idx_a = idx % len(self.audio_files)
        idx_b = random.randint(0, len(self.audio_files) - 1)
        while idx_b == idx_a and len(self.audio_files) > 1:
            idx_b = random.randint(0, len(self.audio_files) - 1)

        if self.cross_segment:
            audio_a, audio_a_alt = self._load_two_segments_with_fallback(idx_a)
            audio_b, audio_b_alt = self._load_two_segments_with_fallback(idx_b)
            return {
                'audio_a': torch.from_numpy(audio_a),
                'audio_a_alt': torch.from_numpy(audio_a_alt),
                'audio_b': torch.from_numpy(audio_b),
                'audio_b_alt': torch.from_numpy(audio_b_alt),
            }
        else:
            audio_a = self._load_segment_with_fallback(idx_a)
            audio_b = self._load_segment_with_fallback(idx_b)
            return {
                'audio_a': torch.from_numpy(audio_a),
                'audio_b': torch.from_numpy(audio_b),
            }

    def _load_segment_with_fallback(self, idx, max_file_retries=3):
        """Load a clip, changing files after repeated density failures."""
        for _ in range(max_file_retries):
            filepath = self.audio_files[idx]
            audio = self._load_segment(filepath)
            if filepath not in self._density_filter_files:
                return audio
            ratio = self._compute_content_ratio(
                audio, self.sample_rate, self.CONTENT_HOP, self.SILENCE_RMS_THRESHOLD
            )
            if ratio >= self.MIN_CONTENT_RATIO:
                return audio
            # 密度不足，换一个随机文件
            idx = random.randint(0, len(self.audio_files) - 1)
        return audio  # 多次换文件仍不满足，用最后一个

    def _load_two_segments_with_fallback(self, idx, max_file_retries=3):
        """Load two distinct clips with optional structure and density rules."""
        for _ in range(max_file_retries):
            filepath = self.audio_files[idx]

            if filepath in self._structured_files:
                seg1, seg2 = self._load_two_structured_segments(filepath)
                return seg1, seg2

            seg1, seg2 = self._load_two_different_segments(filepath)
            if filepath not in self._density_filter_files:
                return seg1, seg2
            # 两个片段都要达标
            r1 = self._compute_content_ratio(
                seg1, self.sample_rate, self.CONTENT_HOP, self.SILENCE_RMS_THRESHOLD
            )
            r2 = self._compute_content_ratio(
                seg2, self.sample_rate, self.CONTENT_HOP, self.SILENCE_RMS_THRESHOLD
            )
            if r1 >= self.MIN_CONTENT_RATIO and r2 >= self.MIN_CONTENT_RATIO:
                return seg1, seg2
            idx = random.randint(0, len(self.audio_files) - 1)
        return seg1, seg2


class FxContrastiveCollator:
    """
    Collator — V3 版本，支持方法 4 (Hard Negative Mining) + cross_segment 模式。

    cross_segment 模式:
        数据集返回 audio_a, audio_a_alt, audio_b, audio_b_alt
        - audio_a: 歌曲 A 的片段 1 (作为干声输入模型)
        - audio_a_alt: 歌曲 A 的片段 2 (施加 FX 后作为 processed 输入模型)
        → 模型输入 = (audio_a, fx(audio_a_alt))，干声和 processed 内容不同

    标准模式:
        数据集返回 audio_a, audio_b
        - audio_a: 歌曲 A 的片段 (同时作为干声和 FX 输入源)
        → 模型输入 = (audio_a, fx(audio_a))，干声和 processed 是同一段
    """

    def __init__(
        self,
        fx_chain,
        shuffle_order=True,
        hard_negative_enabled=False,
        hard_negative_config=None,
        current_epoch=0,
        cross_segment=False,
        fx_sample_mode="uniform",
        conservative_ratio=0.4,
        conservative_range=(0.3, 0.7),
    ):
        self.fx_chain = fx_chain
        self.shuffle_order = shuffle_order
        self.cross_segment = cross_segment

        # V6: 可动态更新的 FX 概率 (初始化为全局 FX_PROB)
        self._fx_prob_override = None

        # FX 参数采样模式
        self.fx_sample_mode = fx_sample_mode
        self.conservative_ratio = conservative_ratio
        self.conservative_range = conservative_range

        # 方法 4 配置
        self.hard_negative_enabled = hard_negative_enabled
        self.hn_config = hard_negative_config or {}
        self.current_epoch = current_epoch

    def set_epoch(self, epoch):
        """更新当前 epoch (用于 curriculum learning)"""
        self.current_epoch = epoch

    def update_fx_prob(self, new_probs: dict):
        """V6: 动态更新 FX 概率 (由训练循环中的动态调度调用)"""
        self._fx_prob_override = new_probs

    def _get_fx_prob_tensor(self):
        """获取当前 FX 概率 tensor，优先使用动态更新的概率"""
        prob_dict = self._fx_prob_override if self._fx_prob_override is not None else FX_PROB
        return torch.tensor([
            prob_dict.get(fx, 0.5) for fx in self.fx_chain.fx_indices.keys()
        ])

    def _sample_fx_params(self, batch_size: int) -> torch.Tensor:
        """
        采样 FX 参数。

        支持三种模式:
          - "uniform":       [0, 1] 均匀随机 (FX-Encoder++ 原版)
          - "conservative":  [low, high] 保守范围
          - "dual":          每个样本独立选择 uniform 或 conservative
        """
        D = self.fx_chain.total_num_param

        if self.fx_sample_mode == "uniform":
            return torch.rand(batch_size, D)

        elif self.fx_sample_mode == "conservative":
            lo, hi = self.conservative_range
            return lo + (hi - lo) * torch.rand(batch_size, D)

        elif self.fx_sample_mode == "dual":
            # 每个样本独立: conservative_ratio 概率用保守, 其余用全范围
            lo, hi = self.conservative_range
            params = torch.rand(batch_size, D)  # 先全部 [0,1]
            # 决定哪些样本用保守模式
            mask = torch.rand(batch_size) < self.conservative_ratio  # (B,)
            if mask.any():
                n_cons = mask.sum().item()
                params[mask] = lo + (hi - lo) * torch.rand(n_cons, D)
            return params

        else:
            raise ValueError(f"Unknown fx_sample_mode: {self.fx_sample_mode}")

    def _get_hard_ratio(self):
        """根据 curriculum 获取当前 hard negative 比例"""
        base_ratio = self.hn_config.get("hard_ratio", 0.5)

        if not self.hn_config.get("curriculum", False):
            return base_ratio

        start_epoch = self.hn_config.get("curriculum_start_epoch", 0)
        end_epoch = self.hn_config.get("curriculum_end_epoch", 50)

        if self.current_epoch < start_epoch:
            return 0.0
        elif self.current_epoch >= end_epoch:
            return base_ratio
        else:
            progress = (self.current_epoch - start_epoch) / max(end_epoch - start_epoch, 1)
            return base_ratio * progress

    def _generate_hard_negatives(self, B, nn_param_shared, activate_shared):
        """
        生成 hard negatives。

        策略:
        - same_activate: 保持 activate 一致，只随机化参数
        - param_perturb: 在 shared 参数基础上加高斯扰动
        - curriculum: 同 param_perturb 但扰动随 epoch 减小
        """
        strategy = self.hn_config.get("strategy", "same_activate")
        hard_ratio = self._get_hard_ratio()
        n_hard = int(B * hard_ratio)
        n_easy = B - n_hard

        # Easy negatives: 完全随机 (和 V2 一样)
        fx_prob_tensor = self._get_fx_prob_tensor()

        nn_param_easy = torch.rand(n_easy, self.fx_chain.total_num_param)
        activate_easy = torch.bernoulli(fx_prob_tensor.unsqueeze(0).expand(n_easy, -1))

        if n_hard == 0:
            return nn_param_easy, activate_easy

        # Hard negatives
        if strategy == "same_activate":
            # 保持 activate 一致，只随机化参数
            nn_param_hard = torch.rand(n_hard, self.fx_chain.total_num_param)
            activate_hard = activate_shared[:n_hard].clone()

        elif strategy in ("param_perturb", "curriculum"):
            # 在 shared 参数基础上加扰动
            perturb_std = self.hn_config.get("perturb_std", 0.1)

            if strategy == "curriculum":
                # 扰动随 epoch 增大 → 越来越难 (扰动越小越难区分)
                end_epoch = self.hn_config.get("curriculum_end_epoch", 50)
                progress = min(self.current_epoch / max(end_epoch, 1), 1.0)
                # 初期扰动大 (容易), 后期扰动小 (困难)
                perturb_std = perturb_std * (2.0 - progress)

            noise = torch.randn(n_hard, self.fx_chain.total_num_param) * perturb_std
            nn_param_hard = (nn_param_shared[:n_hard] + noise).clamp(0, 1)
            activate_hard = activate_shared[:n_hard].clone()

        else:
            raise ValueError(f"Unknown hard negative strategy: {strategy}")

        # 合并 hard + easy
        nn_param_diff = torch.cat([nn_param_hard, nn_param_easy], dim=0)
        activate_diff = torch.cat([activate_hard, activate_easy], dim=0)

        # 打乱顺序 (避免 batch 前半段全是 hard)
        perm = torch.randperm(B)
        nn_param_diff = nn_param_diff[perm]
        activate_diff = activate_diff[perm]

        return nn_param_diff, activate_diff

    def __call__(self, batch):
        audio_a = torch.stack([item['audio_a'] for item in batch])
        audio_b = torch.stack([item['audio_b'] for item in batch])
        B = audio_a.shape[0]

        # cross_segment 模式: 额外取出 alt 片段 (用于施加 FX)
        if self.cross_segment:
            audio_a_alt = torch.stack([item['audio_a_alt'] for item in batch])
            audio_b_alt = torch.stack([item['audio_b_alt'] for item in batch])
        else:
            audio_a_alt = None
            audio_b_alt = None

        # 共享 FX 参数 (正样本对)
        nn_param_shared = self._sample_fx_params(B)
        fx_prob_tensor = self._get_fx_prob_tensor()
        activate_shared = torch.bernoulli(fx_prob_tensor.unsqueeze(0).expand(B, -1))
        # V5: 不再强制最少激活数，允许 activate 全为 0
        # V6: fx_prob_tensor 可能已被动态调度更新

        # 负样本 FX 参数
        if self.hard_negative_enabled:
            nn_param_diff, activate_diff = self._generate_hard_negatives(
                B, nn_param_shared, activate_shared,
            )
        else:
            # 标准随机负样本 (V2 行为)
            nn_param_diff = self._sample_fx_params(B)
            activate_diff = torch.bernoulli(fx_prob_tensor.unsqueeze(0).expand(B, -1))

        # 效果链顺序
        if self.shuffle_order:
            processors_order = FX_CHAIN_ORDER.copy()
            random.shuffle(processors_order)
        else:
            processors_order = FX_CHAIN_ORDER

        if self.shuffle_order:
            processors_order_diff = FX_CHAIN_ORDER.copy()
            random.shuffle(processors_order_diff)
        else:
            processors_order_diff = processors_order

        result = {
            'audio_a': audio_a,
            'audio_b': audio_b,
            'nn_param_shared': nn_param_shared,
            'nn_param_diff': nn_param_diff,
            'activate_shared': activate_shared,
            'activate_diff': activate_diff,
            'processors_order': processors_order,
            'processors_order_diff': processors_order_diff,
        }

        # cross_segment 模式: 额外传递 alt 音频
        if self.cross_segment:
            result['audio_a_alt'] = audio_a_alt
            result['audio_b_alt'] = audio_b_alt

        return result


def create_dataloader(
    audio_dir: str = AUDIO_DIR,
    batch_size: int = 16,
    num_workers: int = 4,
    fx_chain=None,
    shuffle_order: bool = True,
    hard_negative_enabled: bool = False,
    hard_negative_config: dict = None,
    cross_segment: bool = False,
):
    """创建训练用的 DataLoader"""
    dataset = AudioSegmentDataset(audio_dir=audio_dir, cross_segment=cross_segment)

    collator = FxContrastiveCollator(
        fx_chain=fx_chain,
        shuffle_order=shuffle_order,
        hard_negative_enabled=hard_negative_enabled,
        hard_negative_config=hard_negative_config,
        cross_segment=cross_segment,
    ) if fx_chain is not None else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    return dataloader
