"""
08 Parameter Matching — 数据准备

从 FxNorm 预处理后的 MUSDB18/MedleyDB stems 生成 (clean, reference, target) 三元组。

三元组定义 (论文 Section 4.2):
  - clean: 未处理的干声 (content_A)
  - target: 对干声应用效果链后的音频 fx(content_A, params_P)
  - reference: 对另一个片段应用相同效果链后的音频 fx(content_B, params_P)
  → clean/target 内容相同效果不同, reference/target 内容不同效果相同

数据集 (论文 Section 4.2):
  - MUSDB18: drums, bass, vocals, other (各 100 samples)
  - MedleyDB: mandolin, alto_saxophone, horn, trumpet (各 100 samples)

响度归一化 (论文 Section 3.2):
  - 单轨: 随机 -18 ~ -14 dB LUFS
  - 使用 pyloudnorm 进行 LUFS 归一化

7 种效果器 (论文 Section 4.2, 不含 reverb):
  EQ → Multiband Compressor → Stereo Imager → Gain → Distortion → Delay → Limiter
"""

import os
import json
import random
import argparse
import gc
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from .config import (
    MUSDB18_ROOT, MEDLEYDB_ROOT,
    FXNORM_MUSDB_DIR, FXNORM_MEDLEYDB_DIR,
    FXNORM_MUSDB_NPY_DIR,
    SAMPLE_RATE, SEGMENT_DURATION, SEGMENT_SAMPLES,
    FX_CHAIN_ORDER, FX_PROB, NUM_SAMPLES_PER_INSTRUMENT,
    MUSDB_INSTRUMENTS, MEDLEYDB_INSTRUMENTS, DATASETS,
    LOUDNESS_RANGE, TRIPLETS_DIR, ensure_dirs,
)


# ===================== LUFS 响度归一化 =====================

def setup_meter(sample_rate: int):
    """初始化 pyloudnorm Meter"""
    import pyloudnorm as pyln
    meter = pyln.Meter(sample_rate)
    return meter


def loudness_normalize_numpy(audio_np: np.ndarray, meter, target_lufs: float = None) -> np.ndarray:
    """
    使用 pyloudnorm 对 numpy 音频进行 LUFS 归一化。

    Args:
        audio_np: (channels, samples) numpy array
        meter: pyloudnorm.Meter 实例
        target_lufs: 目标 LUFS 值，None 则随机从 LOUDNESS_RANGE 采样

    Returns:
        normalized: (channels, samples) numpy array
    """
    if target_lufs is None:
        # 论文: random LUFS in [-18, -14] dB
        target_lufs = LOUDNESS_RANGE[0] + (LOUDNESS_RANGE[1] - LOUDNESS_RANGE[0]) * random.random()

    # pyloudnorm 期望 (samples, channels) 格式
    audio_perm = audio_np.T  # (samples, channels)

    # 计算当前响度
    try:
        loudness = meter.integrated_loudness(audio_perm)
    except Exception:
        loudness = -70.0

    # 处理无效 loudness
    if np.isinf(loudness) or loudness < -70:
        loudness = -70.0

    # 计算增益并应用
    delta_loudness = target_lufs - loudness
    gain = 10.0 ** (delta_loudness / 20.0)

    normalized = audio_np * gain

    # 防止 clipping
    max_val = np.abs(normalized).max()
    if max_val > 0.99:
        normalized = normalized / max_val * 0.99

    return normalized.astype(np.float32)


# ===================== 数据加载 =====================

def check_fxnorm_available(fxnorm_dir: str) -> str:
    """
    检查 FxNorm 预处理数据是否可用，返回数据格式。

    Returns:
        "wav": 有 wav 文件
        "npy": 有 npy 文件 (musdb18_fxnorm_npy 格式)
        None: 不可用
    """
    if not os.path.exists(fxnorm_dir):
        return None
    # 检查是否有 npy 文件 (优先)
    for root, dirs, files in os.walk(fxnorm_dir):
        for f in files:
            if f.endswith('.npy'):
                return "npy"
    # 检查是否有 wav 文件
    for root, dirs, files in os.walk(fxnorm_dir):
        for f in files:
            if f.endswith('.wav'):
                return "wav"
    return None


def extract_segments_from_fxnorm_npy(
    npy_dir: str,
    instruments: list,
    split: str = "train",
    segment_samples: int = SEGMENT_SAMPLES,
    max_segments_per_track: int = 3,
) -> tuple:
    """
    从 FxNorm 预处理后的 npy 文件提取片段。

    目录结构:
      npy_dir/
        {split}/
          track_name/
            vocals.npy    # shape: (samples, 2), float32, stereo 44100Hz
            drums.npy
            bass.npy
            other.npy

    Args:
        npy_dir: FxNorm npy root
        instruments: 要提取的乐器列表
        split: "train" or "test"
        segment_samples: 片段长度 (samples)
        max_segments_per_track: 每个 track 最多提取的片段数

    Returns:
        dict: {instrument: [segment_array, ...]}  segment_array shape: (2, samples)
        dict: {instrument: [track_name, ...]}
    """
    segments = {inst: [] for inst in instruments}
    track_names = {inst: [] for inst in instruments}

    split_dir = os.path.join(npy_dir, split)
    if not os.path.exists(split_dir):
        print(f"WARNING: Split dir not found: {split_dir}")
        return segments, track_names

    track_dirs = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    print(f"Found {len(track_dirs)} tracks in {split_dir}")

    for track_dir in tqdm(track_dirs, desc=f"Extracting npy segments ({split})"):
        track_path = os.path.join(split_dir, track_dir)

        for inst in instruments:
            npy_path = os.path.join(track_path, f"{inst}.npy")
            if not os.path.exists(npy_path):
                continue

            try:
                # npy shape: (samples, 2), float32
                audio = np.load(npy_path)

                # 转为 (channels, samples) = (2, samples)
                if audio.ndim == 1:
                    audio = np.stack([audio, audio], axis=0)
                else:
                    audio = audio.T  # (2, samples)

                # 确保 stereo
                if audio.shape[0] == 1:
                    audio = np.repeat(audio, 2, axis=0)

                # 检查有效音频
                if np.abs(audio).max() < 1e-4:
                    continue

                # 提取片段
                _extract_segments(audio, track_dir, inst, segment_samples,
                                  max_segments_per_track, segments, track_names)

            except Exception as e:
                print(f"  Error loading {npy_path}: {e}")

    for inst in instruments:
        n_tracks = len(set(track_names[inst]))
        print(f"  {inst}: {len(segments[inst])} segments from {n_tracks} tracks")

    return segments, track_names


def extract_segments_from_fxnorm(
    fxnorm_dir: str,
    instruments: list,
    segment_samples: int = SEGMENT_SAMPLES,
    max_segments_per_track: int = 3,
) -> tuple:
    """
    从 FxNorm 预处理后的 stems 提取片段 (wav 格式)。

    支持多种目录结构:
      结构 A (按 track):
        fxnorm_dir/
          track_name/
            instrument.wav

      结构 B (按 instrument):
        fxnorm_dir/
          instrument/
            track_name.wav

    Args:
        fxnorm_dir: FxNorm stems 目录
        instruments: 要提取的乐器列表
        segment_samples: 片段长度 (samples)
        max_segments_per_track: 每个 track 最多提取的片段数

    Returns:
        dict: {instrument: [segment_array, ...]}
        dict: {instrument: [track_name, ...]}
    """
    segments = {inst: [] for inst in instruments}
    track_names = {inst: [] for inst in instruments}

    # 检测目录结构
    subdirs = sorted([
        d for d in os.listdir(fxnorm_dir)
        if os.path.isdir(os.path.join(fxnorm_dir, d))
    ])

    if not subdirs:
        print(f"WARNING: No subdirectories in {fxnorm_dir}")
        return segments, track_names

    # 检测是按 track 还是按 instrument 组织
    first_subdir = os.path.join(fxnorm_dir, subdirs[0])
    first_subdir_contents = os.listdir(first_subdir)

    # 如果第一个子目录包含 wav 文件，则是 track 结构
    is_track_structure = any(f.endswith('.wav') for f in first_subdir_contents)

    if is_track_structure:
        # 结构 A: track_name/instrument.wav
        print(f"Detected structure: track/instrument.wav")
        track_dirs = subdirs

        for track_dir in tqdm(track_dirs, desc="Extracting FxNorm segments"):
            track_path = os.path.join(fxnorm_dir, track_dir)

            for inst in instruments:
                # 尝试多种文件名格式
                possible_names = [f"{inst}.wav", f"{inst.lower()}.wav", f"{inst.replace('_', ' ')}.wav"]
                wav_path = None
                for name in possible_names:
                    candidate = os.path.join(track_path, name)
                    if os.path.exists(candidate):
                        wav_path = candidate
                        break

                if wav_path is None:
                    continue

                _load_and_extract_segments(
                    wav_path, track_dir, inst, segment_samples,
                    max_segments_per_track, segments, track_names
                )
    else:
        # 结构 B: instrument/track_name.wav
        print(f"Detected structure: instrument/track.wav")

        for inst in instruments:
            inst_dir = os.path.join(fxnorm_dir, inst)
            if not os.path.isdir(inst_dir):
                # 尝试小写
                inst_dir = os.path.join(fxnorm_dir, inst.lower())
            if not os.path.isdir(inst_dir):
                print(f"  WARNING: Directory not found for {inst}")
                continue

            wav_files = sorted([f for f in os.listdir(inst_dir) if f.endswith('.wav')])

            for wav_file in tqdm(wav_files, desc=f"Extracting {inst} segments"):
                wav_path = os.path.join(inst_dir, wav_file)
                track_name = wav_file.replace('.wav', '')

                _load_and_extract_segments(
                    wav_path, track_name, inst, segment_samples,
                    max_segments_per_track, segments, track_names
                )

    for inst in instruments:
        n_tracks = len(set(track_names[inst]))
        print(f"  {inst}: {len(segments[inst])} segments from {n_tracks} tracks")

    return segments, track_names


def _load_and_extract_segments(
    wav_path: str,
    track_name: str,
    inst: str,
    segment_samples: int,
    max_segments_per_track: int,
    segments: dict,
    track_names: dict,
):
    """从单个 wav 文件加载并提取片段"""
    try:
        audio, sr = sf.read(wav_path)

        # 转为 (channels, samples)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, 0)
        else:
            audio = audio.T

        # 确保 stereo
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)

        # 重采样到目标采样率 (如果不同)
        if sr != SAMPLE_RATE:
            import torchaudio
            audio_t = torch.from_numpy(audio).float()
            audio_t = torchaudio.functional.resample(audio_t, sr, SAMPLE_RATE)
            audio = audio_t.numpy()

        # 检查是否有有效音频
        if np.abs(audio).max() < 1e-4:
            return

        # 提取片段
        _extract_segments(audio, track_name, inst, segment_samples,
                          max_segments_per_track, segments, track_names)

    except Exception as e:
        print(f"  Error processing {wav_path}: {e}")


def extract_segments_from_musdb(
    musdb_root: str,
    instruments: list = None,
    split: str = "test",
    segment_samples: int = SEGMENT_SAMPLES,
    target_sr: int = SAMPLE_RATE,
    max_segments_per_track: int = 3,
) -> tuple:
    """
    从 MUSDB18 原始 stems 逐 track 提取乐器片段 (回退方案)。

    ⚠️ 注意: 论文原版使用 FxNorm 预处理后的 stems，直接从原始 MUSDB 提取
    会导致干音仍带有录制时的固有效果，影响评估准确性。

    Args:
        musdb_root: MUSDB18 根目录
        instruments: 要提取的乐器列表 (None = 全部 MUSDB_INSTRUMENTS)
        split: "train" or "test"
        segment_samples: 片段长度 (samples)
        target_sr: 目标采样率
        max_segments_per_track: 每个 track 最多提取的片段数

    Returns:
        dict: {instrument: [segment_array, ...]}
        dict: {instrument: [track_name, ...]}
    """
    import stempeg

    if instruments is None:
        instruments = MUSDB_INSTRUMENTS

    stem_dir = os.path.join(musdb_root, split)
    stem_files = sorted([f for f in os.listdir(stem_dir) if f.endswith(".stem.mp4")])

    print(f"⚠️ 使用原始 MUSDB18 stems (未经 FxNorm 预处理)")
    print(f"Found {len(stem_files)} tracks in {stem_dir}")

    # MUSDB stem index mapping
    stem_indices = {
        "drums": 1,
        "bass": 2,
        "other": 3,
        "vocals": 4,
    }

    segments = {inst: [] for inst in instruments}
    track_names = {inst: [] for inst in instruments}

    for stem_file in tqdm(stem_files, desc=f"Extracting segments from MUSDB18 {split}"):
        filepath = os.path.join(stem_dir, stem_file)
        track_name = stem_file.replace(".stem.mp4", "")

        try:
            # 读取并重采样到目标采样率
            stems, sr = stempeg.read_stems(filepath, sample_rate=target_sr)
            # stems shape: (num_stems, samples, channels)

            for inst in instruments:
                if inst not in stem_indices:
                    continue

                stem_idx = stem_indices[inst]
                stem_audio = stems[stem_idx]  # (samples, channels)

                # 转为 (channels, samples)
                if stem_audio.ndim == 1:
                    stem_audio = np.expand_dims(stem_audio, 0)
                else:
                    stem_audio = stem_audio.T  # (channels, samples)

                # 确保 stereo
                if stem_audio.shape[0] == 1:
                    stem_audio = np.repeat(stem_audio, 2, axis=0)

                # 检查是否有有效音频 (能量过低的跳过)
                if np.abs(stem_audio).max() < 1e-4:
                    continue

                # 提取片段
                _extract_segments(stem_audio, track_name, inst, segment_samples,
                                  max_segments_per_track, segments, track_names)

            # 释放内存
            del stems
            gc.collect()

        except Exception as e:
            print(f"  Error processing {stem_file}: {e}")
            continue

    for inst in instruments:
        n_tracks = len(set(track_names[inst]))
        print(f"  {inst}: {len(segments[inst])} segments from {n_tracks} tracks")

    return segments, track_names


def _extract_segments(audio, track_name, inst, segment_samples, max_segments_per_track,
                      segments, track_names):
    """从单个音频提取片段的公共逻辑"""
    total_samples = audio.shape[1]

    if total_samples < segment_samples:
        # 循环填充
        repeats = int(np.ceil(segment_samples / total_samples))
        padded = np.tile(audio, (1, repeats))[:, :segment_samples]
        if np.sqrt(np.mean(padded ** 2)) < 0.02:
            return  # 低能量，跳过
        segments[inst].append(padded.astype(np.float32))
        track_names[inst].append(track_name)
    else:
        # 均匀提取多个片段
        n_possible = total_samples - segment_samples
        n_segs = min(max_segments_per_track, max(1, n_possible // segment_samples + 1))

        for j in range(n_segs):
            if n_segs == 1:
                start = random.randint(0, n_possible)
            else:
                start = int(j * n_possible / n_segs)

            seg = audio[:, start:start + segment_samples].astype(np.float32)

            # 跳过静音/低能量片段 (RMS < 0.02 ≈ -34 dBFS)
            if np.sqrt(np.mean(seg ** 2)) < 0.02:
                continue

            segments[inst].append(seg)
            track_names[inst].append(track_name)


# ===================== 效果链 =====================

def setup_fx_chain(device: str = "cpu"):
    """初始化 FX-Encoder++ 的可微分效果链"""
    from relfx.fx_chain.fx_aug import Random_FX_Chain

    fx_chain = Random_FX_Chain(sample_rate=SAMPLE_RATE, device=device)
    total_params = fx_chain.total_num_param

    print(f"FX Chain: {total_params} total params, {len(FX_CHAIN_ORDER)} effects")
    for fx_name in FX_CHAIN_ORDER:
        n = fx_chain.fx_processors[fx_name].num_params
        print(f"  {fx_name}: {n} params")

    return fx_chain, total_params


def apply_fx(fx_chain, audio_np, nn_param, activate, device="cpu", clip=True):
    """应用效果链到音频"""
    audio_t = torch.from_numpy(audio_np).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out, _, _ = fx_chain(
            audio_t, nn_param=nn_param.to(device),
            activate=activate.to(device),
            processors_order=FX_CHAIN_ORDER,
        )

    out_np = out.squeeze(0).cpu().numpy()

    # 防止 clipping
    if clip:
        max_val = np.abs(out_np).max()
        if max_val > 1.0:
            out_np = out_np / max_val * 0.99


    return out_np


# ===================== 三元组生成 =====================

def generate_triplets(
    segments: dict,
    track_names: dict,
    instruments: list,
    fx_chain,
    total_num_params: int,
    num_samples: int,
    meter,
    device: str = "cpu",
    output_dir: str = TRIPLETS_DIR,
):
    """
    生成 (clean, reference, target) 三元组。

    流程 (对照论文 Section 4.2):
    1. 选择两个片段 content_A, content_B (尽量来自不同 track)
    2. 对两个片段做 LUFS 响度归一化 (论文: -18 ~ -14 dB LUFS)
    3. 随机生成效果链参数 P
    4. 应用相同效果链: target = fx(content_A, P), reference = fx(content_B, P)
    5. 保存: clean = content_A_norm (无效果), target, reference

    Args:
        segments: {instrument: [segment_array, ...]}
        track_names: {instrument: [track_name, ...]}
        instruments: 要生成的乐器列表
        fx_chain: 效果链对象
        total_num_params: 总参数数
        num_samples: 每个乐器生成的样本数
        meter: pyloudnorm Meter
        device: 计算设备
        output_dir: 输出目录
    """
    ensure_dirs()
    metadata = {}

    for inst in instruments:
        segs = segments.get(inst, [])
        names = track_names.get(inst, [])

        if len(segs) < 2:
            print(f"WARNING: {inst} has {len(segs)} segments, need >= 2. Skipping.")
            continue

        print(f"\n--- {inst}: {len(segs)} segments ---")

        inst_dir = os.path.join(output_dir, inst)
        os.makedirs(inst_dir, exist_ok=True)

        inst_meta = []

        for i in tqdm(range(num_samples), desc=f"Generating {inst} triplets"):
            # 选择两个片段 (尽量来自不同 track)
            idx_a = random.randint(0, len(segs) - 1)

            # 尝试找不同 track 的片段
            attempts = 0
            idx_b = random.randint(0, len(segs) - 1)
            while attempts < 50 and names[idx_b] == names[idx_a]:
                idx_b = random.randint(0, len(segs) - 1)
                attempts += 1

            seg_a = segs[idx_a]  # content_A
            seg_b = segs[idx_b]  # content_B

            # LUFS 响度归一化 (论文: -18 ~ -14 dB LUFS)
            target_lufs_a = LOUDNESS_RANGE[0] + (LOUDNESS_RANGE[1] - LOUDNESS_RANGE[0]) * random.random()
            target_lufs_b = LOUDNESS_RANGE[0] + (LOUDNESS_RANGE[1] - LOUDNESS_RANGE[0]) * random.random()
            seg_a_norm = loudness_normalize_numpy(seg_a, meter, target_lufs_a)
            seg_b_norm = loudness_normalize_numpy(seg_b, meter, target_lufs_b)

            # 随机效果链参数 P
            # 论文 Section 4.2 + build_musdb.py: 评估时全部效果器强制开启
            activate = torch.ones(1, len(fx_chain.fx_indices))

            # 应用效果链，带 clip 重试 (避免硬 clip 破坏信号)
            max_retries = 10
            success = False
            for retry in range(max_retries):
                nn_param = torch.rand(1, total_num_params)
                try:
                    target = apply_fx(fx_chain, seg_a_norm, nn_param, activate, device, clip=False)
                    reference = apply_fx(fx_chain, seg_b_norm, nn_param, activate, device, clip=False)
                except Exception as e:
                    if retry == 0:
                        print(f"  Error at sample {i}, retry {retry}: {e}")
                    continue

                # 检查是否 clip (peak > 1.0)
                peak_t = np.abs(target).max()
                peak_r = np.abs(reference).max()
                if peak_t <= 1.0 and peak_r <= 1.0:
                    success = True
                    break
                # clip 了，重新采样参数重试

            if not success:
                # 全部重试失败，用最后一次结果 + 温和缩放
                if np.abs(target).max() > 1.0:
                    target = target / np.abs(target).max() * 0.99
                if np.abs(reference).max() > 1.0:
                    reference = reference / np.abs(reference).max() * 0.99

            # 保存
            sample_dir = os.path.join(inst_dir, f"{i:04d}")
            os.makedirs(sample_dir, exist_ok=True)

            # clean = content_A 的干声 (无效果)
            sf.write(os.path.join(sample_dir, "clean.wav"), seg_a_norm.T, SAMPLE_RATE)
            # target = fx(content_A, P)
            sf.write(os.path.join(sample_dir, "target.wav"), target.T, SAMPLE_RATE)
            # reference = fx(content_B, P) - 与 target 相同效果，不同内容
            sf.write(os.path.join(sample_dir, "reference.wav"), reference.T, SAMPLE_RATE)
            # clean_ref = content_B 的干声 (供某些模型使用)
            sf.write(os.path.join(sample_dir, "clean_ref.wav"), seg_b_norm.T, SAMPLE_RATE)

            param_info = {
                "nn_param": nn_param.squeeze(0).tolist(),
                "activate": activate.squeeze(0).tolist(),
                "track_a": names[idx_a],
                "track_b": names[idx_b],
                "fx_chain_order": FX_CHAIN_ORDER,
                "target_lufs_a": target_lufs_a,
                "target_lufs_b": target_lufs_b,
            }
            with open(os.path.join(sample_dir, "params.json"), "w") as f:
                json.dump(param_info, f, indent=2)

            inst_meta.append({
                "sample_id": i,
                "track_a": names[idx_a],
                "track_b": names[idx_b],
            })

        metadata[inst] = {"num_samples": len(inst_meta), "samples": inst_meta}

    # 保存元数据
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ 三元组生成完成: {output_dir}")
    print(f"   采样率: {SAMPLE_RATE}Hz, 时长: {SEGMENT_DURATION}s, 样本数: {SEGMENT_SAMPLES}")
    print(f"   响度归一化: LUFS {LOUDNESS_RANGE[0]} ~ {LOUDNESS_RANGE[1]} dB")
    for inst, info in metadata.items():
        print(f"   {inst}: {info['num_samples']} samples")

    return metadata


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description="08 Parameter Matching — 数据准备")

    # 数据集选择
    parser.add_argument("--dataset", type=str, default="musdb18",
                        choices=["musdb18", "medleydb", "all"],
                        help="数据集: musdb18, medleydb, 或 all (两个都生成)")

    # 路径配置
    parser.add_argument("--fxnorm-musdb-dir", type=str, default=FXNORM_MUSDB_DIR,
                        help="MUSDB18 FxNorm 预处理后的 stems 目录 (wav 格式)")
    parser.add_argument("--fxnorm-musdb-npy-dir", type=str, default=None,
                        help="MUSDB18 FxNorm npy directory")
    parser.add_argument("--fxnorm-medleydb-dir", type=str, default=FXNORM_MEDLEYDB_DIR,
                        help="MedleyDB FxNorm 预处理后的 stems 目录")
    parser.add_argument("--musdb-root", type=str, default=MUSDB18_ROOT,
                        help="MUSDB18 原始数据目录 (回退方案)")
    parser.add_argument("--medleydb-root", type=str, default=MEDLEYDB_ROOT,
                        help="MedleyDB 原始数据目录 (回退方案)")

    # 生成配置
    parser.add_argument("--split", type=str, default="test",
                        help="MUSDB18 split (train/test/all). all=两者合并")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES_PER_INSTRUMENT,
                        help="每个乐器生成的样本数")
    parser.add_argument("--output-dir", type=str, default=TRIPLETS_DIR,
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--max-segs-per-track", type=int, default=3,
                        help="每个 track 最多提取的片段数")
    parser.add_argument("--force-raw", action="store_true",
                        help="强制使用原始数据 (跳过 FxNorm 检查)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("08 Parameter Matching — 数据准备")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"采样率: {SAMPLE_RATE}Hz")
    print(f"片段时长: {SEGMENT_DURATION}s ({SEGMENT_SAMPLES} samples)")
    print(f"响度归一化: LUFS {LOUDNESS_RANGE[0]} ~ {LOUDNESS_RANGE[1]} dB")
    print(f"Samples/instrument: {args.num_samples}")
    print(f"Output: {args.output_dir}")
    print()

    # 确定要处理的数据集
    datasets_to_process = []
    if args.dataset == "all":
        datasets_to_process = ["musdb18", "medleydb"]
    else:
        datasets_to_process = [args.dataset]

    # 初始化 LUFS Meter
    print("Step 0: 初始化 LUFS Meter...")
    meter = setup_meter(SAMPLE_RATE)

    # 初始化效果链 (所有数据集共用)
    print("\nStep 1: 初始化效果链...")
    fx_chain, total_params = setup_fx_chain(device=args.device)

    # 处理每个数据集
    for dataset_name in datasets_to_process:
        print(f"\n{'=' * 60}")
        print(f"处理数据集: {dataset_name.upper()}")
        print(f"{'=' * 60}")

        dataset_config = DATASETS[dataset_name]
        instruments = dataset_config["instruments"]
        fxnorm_dir = args.fxnorm_musdb_dir if dataset_name == "musdb18" else args.fxnorm_medleydb_dir
        raw_dir = args.musdb_root if dataset_name == "musdb18" else args.medleydb_root

        # 输出目录按数据集分开
        output_dir = os.path.join(args.output_dir, dataset_name)

        print(f"乐器: {instruments}")
        print(f"FxNorm dir: {fxnorm_dir}")
        print(f"Output dir: {output_dir}")

        # 提取片段: npy 优先 → wav FxNorm → 原始 MUSDB 回退
        segments = None
        track_names = None

        if not args.force_raw:
            # 优先检查 npy 目录 (--fxnorm-musdb-npy-dir 或 config 中的 FXNORM_MUSDB_NPY_DIR)
            npy_dir = None
            if dataset_name == "musdb18" and args.fxnorm_musdb_npy_dir:
                npy_dir = args.fxnorm_musdb_npy_dir
            elif dataset_name == "musdb18":
                # 自动使用 config 中的 FXNORM_MUSDB_NPY_DIR
                if os.path.exists(FXNORM_MUSDB_NPY_DIR):
                    npy_dir = FXNORM_MUSDB_NPY_DIR
                else:
                    # 回退: 检查 fxnorm_dir 是否是 npy 格式
                    fmt = check_fxnorm_available(fxnorm_dir)
                    if fmt == "npy":
                        npy_dir = fxnorm_dir

            if npy_dir is not None:
                fmt = check_fxnorm_available(npy_dir)
                if fmt == "npy":
                    splits_to_use = ["train", "test"] if args.split == "all" else [args.split]
                    print(f"\nStep 2: 从 FxNorm npy 提取片段 ({npy_dir}, splits={splits_to_use})...")
                    segments = None
                    track_names = None
                    for sp in splits_to_use:
                        seg_sp, tn_sp = extract_segments_from_fxnorm_npy(
                            npy_dir,
                            instruments=instruments,
                            split=sp,
                            max_segments_per_track=args.max_segs_per_track,
                        )
                        if segments is None:
                            segments = seg_sp
                            track_names = tn_sp
                        else:
                            for inst in instruments:
                                segments[inst].extend(seg_sp[inst])
                                track_names[inst].extend(tn_sp[inst])

            # npy 没找到，尝试 wav FxNorm
            if segments is None:
                fmt = check_fxnorm_available(fxnorm_dir)
                if fmt == "wav":
                    print(f"\nStep 2: 从 FxNorm wav stems 提取片段...")
                    segments, track_names = extract_segments_from_fxnorm(
                        fxnorm_dir,
                        instruments=instruments,
                        max_segments_per_track=args.max_segs_per_track,
                    )

        # 回退: 原始 MUSDB18 stems
        if segments is None:
            print(f"\n⚠️ FxNorm stems 不可用")

            if dataset_name == "musdb18":
                splits_to_use = ["train", "test"] if args.split == "all" else [args.split]
                print(f"Step 2: 从 MUSDB18 原始 stems 提取片段 (splits={splits_to_use})...")
                segments = None
                track_names = None
                for sp in splits_to_use:
                    seg_sp, tn_sp = extract_segments_from_musdb(
                        raw_dir,
                        instruments=instruments,
                        split=sp,
                        max_segments_per_track=args.max_segs_per_track,
                    )
                    if segments is None:
                        segments = seg_sp
                        track_names = tn_sp
                    else:
                        for inst in instruments:
                            segments[inst].extend(seg_sp[inst])
                            track_names[inst].extend(tn_sp[inst])
            else:
                print(f"❌ MedleyDB 暂不支持从原始数据提取，请提供 FxNorm 预处理数据")
                print(f"   预期目录: {fxnorm_dir}")
                continue

        # 检查是否有足够的片段
        total_segments = sum(len(segs) for segs in segments.values())
        if total_segments < 2:
            print(f"❌ {dataset_name} 片段不足，跳过")
            continue

        # 生成三元组
        print(f"\nStep 3: 生成 {dataset_name} 三元组...")
        generate_triplets(
            segments, track_names,
            instruments=instruments,
            fx_chain=fx_chain,
            total_num_params=total_params,
            num_samples=args.num_samples,
            meter=meter,
            device=args.device,
            output_dir=output_dir,
        )

    print(f"\n{'=' * 60}")
    print("数据准备完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
