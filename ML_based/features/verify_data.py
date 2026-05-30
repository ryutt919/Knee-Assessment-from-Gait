"""
데이터 산출물 검증 — stride 수·길이 분포·결측률 요약 출력
실행: python features/verify_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
ML = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"


def verify_scalar_stride():
    path = PROCESSED / "stride_level_peaks.parquet"
    df = pd.read_parquet(path)
    print("=" * 60)
    print(f"[stride_level_peaks] {df.shape}")
    print(f"  그룹: {df['group'].value_counts().to_dict()}")
    print(f"  속도: {df['speed'].value_counts().to_dict()}")
    strides_per_subj = df.groupby(["subject_id", "speed"]).size()
    print(f"  stride/subject/speed: mean={strides_per_subj.mean():.1f}, "
          f"min={strides_per_subj.min()}, max={strides_per_subj.max()}")
    miss = df.isnull().mean()
    if miss.max() > 0:
        print(f"  결측률 > 0: {miss[miss > 0].to_dict()}")
    else:
        print("  결측률: 0%")


def verify_waveform_norm():
    path = PROCESSED / "waveforms_stride.parquet"
    df = pd.read_parquet(path)
    print("=" * 60)
    print(f"[waveforms_stride (101pt)] {df.shape}")
    print(f"  그룹: {df['group'].value_counts().to_dict()}")
    print(f"  속도: {df['speed'].value_counts().to_dict()}")
    feat_cols = [c for c in df.columns if c[0].isalpha() and "_0" in c]
    miss = df[feat_cols].isnull().mean().max()
    print(f"  파형 결측률 (max): {miss:.4f}")


def verify_raw_cycles():
    path = PROCESSED / "stride_raw_waveforms.parquet"
    if not path.exists():
        print("=" * 60)
        print("[stride_raw_waveforms] 미생성 — features/extract_raw_cycles.py 실행 필요")
        return

    df = pd.read_parquet(path)
    print("=" * 60)
    print(f"[stride_raw_waveforms (raw padded)] {df.shape}")
    print(f"  그룹: {df['group'].value_counts().to_dict()}")
    print(f"  속도: {df['speed'].value_counts().to_dict()}")

    lens = df["cycle_len"]
    print(f"  cycle 길이: mean={lens.mean():.1f}, std={lens.std():.1f}, "
          f"p5={lens.quantile(0.05):.0f}, p95={lens.quantile(0.95):.0f}, max={lens.max()}")

    mask_cols = [c for c in df.columns if c.startswith("mask_")]
    if mask_cols:
        valid_ratio = df[mask_cols].mean(axis=0).mean()
        print(f"  유효 샘플 비율 (마스크): {valid_ratio:.3f}")

    group_cols = ["subject_id", "speed", "trial_id"]
    if "actual_leg" in df.columns:
        group_cols.append("actual_leg")
    strides_per_trial = df.groupby(group_cols).size()
    print(f"  stride/trial/leg: mean={strides_per_trial.mean():.1f}, "
          f"min={strides_per_trial.min()}, max={strides_per_trial.max()}")
    if "side_basis" in df.columns:
        print(f"  side_basis: {df['side_basis'].value_counts().to_dict()}")


def verify_cycle_waveforms_101():
    path = PROCESSED / "cycle_waveforms_101.parquet"
    if not path.exists():
        print("=" * 60)
        print("[cycle_waveforms_101] 미생성 — features/extract_cycle_waveforms_101.py 실행 필요")
        return

    df = pd.read_parquet(path)
    print("=" * 60)
    print(f"[cycle_waveforms_101 (cycle-level 101pt)] {df.shape}")
    print(f"  그룹: {df['group'].value_counts().to_dict()}")
    print(f"  속도: {df['speed'].value_counts().to_dict()}")
    print(f"  side_basis: {df['side_basis'].value_counts().to_dict()}")
    print(f"  actual_leg: {df['actual_leg'].value_counts().to_dict()}")

    lens = df["cycle_len"]
    print(f"  cycle 길이: mean={lens.mean():.1f}, std={lens.std():.1f}, "
          f"p5={lens.quantile(0.05):.0f}, p95={lens.quantile(0.95):.0f}, max={lens.max()}")

    meta_cols = {
        "subject_id", "group", "speed", "trial_id", "actual_leg", "side_basis",
        "injured_leg", "cycle_idx", "cycle_idx_original", "cycle_len",
        "n_cycles_raw", "n_cycles_trimmed",
    }
    wave_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  파형 컬럼 수: {len(wave_cols)} (기대: 9×101=909)")
    print(f"  파형 결측률 (max): {df[wave_cols].isnull().mean().max():.4f}")

    bad_trim = (df["cycle_idx_original"] < 2).sum()
    print(f"  trim 검증: cycle_idx_original < 2 행={int(bad_trim)}")


def verify_feature_ranking():
    path = PROCESSED / "feature_ranking.csv"
    df = pd.read_csv(path)
    print("=" * 60)
    print(f"[feature_ranking] {df.shape}")
    print(f"  top-5 η² features:")
    top = df.nlargest(5, "eta2")[["feature", "speed", "eta2", "p_adj"]]
    print(top.to_string(index=False))


def run_verification(_cfg=None) -> None:
    """파이프라인에서 호출하는 wrapper."""
    verify_scalar_stride()
    verify_waveform_norm()
    verify_cycle_waveforms_101()
    verify_raw_cycles()
    verify_feature_ranking()
    print("=" * 60)
    print("[verify] 검증 완료")


if __name__ == "__main__":
    run_verification()
