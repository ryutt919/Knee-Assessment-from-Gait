"""
scripts/make_body_part_parquet.py

MVNX 라벨 기반으로 신체 부위를 CLI 인자로 지정해 parquet을 필터링하는 스크립트.
유지할 segment/joint/sensor 키워드를 인자로 받아 해당 컬럼만 남깁니다.

사용법:
    # dry-run (저장 없이 컬럼 변화 확인)
    python scripts/make_body_part_parquet.py --segments LeftHand --sensors LeftHand --dry-run

    # 왼손 → Master_Gait_Dataset_LeftHand.parquet
    python scripts/make_body_part_parquet.py --segments LeftHand --sensors LeftHand

    # 오른손 → Master_Gait_Dataset_RightHand.parquet
    python scripts/make_body_part_parquet.py --segments RightHand --sensors RightHand

사용 가능한 세그먼트 키워드 (MVNX label 기준 부분 매칭, 대소문자 무시):
  Pelvis, L5, L3, T12, T8, Neck, Head
  RightShoulder, RightUpperArm, RightForearm, RightHand
  LeftShoulder,  LeftUpperArm,  LeftForearm,  LeftHand
  RightUpperLeg, RightLowerLeg, RightFoot, RightToe
  LeftUpperLeg,  LeftLowerLeg,  LeftFoot,  LeftToe

사용 가능한 조인트 키워드 (예시, 실제 목록은 MVNX 파싱 결과 참조):
  Shoulder, Elbow, Wrist, Hip, Knee, Ankle, BallFoot, ...

사용 가능한 센서 키워드 (실제 목록은 MVNX 파싱 결과 참조):
  Pelvis, Sternum, Head, RightHand, LeftHand, RightUpperLeg, ...

키워드 미입력 시 해당 타입 컬럼 전체 유지.
"""
import re
import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.embedding_utils import META_COLS

DEFAULT_MVNX = ROOT / "data" / "ACLD" / "ACLD1" / "Gait" / "Fast" / "FAST-001.mvnx"
DEFAULT_INPUT = ROOT / "data" / "processed" / "Master_Gait_Dataset.parquet"


# ── 1. CLI 인자 ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="신체 부위별 parquet 필터링 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mvnx", type=Path, default=DEFAULT_MVNX,
        help=f"라벨 파싱용 MVNX 파일 (기본: {DEFAULT_MVNX.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"원본 parquet (기본: {DEFAULT_INPUT.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="출력 parquet 경로. 미지정 시 --segments 키워드로 자동 생성",
    )
    parser.add_argument(
        "--segments", nargs="*", default=None,
        help="KEEP할 segment label 키워드 (부분 매칭, 미입력 시 전체 유지)",
    )
    parser.add_argument(
        "--joints", nargs="*", default=None,
        help="KEEP할 joint label 키워드 (부분 매칭, 미입력 시 전체 유지)",
    )
    parser.add_argument(
        "--sensors", nargs="*", default=None,
        help="KEEP할 sensor label 키워드 (부분 매칭, 미입력 시 전체 유지)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 저장 없이 컬럼 변화만 출력",
    )
    return parser.parse_args()


# ── 2. MVNX 헤더 파싱 ─────────────────────────────────────────────────────
def parse_mvnx_labels(mvnx_path: Path):
    """MVNX 파일 헤더에서 segment/joint/sensor 라벨 리스트를 반환."""
    segment_labels, joint_labels, sensor_labels = [], [], []
    with open(mvnx_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r'<segment\s[^>]*label="([^"]+)"', line)
            if m:
                segment_labels.append(m.group(1))
            m = re.search(r'<joint\s[^>]*label="([^"]+)"', line)
            if m:
                joint_labels.append(m.group(1))
            m = re.search(r'<sensor\s[^>]*label="([^"]+)"', line)
            if m:
                sensor_labels.append(m.group(1))
            if "<frame " in line:
                break
    return segment_labels, joint_labels, sensor_labels


# ── 3. 키워드 → KEEP 인덱스 ────────────────────────────────────────────────
def resolve_keep_indices(labels: list, keywords) -> set:
    """
    keywords가 None 또는 빈 리스트이면 빈 집합 반환 (전체 DROP).
    있을 경우 case-insensitive 부분 매칭으로 KEEP 인덱스 도출.
    """
    if not keywords:
        return set()
    kws_lower = [kw.lower() for kw in keywords]
    return {i for i, label in enumerate(labels) if any(kw in label.lower() for kw in kws_lower)}


# 센서 컬럼 prefix (MVNX 라벨 없이 prefix 패턴으로 직접 처리)
SENSOR_PREFIXES = ("sensorFreeAcceleration_", "sensorMagneticField_", "sensorOrientation_")


# ── 4. KEEP 컬럼 화이트리스트 생성 ───────────────────────────────────────
def build_keep_columns(
    keep_seg: set,
    keep_jt: set,
    df_cols: list,
    drop_sensors: bool,
) -> list:
    """명시된 segment/joint/sensor에 해당하는 컬럼 + META_COLS만 화이트리스트로 반환."""
    cols = list(META_COLS)

    for seg in keep_seg:
        cols += [f"position_{seg*3+ax}"           for ax in range(3)]
        cols += [f"orientation_{seg*4+q}"          for q  in range(4)]
        cols += [f"velocity_{seg*3+ax}"            for ax in range(3)]
        cols += [f"acceleration_{seg*3+ax}"        for ax in range(3)]
        cols += [f"angularVelocity_{seg*3+ax}"     for ax in range(3)]
        cols += [f"angularAcceleration_{seg*3+ax}" for ax in range(3)]

    for ji in keep_jt:
        cols += [f"jointAngle_{ji*3+ax}" for ax in range(3)]

    if not drop_sensors:
        cols += [c for c in df_cols if c.startswith(SENSOR_PREFIXES)]

    # 실제 존재하는 컬럼만
    existing = set(df_cols)
    return [c for c in cols if c in existing]


# ── 5. 출력 경로 결정 ─────────────────────────────────────────────────────
def resolve_output_path(args) -> Path:
    if args.output:
        return args.output
    parts = "_".join(args.segments) if args.segments else "subset"
    return ROOT / "data" / "processed" / f"Master_Gait_Dataset_{parts}.parquet"


# ── 6. 필터 적용 ──────────────────────────────────────────────────────────
def apply_filter(df_src: pd.DataFrame, keep_cols: list, output_path: Path, dry_run: bool):
    df_out = df_src[keep_cols]

    print(f"\n=== 컬럼 필터 적용 ===")
    print(f"  원본 shape   : {df_src.shape}")
    print(f"  DROP 컬럼 수 : {df_src.shape[1] - df_out.shape[1]}")
    print(f"  결과 shape   : {df_out.shape}")

    if dry_run:
        print(f"\n[DRY RUN] 저장 생략")
        remaining = [c for c in df_out.columns if c not in META_COLS]
        print(f"[DRY RUN] 유지되는 피처 컬럼 ({len(remaining)}개):")
        for c in remaining:
            print(f"    {c}")
        return df_out

    df_out.to_parquet(output_path, index=False)
    print(f"\n저장 완료: {output_path}")
    return df_out


# ── 7. 검증 ───────────────────────────────────────────────────────────────
def validate_output(df_out: pd.DataFrame, df_src: pd.DataFrame):
    print("\n=== 검증 ===")
    all_pass = True

    # 1) 메타 컬럼 완전성
    missing_meta = [c for c in META_COLS if c not in df_out.columns]
    if missing_meta:
        print(f"  [FAIL] 메타 컬럼 누락: {missing_meta}")
        all_pass = False
    else:
        print(f"  [PASS] 메타 컬럼 {len(META_COLS)}개 모두 존재")

    # 2) 피처 컬럼 명명 패턴 ({prefix}_{숫자})
    feature_cols = [c for c in df_out.columns if c not in META_COLS]
    bad_cols = [c for c in feature_cols if not re.fullmatch(r"[a-zA-Z]+_\d+", c)]
    if bad_cols:
        print(f"  [FAIL] 컬럼명 패턴 불일치 ({len(bad_cols)}개): {bad_cols[:5]} ...")
        all_pass = False
    else:
        print(f"  [PASS] 피처 컬럼 {len(feature_cols)}개 명명 패턴 정상")

    # 3) 공통 컬럼 dtype 일치
    common = [c for c in df_out.columns if c in df_src.columns]
    dtype_mismatch = [
        c for c in common if df_out[c].dtype != df_src[c].dtype
    ]
    if dtype_mismatch:
        print(f"  [FAIL] dtype 불일치 컬럼 ({len(dtype_mismatch)}개): {dtype_mismatch[:5]}")
        for c in dtype_mismatch[:3]:
            print(f"    {c}: 원본={df_src[c].dtype}, 결과={df_out[c].dtype}")
        all_pass = False
    else:
        print(f"  [PASS] 공통 컬럼 {len(common)}개 dtype 일치")

    if not all_pass:
        sys.exit(1)


# ── 8. main ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # MVNX 라벨 파싱
    print(f"=== MVNX 라벨 파싱 ===")
    print(f"  파일: {args.mvnx}")
    seg_labels, jt_labels, sn_labels = parse_mvnx_labels(args.mvnx)
    print(f"  세그먼트 {len(seg_labels)}개 / 조인트 {len(jt_labels)}개 / 센서 {len(sn_labels)}개")

    # KEEP 인덱스 결정
    print(f"\n=== KEEP 인덱스 결정 ===")
    keep_seg = resolve_keep_indices(seg_labels, args.segments)
    keep_jt  = resolve_keep_indices(jt_labels,  args.joints)
    drop_sensors = args.sensors is None  # --sensors 미지정 시 전체 drop

    print(f"  KEEP segments ({len(keep_seg)}개): {[seg_labels[i] for i in sorted(keep_seg)]}")
    print(f"  KEEP joints   ({len(keep_jt)}개): {[jt_labels[i]  for i in sorted(keep_jt)]}")
    print(f"  KEEP sensors  : {'전체 DROP (미지정)' if drop_sensors else '전체 유지 (--sensors 지정됨)'}")

    # parquet 로딩
    print(f"\n원본 로딩: {args.input}")
    df_src = pd.read_parquet(args.input)

    # KEEP 컬럼 화이트리스트 생성
    keep_cols = build_keep_columns(keep_seg, keep_jt, list(df_src.columns), drop_sensors)

    # 출력 경로
    output_path = resolve_output_path(args)

    # 필터 적용
    df_out = apply_filter(df_src, keep_cols, output_path, args.dry_run)

    # 검증
    validate_output(df_out, df_src)

    if not args.dry_run:
        print(f"\n완료: {output_path.name}")


if __name__ == "__main__":
    main()
