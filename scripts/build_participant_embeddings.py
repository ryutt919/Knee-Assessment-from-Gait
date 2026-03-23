from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import umap  # type: ignore

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


PACE_ORDER = ["slow", "normal", "fast"]
SHEETS = [
    "Joint Angles ZXY",
    "Segment Position",
    "Center of Mass",
    "Sensor Free Acceleration",
    "Segment Velocity",
    "Segment Orientation - Euler",
]

GROUP_MAP = {
    1: "Healthy adults",
    2: "Healthy adolescents",
    3: "ACLD",
    4: "ACLR",
}


@dataclass
class TrialIndexRow:
    group: str
    participant: str
    pace_condition: str
    file_name: str
    file_stem_lower: str
    file_path: Path
    unknown_pace_flag: int


def canonical_group(value: str) -> str:
    key = re.sub(r"\s+", " ", value.strip().lower())
    if key in {"acld"}:
        return "ACLD"
    if key in {"aclr"}:
        return "ACLR"
    if key in {"healthy adults", "healthy_adults"}:
        return "Healthy adults"
    if key in {"healthy adolescents", "healthy_adolescents"}:
        return "Healthy adolescents"
    return value.strip()


def infer_pace(text: str) -> str:
    s = text.lower()
    if "slow" in s:
        return "slow"
    if "normal" in s or "nrml" in s:
        return "normal"
    if "fast" in s:
        return "fast"
    return "unknown"


def sanitize_token(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip()).strip("_").lower()


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(metadata_path)
    meta = raw.rename(
        columns={
            "ID": "participant",
            "Group": "group_code",
            "Sex": "sex",
            "Age": "age",
            "Weight": "weight_kg",
            "Height": "height_m",
            "Injured leg": "injured_leg",
        }
    ).copy()
    meta["participant"] = meta["participant"].astype(str).str.strip()
    meta["group"] = meta["group_code"].map(GROUP_MAP).fillna("Unknown")
    meta["sex"] = meta["sex"].replace({"FeMale": "Female", "Male": "Male"}).fillna("Unknown")
    for col in ("age", "weight_kg", "height_m"):
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta["bmi"] = meta["weight_kg"] / (meta["height_m"] ** 2)
    meta["injury_status"] = np.select(
        [meta["group"].eq("ACLD"), meta["group"].eq("ACLR")],
        ["ACL deficient", "Post-op ACLR"],
        default="Healthy",
    )
    meta["injured_leg"] = meta["injured_leg"].fillna("none").astype(str).str.strip().str.lower()
    keep_cols = [
        "participant",
        "group",
        "sex",
        "age",
        "weight_kg",
        "height_m",
        "bmi",
        "injury_status",
        "injured_leg",
    ]
    return meta[keep_cols]


def collect_trial_index(data_dir: Path, max_trials: int | None = None) -> pd.DataFrame:
    rows: list[TrialIndexRow] = []
    for xlsx_path in sorted(data_dir.rglob("*.xlsx")):
        rel = xlsx_path.relative_to(data_dir)
        parts = rel.parts
        if len(parts) < 3:
            continue
        group = canonical_group(parts[0])
        participant = parts[1].strip()
        pace = infer_pace(str(rel))
        rows.append(
            TrialIndexRow(
                group=group,
                participant=participant,
                pace_condition=pace,
                file_name=xlsx_path.name,
                file_stem_lower=xlsx_path.stem.lower(),
                file_path=xlsx_path,
                unknown_pace_flag=int(pace == "unknown"),
            )
        )
        if max_trials is not None and len(rows) >= max_trials:
            break
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        raise RuntimeError(f"No xlsx trial files found under: {data_dir}")
    return df


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Frame" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Frame"})
    out["Frame"] = pd.to_numeric(out["Frame"], errors="coerce")
    out = out.dropna(subset=["Frame"]).reset_index(drop=True)
    for c in out.columns:
        if c == "Frame":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def decimate_frame(df: pd.DataFrame, decimation_factor: int) -> pd.DataFrame:
    if decimation_factor <= 1:
        return df
    if df.empty:
        return df
    mod = np.mod(df["Frame"].to_numpy(dtype=float), float(decimation_factor))
    mask = np.isclose(mod, 0.0)
    out = df.loc[mask].copy()
    return out


def describe_series(s: pd.Series) -> dict[str, float]:
    x = s.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return {k: float("nan") for k in ["mean", "std", "min", "max", "range", "median", "iqr", "rms", "energy"]}
    q75, q25 = np.percentile(x, [75, 25])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)),
        "median": float(np.median(x)),
        "iqr": float(q75 - q25),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "energy": float(np.mean(np.square(x))),
    }


def right_counterpart_name(name: str) -> str | None:
    candidates = [
        re.sub(r"\bleft\b", "right", name, flags=re.IGNORECASE),
        re.sub(r"\bl\b", "r", name, flags=re.IGNORECASE),
        name.replace("L_", "R_").replace("l_", "r_"),
    ]
    for cand in candidates:
        if cand != name:
            return cand
    return None


def base_left_name(name: str) -> str:
    s = re.sub(r"\bleft\b", "", name, flags=re.IGNORECASE)
    s = re.sub(r"\bl\b", "", s, flags=re.IGNORECASE)
    s = s.replace("L_", "").replace("l_", "")
    return sanitize_token(s)


def extract_sheet_features(df: pd.DataFrame, sheet_name: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    numeric_cols = [c for c in df.columns if c != "Frame"]
    lc_to_col = {c.lower(): c for c in numeric_cols}
    sheet_token = sanitize_token(sheet_name)

    for col in numeric_cols:
        stats = describe_series(df[col])
        col_token = sanitize_token(col)
        for stat_name, value in stats.items():
            feats[f"{sheet_token}__{col_token}__{stat_name}"] = value

    used_pairs: set[tuple[str, str]] = set()
    for col in numeric_cols:
        if "left" not in col.lower() and not re.search(r"\bl\b|^l_", col.lower()):
            continue
        right_name = right_counterpart_name(col)
        if right_name is None:
            continue
        right_col = lc_to_col.get(right_name.lower())
        if right_col is None:
            continue
        pair_key = tuple(sorted([col.lower(), right_col.lower()]))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        left_values = pd.to_numeric(df[col], errors="coerce")
        right_values = pd.to_numeric(df[right_col], errors="coerce")
        mask = left_values.notna() & right_values.notna()
        if not mask.any():
            continue
        diff = left_values[mask] - right_values[mask]
        base = base_left_name(col)
        feats[f"{sheet_token}__asym__{base}__mean_diff"] = float(diff.mean())
        feats[f"{sheet_token}__asym__{base}__abs_mean_diff"] = float(diff.abs().mean())
        feats[f"{sheet_token}__asym__{base}__std_diff"] = float(diff.std(ddof=0))
    return feats


def load_existing_summary(summary_csv: Path) -> pd.DataFrame | None:
    if not summary_csv.exists():
        return None
    summary = pd.read_csv(summary_csv)
    required = {"group", "participant", "pace_condition"}
    if not required.issubset(summary.columns):
        return None
    summary = summary.copy()
    summary["group"] = summary["group"].astype(str).map(canonical_group)
    summary["participant"] = summary["participant"].astype(str).str.strip()
    summary["pace_condition"] = summary["pace_condition"].astype(str).str.lower().str.strip()
    if "source_file" in summary.columns:
        summary["file_stem_lower"] = (
            summary["source_file"].astype(str).apply(lambda x: Path(x).stem.lower())
        )
    return summary


def _process_trial_row(args: tuple[dict[str, object], float, int]) -> dict[str, object]:
    row, source_hz, decimation_factor = args
    record: dict[str, object] = {
        "group": row["group"],
        "participant": row["participant"],
        "pace_condition": row["pace_condition"],
        "file_name": row["file_name"],
        "file_stem_lower": row["file_stem_lower"],
        "file_path": str(row["file_path"]),
        "unknown_pace_flag": row["unknown_pace_flag"],
    }
    valid_sheets = 0
    missing_sheets = 0
    max_count_original = 0
    max_count_resampled = 0
    for sheet in SHEETS:
        try:
            sdf = pd.read_excel(row["file_path"], sheet_name=sheet)
            sdf = to_numeric_frame(sdf)
            if sdf.empty:
                missing_sheets += 1
                continue
            max_count_original = max(max_count_original, int(len(sdf)))
            sdf = decimate_frame(sdf, decimation_factor=decimation_factor)
            if sdf.empty:
                missing_sheets += 1
                continue
            valid_sheets += 1
            max_count_resampled = max(max_count_resampled, int(len(sdf)))
            record.update(extract_sheet_features(sdf, sheet))
        except Exception:
            missing_sheets += 1
            continue

    trial_frames_original = int(max_count_original)
    trial_frames_resampled = int(max_count_resampled)
    record["trial_n_frames_original"] = trial_frames_original
    record["trial_n_frames_resampled"] = trial_frames_resampled
    record["trial_n_frames"] = trial_frames_resampled
    # duration is intentionally kept in original time scale
    record["trial_duration_s"] = float(trial_frames_original / source_hz) if trial_frames_original > 0 else np.nan
    record["valid_sheet_count"] = valid_sheets
    record["missing_sheet_count"] = missing_sheets
    record["source_hz"] = source_hz
    record["decimation_factor"] = decimation_factor
    return record


def build_trial_feature_table(
    trials: pd.DataFrame,
    source_hz: float,
    target_hz: float,
    decimation_factor: int,
    existing_summary: pd.DataFrame | None = None,
    workers: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    input_rows = trials.to_dict(orient="records")
    rows: list[dict[str, object]] = []
    if workers <= 1:
        for i, r in enumerate(input_rows, start=1):
            rows.append(_process_trial_row((r, source_hz, decimation_factor)))
            if verbose and i % 25 == 0:
                print(f"processed trials: {i}/{len(input_rows)}")
    else:
        mapped = [(r, source_hz, decimation_factor) for r in input_rows]
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for i, out_row in enumerate(ex.map(_process_trial_row, mapped), start=1):
                rows.append(out_row)
                if verbose and i % 25 == 0:
                    print(f"processed trials: {i}/{len(input_rows)}")

    out = pd.DataFrame(rows)

    if existing_summary is not None and not existing_summary.empty:
        numeric_summary_cols = [
            c
            for c in existing_summary.columns
            if c not in {"group", "participant", "pace_condition", "source_file", "file_stem_lower"}
            and pd.api.types.is_numeric_dtype(existing_summary[c])
        ]
        if "file_stem_lower" in existing_summary.columns:
            merged = out.merge(
                existing_summary[["group", "participant", "pace_condition", "file_stem_lower"] + numeric_summary_cols],
                on=["group", "participant", "pace_condition", "file_stem_lower"],
                how="left",
                suffixes=("", "_summary"),
            )
        else:
            per_pace_summary = (
                existing_summary.groupby(["group", "participant", "pace_condition"], dropna=False)[numeric_summary_cols]
                .mean()
                .reset_index()
            )
            merged = out.merge(
                per_pace_summary,
                on=["group", "participant", "pace_condition"],
                how="left",
                suffixes=("", "_summary"),
            )
        for c in numeric_summary_cols:
            if c in merged.columns:
                merged = merged.rename(columns={c: f"summary__{sanitize_token(c)}"})
        out = merged

    return out


def aggregate_participant_features(trial_features: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["group", "participant", "pace_condition", "file_name", "file_stem_lower", "file_path"]
    numeric_cols = [c for c in trial_features.columns if c not in key_cols and pd.api.types.is_numeric_dtype(trial_features[c])]
    base_index = trial_features[["group", "participant"]].drop_duplicates().reset_index(drop=True)

    pace_tables = []
    for pace in PACE_ORDER:
        pace_df = trial_features[trial_features["pace_condition"] == pace]
        agg = pace_df.groupby(["group", "participant"], dropna=False)[numeric_cols].mean().reset_index()
        rename_map = {c: f"pace_{pace}__{c}" for c in numeric_cols}
        agg = agg.rename(columns=rename_map)
        count_df = (
            pace_df.groupby(["group", "participant"], dropna=False)
            .size()
            .reset_index(name=f"pace_{pace}__trial_count")
        )
        agg = agg.merge(count_df, on=["group", "participant"], how="left")
        pace_tables.append(agg)

    participant = base_index.copy()
    for t in pace_tables:
        participant = participant.merge(t, on=["group", "participant"], how="left")

    for pace in PACE_ORDER:
        cnt_col = f"pace_{pace}__trial_count"
        participant[cnt_col] = participant[cnt_col].fillna(0).astype(int)
        participant[f"missing_pace_{pace}"] = (participant[cnt_col] == 0).astype(int)

    participant["pace_coverage_count"] = participant[[f"pace_{p}__trial_count" for p in PACE_ORDER]].gt(0).sum(axis=1)
    participant["missing_pace_any"] = (participant["pace_coverage_count"] < len(PACE_ORDER)).astype(int)

    across_blocks: dict[str, pd.Series] = {}
    for base_col in numeric_cols:
        pace_cols = [f"pace_{p}__{base_col}" for p in PACE_ORDER if f"pace_{p}__{base_col}" in participant.columns]
        if not pace_cols:
            continue
        across_blocks[f"across_pace__{base_col}__mean"] = participant[pace_cols].mean(axis=1, skipna=True)
        across_blocks[f"across_pace__{base_col}__std"] = participant[pace_cols].std(axis=1, skipna=True, ddof=0)
        across_blocks[f"across_pace__{base_col}__range"] = participant[pace_cols].max(axis=1, skipna=True) - participant[
            pace_cols
        ].min(axis=1, skipna=True)
    if across_blocks:
        participant = pd.concat([participant, pd.DataFrame(across_blocks)], axis=1)
    return participant


def merge_metadata(participant_features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = participant_features.merge(
        metadata,
        on=["participant"],
        how="left",
        suffixes=("", "__meta"),
    )
    if "group__meta" in merged.columns:
        merged["meta_group_mismatch"] = (
            merged["group__meta"].notna() & (merged["group__meta"].astype(str) != merged["group"].astype(str))
        ).astype(int)
    else:
        merged["meta_group_mismatch"] = 0
    return merged


def build_embedding_matrix(participant_table: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    id_cols = ["group", "participant"]
    passthrough_cols = [c for c in ["injury_status", "sex", "injured_leg", "pace_coverage_count"] if c in participant_table.columns]

    feature_df = participant_table.copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    for col in feature_df.columns:
        if col in id_cols:
            continue
        if feature_df[col].dtype == "object":
            feature_df[col] = feature_df[col].astype(str)
    drop_all_nan = [c for c in feature_df.columns if c not in id_cols and feature_df[c].isna().all()]
    feature_df = feature_df.drop(columns=drop_all_nan)

    cat_cols = [c for c in feature_df.columns if c not in id_cols and feature_df[c].dtype == "object"]
    num_cols = [c for c in feature_df.columns if c not in id_cols and c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
    )

    x_pre = preprocessor.fit_transform(feature_df)
    n_samples, n_features = x_pre.shape
    n_components = int(min(32, max(2, n_samples - 1), n_features))
    if n_samples <= 2:
        n_components = int(min(2, n_features))
    pca = PCA(n_components=n_components, random_state=random_state)
    x_emb = pca.fit_transform(x_pre)

    emb_cols = [f"emb_{i + 1:03d}" for i in range(x_emb.shape[1])]
    embeddings = participant_table[id_cols + passthrough_cols].copy()
    for i, col in enumerate(emb_cols):
        embeddings[col] = x_emb[:, i]

    return embeddings, x_emb, {
        "preprocessor": preprocessor,
        "pca": pca,
        "drop_all_nan_columns": drop_all_nan,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def build_umap_table(
    embeddings: pd.DataFrame,
    x_emb: np.ndarray,
    random_state: int,
    n_neighbors: int = 15,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base_cols = [c for c in ["group", "participant", "injury_status", "sex", "pace_coverage_count"] if c in embeddings.columns]
    out = embeddings[base_cols].copy()

    if x_emb.shape[0] < 3 or not HAS_UMAP:
        out["umap_x"] = x_emb[:, 0] if x_emb.shape[1] > 0 else 0.0
        out["umap_y"] = x_emb[:, 1] if x_emb.shape[1] > 1 else 0.0
        out["umap_z"] = x_emb[:, 2] if x_emb.shape[1] > 2 else 0.0
        return out, {
            "method": "pca_fallback",
            "has_umap": bool(HAS_UMAP),
            "reason": "Insufficient samples or UMAP unavailable",
            "umap_2d": None,
            "umap_3d": None,
            "n_neighbors": None,
            "min_dist": None,
        }

    nn = int(max(2, min(n_neighbors, x_emb.shape[0] - 1)))
    umap_2d = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=nn,
        min_dist=0.1,
    )
    umap_3d = umap.UMAP(
        n_components=3,
        random_state=random_state,
        n_neighbors=nn,
        min_dist=0.1,
    )
    z2 = umap_2d.fit_transform(x_emb)
    z3 = umap_3d.fit_transform(x_emb)
    out["umap_x"] = z2[:, 0]
    out["umap_y"] = z2[:, 1]
    out["umap_z"] = z3[:, 2]

    return out, {
        "method": "umap",
        "has_umap": True,
        "reason": None,
        "umap_2d": umap_2d,
        "umap_3d": umap_3d,
        "n_neighbors": nn,
        "min_dist": 0.1,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(
    out_path: Path,
    *,
    data_dir: Path,
    metadata_path: Path,
    summary_path: Path,
    trial_index: pd.DataFrame,
    trial_features: pd.DataFrame,
    participant_table: pd.DataFrame,
    embeddings: pd.DataFrame,
    umap_table: pd.DataFrame,
    model_info: dict[str, object],
    umap_info: dict[str, object],
    random_state: int,
    source_hz: float,
    target_hz: float,
    decimation_factor: int,
) -> None:
    used_sheets = [s for s in SHEETS]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path),
        "random_state": random_state,
        "sampling": {
            "source_hz": source_hz,
            "target_hz": target_hz,
            "decimation_factor": decimation_factor,
            "duration_policy": "original_time",
        },
        "sheets": used_sheets,
        "pace_order": PACE_ORDER,
        "counts": {
            "trial_rows": int(len(trial_index)),
            "participant_rows": int(participant_table[["group", "participant"]].drop_duplicates().shape[0]),
            "embedding_rows": int(len(embeddings)),
            "embedding_dims": int(len([c for c in embeddings.columns if c.startswith("emb_")])),
            "umap_rows": int(len(umap_table)),
            "unknown_pace_trials": int((trial_index["pace_condition"] == "unknown").sum()),
        },
        "missingness": {
            "trial_features_all_nan_rows": int(trial_features.drop(columns=["group", "participant", "pace_condition", "file_name", "file_stem_lower", "file_path"], errors="ignore").isna().all(axis=1).sum()),
            "participant_missing_pace_any": int(participant_table.get("missing_pace_any", pd.Series(dtype=int)).sum()),
        },
        "preprocessing": {
            "numeric_columns": len(model_info["num_cols"]),
            "categorical_columns": len(model_info["cat_cols"]),
            "dropped_all_nan_columns": model_info["drop_all_nan_columns"],
            "pca_n_components": int(model_info["pca"].n_components_),
        },
        "umap": {
            "method": umap_info["method"],
            "has_umap": umap_info["has_umap"],
            "reason": umap_info["reason"],
            "n_neighbors": umap_info["n_neighbors"],
            "min_dist": umap_info["min_dist"],
        },
    }
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build participant-level gait embeddings from metadata + xlsx timeseries.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--summary-csv", type=Path, default=Path("notebooks") / "gait_analysis_global.csv")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--source-hz", type=float, default=100.0)
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_sampling_args(source_hz: float, target_hz: float) -> int:
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("source_hz and target_hz must be positive.")
    if target_hz > source_hz:
        raise ValueError(f"target_hz({target_hz}) must be <= source_hz({source_hz}).")
    ratio = source_hz / target_hz
    if not np.isclose(ratio, round(ratio)):
        raise ValueError(
            f"source_hz({source_hz}) must be divisible by target_hz({target_hz}) for integer decimation."
        )
    return int(round(ratio))


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    metadata_path = data_dir / "ID.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    decimation_factor = validate_sampling_args(args.source_hz, args.target_hz)

    ensure_dir(out_dir)
    ensure_dir(out_dir / "models")

    metadata = load_metadata(metadata_path)
    trials = collect_trial_index(data_dir, max_trials=args.max_trials)
    existing_summary = load_existing_summary(args.summary_csv.resolve())
    trial_features = build_trial_feature_table(
        trials=trials,
        source_hz=args.source_hz,
        target_hz=args.target_hz,
        decimation_factor=decimation_factor,
        existing_summary=existing_summary,
        workers=max(1, args.workers),
        verbose=args.verbose,
    )
    participant_table = aggregate_participant_features(trial_features)
    participant_table = merge_metadata(participant_table, metadata)

    embeddings, x_emb, model_info = build_embedding_matrix(participant_table, random_state=args.random_state)
    umap_table, umap_info = build_umap_table(
        embeddings=embeddings,
        x_emb=x_emb,
        random_state=args.random_state,
    )

    embeddings_path = out_dir / "participant_embeddings.csv"
    umap_path = out_dir / "participant_umap.csv"
    features_path = out_dir / "participant_features.csv"
    trial_features_path = out_dir / "trial_features.csv"
    manifest_path = out_dir / "feature_manifest.json"

    embeddings.to_csv(embeddings_path, index=False)
    umap_table.to_csv(umap_path, index=False)
    participant_table.to_csv(features_path, index=False)
    trial_features.to_csv(trial_features_path, index=False)

    joblib.dump(model_info["preprocessor"], out_dir / "models" / "preprocessor.joblib")
    joblib.dump(model_info["pca"], out_dir / "models" / "pca.joblib")
    if umap_info["umap_2d"] is not None:
        joblib.dump(umap_info["umap_2d"], out_dir / "models" / "umap_2d.joblib")
    if umap_info["umap_3d"] is not None:
        joblib.dump(umap_info["umap_3d"], out_dir / "models" / "umap_3d.joblib")

    write_manifest(
        out_path=manifest_path,
        data_dir=data_dir,
        metadata_path=metadata_path,
        summary_path=args.summary_csv.resolve(),
        trial_index=trials,
        trial_features=trial_features,
        participant_table=participant_table,
        embeddings=embeddings,
        umap_table=umap_table,
        model_info=model_info,
        umap_info=umap_info,
        random_state=args.random_state,
        source_hz=args.source_hz,
        target_hz=args.target_hz,
        decimation_factor=decimation_factor,
    )

    if args.verbose:
        print(f"metadata rows: {len(metadata)}")
        print(f"trial rows: {len(trials)}")
        print(f"trial features rows: {len(trial_features)}")
        print(f"participant rows: {len(participant_table)}")
        print(f"embedding dims: {len([c for c in embeddings.columns if c.startswith('emb_')])}")
        print(
            "sampling:"
            f" source_hz={args.source_hz}, target_hz={args.target_hz}, decimation_factor={decimation_factor}"
        )
        print(f"saved: {embeddings_path}")
        print(f"saved: {umap_path}")
        print(f"saved: {manifest_path}")


if __name__ == "__main__":
    main()
