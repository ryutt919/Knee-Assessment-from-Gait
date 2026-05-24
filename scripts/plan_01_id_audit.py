from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_PATH = DATA_DIR / "ID.csv"

GROUP_DIRS = {
    "ACLD": DATA_DIR / "ACLD",
    "ACLR": DATA_DIR / "ACLR",
    "HA": DATA_DIR / "Healthy adults",
}

GROUP_CODES = {
    "HA": 1,
    "HK": 2,
    "ACLD": 3,
    "ACLR": 4,
}

ALIAS_SUGGESTIONS = {
    "ACLR38": "ACLR36",
}

OUT_REPORT = OUT_DIR / "id_audit_report.csv"
OUT_ALIAS = OUT_DIR / "id_alias_map.csv"
OUT_PAIRING = OUT_DIR / "id_pairing_summary.csv"


def list_subject_dirs(base_dir: Path, prefix: str) -> list[str]:
    if not base_dir.exists():
        return []
    pattern = re.compile(rf"^{re.escape(prefix)}\d+$")
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir() and pattern.match(p.name)])


def extract_num(subject_id: str) -> int | None:
    match = re.search(r"(\d+)$", str(subject_id))
    return int(match.group(1)) if match else None


def main() -> int:
    if not ID_PATH.exists():
        raise FileNotFoundError(f"ID.csv not found: {ID_PATH}")

    id_df = pd.read_csv(ID_PATH)
    id_df["ID"] = id_df["ID"].astype(str).str.strip()

    id_sets = {
        "HA": set(id_df[id_df["Group"] == GROUP_CODES["HA"]]["ID"]),
        "ACLD": set(id_df[id_df["Group"] == GROUP_CODES["ACLD"]]["ID"]),
        "ACLR": set(id_df[id_df["Group"] == GROUP_CODES["ACLR"]]["ID"]),
    }

    rows: list[dict[str, object]] = []

    for group, base_dir in GROUP_DIRS.items():
        folder_ids = list_subject_dirs(base_dir, group)
        csv_ids = id_sets[group]

        for sid in folder_ids:
            rows.append(
                {
                    "group": group,
                    "source": "folder",
                    "subject_id": sid,
                    "exists_in_id_csv": sid in csv_ids,
                    "exists_in_folder": True,
                    "note": "",
                }
            )

        for sid in sorted(csv_ids):
            if sid not in folder_ids:
                rows.append(
                    {
                        "group": group,
                        "source": "id_csv",
                        "subject_id": sid,
                        "exists_in_id_csv": True,
                        "exists_in_folder": False,
                        "note": "missing_folder",
                    }
                )

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(OUT_REPORT, index=False, encoding="utf-8-sig")

    alias_rows: list[dict[str, object]] = []
    for src, canon in ALIAS_SUGGESTIONS.items():
        src_group = "ACLR" if src.startswith("ACLR") else "ACLD" if src.startswith("ACLD") else "UNKNOWN"
        alias_rows.append(
            {
                "source_id": src,
                "canonical_id": canon,
                "group": src_group,
                "source_folder_exists": (GROUP_DIRS.get(src_group, Path()).joinpath(src).exists() if src_group in GROUP_DIRS else False),
                "canonical_folder_exists": (GROUP_DIRS.get(src_group, Path()).joinpath(canon).exists() if src_group in GROUP_DIRS else False),
                "source_in_id_csv": src in id_sets.get(src_group, set()),
                "canonical_in_id_csv": canon in id_sets.get(src_group, set()),
                "action": "rename_folder_and_update_id_csv",
                "reason": "known_alias",
            }
        )

    alias_df = pd.DataFrame(alias_rows)
    alias_df.to_csv(OUT_ALIAS, index=False, encoding="utf-8-sig")

    acld_nums = {extract_num(sid) for sid in id_sets["ACLD"] if extract_num(sid) is not None}
    aclr_nums = {extract_num(sid) for sid in id_sets["ACLR"] if extract_num(sid) is not None}
    unmatched_acld = sorted(acld_nums - aclr_nums)
    unmatched_aclr = sorted(aclr_nums - acld_nums)

    pairing_rows = []
    for num in sorted(acld_nums | aclr_nums):
        pairing_rows.append(
            {
                "numeric_id": num,
                "in_acld": num in acld_nums,
                "in_aclr": num in aclr_nums,
                "pair_status": "paired" if num in acld_nums and num in aclr_nums else "unpaired",
            }
        )

    pd.DataFrame(pairing_rows).to_csv(OUT_PAIRING, index=False, encoding="utf-8-sig")

    print("=== ID audit summary ===")
    print(f"Report: {OUT_REPORT}")
    print(f"Alias map: {OUT_ALIAS}")
    print(f"Pairing summary: {OUT_PAIRING}")
    print(f"Unmatched ACLD numeric IDs: {unmatched_acld}")
    print(f"Unmatched ACLR numeric IDs: {unmatched_aclr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
