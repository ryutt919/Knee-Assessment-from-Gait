"""
step001: ID 정정
- data/ACLR/ACLR38 → data/ACLR/ACLR36 폴더 rename
- data/ID.csv ACLR36 행 존재 확인 (없으면 ACLD36 인구통계 복사해 추가)

검증: ACLR36 폴더 존재, ACLR38 없음, ID.csv ACLR36 행 있음
"""
import os
import shutil
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ACLR_DIR = DATA_DIR / "ACLR"
ID_CSV = DATA_DIR / "ID.csv"


def fix_folder():
    src = ACLR_DIR / "ACLR38"
    dst = ACLR_DIR / "ACLR36"

    if dst.exists() and not src.exists():
        print(f"[001] ACLR36 이미 존재, ACLR38 없음 — rename 불필요")
        return

    if not src.exists():
        print(f"[001] ACLR38 폴더 없음 — 스킵")
        return

    if dst.exists():
        print(f"[001] 경고: ACLR36 이미 존재. ACLR38 → ACLR36 내용 이동 후 삭제")
        for item in src.iterdir():
            shutil.move(str(item), str(dst / item.name))
        src.rmdir()
    else:
        os.rename(src, dst)

    print(f"[001] ✅ ACLR38 → ACLR36 rename 완료")


def fix_id_csv():
    df = pd.read_csv(ID_CSV)

    if "ACLR36" in df["ID"].values:
        print(f"[001] ID.csv ACLR36 행 이미 존재 — 스킵")
        return

    acld36 = df[df["ID"] == "ACLD36"]
    if acld36.empty:
        print(f"[001] 경고: ACLD36 행 없음 — ACLR36 추가 불가")
        return

    new_row = acld36.iloc[0].copy()
    new_row["ID"] = "ACLR36"
    new_row["Group"] = 4

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(ID_CSV, index=False)
    print(f"[001] ✅ ID.csv ACLR36 행 추가 완료")


def verify():
    assert (ACLR_DIR / "ACLR36").exists(), "ACLR36 폴더 없음"
    assert not (ACLR_DIR / "ACLR38").exists(), "ACLR38 폴더가 여전히 존재함"
    df = pd.read_csv(ID_CSV)
    assert "ACLR36" in df["ID"].values, "ID.csv에 ACLR36 행 없음"
    print(f"[001] ✅ 검증 통과")


if __name__ == "__main__":
    fix_folder()
    fix_id_csv()
    verify()
