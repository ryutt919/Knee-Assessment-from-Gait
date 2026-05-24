"""
step002: id_pairing_summary.csv 재생성
- ID.csv 기반으로 ACLD/ACLR 숫자 ID로 매칭
- paired=27, ACLR36 포함 확인

출력: data/processed/id_pairing_summary.csv
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ID_CSV = DATA_DIR / "ID.csv"
OUT_PATH = PROCESSED_DIR / "id_pairing_summary.csv"


def run():
    df = pd.read_csv(ID_CSV)

    acld = df[df["Group"] == 3].copy()
    aclr = df[df["Group"] == 4].copy()

    acld["num"] = acld["ID"].str.extract(r"(\d+)$").astype(int)
    aclr["num"] = aclr["ID"].str.extract(r"(\d+)$").astype(int)

    paired = pd.merge(
        acld[["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]],
        aclr[["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]],
        on=["num", "Sex", "Age", "Weight", "Height", "Injured leg"],
        suffixes=("_ACLD", "_ACLR"),
    )
    paired["pair_status"] = "paired"

    unpaired_nums = set(acld["num"]) - set(paired["num"])
    unpaired = acld[acld["num"].isin(unpaired_nums)][["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]].copy()
    unpaired = unpaired.rename(columns={"ID": "ID_ACLD"})
    unpaired["ID_ACLR"] = None
    unpaired["pair_status"] = "ACLD_only"

    result = pd.concat([paired, unpaired], ignore_index=True)
    result.to_csv(OUT_PATH, index=False)

    n_paired = (result["pair_status"] == "paired").sum()
    print(f"[002] paired={n_paired}, ACLD_only={(result['pair_status']=='ACLD_only').sum()}")
    print(f"[002] paired subjects: {paired['ID_ACLR'].tolist()}")

    assert n_paired == 27, f"paired 수 {n_paired} != 27"
    print(f"[002] ✅ id_pairing_summary.csv 저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    run()
