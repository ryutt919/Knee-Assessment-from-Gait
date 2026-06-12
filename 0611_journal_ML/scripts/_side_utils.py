"""
_side_utils.py — Unified injured / contralateral side mapping (shared helper).

Two source files encode "side" differently:
  - stride_level_peaks.parquet : side ∈ {injured, contralateral}
  - waveforms_stride.parquet    : side ∈ {Right, Left}

This module standardizes both to {inj, con} so every script in the pipeline
groups strides/waveforms by the SAME biomechanical convention.

Convention
----------
- side already symbolic ({injured, contralateral}) → mapped directly.
- side anatomical ({Right, Left}) → compared against the subject's injured leg.
- HA (healthy) subjects have no injured leg → pseudo-injured = Right
  (con = Left), matching the project-wide HA convention.

History
-------
Before this helper, 02b compared stride side (injured/contralateral) against
injured_leg (Right/Left) → never matched → every stride collapsed to `con`,
silently dropping all injured-side variability features.
"""
from __future__ import annotations

import pandas as pd

# Recognized literals (lower-cased for comparison)
_SYMBOLIC = {"injured": "inj", "contralateral": "con"}
_ANATOMICAL = {"right", "left"}

# HA pseudo-injured side (Right = inj, Left = con)
HA_PSEUDO_INJURED = "Right"


def build_injured_leg_map(scalar_df: pd.DataFrame) -> dict:
    """subject_id → injured leg ('Right'/'Left'). HA / missing → HA_PSEUDO_INJURED.

    Reads the subject-level `injured_leg` column from features_scalar.csv
    (one value per subject, repeated across speeds).
    """
    sub = scalar_df.drop_duplicates("subject_id")
    mapping: dict = {}
    for _, row in sub.iterrows():
        leg = row.get("injured_leg")
        if pd.isna(leg) or str(leg).strip() == "" or str(leg).strip().lower() == "nan":
            leg = HA_PSEUDO_INJURED
        mapping[row["subject_id"]] = str(leg).strip()
    return mapping


def to_inj_con(side_value, injured_leg) -> str | None:
    """Map a single `side` value to 'inj' or 'con'.

    Parameters
    ----------
    side_value : the row's `side` (e.g. 'injured', 'contralateral', 'Right', 'Left')
    injured_leg : the subject's injured leg ('Right'/'Left'); for HA pass the
                  pseudo-injured side (HA_PSEUDO_INJURED).

    Returns 'inj' / 'con', or None if the value is unrecognized.
    """
    if side_value is None:
        return None
    s = str(side_value).strip().lower()

    # Symbolic encoding → direct
    if s in _SYMBOLIC:
        return _SYMBOLIC[s]

    # Anatomical encoding → compare with injured leg
    if s in _ANATOMICAL:
        leg = str(injured_leg).strip().lower() if injured_leg is not None else ""
        if leg not in _ANATOMICAL:
            leg = HA_PSEUDO_INJURED.lower()  # fall back to HA convention
        return "inj" if s == leg else "con"

    return None


def add_inj_con_column(df: pd.DataFrame, inj_map: dict,
                       side_col: str = "side", out_col: str = "side_std") -> pd.DataFrame:
    """Return a copy of `df` with a standardized {inj, con} column `out_col`.

    `inj_map` maps subject_id → injured leg (use build_injured_leg_map).
    Rows whose side cannot be resolved get NaN in `out_col`.
    """
    df = df.copy()
    df[out_col] = df.apply(
        lambda r: to_inj_con(r.get(side_col), inj_map.get(r.get("subject_id"), HA_PSEUDO_INJURED)),
        axis=1,
    )
    return df
