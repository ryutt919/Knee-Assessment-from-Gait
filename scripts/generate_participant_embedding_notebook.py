from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "participant_embedding_explorer.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


cells = [
    md(
        """# Participant Embedding Explorer

This notebook explores participant-level embeddings generated from metadata + xlsx timeseries.
"""
    ),
    code(
        """from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

sns.set_theme(style="whitegrid")

ROOT = Path.cwd()
if not (ROOT / "artifacts").exists():
    ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else ROOT

ARTIFACTS = ROOT / "artifacts"
EMB_PATH = ARTIFACTS / "participant_embeddings.csv"
UMAP_PATH = ARTIFACTS / "participant_umap.csv"
MANIFEST_PATH = ARTIFACTS / "feature_manifest.json"

print("Artifacts:", ARTIFACTS)
print("Embeddings exists:", EMB_PATH.exists())
print("UMAP exists:", UMAP_PATH.exists())
print("Manifest exists:", MANIFEST_PATH.exists())
"""
    ),
    code(
        """emb = pd.read_csv(EMB_PATH)
um = pd.read_csv(UMAP_PATH)

display(emb.head())
display(um.head())
"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    data=um,
    x="umap_x",
    y="umap_y",
    hue="group" if "group" in um.columns else None,
    style="injury_status" if "injury_status" in um.columns else None,
    s=70,
    ax=ax,
)
ax.set_title("Participant UMAP (2D)")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """if "missing_pace_any" in emb.columns:
    print("Missing pace(any) participants:", int(emb["missing_pace_any"].sum()))
if "pace_coverage_count" in emb.columns:
    print(emb["pace_coverage_count"].value_counts(dropna=False).sort_index())
"""
    ),
    code(
        """emb_cols = [c for c in emb.columns if c.startswith("emb_")]
X = emb[emb_cols].to_numpy(dtype=float)
nn = NearestNeighbors(n_neighbors=min(6, len(emb)), metric="euclidean")
nn.fit(X)

def nearest_participants(participant_id: str, top_k: int = 5):
    if participant_id not in set(emb["participant"].astype(str)):
        raise ValueError(f"participant not found: {participant_id}")
    idx = emb.index[emb["participant"].astype(str) == participant_id][0]
    dist, ind = nn.kneighbors(X[idx].reshape(1, -1), n_neighbors=min(top_k + 1, len(emb)))
    rows = emb.iloc[ind[0]].copy()
    rows["distance"] = dist[0]
    return rows[["group", "participant", "injury_status", "pace_coverage_count", "distance"] if "injury_status" in rows.columns else ["group", "participant", "distance"]]

# Example:
# nearest_participants("HA1", top_k=5)
"""
    ),
    md(
        """## Bias / Coverage quick checks
- Group counts in embedding outputs
- Pace coverage distribution
- Optional nearest-neighbor spot checks by participant
"""
    ),
    code(
        """display(emb.groupby("group").size().rename("participants").to_frame())
if "injury_status" in emb.columns:
    display(emb.groupby(["group", "injury_status"]).size().rename("n").to_frame())
"""
    ),
]


nb = nbf.v4.new_notebook(
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
)
nb["cells"] = cells
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)

print(f"Wrote notebook: {NOTEBOOK_PATH}")
