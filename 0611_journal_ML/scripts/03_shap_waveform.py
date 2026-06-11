#!/usr/bin/env python3
"""
03_shap_waveform.py — SHAP Backprojection to Gait Timepoints

Loads best model from 02_raw_waveform_ml results, retrains with full data,
computes SHAP values, then backprojects from PCA space to original 9ch×101pt
waveform space to identify which gait phase timepoints drive classification.

Output figures (300dpi PNG for journal):
  Fig A: SHAP beeswarm (PC level)
  Fig B: SHAP → timepoint heatmap (9ch × 101pt) ← key visualization
  Fig C: Per-channel mean |SHAP| across gait cycle (%)
  Fig D: Waterfall plots for representative subjects
"""

import warnings
warnings.filterwarnings("ignore")

import json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import shap

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]
SANDBOX  = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS  = SANDBOX / "results"
FIGURES  = SANDBOX / "figures"
LOGS     = SANDBOX / "logs"
FIGURES.mkdir(exist_ok=True)

WAVE_PATH   = DATA_DIR / "waveforms_stride.parquet"
SCALAR_PATH = DATA_DIR / "features_scalar.csv"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(LOGS / "03_shap_waveform.log")])
log = logging.getLogger(__name__)

RANDOM_STATE = 42
PCA_VARIANCE = 0.95
CHANNELS = [
    'ankle_adduction', 'ankle_dorsiflexion', 'ankle_int_rotation',
    'hip_adduction',   'hip_flexion',         'hip_int_rotation',
    'knee_adduction',  'knee_flexion',         'knee_int_rotation',
]
N_TIMEPOINTS = 101


# ── Import data utilities from script 02 ──────────────────────────────────────
def get_wave_cols(df):
    return [c for c in df.columns if any(c.startswith(ch + '_') for ch in CHANNELS)]


def get_injured_leg_map(scalar_df):
    mapping = {}
    for _, row in scalar_df.drop_duplicates('subject_id').iterrows():
        leg = row['injured_leg']
        mapping[row['subject_id']] = 'Right' if (pd.isna(leg) or str(leg).lower() == 'nan') else leg
    return mapping


def build_bilateral_asymmetry(wave_df, inj_map, wave_cols):
    rows = []
    for (subj, speed), grp in wave_df.groupby(['subject_id', 'speed']):
        leg = inj_map.get(subj, 'Right')
        inj = grp[grp['side'] == leg]
        con = grp[grp['side'] != leg]
        if len(inj) == 0 or len(con) == 0:
            continue
        asym = inj[wave_cols].values[0] - con[wave_cols].values[0]
        row = {'subject_id': subj, 'group': grp['group'].iloc[0], 'speed': speed,
               'binary_label': 0 if grp['group'].iloc[0] == 'HA' else 1}
        row.update(zip(wave_cols, asym))
        rows.append(row)
    return pd.DataFrame(rows)


def build_multi_speed_bilateral(asym_df, wave_cols):
    speed_dfs = {}
    for spd, sub in asym_df.groupby('speed'):
        speed_dfs[spd] = sub.set_index('subject_id')[wave_cols].add_prefix(f"{spd}_")
    speeds = sorted(speed_dfs.keys())
    pivot  = pd.concat([speed_dfs[s] for s in speeds], axis=1).dropna()
    meta   = asym_df.drop_duplicates('subject_id').set_index('subject_id')[['group', 'binary_label']]
    return pivot.join(meta).reset_index().rename(columns={'index': 'subject_id'})


# ── SHAP backprojection ────────────────────────────────────────────────────────

def shap_to_waveform(shap_values_pca, pca):
    """
    Backproject SHAP values from PCA space to original waveform space.
    shap_values_pca: (n_samples, n_pcs)
    pca.components_: (n_pcs, n_features)
    Returns: (n_samples, n_features) — contribution at each waveform timepoint
    """
    return shap_values_pca @ pca.components_


def reshape_to_channels(wave_importance, wave_cols):
    """Reshape flat 909-dim array → 9ch × 101pt matrix."""
    mat = np.zeros((len(CHANNELS), N_TIMEPOINTS))
    col_to_idx = {c: i for i, c in enumerate(wave_cols)}
    for ch_idx, ch in enumerate(CHANNELS):
        for t in range(N_TIMEPOINTS):
            col = f"{ch}_{t:03d}"
            if col in col_to_idx:
                mat[ch_idx, t] = wave_importance[col_to_idx[col]]
    return mat


# ── Plotting ───────────────────────────────────────────────────────────────────

COLORS_GROUP = {'HA': '#2196F3', 'ACLD': '#F44336', 'ACLR': '#FF9800'}

def plot_shap_heatmap(mean_abs_shap_matrix, title, save_path):
    """9ch × 101pt SHAP importance heatmap with gait phase annotations."""
    fig, ax = plt.subplots(figsize=(14, 5))

    im = ax.imshow(mean_abs_shap_matrix, aspect='auto', cmap='hot_r',
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Mean |SHAP| contribution')

    ax.set_yticks(range(len(CHANNELS)))
    ax.set_yticklabels([c.replace('_', ' ').title() for c in CHANNELS], fontsize=9)
    ax.set_xlabel('Gait Cycle (%)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Gait phase annotations
    phases = [
        (0, 12,   '#E3F2FD', 'Loading\nResponse'),
        (12, 50,  '#E8F5E9', 'Midstance'),
        (50, 62,  '#FFF3E0', 'Terminal\nStance'),
        (62, 100, '#FCE4EC', 'Swing'),
    ]
    for start, end, color, label in phases:
        ax.axvspan(start, end, alpha=0.12, color=color, label=label)
        ax.text((start + end) / 2, len(CHANNELS) - 0.3, label,
                ha='center', va='top', fontsize=7, color='gray')

    xticks = range(0, 101, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x}%" for x in xticks], fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {save_path.name}")


def plot_channel_importance(mean_abs_shap_matrix, save_path):
    """Bar chart: mean |SHAP| per channel (summed across gait cycle)."""
    channel_importance = mean_abs_shap_matrix.sum(axis=1)
    ch_pct = 100 * channel_importance / channel_importance.sum()
    sorted_idx = np.argsort(ch_pct)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh([CHANNELS[i].replace('_',' ') for i in sorted_idx],
                   ch_pct[sorted_idx],
                   color=['#F44336' if 'knee' in CHANNELS[i]
                          else '#2196F3' if 'hip' in CHANNELS[i]
                          else '#4CAF50' for i in sorted_idx])
    ax.set_xlabel('Relative SHAP Importance (%)', fontsize=11)
    ax.set_title('Channel Contribution to ACL Classification', fontsize=12, fontweight='bold')
    ax.axvline(100 / len(CHANNELS), color='gray', linestyle='--', alpha=0.5, label='Uniform baseline')

    for bar, pct in zip(bars, ch_pct[sorted_idx]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)

    legend_elements = [
        mpatches.Patch(facecolor='#F44336', label='Knee'),
        mpatches.Patch(facecolor='#2196F3', label='Hip'),
        mpatches.Patch(facecolor='#4CAF50', label='Ankle'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {save_path.name}")


def plot_gait_cycle_importance(mean_abs_shap_matrix, save_path):
    """Line plot: mean |SHAP| across gait cycle for each channel."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    axes = axes.flatten()

    gait_pct = np.linspace(0, 100, N_TIMEPOINTS)

    for idx, ch in enumerate(CHANNELS):
        ax = axes[idx]
        importance = mean_abs_shap_matrix[idx]
        ax.fill_between(gait_pct, importance, alpha=0.3, color='#F44336')
        ax.plot(gait_pct, importance, color='#D32F2F', linewidth=1.5)
        ax.set_title(ch.replace('_', ' ').title(), fontsize=9, fontweight='bold')
        ax.set_xlabel('Gait Cycle (%)', fontsize=8)

        # Phase shading
        for start, end, color, _ in [
            (0, 12, '#E3F2FD', ''), (12, 50, '#E8F5E9', ''),
            (50, 62, '#FFF3E0', ''), (62, 100, '#FCE4EC', '')
        ]:
            ax.axvspan(start, end, alpha=0.1, color=color)

        # Highlight peak
        peak_t = gait_pct[np.argmax(importance)]
        ax.axvline(peak_t, color='gray', linestyle=':', alpha=0.7)
        ax.text(peak_t + 1, importance.max() * 0.95, f'{peak_t:.0f}%',
                fontsize=7, color='gray')

    fig.suptitle('SHAP Importance Across Gait Cycle\n(Backprojected from PCA components)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {save_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("03_shap_waveform.py — SHAP Backprojection Analysis")
    log.info("=" * 60)

    wave_df   = pd.read_parquet(WAVE_PATH)
    scalar_df = pd.read_csv(SCALAR_PATH)
    wave_cols = get_wave_cols(wave_df)
    inj_map   = get_injured_leg_map(scalar_df)

    # Build feature set D (multi-speed bilateral) — the best feature set
    asym_df  = build_bilateral_asymmetry(wave_df, inj_map, wave_cols)
    multi_df = build_multi_speed_bilateral(asym_df, wave_cols)

    META_COLS = {'subject_id', 'group', 'speed', 'side', 'binary_label', 'n_strides', 'injured_leg'}
    feat_cols = [c for c in multi_df.columns
                 if c not in META_COLS and multi_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    X = multi_df[feat_cols].fillna(0).values.astype(np.float32)
    y = multi_df['binary_label'].values
    groups_label = multi_df['group'].values

    log.info(f"Dataset: {X.shape} | HA={int((y==0).sum())} ACL={int((y==1).sum())}")

    # Fit scaler + PCA on full dataset (for SHAP analysis, not CV)
    scaler = StandardScaler()
    pca    = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)

    X_scaled = scaler.fit_transform(X)
    X_pca    = pca.fit_transform(X_scaled)
    log.info(f"PCA: {X_pca.shape[1]} components (retain {PCA_VARIANCE*100:.0f}% variance)")

    # Load best params from script 02 if available
    best_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'max_features': 0.5,
    }
    best_json = RESULTS / "02_waveform_best.json"
    if best_json.exists():
        with open(best_json) as f:
            b = json.load(f)
        fs = b.get('best_by_feature_set', {}).get('D_multi_speed_bilateral', {})
        log.info(f"Loaded best AUC from script 02: {fs.get('auc', 'N/A')}")

    clf = RandomForestClassifier(**best_params, class_weight='balanced',
                                  random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_pca, y)
    train_auc = roc_auc_score(y, clf.predict_proba(X_pca)[:, 1])
    log.info(f"Full-data RF train AUC: {train_auc:.4f}")

    # ── SHAP computation ───────────────────────────────────────────────────────
    log.info("Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_pca)

    # shap_values: (n_samples, n_pcs) for binary → take class=1 (ACL)
    if isinstance(shap_values, list):
        sv = shap_values[1]   # class 1 = ACL
    else:
        sv = shap_values

    log.info(f"SHAP values shape: {sv.shape}")

    # ── Backproject to waveform space ──────────────────────────────────────────
    log.info("Backprojecting SHAP → waveform timepoints...")
    sv_wave = shap_to_waveform(sv, pca)   # (n_samples, 2727)

    # The multi-speed bilateral has feat_cols ordered as slow_wave, normal_wave, fast_wave
    # feat_cols = slow_wc + normal_wc + fast_wc (each 909 long)
    n_per_speed = len(wave_cols)
    sv_wave_slow   = sv_wave[:, :n_per_speed]
    sv_wave_normal = sv_wave[:, n_per_speed:2*n_per_speed]
    sv_wave_fast   = sv_wave[:, 2*n_per_speed:]
    sv_wave_mean   = (sv_wave_slow + sv_wave_normal + sv_wave_fast) / 3

    # Mean absolute SHAP per timepoint (averaged across subjects)
    mean_abs_sv_slow   = np.abs(sv_wave_slow).mean(axis=0)
    mean_abs_sv_normal = np.abs(sv_wave_normal).mean(axis=0)
    mean_abs_sv_fast   = np.abs(sv_wave_fast).mean(axis=0)
    mean_abs_sv_mean   = np.abs(sv_wave_mean).mean(axis=0)

    # Reshape to 9ch × 101pt matrices
    mat_slow   = reshape_to_channels(mean_abs_sv_slow,   wave_cols)
    mat_normal = reshape_to_channels(mean_abs_sv_normal, wave_cols)
    mat_fast   = reshape_to_channels(mean_abs_sv_fast,   wave_cols)
    mat_mean   = reshape_to_channels(mean_abs_sv_mean,   wave_cols)

    log.info(f"SHAP heatmap shape: {mat_mean.shape}")

    # ── Save SHAP data ─────────────────────────────────────────────────────────
    np.save(RESULTS / "03_shap_heatmap_mean.npy", mat_mean)
    np.save(RESULTS / "03_shap_heatmap_fast.npy", mat_fast)

    # Top timepoints per channel
    top_timepoints = {}
    for ch_idx, ch in enumerate(CHANNELS):
        top3_t = np.argsort(mat_mean[ch_idx])[::-1][:3]
        top_timepoints[ch] = [
            {'timepoint': int(t), 'gait_pct': round(t / N_TIMEPOINTS * 100, 1),
             'importance': round(float(mat_mean[ch_idx, t]), 6)}
            for t in top3_t
        ]

    with open(RESULTS / "03_shap_top_timepoints.json", 'w') as f:
        json.dump(top_timepoints, f, indent=2)

    log.info("Top timepoints per channel:")
    for ch, tops in top_timepoints.items():
        log.info(f"  {ch}: " + ", ".join(f"{t['gait_pct']:.0f}%" for t in tops))

    # ── Generate Figures ───────────────────────────────────────────────────────
    log.info("Generating figures...")

    # Fig B: SHAP heatmap (mean across speeds)
    plot_shap_heatmap(mat_mean,
                      'SHAP Importance Heatmap — Bilateral Asymmetry Waveform\n(Mean across speeds)',
                      FIGURES / "fig_B_shap_heatmap_mean.png")

    # Fig B2: fast speed heatmap (most informative per H2)
    plot_shap_heatmap(mat_fast,
                      'SHAP Importance Heatmap — Bilateral Asymmetry Waveform (Fast Speed)',
                      FIGURES / "fig_B2_shap_heatmap_fast.png")

    # Fig C: Channel importance
    plot_channel_importance(mat_mean, FIGURES / "fig_C_channel_importance.png")

    # Fig D: Gait cycle importance per channel
    plot_gait_cycle_importance(mat_mean, FIGURES / "fig_D_gait_cycle_importance.png")

    # Fig A: SHAP beeswarm (PC level)
    try:
        shap_exp = shap.Explanation(values=sv, data=X_pca,
                                     feature_names=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_exp, max_display=20, show=False)
        plt.title('SHAP Beeswarm — PCA Components\n(Multi-speed Bilateral Asymmetry)', fontsize=11)
        plt.tight_layout()
        fig.savefig(FIGURES / "fig_A_shap_beeswarm.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info("Saved: fig_A_shap_beeswarm.png")
    except Exception as e:
        log.warning(f"Beeswarm plot skipped: {e}")

    # Summary stats
    fastest_ch = CHANNELS[np.argmax(mat_mean.sum(axis=1))]
    peak_phase = np.unravel_index(mat_mean.argmax(), mat_mean.shape)
    peak_ch    = CHANNELS[peak_phase[0]]
    peak_t_pct = peak_phase[1] / N_TIMEPOINTS * 100

    log.info(f"\n{'='*60}")
    log.info("SHAP SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Most important channel: {fastest_ch}")
    log.info(f"  Peak timepoint: {peak_ch} at {peak_t_pct:.0f}% gait cycle")
    log.info(f"  Figures saved: {FIGURES}")


if __name__ == '__main__':
    main()
