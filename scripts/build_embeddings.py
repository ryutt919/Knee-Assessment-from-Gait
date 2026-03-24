"""
scripts/build_embeddings.py

1D-CNN+GAP / AutoEncoder / Centroid 기반 보행 임베딩 파이프라인
Trial 단위(1파일=1점)로 임베딩을 생성하고 Parquet으로 저장.

사용법:
    python scripts/build_embeddings.py \
        --input_path data/processed/Master_Gait_Dataset.parquet \
        --model cnn \
        --target_hz 100 \
        --latent_dim 128

--model 옵션: cnn | ae | centroid
"""
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import umap

from utils.embedding_utils import (
    load_parquet,
    split_by_trial,
    downsample,
    build_windows,
)

# ─────────────────────────────────────────────
# 1. 1D-CNN + Global Average Pooling 모델
# ─────────────────────────────────────────────
class CNN1DEncoder(nn.Module):
    """
    가변 길이 시계열을 받아 Global Average Pooling으로 고정 벡터로 압축.
    입력: (batch, n_features, n_frames)  ← Conv1d는 채널이 두번째
    출력: (batch, latent_dim)
    """
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=7, padding=3),  # 로컬 패턴 추출 (넓게)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),         # 중간 패턴 정제
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, latent_dim, kernel_size=3, padding=1),  # 최종 피처 맵
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        # Global Average Pooling: 시간 축(dim=2) 평균 → (batch, latent_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv_layers(x)    # (batch, latent_dim, T)
        x = self.gap(x)            # (batch, latent_dim, 1)
        return x.squeeze(-1)       # (batch, latent_dim)


# ─────────────────────────────────────────────
# 2. AutoEncoder (1D-CNN 기반 Bottleneck 비지도 학습)
# ─────────────────────────────────────────────
class CNN1DAutoEncoder(nn.Module):
    """
    윈도우 기반 시계열 AutoEncoder.
    고정 길이(window_size) 입력을 잠재 벡터(latent_dim)로 압축했다 복원.
    인코더 출력을 임베딩으로 사용함.
    """
    def __init__(self, window_size: int, n_features: int, latent_dim: int):
        super().__init__()
        flat_dim = window_size * n_features

        # 인코더: 평탄화(Flatten) → 병목(Bottleneck)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        # 디코더: 병목 → 복원 (학습 목적으로만 사용)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, flat_dim),
        )
        self.window_size = window_size
        self.n_features = n_features

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat.view(-1, self.window_size, self.n_features), z


# ─────────────────────────────────────────────
# 3. 임베딩 추출 함수들
# ─────────────────────────────────────────────

def embed_with_cnn(trials, n_features, latent_dim, device):
    """
    1D-CNN + GAP: Trial 전체 시계열을 통째로 넣어 1개의 벡터 산출.
    학습 없이 랜덤 초기화 모델로 구조적 임베딩(특성 공간 탐색용)을 추출함.
    재현성 보장을 위해 seed를 고정함.
    """
    torch.manual_seed(42)
    model = CNN1DEncoder(n_features, latent_dim).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for trial in tqdm(trials, desc="[CNN] 임베딩 추출 중"):
            data = trial["data"]            # (T, F)
            x = torch.tensor(data.T, dtype=torch.float32).unsqueeze(0).to(device)  # (1, F, T)
            vec = model(x).cpu().numpy().squeeze()   # (latent_dim,)
            embeddings.append(vec)
    return np.array(embeddings)


def embed_with_autoencoder(trials, n_features, latent_dim, window_size, device, epochs=30):
    """
    AutoEncoder: 윈도우 조각들로 모델을 먼저 훈련한 뒤, 
    Trial 전체의 조각들을 인코딩하고 평균(Centroid)으로 1개의 벡터를 결정.
    """
    # 1) 전체 Trial의 윈도우 조각 수집 (학습용)
    print("[AE] 윈도우 데이터 구축 중...")
    all_windows = []
    trial_window_map = []  # 각 trial이 몇 번째 윈도우인지 범위(index) 기록
    idx = 0
    for trial in trials:
        windows = build_windows(trial["data"], window_size=window_size, step=window_size // 2)
        all_windows.append(windows)
        trial_window_map.append((idx, idx + len(windows)))
        idx += len(windows)

    X = np.concatenate(all_windows, axis=0).astype(np.float32)  # (total_windows, window_size, F)
    X_flat_size = window_size * n_features

    tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=256, shuffle=True)

    # 2) AutoEncoder 학습
    torch.manual_seed(42)
    model = CNN1DAutoEncoder(window_size, n_features, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)          # (B, window_size, F)
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}")

    # 3) 인코더로 각 Trial의 대표 임베딩산출 (윈도우 평균)
    model.eval()
    all_latents = []
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        _, latents = model(X_tensor)  # (total_windows, latent_dim)
        latents = latents.cpu().numpy()

    embeddings = []
    for (start, end) in trial_window_map:
        emb = latents[start:end].mean(axis=0)  # 윈도우 평균 = Trial 대표 임베딩
        embeddings.append(emb)
    return np.array(embeddings)


def embed_with_centroid(trials, n_features, latent_dim, window_size):
    """
    Centroid: PCA로 윈도우 조각들을 잠재 공간에 투영한 뒤,
    같은 Trial에 속한 조각들의 무게중심(Centroid)을 대표 임베딩으로 산출.
    """
    from sklearn.decomposition import PCA

    print("[Centroid] 윈도우 데이터 구축 중...")
    all_windows = []
    trial_window_map = []
    idx = 0
    for trial in trials:
        windows = build_windows(trial["data"], window_size=window_size, step=window_size // 2)
        all_windows.append(windows)
        trial_window_map.append((idx, idx + len(windows)))
        idx += len(windows)

    X = np.concatenate(all_windows, axis=0)  # (total_windows, window_size, F)
    X_flat = X.reshape(X.shape[0], -1)        # (total_windows, window_size * F)

    # 차원이 너무 크므로 PCA로 latent_dim까지 압축
    print(f"  PCA 압축 중... ({X_flat.shape[1]} → {latent_dim})")
    pca = PCA(n_components=min(latent_dim, X_flat.shape[0]-1, X_flat.shape[1]))
    X_pca = pca.fit_transform(X_flat)  # (total_windows, latent_dim)

    embeddings = []
    for (start, end) in tqdm(trial_window_map, desc="[Centroid] 무게중심 계산 중"):
        centroid = X_pca[start:end].mean(axis=0)
        embeddings.append(centroid)
    return np.array(embeddings)


# ─────────────────────────────────────────────
# 4. UMAP 2D 축소 및 결과 저장
# ─────────────────────────────────────────────

def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """UMAP으로 고차원 임베딩 → 2D 좌표 변환"""
    print("  📉 UMAP 2D 차원 축소 중...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(embeddings)


def save_results(trials, embeddings_high, coords_2d, output_path: Path):
    """임베딩 벡터(고차원) + 2D 좌표 + 메타 정보를 Parquet으로 저장"""
    records = []
    for i, trial in enumerate(trials):
        row = {**trial["meta"]}
        # 고차원 임베딩을 각 차원별 컬럼으로 저장
        for j, val in enumerate(embeddings_high[i]):
            row[f"emb_{j}"] = float(val)
        row["umap_x"] = float(coords_2d[i, 0])
        row["umap_y"] = float(coords_2d[i, 1])
        records.append(row)

    df_out = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"  💾 결과 저장 완료 → {output_path}")
    return df_out


# ─────────────────────────────────────────────
# 5. 메인 실행 엔트리
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="보행 시계열 임베딩 파이프라인")
    parser.add_argument("--input_path",  type=Path,
                        default=ROOT / "data" / "processed" / "Master_Gait_Dataset.parquet",
                        help="입력 Parquet 파일 경로")
    parser.add_argument("--model",       type=str, default="cnn",
                        choices=["cnn", "ae", "centroid"],
                        help="임베딩 모델: cnn | ae | centroid")
    parser.add_argument("--target_hz",   type=int, default=100,
                        help="다운샘플링 목표 주파수 (기본 100Hz = 변경 없음)")
    parser.add_argument("--latent_dim",  type=int, default=128,
                        help="임베딩 벡터 차원 수 (기본 128)")
    parser.add_argument("--window_size", type=int, default=100,
                        help="윈도우 조각 크기 (ae / centroid 전용, 기본 100프레임)")
    parser.add_argument("--epochs",      type=int, default=20,
                        help="AutoEncoder 학습 에폭 수 (ae 전용, 기본 20)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 보행 임베딩 파이프라인 시작")
    print(f"   모델   : {args.model.upper()}")
    print(f"   데이터 : {args.input_path.name}")
    print(f"   Device : {device}\n")

    # ── 데이터 로딩 ──
    df = load_parquet(args.input_path)

    # ── Trial 단위 분리 ──
    trials = split_by_trial(df)
    print(f"\n  ✅ Trial 수: {len(trials)}개 (피험자 단위 시행)\n")

    # ── 다운샘플링 ──
    if args.target_hz < 100:
        print(f"  📉 다운샘플링 {100}Hz → {args.target_hz}Hz (stride={int(100/args.target_hz)})")
        for t in trials:
            t["data"] = downsample(t["data"], source_hz=100, target_hz=args.target_hz)

    # ── 피처 정규화 (표준화) ──
    print("  🔧 StandardScaler 정규화 중...")
    all_data = np.concatenate([t["data"] for t in trials], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_data)
    for t in trials:
        t["data"] = scaler.transform(t["data"]).astype(np.float32)

    n_features = trials[0]["data"].shape[1]
    print(f"  피처 수 : {n_features}\n")

    # ── 임베딩 추출 ──
    if args.model == "cnn":
        embeddings = embed_with_cnn(trials, n_features, args.latent_dim, device)
    elif args.model == "ae":
        embeddings = embed_with_autoencoder(
            trials, n_features, args.latent_dim, args.window_size, device, args.epochs
        )
    elif args.model == "centroid":
        embeddings = embed_with_centroid(trials, n_features, args.latent_dim, args.window_size)

    print(f"\n  ✅ 임베딩 완료: {embeddings.shape}")

    # ── UMAP 2D 축소 ──
    coords_2d = reduce_to_2d(embeddings)

    # ── 결과 저장 ──
    dataset_tag = args.input_path.stem  # 파일명(확장자 제외)
    out_filename = f"embedding_results_{args.model}_{dataset_tag}.parquet"
    output_path = ROOT / "data" / "processed" / out_filename
    save_results(trials, embeddings, coords_2d, output_path)

    print(f"\n🎉 완료! 결과 파일: data/processed/{out_filename}")


if __name__ == "__main__":
    main()
