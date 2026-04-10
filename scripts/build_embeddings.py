"""
scripts/build_embeddings.py

1D-CNN+GAP / AutoEncoder / Centroid 기반 보행 임베딩 파이프라인
Trial 단위(1파일=1점)로 임베딩을 생성하고 Parquet으로 저장.

사용법:
    python scripts/build_embeddings.py --input_path data/processed/Master_Gait_Dataset_lower.parquet --model cnn
    python scripts/build_embeddings.py --input_path data/processed/Master_Gait_Dataset_lower.parquet --model ae
    python scripts/build_embeddings.py --input_path data/processed/Master_Gait_Dataset_lower.parquet --model centroid

    # 여러 모델 순차 실행
    python scripts/build_embeddings.py --input_path data/processed/Master_Gait_Dataset_lower.parquet --model cnn ae centroid

--model 옵션: cnn | ae | centroid (여러 개 나열 가능)
"""

import argparse
import sys
import gc
import warnings
from pathlib import Path

# 불필요한 경고 메시지(UMAP, TensorFlow 등) 숨기기
warnings.filterwarnings("ignore")

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
    load_trials_chunked,
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
            nn.Conv1d(128, 256, kernel_size=5, padding=2),  # 중간 패턴 정제
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, latent_dim, kernel_size=3, padding=1),  # 최종 피처 맵
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        # Global Average Pooling: 시간 축(dim=2) 평균 → (batch, latent_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv_layers(x)  # (batch, latent_dim, T)
        x = self.gap(x)  # (batch, latent_dim, 1)
        return x.squeeze(-1)  # (batch, latent_dim)


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
            data = trial["data"]  # (T, F)
            x = torch.tensor(data.T, dtype=torch.float32).unsqueeze(0).to(device)  # (1, F, T)
            vec = model(x).cpu().numpy().squeeze()  # (latent_dim,)
            embeddings.append(vec)
    return np.array(embeddings)


def embed_with_autoencoder(trials, n_features, latent_dim, window_size, device, epochs=30):
    """
    AutoEncoder: 윈도우 조각들로 학습한 뒤 평균(Centroid) 인코딩 벡터 산출 (OOM 방지 Dataset 적용)
    """

    class WindowDataset(torch.utils.data.Dataset):
        def __init__(self, trials_list):
            self.windows = []
            self.index_map = []
            for t_idx, trial in enumerate(trials_list):
                w = build_windows(trial["data"], window_size=window_size, step=window_size // 2)
                self.windows.append(w)
                for w_idx in range(len(w)):
                    self.index_map.append((t_idx, w_idx))

        def __len__(self):
            return len(self.index_map)

        def __getitem__(self, idx):
            t_idx, w_idx = self.index_map[idx]
            return torch.tensor(self.windows[t_idx][w_idx], dtype=torch.float32)

    print("[AE] 윈도우 데이터셋 구축 중 (OOM 최적화)...")
    dataset = WindowDataset(trials)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    torch.manual_seed(42)
    model = CNN1DAutoEncoder(window_size, n_features, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # 학습률 하향 조정 (지그재그 방지)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # Best 모델 저장 (Loss가 오르더라도 가장 좋았던 시점 기억)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} (Best: {best_loss:.4f})")

    # 가장 낮은 Loss를 기록했던 시점의 모델로 복원
    if best_model_state is not None:
        print(f"  ✅ Best Loss({best_loss:.4f}) 시점의 가중치로 복원합니다.")
        model.load_state_dict(best_model_state)

    # 인코더로 각 Trial의 평균 임베딩 도출
    model.eval()
    embeddings = []
    with torch.no_grad():
        for t_idx, trial in tqdm(enumerate(trials), desc="[AE] 임베딩 추출", total=len(trials)):
            w = dataset.windows[t_idx]  # (num_windows, W, F)
            w_tensor = torch.tensor(w, dtype=torch.float32).to(device)
            _, latents = model(w_tensor)
            embeddings.append(latents.cpu().numpy().mean(axis=0))

    return np.array(embeddings)


def embed_with_centroid(trials, n_features, latent_dim, window_size):
    """
    Centroid: IncrementalPCA로 OOM 없이 윈도우 조각들을 투영하고 평균(Centroid) 벡터를 산출.
    """
    from sklearn.decomposition import IncrementalPCA

    print(f"[Centroid] Incremental PCA 학습 준비 (Batch chunking)...")
    latent_dim = min(latent_dim, len(trials))
    ipca = IncrementalPCA(n_components=latent_dim, batch_size=max(latent_dim * 2, 512))

    buffer = []
    buffer_sz = 0
    batch_limit = max(latent_dim * 2, 1024)

    for trial in tqdm(trials, desc="  [Centroid] IPCA Fitting"):
        w = build_windows(trial["data"], window_size=window_size, step=window_size // 2)
        w_flat = w.reshape(w.shape[0], -1)
        buffer.append(w_flat)
        buffer_sz += w_flat.shape[0]

        if buffer_sz >= batch_limit:
            X_batch = np.concatenate(buffer, axis=0)
            ipca.partial_fit(X_batch)
            buffer = []
            buffer_sz = 0

    if buffer:
        X_batch = np.concatenate(buffer, axis=0)
        if X_batch.shape[0] >= latent_dim:
            ipca.partial_fit(X_batch)

    embeddings = []
    for trial in tqdm(trials, desc="  [Centroid] 공간 투영 및 무게중심 계산"):
        w = build_windows(trial["data"], window_size=window_size, step=window_size // 2)
        w_flat = w.reshape(w.shape[0], -1)
        z = ipca.transform(w_flat)
        embeddings.append(z.mean(axis=0))

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
    parser.add_argument("--input_path", type=Path, default=ROOT / "data" / "processed" / "Master_Gait_Dataset.parquet", help="입력 Parquet 파일 경로")
    parser.add_argument(
        "--model",
        type=str,
        default=["cnn"],
        nargs="+",
        choices=["cnn", "ae", "centroid"],
        help="임베딩 모델: cnn | ae | centroid (여러 개 나열 시 순차 실행)",
    )
    parser.add_argument("--target_hz", type=int, default=100, help="다운샘플링 목표 주파수 (기본 100Hz = 변경 없음)")
    parser.add_argument("--latent_dim", type=int, default=128, help="임베딩 벡터 차원 수 (기본 128)")
    parser.add_argument("--window_size", type=int, default=100, help="윈도우 조각 크기 (ae / centroid 전용, 기본 100프레임)")
    parser.add_argument("--epochs", type=int, default=20, help="AutoEncoder 학습 에폭 수 (ae 전용, 기본 20)")
    return parser.parse_args()


def run_model(model_name, trials, n_features, args, device):
    """단일 모델 임베딩 추출 → UMAP → 저장"""
    print(f"\n{'=' * 55}")
    print(f"  모델: {model_name.upper()}")
    print(f"{'=' * 55}")

    if model_name == "cnn":
        embeddings = embed_with_cnn(trials, n_features, args.latent_dim, device)
    elif model_name == "ae":
        embeddings = embed_with_autoencoder(trials, n_features, args.latent_dim, args.window_size, device, args.epochs)
    elif model_name == "centroid":
        embeddings = embed_with_centroid(trials, n_features, args.latent_dim, args.window_size)

    print(f"\n  ✅ 임베딩 완료: {embeddings.shape}")

    coords_2d = reduce_to_2d(embeddings)

    dataset_tag = args.input_path.stem
    out_filename = f"embedding_results_{model_name}_{dataset_tag}.parquet"
    output_path = ROOT / "data" / "processed" / out_filename
    save_results(trials, embeddings, coords_2d, output_path)

    print(f"\n🎉 완료! 결과 파일: data/processed/{out_filename}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = args.model  # list
    print(f"\n🚀 보행 임베딩 파이프라인 시작")
    print(f"   모델   : {' → '.join(m.upper() for m in models)}")
    print(f"   데이터 : {args.input_path.name}")
    print(f"   Device : {device}\n")

    # ── 데이터 로딩 및 Trial 분리 (Chunk Streaming) ──
    trials = load_trials_chunked(args.input_path)
    print(f"\n  ✅ Trial 수: {len(trials)}개 (피험자 단위 시행)\n")

    # ── 다운샘플링 ──
    if args.target_hz < 100:
        print(f"  📉 다운샘플링 {100}Hz → {args.target_hz}Hz (stride={int(100 / args.target_hz)})")
        for t in trials:
            t["data"] = downsample(t["data"], source_hz=100, target_hz=args.target_hz)

    # ── 피처 정규화 (Incremental Scaling) ──
    # 전체를 묶어서 fit() 하면 6GB+ 데이터에서 MemoryError 발생 가능.
    # 따라서 partial_fit()으로 각 Trial씩 순차적으로 학습하여 메모리 절약.
    print("  🔧 StandardScaler 정규화 중 (Incremental Scaling)...")
    scaler = StandardScaler()
    for t in tqdm(trials, desc="  [Scaling] Fitting"):
        scaler.partial_fit(t["data"])

    for t in tqdm(trials, desc="  [Scaling] Transforming"):
        t["data"] = scaler.transform(t["data"]).astype(np.float32)

    n_features = trials[0]["data"].shape[1]
    print(f"  피처 수 : {n_features}\n")

    # ── 모델별 순차 실행 ──
    for model_name in models:
        run_model(model_name, trials, n_features, args, device)
        gc.collect()


if __name__ == "__main__":
    main()
