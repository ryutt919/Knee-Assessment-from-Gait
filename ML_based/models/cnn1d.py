"""
1D-CNN 분류 모델 (PyTorch)
입력: (B, 9, T) — 9채널 × T 타임스텝 (raw_padded or norm_101)
하드웨어 가속: CUDA (NVIDIA) / MPS (Apple Silicon) / CPU 자동 감지
"""
from __future__ import annotations
import os
from copy import deepcopy

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from utils.device import get_device
from .base import BaseModel


class CNN1DNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, num_layers: int,
                 channels: int, kernel_size: int, dilation: int,
                 dropout: float, speed_dim: int = 0):
        super().__init__()
        layers = []
        ch_in = in_channels
        for i in range(num_layers):
            layers += [
                nn.Conv1d(ch_in, channels, kernel_size,
                          padding=(kernel_size - 1) * dilation // 2, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            ch_in = channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        fc_in = channels + speed_dim
        self.fc = nn.Linear(fc_in, n_classes)

    def forward(self, x: torch.Tensor, speed: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, 9, T)
        if mask is not None:
            # mask: (B, 9, T) — False 위치를 0으로
            x = x * mask.float()
        feat = self.pool(self.conv(x)).squeeze(-1)  # (B, channels)
        if speed is not None:
            feat = torch.cat([feat, speed], dim=1)
        return self.fc(feat)


class CNN1D(BaseModel):
    name = "cnn1d"

    def __init__(self, n_classes: int = 2, num_layers: int = 3, channels: int = 64,
                 kernel_size: int = 5, dilation: int = 1, dropout: float = 0.3,
                 lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 32,
                 max_epochs: int = 200, patience: int = 15, min_delta: float = 0.001,
                 speed_dim: int = 0, force_cpu: bool = False, **kw):
        self.params = dict(
            n_classes=n_classes, num_layers=num_layers, channels=channels,
            kernel_size=kernel_size, dilation=dilation, dropout=dropout,
            speed_dim=speed_dim,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.device = get_device(force_cpu=force_cpu)
        self.net: CNN1DNet | None = None

    def _build(self, in_channels: int):
        self.net = CNN1DNet(in_channels=in_channels, **self.params).to(self.device)

    def _make_loader(self, X, speed, mask, y, shuffle: bool) -> DataLoader:
        # MPS Segfault 방지를 위해 X를 contiguous array로 변환
        X = np.ascontiguousarray(X)
        tensors = [torch.tensor(X, dtype=torch.float32)]
        tensors.append(torch.tensor(speed, dtype=torch.float32) if speed is not None
                       else torch.zeros(len(X), 0))
        tensors.append(torch.tensor(mask, dtype=torch.bool) if mask is not None
                       else torch.ones(*X.shape, dtype=torch.bool))
        tensors.append(torch.tensor(y, dtype=torch.long))
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          drop_last=False, num_workers=0)

    def fit(self, X: np.ndarray, y: np.ndarray,
            speed: np.ndarray | None = None, mask: np.ndarray | None = None,
            val_data: tuple | None = None, trial=None, mlflow_run=None,
            verbose: int = 10, **kw):
        """verbose: 몇 epoch마다 진행 출력 (0=비활성, 1=매 epoch)"""
        self._build(in_channels=X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        loader = self._make_loader(X, speed, mask, y, shuffle=True)

        best_f1, best_weights, patience_counter = -1.0, None, 0
        print(f"  [CNN1D] 학습 시작: {len(X)}샘플, {self.max_epochs}epochs, device={self.device}")

        for epoch in range(self.max_epochs):
            self.net.train()
            for batch in loader:
                xb, sb, mb, yb = [t.to(self.device) for t in batch]
                opt.zero_grad()
                logits = self.net(xb, sb if sb.shape[1] > 0 else None,
                                  mb if mb.all() == False else None)
                criterion(logits, yb).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                opt.step()

            if val_data is not None:
                X_v, sp_v, mk_v, y_v = val_data
                val_preds = self._predict_raw(X_v, sp_v, mk_v)
                val_f1 = f1_score(y_v, val_preds, average="macro", zero_division=0)

                if mlflow_run:
                    mlflow.log_metrics({"val_macro_f1": val_f1}, step=epoch)

                if verbose and (epoch % max(1, verbose) == 0 or epoch == self.max_epochs - 1):
                    print(f"    epoch {epoch+1:>4}/{self.max_epochs}  "
                          f"val_f1={val_f1:.4f}  best={best_f1:.4f}  "
                          f"patience={patience_counter}/{self.patience}")

                if val_f1 > best_f1 + self.min_delta:
                    best_f1 = val_f1
                    best_weights = deepcopy(self.net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if mlflow_run:
                            mlflow.log_param("early_stop_epoch", epoch)
                        print(f"    Early stop at epoch {epoch+1}")
                        break

                if trial is not None:
                    trial.report(val_f1, step=epoch)
                    import optuna
                    if trial.should_prune():
                        raise optuna.TrialPruned()

        if best_weights is not None:
            self.net.load_state_dict(best_weights)
        return self

    def _predict_raw(self, X, speed=None, mask=None) -> np.ndarray:
        self.net.eval()
        loader = self._make_loader(X, speed, mask,
                                   np.zeros(len(X), dtype=int), shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                xb, sb, mb, _ = [t.to(self.device) for t in batch]
                logits = self.net(xb, sb if sb.shape[1] > 0 else None, None)
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    def predict(self, X, speed=None, mask=None) -> np.ndarray:
        return self._predict_raw(X, speed, mask)

    def predict_proba(self, X, speed=None, mask=None) -> np.ndarray:
        self.net.eval()
        loader = self._make_loader(X, speed, mask,
                                   np.zeros(len(X), dtype=int), shuffle=False)
        probs = []
        with torch.no_grad():
            for batch in loader:
                xb, sb, mb, _ = [t.to(self.device) for t in batch]
                logits = self.net(xb, sb if sb.shape[1] > 0 else None, None)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs)
