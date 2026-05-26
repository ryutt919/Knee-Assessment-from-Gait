"""
Transformer Encoder 분류 모델 (PyTorch)
입력: (B, T, 9) — T 타임스텝 × 9채널
하드웨어 가속: CUDA (NVIDIA) / MPS (Apple Silicon) / CPU 자동 감지
"""
from __future__ import annotations
import math
from copy import deepcopy

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from utils.device import get_device
from .base import BaseModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerNet(nn.Module):
    def __init__(self, in_channels: int, d_model: int, n_heads: int,
                 num_layers: int, dim_feedforward: int, dropout: float,
                 attention_dropout: float, n_classes: int, speed_dim: int = 0):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        fc_in = d_model + speed_dim
        self.fc = nn.Linear(fc_in, n_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, speed: torch.Tensor | None = None,
                src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, 9)
        B = x.size(0)
        x = self.proj(x)  # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)
        x = self.pos_enc(x)

        # padding mask: (B, T+1), float "-inf"로 변환 (MPS Segfault 방지)
        if src_key_padding_mask is not None and src_key_padding_mask.any():
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            bool_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
            # True(패딩) -> -inf, False(유효) -> 0.0
            float_mask = torch.zeros_like(bool_mask, dtype=x.dtype)
            float_mask.masked_fill_(bool_mask, float("-inf"))
            src_key_padding_mask = float_mask
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls_out = self.norm(x[:, 0])  # CLS token
        if speed is not None:
            cls_out = torch.cat([cls_out, speed], dim=1)
        return self.fc(cls_out)


class TransformerModel(BaseModel):
    name = "transformer"

    def __init__(self, n_classes: int = 2, d_model: int = 64, n_heads: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.3, attention_dropout: float = 0.1,
                 lr: float = 1e-4, weight_decay: float = 1e-3,
                 warmup_ratio: float = 0.05, batch_size: int = 16,
                 max_epochs: int = 200, patience: int = 15, min_delta: float = 0.001,
                 speed_dim: int = 0, force_cpu: bool = False, **kw):
        self.params = dict(
            d_model=d_model, n_heads=n_heads, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            attention_dropout=attention_dropout, n_classes=n_classes,
            speed_dim=speed_dim,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.device = get_device(force_cpu=force_cpu)
        self.net: TransformerNet | None = None

    def _build(self, in_channels: int):
        self.net = TransformerNet(in_channels=in_channels, **self.params).to(self.device)

    def _make_loader(self, X, speed, mask, y, shuffle: bool) -> DataLoader:
        # X: (N, 9, T) → transpose to (N, T, 9) and make memory contiguous (MPS segfault 방지)
        X_t = np.ascontiguousarray(X.transpose(0, 2, 1))
        tensors = [torch.tensor(X_t, dtype=torch.float32)]
        tensors.append(torch.tensor(speed, dtype=torch.float32) if speed is not None
                       else torch.zeros(len(X), 0))
        # padding mask (N, T): True where padded
        if mask is not None:
            pad_mask = ~mask[:, 0, :]  # (N, T) False=유효 → True=패딩
            tensors.append(torch.tensor(pad_mask, dtype=torch.bool))
        else:
            tensors.append(torch.zeros(len(X), X_t.shape[1], dtype=torch.bool))
        tensors.append(torch.tensor(y, dtype=torch.long))
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def fit(self, X: np.ndarray, y: np.ndarray,
            speed: np.ndarray | None = None, mask: np.ndarray | None = None,
            val_data: tuple | None = None, trial=None, mlflow_run=None,
            verbose: int = 10, **kw):
        """verbose: 몇 epoch마다 진행 출력 (0=비활성, 1=매 epoch)"""
        self._build(in_channels=X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_steps  = self.max_epochs * max(1, len(X) // self.batch_size)
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )

        criterion = nn.CrossEntropyLoss()
        loader = self._make_loader(X, speed, mask, y, shuffle=True)
        best_f1, best_weights, patience_counter = -1.0, None, 0
        print(f"  [Transformer] 학습 시작: {len(X)}샘플, {self.max_epochs}epochs, "
              f"warmup={warmup_steps}steps, device={self.device}")

        for epoch in range(self.max_epochs):
            self.net.train()
            for batch in loader:
                xb, sb, pmb, yb = [t.to(self.device) for t in batch]
                opt.zero_grad()
                logits = self.net(xb, sb if sb.shape[1] > 0 else None, pmb)
                criterion(logits, yb).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                opt.step()
                if epoch == 0:
                    scheduler.step()

            if val_data is not None:
                X_v, sp_v, mk_v, y_v = val_data
                val_preds = self._predict_raw(X_v, sp_v, mk_v)
                val_f1 = f1_score(y_v, val_preds, average="macro", zero_division=0)
                lr_now  = scheduler.get_last_lr()[0]

                if mlflow_run:
                    mlflow.log_metrics({"val_macro_f1": val_f1, "lr": lr_now}, step=epoch)

                if verbose and (epoch % max(1, verbose) == 0 or epoch == self.max_epochs - 1):
                    print(f"    epoch {epoch+1:>4}/{self.max_epochs}  "
                          f"val_f1={val_f1:.4f}  best={best_f1:.4f}  "
                          f"lr={lr_now:.2e}  patience={patience_counter}/{self.patience}")

                if val_f1 > best_f1 + self.min_delta:
                    best_f1, best_weights, patience_counter = val_f1, deepcopy(self.net.state_dict()), 0
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
        loader = self._make_loader(X, speed, mask, np.zeros(len(X), dtype=int), shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                xb, sb, pmb, _ = [t.to(self.device) for t in batch]
                logits = self.net(xb, sb if sb.shape[1] > 0 else None, pmb)
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    def predict(self, X, speed=None, mask=None) -> np.ndarray:
        return self._predict_raw(X, speed, mask)

    def predict_proba(self, X, speed=None, mask=None) -> np.ndarray:
        self.net.eval()
        loader = self._make_loader(X, speed, mask, np.zeros(len(X), dtype=int), shuffle=False)
        probs = []
        with torch.no_grad():
            for batch in loader:
                xb, sb, pmb, _ = [t.to(self.device) for t in batch]
                logits = self.net(xb, sb if sb.shape[1] > 0 else None, pmb)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs)
