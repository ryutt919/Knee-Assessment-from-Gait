"""
하드웨어 가속 디바이스 자동 감지
- Mac (Apple Silicon): MPS (Metal Performance Shaders)
- Windows/Linux (NVIDIA): CUDA
- fallback: CPU

환경변수 FORCE_CPU=1 이면 항상 CPU 반환 (소규모 스모크 테스트 등에서 사용).
MPS는 처음 실행 시 Metal 셰이더 컴파일 오버헤드가 있어 소규모 데이터에서는
CPU가 오히려 빠를 수 있다.
"""
import os
import torch

# 환경변수로 CPU 강제 지정 가능
_FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
_MPS_WARMED_UP = False


def _warmup_mps(device: "torch.device") -> None:
    """MPS Metal 셰이더 사전 컴파일 — Transformer 이동 시 segfault 방지.

    TransformerEncoderLayer 포함 미니 forward pass를 실행해서
    MultiheadAttention · LayerNorm · Linear 관련 셰이더를 미리 컴파일.
    """
    global _MPS_WARMED_UP
    if _MPS_WARMED_UP:
        return
    import torch.nn as nn
    d = 32
    enc = nn.TransformerEncoderLayer(
        d_model=d, nhead=2, dim_feedforward=64,
        dropout=0.0, batch_first=True,
    ).to(device)
    x = torch.zeros(1, 4, d, device=device)
    with torch.no_grad():
        _ = enc(x)
    del enc, x, _
    _MPS_WARMED_UP = True


def get_device(verbose: bool = True, force_cpu: bool = False) -> torch.device:
    """
    force_cpu=True 또는 FORCE_CPU=1 환경변수 설정 시 CPU 반환.
    소규모 테스트에서 MPS 셰이더 컴파일 오버헤드 회피용.
    """
    import multiprocessing
    omp_threads = os.environ.get("OMP_NUM_THREADS", None)
    phys_cores = multiprocessing.cpu_count()
    cpu_info = (
        f"물리 코어={phys_cores}, "
        f"OMP_NUM_THREADS={omp_threads if omp_threads else '미설정(전체)'}개 사용"
    )

    if force_cpu or _FORCE_CPU:
        if verbose:
            print(f"[device] CPU 강제 사용 (force_cpu=True / FORCE_CPU=1) | {cpu_info}")
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            print(f"[device] CUDA 사용: {name}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        _warmup_mps(device)
        if verbose:
            print(f"[device] Apple MPS 사용 (Apple Silicon) | {cpu_info}")
    else:
        device = torch.device("cpu")
        if verbose:
            print(f"[device] CPU 사용 (하드웨어 가속 없음) | {cpu_info}")
    return device


def to_device(obj, device: torch.device):
    """모델 또는 텐서를 지정 디바이스로 이동."""
    return obj.to(device)
