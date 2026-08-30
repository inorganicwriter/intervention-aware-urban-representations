"""Deterministic and explicit PyTorch runtime configuration."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


class TorchUnavailableError(RuntimeError):
    """Raised when the optional GPU backend is used without PyTorch."""


def require_torch() -> Any:
    """Import PyTorch lazily so the base package remains a light dependency."""
    # CUDA deterministic matrix multiplication requires this to be present
    # before the first cuBLAS handle is created.  ``setdefault`` preserves an
    # explicit operator choice while making spawned GPU workers reproducible.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise TorchUnavailableError(
            "The causal GPU backend requires PyTorch. Install the project 'ml' extra "
            "or a CUDA-enabled PyTorch build."
        ) from exc
    return torch


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Numerical and device policy for a causal estimator worker."""

    device: str = "auto"
    dtype: str = "float64"
    deterministic: bool = True
    allow_tf32: bool = False
    memory_fraction: float = 0.85
    chunk_size: int = 65_536
    seed: int = 20260823

    def __post_init__(self) -> None:
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if not 0 < self.memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0, 1]")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")


class TorchRuntime:
    """Resolved PyTorch runtime shared by all three estimators.

    Formal estimation defaults to float64 and TF32 is disabled.  A caller may
    explicitly select float32 for screening benchmarks, but that choice is
    recorded in provenance and is never made automatically.
    """

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.torch = require_torch()
        self.device = self._resolve_device(self.config.device)
        self.dtype = getattr(self.torch, self.config.dtype)
        self._configure()

    def _resolve_device(self, requested: str) -> Any:
        if requested == "auto":
            requested = "cuda:0" if self.torch.cuda.is_available() else "cpu"
        device = self.torch.device(requested)
        if device.type == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
        if device.type == "cuda":
            count = self.torch.cuda.device_count()
            index = 0 if device.index is None else device.index
            if index >= count:
                raise ValueError(f"CUDA device index {index} is outside available range 0..{count - 1}")
            # torch.cuda.set_device requires an explicit index even though
            # torch.device("cuda") is otherwise a valid device target.
            device = self.torch.device("cuda", index)
        return device

    def _configure(self) -> None:
        torch = self.torch
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.cuda.set_per_process_memory_fraction(
                self.config.memory_fraction,
                device=self.device,
            )
            torch.backends.cuda.matmul.allow_tf32 = self.config.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.config.allow_tf32
            torch.backends.cudnn.deterministic = self.config.deterministic
            torch.backends.cudnn.benchmark = False
        if self.config.deterministic:
            torch.use_deterministic_algorithms(True)

    def tensor(self, value: Any, *, dtype: Any | None = None) -> Any:
        """Create a tensor without exposing read-only NumPy storage to PyTorch.

        ``torch.as_tensor`` may share NumPy storage.  Arrow- and Parquet-backed
        arrays can be read-only, while PyTorch tensors do not carry a read-only
        flag.  Copy only those arrays so writable inputs retain the existing
        zero-copy CPU path and read-only inputs cannot be mutated through a
        tensor view.
        """
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            return self.torch.tensor(
                value, dtype=dtype or self.dtype, device=self.device
            )
        return self.torch.as_tensor(value, dtype=dtype or self.dtype, device=self.device)

    def empty_cache(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def metadata(self) -> dict[str, Any]:
        """Return serialisable numerical provenance for estimator artifacts."""
        data: dict[str, Any] = asdict(self.config)
        data["resolved_device"] = str(self.device)
        data["torch_version"] = self.torch.__version__
        data["cuda_version"] = self.torch.version.cuda
        data["cudnn_version"] = self.torch.backends.cudnn.version()
        data["python_version"] = platform.python_version()
        data["python_implementation"] = platform.python_implementation()
        data["python_executable"] = str(sys.executable)
        if self.device.type == "cuda":
            data["gpu_name"] = self.torch.cuda.get_device_name(self.device)
            data["gpu_capability"] = list(self.torch.cuda.get_device_capability(self.device))
        else:
            data["gpu_name"] = None
            data["gpu_capability"] = None
        return data


def is_cuda_oom(error: BaseException) -> bool:
    """Recognise CUDA allocation errors without importing torch eagerly."""
    text = str(error).lower()
    return "cuda" in text and ("out of memory" in text or "memory allocation" in text)
