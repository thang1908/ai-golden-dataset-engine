"""ONNX Runtime session construction, shared by both towers.

Two things this module exists to hold:

* **Device selection that cannot silently downgrade.** ``cuda`` *requires* the
  CUDA provider. A deployment that asks for the GPU and quietly gets the CPU is
  the failure mode this prevents — it does not error, it just runs ten times
  slower forever. ``auto`` is explicitly permitted to fall back, because that is
  what the caller asked for.
* **Lazy import.** ``onnxruntime`` is imported inside the function, so the
  taxonomy and preprocessing halves of this package stay importable on a machine
  with no runtime and no model files.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any


class ModelUnavailable(RuntimeError):
    """The model could not be loaded, or violates its serving contract."""


class Device(StrEnum):
    AUTO = "auto"
    """CUDA when the provider is present, CPU otherwise."""

    CPU = "cpu"
    CUDA = "cuda"


def build_session(
    model_path: Path,
    *,
    device: Device | str = Device.AUTO,
    intra_op_threads: int | None = None,
) -> Any:
    """Open one ONNX Runtime session on the requested device."""
    path = Path(model_path)
    device = Device(device)

    try:
        import onnxruntime
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ModelUnavailable("onnxruntime is not installed; see requirements.txt") from error

    if not path.is_file():
        raise ModelUnavailable(f"model not found at {path}")

    available = set(onnxruntime.get_available_providers())
    if device is Device.CUDA:
        if "CUDAExecutionProvider" not in available:
            raise ModelUnavailable(
                "device='cuda' was requested but CUDAExecutionProvider is unavailable; "
                f"installed providers: {sorted(available)}. Install onnxruntime-gpu, or "
                "pass device='auto' to accept CPU."
            )
        # ORT intentionally places a few shape/bookkeeping nodes on CPU. CUDA
        # stays first for the transformer compute; requiring its availability
        # above is what stops a GPU deployment from becoming CPU-only.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device is Device.AUTO and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    options = onnxruntime.SessionOptions()
    if intra_op_threads is not None:
        if intra_op_threads < 1:
            raise ValueError("intra_op_threads must be at least 1")
        options.intra_op_num_threads = intra_op_threads

    try:
        return onnxruntime.InferenceSession(str(path), sess_options=options, providers=providers)
    except Exception as error:
        raise ModelUnavailable(
            f"could not load {path.name}: {type(error).__name__}: {error}"
        ) from error


def describe_io(session: Any) -> dict[str, Any]:
    """Session inputs and outputs, for error messages and diagnostics."""
    return {
        "providers": list(session.get_providers()),
        "inputs": [
            {"name": item.name, "type": item.type, "shape": list(item.shape)}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "type": item.type, "shape": list(item.shape)}
            for item in session.get_outputs()
        ],
    }
