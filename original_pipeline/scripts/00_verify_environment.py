"""Workstation Python, GPU ve kritik paket doğrulaması."""
from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = [
    "numpy", "pandas", "cv2", "albumentations", "sklearn", "skimage",
    "torch", "torchvision", "segmentation_models_pytorch", "timm", "yaml",
]


def main() -> None:
    print("=" * 72)
    print("MYELOFIBROSIS PIPELINE | ORTAM DOGRULAMA")
    print("=" * 72)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    failures: list[str] = []

    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {name:<32} {version}")
        except Exception as exc:  # Kullanıcıya paket adıyla doğrudan hata verilir.
            failures.append(name)
            print(f"[FAIL] {name:<30} {type(exc).__name__}: {exc}")

    try:
        import torch
        print("-" * 72)
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA runtime:   {torch.version.cuda}")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"GPU:                    {torch.cuda.get_device_name(device)}")
            props = torch.cuda.get_device_properties(device)
            print(f"VRAM:                   {props.total_memory / 1024**3:.1f} GB")
            probe = torch.randn(1024, 1024, device=device)
            _ = probe @ probe.T
            torch.cuda.synchronize()
            print("[OK] CUDA matmul smoke test")
        else:
            failures.append("CUDA/GPU")
            print("[FAIL] CUDA kullanilamiyor. CPU ile egitim baslatmayin.")
    except Exception as exc:
        failures.append("PyTorch GPU test")
        print(f"[FAIL] PyTorch GPU test: {type(exc).__name__}: {exc}")

    print("=" * 72)
    if failures:
        print("DOGRULAMA BASARISIZ. Duzenlenecek kalemler:", ", ".join(failures))
        raise SystemExit(1)
    print("DOGRULAMA BASARILI. Veri kontrolune gecebilirsiniz.")


if __name__ == "__main__":
    main()
