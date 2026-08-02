# fused-ssim — ROCm 7.2 / RDNA4 / gfx1201

<details>
<summary><strong>ROCm / gfx1201 port note</strong></summary>

- HIP/ROCm port of `fused-ssim` (Rahul Goel), a fully-fused differentiable SSIM.
- Target: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201`.
- Stack: ROCm 7.2, Python 3.12, PyTorch `2.13.0+rocm7.2`.
- Context: SSIM training-loss helper of the AMD Gaussian Splatting ROCm/gfx1201 stack.
- License: MIT — see [`LICENSE`](./LICENSE).
</details>

---

## What this is

`fused-ssim` is an efficient, fully-fused, **differentiable SSIM** (Structural
Similarity) implementation. This repository is a HIP/ROCm build of that
extension, adapted and validated for AMD RDNA4 / `gfx1201` under ROCm 7.2.

It is **not** a new algorithm — it is an AMD enablement port of the original
implementation, so that SSIM-based training (e.g. the 3D Gaussian Splatting
pipeline) can run on Radeon RDNA4 hardware.

## What it does

SSIM measures the perceptual similarity between two images. Because it is
differentiable here, it can be used directly as a training loss.

The "fused" implementation is fast because:

- the SSIM convolutions are computed on-chip in a single fused pass, without
  writing intermediate results back to global memory,
- the 11×11 Gaussian kernel is separable and symmetric, cutting the number of
  operations,
- backpropagation through a Gaussian convolution is itself a Gaussian
  convolution, so the backward pass reuses the same fast path.

In 3D Gaussian Splatting it is used as (part of) the image reconstruction loss
during optimization.

## Usage

```python
import torch
from fused_ssim import fused_ssim

# On ROCm, "cuda" maps to the AMD GPU through the HIP runtime.
device = "cuda"

# images: [B, C, H, W], normalized to [0, 1]
gt_image = torch.rand(2, 3, 1080, 1920, device=device)
predicted_image = torch.nn.Parameter(torch.rand_like(gt_image))

# Differentiable w.r.t. the predicted image (the first argument):
ssim_value = fused_ssim(predicted_image, gt_image)

loss = 1.0 - ssim_value
loss.backward()
```

## Where it fits

```text
gaussian-splatting (training / rendering pipeline)
        │
        ├── diff-gaussian-rasterization   (rendering + gradients)
        ├── simple-knn                    (initial Gaussian scales)
        └── fused-ssim                    ← this repo (ROCm/gfx1201 port)
                └── differentiable SSIM used in the reconstruction loss
```

Companion repositories in the AMD ROCm/gfx1201 stack:

- [`amd-gaussian-splatting-rocm72-gfx1201`](https://github.com/Painter3000/amd-gaussian-splatting-rocm72-gfx1201)
- [`diff-gaussian-rasterization-rocm72-gfx1201`](https://github.com/Painter3000/diff-gaussian-rasterization-rocm72-gfx1201)
- [`simple-knn-rocm72-gfx1201`](https://github.com/Painter3000/simple-knn-rocm72-gfx1201)

## Installation

This fork targets a ROCm-enabled PyTorch environment. It does **not** install
AMD drivers, ROCm, or PyTorch — you need a working ROCm PyTorch environment
first.

Verify your environment:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Arch:", torch.cuda.get_device_properties(0).gcnArchName)
PY
```

Build and install from the repository root:

```bash
export PYTORCH_ROCM_ARCH=gfx1201
pip install . --no-build-isolation -v
```

> **Note:** PyTorch keeps the `cuda` device name and the `CUDAExtension` build
> interface for ROCm/HIP builds. Seeing `cuda` in Python or in `setup.py` does
> **not** mean NVIDIA CUDA is being used.

## Validated environment

```text
GPU:        AMD Radeon AI PRO R9700 / gfx1201 / RDNA4
ROCm:       7.2
PyTorch:    2.13.0+rocm7.2
Python:     3.12
```

## Performance

The upstream `fused-ssim` reports roughly a **5–8× speedup** over the previously
fastest differentiable SSIM implementation,
[pytorch-msssim](https://github.com/VainF/pytorch-msssim).

This ROCm/gfx1201 fork preserves that implementation path, so the figure above
should be read as **upstream context** — not as a measured RDNA4 benchmark —
unless a separate ROCm/RDNA4 benchmark is documented here.

## Source and attribution

- **Upstream:** [`rahul-goel/fused-ssim`](https://github.com/rahul-goel/fused-ssim)
  by Rahul Goel — "Lightning fast differentiable SSIM".
- Also used as a submodule of
  [`graphdeco-inria/gaussian-splatting`](https://github.com/graphdeco-inria/gaussian-splatting).
- This repository contains that upstream code with HIP/ROCm adaptations for
  RDNA4 / `gfx1201`. The SSIM logic and kernels originate upstream; the changes
  here are limited to AMD/ROCm build and runtime enablement.

## License

MIT License — see [`LICENSE`](./LICENSE). The MIT terms apply to both the
upstream code and this port.
