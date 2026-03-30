# setup.py — owns ONLY the CUDA extension build logic.
# All project metadata lives in pyproject.toml.
#
# Fast CPU-only install (no CUDA compilation):
#   MEMOPT_SKIP_CUDA=1 pip install -e . --no-build-isolation
#
# Full install with CUDA extension (requires nvcc on PATH):
#   pip install -e . --no-build-isolation
#
# Force build even if nvcc and PyTorch CUDA major versions differ:
#   MEMOPT_FORCE_CUDA=1 pip install -e . --no-build-isolation

import os
import re
import shutil
import subprocess
from setuptools import setup

# ---------------------------------------------------------------------------
# CUDA availability check — uses shutil.which(), NOT torch CUDA APIs.
# Safe to call at module-evaluation time: no device queries, no SIGSEGV.
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Return True when nvcc is on PATH and MEMOPT_SKIP_CUDA is not set.

    MEMOPT_FORCE_CUDA=1  — bypass the nvcc/PyTorch version-match check and
                           attempt the build regardless (useful when nvcc and
                           torch CUDA major versions differ but are ABI-compat).
    MEMOPT_SKIP_CUDA=1   — unconditionally skip the CUDA extension build.
    """
    if os.environ.get("MEMOPT_SKIP_CUDA", "0") == "1":
        return False
    if shutil.which("nvcc") is None:
        print("[memopt] nvcc not found on PATH — skipping CUDA extension build.")
        return False
    # Short-circuit: user explicitly wants to force the build.
    if os.environ.get("MEMOPT_FORCE_CUDA", "0") == "1":
        print("[memopt] MEMOPT_FORCE_CUDA=1 — skipping version-match check.")
        return True
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        match = re.search(r"release (\d+\.\d+)", result.stdout)
        if not match:
            return False
        nvcc_ver = match.group(1)

        import torch  # noqa: PLC0415 — deferred to avoid build-time CUDA init

        torch_cuda_ver = torch.version.cuda  # string attribute, no device query
        if torch_cuda_ver is None:
            # CPU-only PyTorch build
            return False

        nvcc_major = nvcc_ver.split(".")[0]
        torch_major = torch_cuda_ver.split(".")[0]
        if nvcc_major != torch_major:
            print(
                f"[memopt] WARNING: nvcc {nvcc_ver} != "
                f"PyTorch CUDA {torch_cuda_ver} (major version mismatch).\n"
                f"[memopt] Skipping CUDA build to avoid a broken extension.\n"
                f"[memopt] To override, re-run with:  "
                f"MEMOPT_FORCE_CUDA=1 pip install -e . --no-build-isolation"
            )
            return False

        return True
    except Exception:  # noqa: BLE001 — any failure → skip CUDA build safely
        return False


# ---------------------------------------------------------------------------
# Extension builder — torch imports are deferred inside this function so
# they are never executed during pip's "Getting requirements to build
# editable" phase.  They only run when setuptools actually invokes build_ext.
# ---------------------------------------------------------------------------

def _make_extension():
    """Return (ext_modules, cmdclass) for the memopt_C CUDA extension.

    All torch.utils.cpp_extension imports are local to this function so that
    importing setup.py itself never triggers a CUDA device query.
    """
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: PLC0415

    ext = CUDAExtension(
        name="memopt_C",
        sources=[
            "src/csrc/extension.cpp",
            "src/csrc/kv_cache.cu",
            "src/csrc/attention.cu",
            "src/csrc/quantization.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
            ],
        },
    )
    return [ext], {"build_ext": BuildExtension}


# ---------------------------------------------------------------------------
# Conditionally include the CUDA extension
# ---------------------------------------------------------------------------

if _cuda_available():
    ext_modules, cmdclass = _make_extension()
else:
    ext_modules, cmdclass = [], {}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
