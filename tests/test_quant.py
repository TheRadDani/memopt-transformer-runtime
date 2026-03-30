"""INT4 quantization round-trip validation tests for MemOpt.

This module validates ``memopt_C.quantize_kv_cache_int4`` and
``memopt_C.dequantize_kv_cache_int4`` — the W4A16 KV-cache compression
kernels — against a pure-Python reference implementation.

Coverage:
    A. Shape & dtype contracts — packed/decompressed shapes, element dtypes.
    B. Scale correctness     — CUDA scales vs. Python reference (atol=1e-4).
    C. MSE round-trip        — reconstruction error within W4A16 thresholds.
    D. Edge cases            — zeros, saturated values, single group, large range.
    E. Determinism           — bit-identical outputs on repeated calls.
    F. Reference equivalence — q-codes within ±1, values within one quant step.
    G. Error guards          — RuntimeError on contract violations.

All tests require a compiled ``memopt_C`` extension and a CUDA-capable GPU;
the entire suite is skipped when either dependency is absent.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

memopt_C = pytest.importorskip("memopt_C")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

GROUP_SIZE: int = 64  # Only supported group size in the CUDA kernel
DEVICE: str = "cuda"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def make_cache(
    shape: Tuple[int, ...],
    distribution: str,
    dtype: torch.dtype = torch.float16,
    device: str = DEVICE,
) -> torch.Tensor:
    """Create a contiguous test tensor with a given shape and value distribution.

    Args:
        shape: Desired tensor shape, e.g. ``(B, S, D)`` for KV-cache inputs.
            The last dimension must be even; total elements must be divisible
            by ``GROUP_SIZE``.
        distribution: One of:
            - ``"uniform_unit"``  — ``Uniform(-1, 1)``
            - ``"uniform_pos"``   — ``Uniform(0, 1)``
            - ``"uniform_neg"``   — ``Uniform(-1, 0)``
            - ``"uniform_small"`` — ``Uniform(-0.1, 0.1)``
            - ``"normal_std"``    — ``Normal(0, 1)``
            - ``"normal_kv"``     — ``Normal(0, 0.02)`` (typical post-layer-norm)
            - ``"zeros"``         — all zeros
            - ``"constant"``      — all 0.5
        dtype: Element dtype (default ``torch.float16``).
        device: Target device (default ``"cuda"``).

    Returns:
        Contiguous tensor of shape ``shape`` on ``device`` with ``dtype``.
    """
    if distribution == "uniform_unit":
        t = torch.rand(shape, device=device).mul(2).sub(1)
    elif distribution == "uniform_pos":
        t = torch.rand(shape, device=device)
    elif distribution == "uniform_neg":
        t = torch.rand(shape, device=device).neg()
    elif distribution == "uniform_small":
        t = torch.rand(shape, device=device).mul(0.2).sub(0.1)
    elif distribution == "normal_std":
        t = torch.randn(shape, device=device)
    elif distribution == "normal_kv":
        t = torch.randn(shape, device=device).mul(0.02)
    elif distribution == "zeros":
        t = torch.zeros(shape, device=device)
    elif distribution == "constant":
        t = torch.full(shape, 0.5, device=device)
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}")
    return t.to(dtype).contiguous()


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute scalar mean-squared error between two tensors.

    Args:
        a: First tensor, any shape; must be broadcastable with ``b``.
        b: Second tensor, any shape; must be broadcastable with ``a``.

    Returns:
        Scalar MSE as a Python float.
    """
    return ((a.float() - b.float()) ** 2).mean().item()


def reference_quant(
    cache_fp16: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-Python group-wise symmetric INT4 quantization reference.

    Implements the same algorithm as the CUDA kernel:
    ``scale = max(abs(group)) / 7.0``,
    ``q = clamp(round(val / scale) + 8, 0, 15)``.

    Args:
        cache_fp16: FP16 tensor of any shape. Total elements must be divisible
            by ``group_size``.
        group_size: Number of elements per quantization group (must be 64).

    Returns:
        A 3-tuple ``(q_codes, scales, reconstructed)`` where:

        - ``q_codes``      — int32 tensor, same numel as input, values in [0, 15].
        - ``scales``       — float32 tensor, shape ``[total_elements / group_size]``.
        - ``reconstructed``— float32 tensor, same shape as input; dequantized values.
    """
    flat = cache_fp16.float().reshape(-1)
    num_groups = flat.numel() // group_size
    groups = flat.reshape(num_groups, group_size)

    max_abs = groups.abs().max(dim=1, keepdim=True).values
    scale = (max_abs / 7.0).clamp(min=1e-8)  # fp32, shape [num_groups, 1]

    q = groups.div(scale).round().add(8).clamp(0, 15).to(torch.int32)
    reconstructed = (q.float() - 8) * scale
    return q, scale.squeeze(1), reconstructed.reshape(cache_fp16.shape)


# ---------------------------------------------------------------------------
# Parametrization constants
# ---------------------------------------------------------------------------

# KV-cache-like shapes: (B, S, D) or (B, H, S, D) whose total elem % 64 == 0
_SHAPES = [
    (128,),         # flat 1-D
    (4, 128),       # 2-D
    (2, 8, 64),     # 3-D: B=2, H=8, D=64 — minimal KV-cache
    (2, 4, 8, 64),  # 4-D: B=2, num_heads=4, S=8, D=64
]

_DISTRIBUTIONS_SCALE = [
    "uniform_unit",
    "normal_std",
    "uniform_small",
]

# Number of groups → flat shape [num_groups * GROUP_SIZE]
_GROUP_COUNTS = [1, 4, 16, 128]

# KV-cache shapes for MSE parametrization (total elem divisible by GROUP_SIZE)
_KV_SHAPES = [
    (2, 8, 64),    # 1024 elements = 16 groups
    (4, 16, 128),  # 8192 elements = 128 groups
]


# ===========================================================================
# A. Shape & dtype contracts
# ===========================================================================


@pytest.mark.parametrize("shape", _SHAPES)
def test_packed_shape(shape: Tuple[int, ...]) -> None:
    """Verify packed tensor has last dim halved relative to the input.

    The CUDA kernel packs two INT4 nibbles per UInt8 byte, so the packed
    tensor must satisfy: ``packed.shape == (*input.shape[:-1], input.shape[-1] // 2)``.

    Args:
        shape: Input tensor shape, e.g. ``(B, S, D)``.
    """
    torch.manual_seed(42)
    cache = make_cache(shape, "uniform_unit")
    packed, _ = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)

    expected_shape = shape[:-1] + (shape[-1] // 2,)
    assert packed.shape == torch.Size(expected_shape), (
        f"shape={shape}: expected packed shape {expected_shape}, got {tuple(packed.shape)}"
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_decompressed_shape(shape: Tuple[int, ...]) -> None:
    """Verify dequantized tensor exactly restores the original shape.

    The dequantize kernel unpacks each byte back into two FP16 values, so the
    output shape must equal the original input shape.

    Args:
        shape: Input tensor shape.
    """
    torch.manual_seed(42)
    cache = make_cache(shape, "uniform_unit")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    assert decompressed.shape == torch.Size(shape), (
        f"shape={shape}: expected decompressed shape {shape}, got {tuple(decompressed.shape)}"
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_packed_dtype(shape: Tuple[int, ...]) -> None:
    """Verify packed tensor has dtype UInt8.

    INT4 values are stored two-per-byte in a UInt8 tensor; any other dtype
    would indicate a packing error in the CUDA kernel.

    Args:
        shape: Input tensor shape.
    """
    torch.manual_seed(42)
    cache = make_cache(shape, "uniform_unit")
    packed, _ = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)

    assert packed.dtype == torch.uint8, (
        f"shape={shape}: expected packed dtype uint8, got {packed.dtype}"
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_decompressed_dtype(shape: Tuple[int, ...]) -> None:
    """Verify dequantized tensor has dtype float16 (W4A16 output requirement).

    The dequantize kernel must emit FP16 so that the result can be fed directly
    into FP16 attention without a separate cast.

    Args:
        shape: Input tensor shape.
    """
    torch.manual_seed(42)
    cache = make_cache(shape, "uniform_unit")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    assert decompressed.dtype == torch.float16, (
        f"shape={shape}: expected decompressed dtype float16, got {decompressed.dtype}"
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_scales_shape(shape: Tuple[int, ...]) -> None:
    """Verify scales tensor has shape ``[total_elements / GROUP_SIZE]``.

    One scale factor is computed per group of GROUP_SIZE elements; the flat
    scales tensor must have exactly ``numel(input) / GROUP_SIZE`` entries.

    Args:
        shape: Input tensor shape.
    """
    torch.manual_seed(42)
    cache = make_cache(shape, "uniform_unit")
    _, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)

    import math
    total_elements = math.prod(shape)
    expected_num_scales = total_elements // GROUP_SIZE
    assert scales.shape == torch.Size([expected_num_scales]), (
        f"shape={shape}: expected scales shape [{expected_num_scales}], got {tuple(scales.shape)}"
    )


# ===========================================================================
# B. Scale correctness
# ===========================================================================


@pytest.mark.parametrize("distribution", _DISTRIBUTIONS_SCALE)
@pytest.mark.parametrize("shape", [(4, 128), (2, 8, 64)])
def test_scale_values(shape: Tuple[int, ...], distribution: str) -> None:
    """Verify CUDA-computed scales match the Python reference within atol=1e-4.

    The scale for each group of GROUP_SIZE elements is ``max(abs(group)) / 7.0``
    (clamped to ≥ 1e-8 for numerical safety).  This test checks the CUDA
    implementation matches this formula to float32 precision.

    Args:
        shape: Input tensor shape.
        distribution: Value distribution; see ``make_cache`` for options.
    """
    torch.manual_seed(42)
    cache = make_cache(shape, distribution)
    _, cuda_scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)

    _, ref_scales, _ = reference_quant(cache)

    assert torch.allclose(cuda_scales.float(), ref_scales.to(DEVICE), atol=1e-4, rtol=1e-4), (
        f"shape={shape}, dist={distribution}: max scale diff = "
        f"{(cuda_scales.float() - ref_scales.to(DEVICE)).abs().max().item():.6e}"
    )


# ===========================================================================
# C. MSE round-trip accuracy
# ===========================================================================


@pytest.mark.parametrize("num_groups", _GROUP_COUNTS)
def test_mse_uniform_unit_range(num_groups: int) -> None:
    """Verify MSE < 5e-3 for Uniform(-1, 1) inputs across group counts.

    For symmetric INT4 with 16 levels (range ±7 from zero_point), the
    theoretical maximum MSE on a uniform input is roughly ``(2/14)^2 / 3 ≈ 1e-3``,
    so 5e-3 is a generous industry-standard threshold for W4A16 KV compression.

    Args:
        num_groups: Number of quantization groups; shape is ``[num_groups * GROUP_SIZE]``.
    """
    torch.manual_seed(42)
    shape = (num_groups * GROUP_SIZE,)
    cache = make_cache(shape, "uniform_unit")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    error = mse(cache, decompressed)
    assert error < 5e-3, (
        f"num_groups={num_groups}: uniform(-1,1) MSE={error:.6e} exceeds threshold 5e-3"
    )


@pytest.mark.parametrize("num_groups", _GROUP_COUNTS)
def test_mse_normal_standard(num_groups: int) -> None:
    """Verify MSE < 5e-2 for Normal(0, 1) inputs across group counts.

    Standard normal has heavier tails than uniform; the INT4 dynamic range
    covers only ±7*scale, so outliers incur clipping error.  The 5e-2 threshold
    reflects industry acceptance for KV-cache INT4 on activations.

    Args:
        num_groups: Number of quantization groups.
    """
    torch.manual_seed(42)
    shape = (num_groups * GROUP_SIZE,)
    cache = make_cache(shape, "normal_std")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    error = mse(cache, decompressed)
    assert error < 5e-2, (
        f"num_groups={num_groups}: normal(0,1) MSE={error:.6e} exceeds threshold 5e-2"
    )


@pytest.mark.parametrize("num_groups", _GROUP_COUNTS)
def test_mse_small_range(num_groups: int) -> None:
    """Verify MSE < 5e-5 for Uniform(-0.1, 0.1) inputs across group counts.

    Small-range inputs have a tight dynamic range, so the quantization step
    is proportionally tiny (scale ≈ 0.1/7 ≈ 0.014); MSE should be much lower
    than the unit-range case.

    Args:
        num_groups: Number of quantization groups.
    """
    torch.manual_seed(42)
    shape = (num_groups * GROUP_SIZE,)
    cache = make_cache(shape, "uniform_small")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    error = mse(cache, decompressed)
    assert error < 5e-5, (
        f"num_groups={num_groups}: uniform(-0.1,0.1) MSE={error:.6e} exceeds threshold 5e-5"
    )


@pytest.mark.parametrize("num_groups", _GROUP_COUNTS)
def test_mse_kv_cache_like(num_groups: int) -> None:
    """Verify relative MSE < 2% for Normal(0, 0.02) inputs (typical KV activations).

    Post-layer-norm KV tensors commonly have std ≈ 0.02.  Absolute MSE is
    tiny, so we use relative MSE (MSE / variance) to normalise for signal power.
    A relative MSE < 2% is considered acceptable for research-grade compression.

    Args:
        num_groups: Number of quantization groups.
    """
    torch.manual_seed(42)
    shape = (num_groups * GROUP_SIZE,)
    cache = make_cache(shape, "normal_kv")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    variance = cache.float().var().item()
    if variance < 1e-10:
        # Degenerate case: near-constant input; absolute MSE must be ~0
        error = mse(cache, decompressed)
        assert error < 1e-6, (
            f"num_groups={num_groups}: near-zero-variance input, "
            f"absolute MSE={error:.6e} unexpectedly large"
        )
        return

    relative_mse = mse(cache, decompressed) / variance
    assert relative_mse < 0.02, (
        f"num_groups={num_groups}: normal(0,0.02) relative MSE={relative_mse:.6e} "
        f"exceeds threshold 0.02 (2%)"
    )


@pytest.mark.parametrize("shape", _KV_SHAPES)
def test_mse_kv_shape_uniform(shape: Tuple[int, ...]) -> None:
    """Verify MSE < 5e-3 for KV-cache-shaped tensors with Uniform(-1, 1).

    Checks that quantization accuracy holds for realistic multi-dimensional
    KV-cache tensor shapes ``(B, H, D)`` and ``(B, H, S, D)``, not just flat
    1-D arrays.

    Args:
        shape: KV-cache-like tensor shape.
    """
    torch.manual_seed(42)
    cache = make_cache(shape, "uniform_unit")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    error = mse(cache, decompressed)
    assert error < 5e-3, (
        f"shape={shape}: uniform(-1,1) MSE={error:.6e} exceeds threshold 5e-3"
    )


# ===========================================================================
# D. Edge cases
# ===========================================================================


def test_all_zeros() -> None:
    """Verify that an all-zero input round-trips to all zeros with MSE = 0.

    When all values are zero, each group's scale is clamped to min (1e-8).
    Quantising 0.0 gives q = round(0 / scale) + 8 = 8, so dequantising
    gives (8 - 8) * scale = 0.  The round-trip must be exactly zero.
    """
    torch.manual_seed(42)
    shape = (4, 128)
    cache = make_cache(shape, "zeros")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    error = mse(cache, decompressed)
    assert error == 0.0, (
        f"all-zeros input: expected MSE=0.0, got MSE={error:.6e}"
    )
    assert torch.all(decompressed == 0), (
        "all-zeros input: decompressed tensor contains non-zero values"
    )


def test_all_same_value() -> None:
    """Verify that a constant-value input round-trips with near-zero MSE.

    When all elements equal a constant c, the group max is |c|, scale = |c|/7,
    and q = round(c / (|c|/7)) + 8 = 7 + 8 = 15 (for c > 0).  Dequantising
    gives (15-8) * (c/7) = c exactly, so MSE should be 0 or extremely small
    (limited only by FP16 precision of the constant itself).
    """
    torch.manual_seed(42)
    shape = (4, 128)
    cache = make_cache(shape, "constant")  # all 0.5
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    error = mse(cache, decompressed)
    assert error < 1e-6, (
        f"constant-value (0.5) input: MSE={error:.6e} exceeds near-zero threshold 1e-6"
    )


def test_positive_only() -> None:
    """Verify MSE < 5e-3 for Uniform(0, 1) (all-positive) inputs.

    The symmetric quantizer centres around zero_point=8, so positive-only
    inputs use roughly half the INT4 range [8,15].  This should still achieve
    acceptable accuracy.
    """
    torch.manual_seed(42)
    shape = (4, 128)
    cache = make_cache(shape, "uniform_pos")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    error = mse(cache, decompressed)
    assert error < 5e-3, (
        f"uniform(0,1) input: MSE={error:.6e} exceeds threshold 5e-3"
    )


def test_negative_only() -> None:
    """Verify MSE < 5e-3 for Uniform(-1, 0) (all-negative) inputs.

    The mirror case of ``test_positive_only``; negative-only values use the
    lower INT4 range [0, 8).  Symmetry of the algorithm means accuracy should
    match the positive-only case.
    """
    torch.manual_seed(42)
    shape = (4, 128)
    cache = make_cache(shape, "uniform_neg")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    error = mse(cache, decompressed)
    assert error < 5e-3, (
        f"uniform(-1,0) input: MSE={error:.6e} exceeds threshold 5e-3"
    )


def test_single_group() -> None:
    """Verify correctness for exactly one GROUP_SIZE group (64 elements).

    With a single group, ``scales`` should be a 1-element tensor and
    ``packed`` should have 32 elements (64 INT4 nibbles packed into 32 bytes).
    """
    torch.manual_seed(42)
    shape = (GROUP_SIZE,)
    cache = make_cache(shape, "uniform_unit")
    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)

    assert scales.numel() == 1, (
        f"single group: expected scales.numel()=1, got {scales.numel()}"
    )
    assert packed.numel() == GROUP_SIZE // 2, (
        f"single group: expected packed.numel()={GROUP_SIZE // 2}, got {packed.numel()}"
    )

    # Also verify round-trip accuracy for this minimal case
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)
    error = mse(cache, decompressed)
    assert error < 5e-3, (
        f"single group round-trip: MSE={error:.6e} exceeds threshold 5e-3"
    )


def test_max_representable_values() -> None:
    """Verify that ±7.0 inputs (max INT4 range) round-trip exactly.

    The value +7.0 → scale = 7/7 = 1.0 → q = round(7/1)+8 = 15 → deq = (15-8)*1 = 7.
    The value -7.0 → q = round(-7/1)+8 = 1 → deq = (1-8)*1 = -7.
    Both extremes should be exactly representable with zero error.
    """
    torch.manual_seed(42)
    # Fill alternating ±7.0 so every group has scale = 1.0
    shape = (GROUP_SIZE,)
    cache = torch.empty(shape, dtype=torch.float16, device=DEVICE)
    cache[::2] = 7.0
    cache[1::2] = -7.0
    cache = cache.contiguous()

    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    error = mse(cache, decompressed)
    assert error < 1e-4, (
        f"±7.0 values: expected near-exact round-trip, MSE={error:.6e}"
    )


def test_large_values() -> None:
    """Verify bounded relative MSE for Uniform(-100, 100) large-magnitude inputs.

    Large values force large scales, which are valid for the symmetric algorithm.
    The relative MSE (MSE / variance) must stay below 5% to confirm the kernel
    handles wide dynamic ranges without overflow or precision collapse.
    """
    torch.manual_seed(42)
    shape = (4, 128)
    # Generate in float32, clip to float16 range, then cast
    raw = (torch.rand(shape, device=DEVICE).mul(200).sub(100)).clamp(-65504, 65504)
    cache = raw.to(torch.float16).contiguous()

    packed, scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    decompressed = memopt_C.dequantize_kv_cache_int4(packed, scales, GROUP_SIZE)

    variance = cache.float().var().item()
    relative_mse = mse(cache, decompressed) / variance
    assert relative_mse < 0.05, (
        f"large-range input: relative MSE={relative_mse:.6e} exceeds threshold 0.05 (5%)"
    )


# ===========================================================================
# E. Determinism
# ===========================================================================


def test_determinism() -> None:
    """Verify quantize→dequantize produces bit-identical outputs on repeated calls.

    Runs the quantize kernel twice on the same tensor and asserts:
    - packed tensors are ``torch.equal`` (bitwise identical).
    - scales tensors are ``torch.equal`` (bitwise identical).

    Non-determinism would indicate race conditions or undefined CUDA behaviour
    in the kernel, which would break reproducibility across inference runs.
    """
    torch.manual_seed(42)
    shape = (4, 128)
    cache = make_cache(shape, "normal_std")

    packed_a, scales_a = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    packed_b, scales_b = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)

    assert torch.equal(packed_a, packed_b), (
        "Quantize kernel is non-deterministic: packed outputs differ across calls"
    )
    assert torch.equal(scales_a, scales_b), (
        "Quantize kernel is non-deterministic: scales outputs differ across calls"
    )


# ===========================================================================
# F. Python reference equivalence
# ===========================================================================


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a UInt8 tensor into INT32 nibble codes in [0, 15].

    Each byte stores two nibbles: ``lo = byte & 0xF``, ``hi = (byte >> 4) & 0xF``.
    The output interleaves lo/hi in the order they appear in the flat packed
    tensor, matching the kernel's packing convention.

    Args:
        packed: UInt8 CUDA tensor of any shape.

    Returns:
        Int32 tensor of shape ``[packed.numel() * 2]`` with values in [0, 15].
    """
    flat = packed.reshape(-1).to(torch.int32)
    lo = flat & 0xF
    hi = (flat >> 4) & 0xF
    # Interleave: element 2i = lo[i], element 2i+1 = hi[i]
    out = torch.empty(flat.numel() * 2, dtype=torch.int32, device=packed.device)
    out[0::2] = lo
    out[1::2] = hi
    return out


def test_reference_equivalence() -> None:
    """Verify CUDA kernel output is consistent with the Python reference.

    Two properties are checked:
    1. **Q-code agreement**: unpacked integer codes from the CUDA kernel must
       be within ±1 of the Python reference codes (rounding may differ by at
       most 1 for halfway cases).
    2. **Value agreement**: dequantized values must be within ``scale/2`` of
       the Python reference reconstructed values element-wise (i.e., within
       one quantization step of each other).

    This tolerance acknowledges that CUDA round-half-to-even may differ from
    Python's round-half-away-from-zero for exact halfway values.
    """
    torch.manual_seed(42)
    shape = (4, 128)
    cache = make_cache(shape, "uniform_unit")

    # CUDA path
    packed, cuda_scales = memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
    cuda_decompressed = memopt_C.dequantize_kv_cache_int4(packed, cuda_scales, GROUP_SIZE)

    # Python reference path
    ref_q, ref_scales, ref_reconstructed = reference_quant(cache)

    # 1. Q-code agreement: unpack CUDA nibbles and compare to reference codes
    cuda_q = _unpack_nibbles(packed)  # shape [total_elements]
    ref_q_flat = ref_q.reshape(-1).to(DEVICE)
    q_diff = (cuda_q.int() - ref_q_flat.int()).abs()
    assert q_diff.max().item() <= 1, (
        f"Q-code max deviation from reference = {q_diff.max().item()} (expected ≤ 1)"
    )

    # 2. Value agreement: within half a quantization step (scale/2)
    # Expand scales to per-element tolerance: each group of GROUP_SIZE elements
    # shares one scale value.
    cuda_scales_expanded = cuda_scales.repeat_interleave(GROUP_SIZE).reshape(shape)
    tolerance = cuda_scales_expanded.float() / 2.0  # half a quantization step

    abs_diff = (cuda_decompressed.float() - ref_reconstructed.to(DEVICE)).abs()
    max_violation = (abs_diff - tolerance).clamp(min=0).max().item()
    assert max_violation == 0.0, (
        f"CUDA dequantized values deviate from reference by more than scale/2; "
        f"max excess = {max_violation:.6e}"
    )


# ===========================================================================
# G. Error guards (contract violations → RuntimeError)
# ===========================================================================


def test_error_wrong_group_size() -> None:
    """Verify RuntimeError when group_size != 64.

    The CUDA kernel enforces group_size == 64 via TORCH_CHECK.  Any other value
    must raise a RuntimeError at the Python boundary.
    """
    cache = make_cache((4, 128), "uniform_unit")
    with pytest.raises(RuntimeError):
        memopt_C.quantize_kv_cache_int4(cache, 32)


def test_error_wrong_dtype_fp32() -> None:
    """Verify RuntimeError when input dtype is float32 instead of float16.

    The kernel is a W4A16 implementation and only accepts FP16 input.
    Passing FP32 must be rejected to prevent silent precision surprises.
    """
    cache = torch.rand(4, 128, dtype=torch.float32, device=DEVICE).contiguous()
    with pytest.raises(RuntimeError):
        memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)


def test_error_cpu_tensor() -> None:
    """Verify RuntimeError when input tensor is on CPU instead of CUDA.

    The quantize kernel is CUDA-only; CPU tensors must be rejected with a
    clear error rather than a segfault or silent wrong result.
    """
    cache = torch.rand(4, 128, dtype=torch.float16).contiguous()  # CPU
    with pytest.raises(RuntimeError):
        memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)


def test_error_non_contiguous() -> None:
    """Verify RuntimeError when input tensor is non-contiguous.

    The CUDA kernel assumes a contiguous memory layout for efficient coalesced
    reads.  A transposed or strided tensor must be rejected.
    """
    base = torch.rand(128, 4, dtype=torch.float16, device=DEVICE)
    non_contiguous = base.t()  # transposed → non-contiguous
    assert not non_contiguous.is_contiguous(), "Precondition failed: tensor is contiguous"
    with pytest.raises(RuntimeError):
        memopt_C.quantize_kv_cache_int4(non_contiguous, GROUP_SIZE)


def test_error_odd_last_dim() -> None:
    """Verify RuntimeError when the last dimension is odd (cannot pack INT4 pairs).

    Each output byte stores two INT4 nibbles; an odd last dimension would leave
    one element without a pair, which is undefined.  The kernel must reject this.
    """
    cache = torch.rand(4, 3, dtype=torch.float16, device=DEVICE).contiguous()
    with pytest.raises(RuntimeError):
        memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)


def test_error_not_divisible_by_group() -> None:
    """Verify RuntimeError when total elements are not divisible by GROUP_SIZE.

    Each quantization group must contain exactly GROUP_SIZE=64 elements.
    A tensor with 100 total elements (100 % 64 != 0) cannot be partitioned
    into complete groups and must be rejected.
    """
    # 100 elements: not divisible by 64; last dim must be even (use 4 cols, 25 rows → 100 total)
    # But last dim must be even: use shape (50, 2) → 100 elements, last dim = 2 (even)
    cache = torch.rand(50, 2, dtype=torch.float16, device=DEVICE).contiguous()
    assert cache.numel() == 100 and cache.numel() % GROUP_SIZE != 0
    with pytest.raises(RuntimeError):
        memopt_C.quantize_kv_cache_int4(cache, GROUP_SIZE)
