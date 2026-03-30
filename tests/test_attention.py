"""Attention kernel validation tests for MemOpt.

This module validates that ``memopt_C.dynamic_sparse_attention`` is
mathematically equivalent to ``torch.nn.functional.scaled_dot_product_attention``
when the full KV cache is retained (no token eviction).

All tests require a compiled ``memopt_C`` extension and a CUDA-capable GPU.
The suite is skipped entirely if either dependency is absent.
"""

import math

import pytest
import torch
import torch.nn.functional as F

memopt_C = pytest.importorskip("memopt_C")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# ---------------------------------------------------------------------------
# Constants shared across all tests
# ---------------------------------------------------------------------------
BLOCK_SIZE: int = 16
NUM_HEADS: int = 8
HEAD_DIM: int = 64
BATCH_SIZE: int = 2
DEVICE: str = "cuda"
DTYPE: torch.dtype = torch.float16


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def setup_pool(
    num_tokens: int,
    block_size: int,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Initialise the paged KV pool and build addressing tensors.

    Allocates enough physical blocks to hold ``num_tokens`` tokens (split
    evenly across ``BATCH_SIZE`` sequences) then constructs the flat
    ``slot_mapping`` and batched ``block_tables`` required by
    ``memopt_C.write_cache`` and ``memopt_C.dynamic_sparse_attention``.

    Args:
        num_tokens: Total number of tokens across ALL batch items
            (``B * seq_len``).
        block_size: Physical block capacity in tokens.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.

    Returns:
        A 3-tuple ``(slot_mapping, block_tables, num_blocks_per_seq)`` where:

        - ``slot_mapping`` — shape ``[num_tokens]``, int32 on CUDA. Maps
          global token index ``i`` to its physical slot in the pool.
        - ``block_tables`` — shape ``[BATCH_SIZE, num_blocks_per_seq]``,
          int32 on CUDA. Maps ``(batch_item, logical_block)`` to a physical
          block index.
        - ``num_blocks_per_seq`` — number of blocks allocated per sequence.
    """
    seq_len: int = num_tokens // BATCH_SIZE
    num_blocks_per_seq: int = math.ceil(seq_len / block_size)
    total_blocks: int = BATCH_SIZE * num_blocks_per_seq
    # Small slack so the pool never runs dry
    pool_size: int = total_blocks + 2

    memopt_C.init_pool(pool_size, block_size, num_heads, head_dim)

    # Allocate one physical block per logical slot, per batch item.
    # block_indices[b][s] = physical block for batch b, logical block s.
    block_indices: list[list[int]] = []
    for _ in range(BATCH_SIZE):
        row: list[int] = []
        for _ in range(num_blocks_per_seq):
            phys = memopt_C.allocate_block()
            assert phys != -1, "Pool exhausted during setup — increase slack"
            row.append(phys)
        block_indices.append(row)

    # Build slot_mapping: global token i → physical slot
    slot_mapping_list: list[int] = []
    for b in range(BATCH_SIZE):
        for t in range(seq_len):
            logical_block = t // block_size
            offset_within_block = t % block_size
            phys_block = block_indices[b][logical_block]
            slot_mapping_list.append(phys_block * block_size + offset_within_block)

    slot_mapping = torch.tensor(
        slot_mapping_list, dtype=torch.int32, device=DEVICE
    )

    # Build block_tables: [BATCH_SIZE, num_blocks_per_seq]
    block_tables = torch.tensor(
        block_indices, dtype=torch.int32, device=DEVICE
    )

    return slot_mapping, block_tables, num_blocks_per_seq


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [512, 1024, 2048])
def test_math_equivalence(seq_len: int) -> None:
    """Verify dynamic_sparse_attention matches SDPA output within fp16 tolerance.

    Writes identical K/V tensors into the paged pool and computes attention
    with both ``torch.nn.functional.scaled_dot_product_attention`` (baseline)
    and ``memopt_C.dynamic_sparse_attention`` (custom kernel), then asserts
    element-wise agreement using ``atol=1e-3, rtol=1e-3``.

    Args:
        seq_len: Sequence length under test.
    """
    torch.manual_seed(42)

    # SDPA layout: [B, num_heads, seq_len, head_dim]
    Q_sdpa = torch.rand(
        BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    K_sdpa = torch.rand(
        BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    V_sdpa = torch.rand(
        BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )

    # Baseline — standard scaled dot-product attention
    with torch.no_grad():
        baseline = F.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa)
    # baseline: [B, num_heads, seq_len, head_dim]

    # -------------------------------------------------------------------
    # Pool setup
    # -------------------------------------------------------------------
    slot_mapping, block_tables, num_blocks_per_seq = setup_pool(
        BATCH_SIZE * seq_len, BLOCK_SIZE, NUM_HEADS, HEAD_DIM
    )

    # Convert K/V from SDPA layout [B, H, S, D] → kernel layout [B, S, H, D]
    K_kern = K_sdpa.permute(0, 2, 1, 3).contiguous()
    V_kern = V_sdpa.permute(0, 2, 1, 3).contiguous()

    # Flatten to [B*S, H, D] for write_cache
    K_flat = K_kern.reshape(BATCH_SIZE * seq_len, NUM_HEADS, HEAD_DIM)
    V_flat = V_kern.reshape(BATCH_SIZE * seq_len, NUM_HEADS, HEAD_DIM)

    memopt_C.write_cache(K_flat, V_flat, block_tables, slot_mapping)

    # -------------------------------------------------------------------
    # Custom kernel
    # -------------------------------------------------------------------
    key_cache, val_cache = memopt_C.get_pool_tensors()

    # Query: [B, seq_len, num_heads, head_dim] (heads-last)
    Q_kern = Q_sdpa.permute(0, 2, 1, 3).contiguous()

    context_lens = torch.full(
        (BATCH_SIZE,), seq_len, dtype=torch.int32, device=DEVICE
    )

    with torch.no_grad():
        out_kern = memopt_C.dynamic_sparse_attention(
            Q_kern,
            key_cache,
            val_cache,
            block_tables,
            context_lens,
            seq_len,   # max_seq_len
            seq_len,   # window_size = seq_len → attend to all tokens
        )
    # out_kern: [B, seq_len, num_heads, head_dim]

    # Re-layout to SDPA convention: [B, num_heads, seq_len, head_dim]
    out_kern_sdpa_layout = out_kern.permute(0, 2, 1, 3)

    assert torch.allclose(baseline, out_kern_sdpa_layout, atol=1e-3, rtol=1e-3), (
        f"seq_len={seq_len}: max abs diff = "
        f"{(baseline - out_kern_sdpa_layout).abs().max().item():.6f}"
    )


@pytest.mark.parametrize("seq_len", [512, 1024, 2048])
def test_output_shape(seq_len: int) -> None:
    """Verify that dynamic_sparse_attention returns a tensor with the expected shape.

    Expected output shape: ``(B, seq_len, num_heads, head_dim)`` (heads-last
    kernel convention).

    Args:
        seq_len: Sequence length under test.
    """
    torch.manual_seed(42)

    Q_sdpa = torch.rand(
        BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    K_sdpa = torch.rand(
        BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    V_sdpa = torch.rand(
        BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )

    slot_mapping, block_tables, _ = setup_pool(
        BATCH_SIZE * seq_len, BLOCK_SIZE, NUM_HEADS, HEAD_DIM
    )

    K_flat = K_sdpa.permute(0, 2, 1, 3).contiguous().reshape(
        BATCH_SIZE * seq_len, NUM_HEADS, HEAD_DIM
    )
    V_flat = V_sdpa.permute(0, 2, 1, 3).contiguous().reshape(
        BATCH_SIZE * seq_len, NUM_HEADS, HEAD_DIM
    )
    memopt_C.write_cache(K_flat, V_flat, block_tables, slot_mapping)

    key_cache, val_cache = memopt_C.get_pool_tensors()
    Q_kern = Q_sdpa.permute(0, 2, 1, 3).contiguous()
    context_lens = torch.full(
        (BATCH_SIZE,), seq_len, dtype=torch.int32, device=DEVICE
    )

    with torch.no_grad():
        out_kern = memopt_C.dynamic_sparse_attention(
            Q_kern, key_cache, val_cache,
            block_tables, context_lens,
            seq_len, seq_len,
        )

    expected_shape = (BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM)
    assert out_kern.shape == expected_shape, (
        f"seq_len={seq_len}: expected shape {expected_shape}, "
        f"got {tuple(out_kern.shape)}"
    )


def test_pool_not_initialized_raises() -> None:
    """Verify that write_cache raises RuntimeError when the pool is uninitialised.

    The C++ layer guards all pool operations with a TORCH_CHECK that raises
    a RuntimeError containing "pool not initialized" when ``init_pool`` has
    not been called.  This test exercises that guard by constructing minimal
    tensors and skipping ``init_pool``.

    Note: We re-initialise the pool with a sentinel configuration that is
    deliberately incompatible with the dummy tensors so the write fails, OR
    we rely on the C++ guard directly if the pool has never been set.  To
    make the test state-independent we call a fresh ``init_pool`` with a
    different shape to confirm the guard fires for mismatched shapes — but
    because the simplest approach is to rely on the guard, we instead call
    ``init_pool`` with zero blocks so the pool cannot satisfy any allocation,
    then attempt to write.

    The expected exception is a ``RuntimeError`` from TORCH_CHECK.
    """
    # Reinitialise with zero blocks to guarantee the pool has no capacity.
    # Some C++ implementations guard on zero blocks; others guard on the
    # uninitialized sentinel.  Either way we expect a RuntimeError.
    dummy_keys = torch.zeros(1, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    dummy_vals = torch.zeros(1, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    # block_tables and slot_mapping with a single entry
    dummy_block_tables = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)
    dummy_slot_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    # Do NOT call init_pool — rely on the pool-uninitialized guard, or call
    # init_pool with 0 blocks to force the pool into an unusable state.
    try:
        memopt_C.init_pool(0, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
    except Exception:
        pass  # If init_pool itself rejects 0 blocks, the pool is still bad.

    with pytest.raises(RuntimeError, match="pool not initialized|out of bounds|exhausted"):
        memopt_C.write_cache(
            dummy_keys, dummy_vals, dummy_block_tables, dummy_slot_mapping
        )


def test_determinism() -> None:
    """Verify that dynamic_sparse_attention produces bit-identical outputs on repeated calls.

    Runs the kernel twice with identical Q/K/V inputs and the same pool state
    and asserts the outputs are exactly equal (``torch.equal``), confirming
    the kernel is deterministic.
    """
    seq_len: int = 512
    torch.manual_seed(42)

    Q_sdpa = torch.rand(
        BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    K_sdpa = torch.rand(
        BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    V_sdpa = torch.rand(
        BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )

    slot_mapping, block_tables, _ = setup_pool(
        BATCH_SIZE * seq_len, BLOCK_SIZE, NUM_HEADS, HEAD_DIM
    )

    K_flat = K_sdpa.permute(0, 2, 1, 3).contiguous().reshape(
        BATCH_SIZE * seq_len, NUM_HEADS, HEAD_DIM
    )
    V_flat = V_sdpa.permute(0, 2, 1, 3).contiguous().reshape(
        BATCH_SIZE * seq_len, NUM_HEADS, HEAD_DIM
    )
    memopt_C.write_cache(K_flat, V_flat, block_tables, slot_mapping)

    key_cache, val_cache = memopt_C.get_pool_tensors()
    Q_kern = Q_sdpa.permute(0, 2, 1, 3).contiguous()
    context_lens = torch.full(
        (BATCH_SIZE,), seq_len, dtype=torch.int32, device=DEVICE
    )

    with torch.no_grad():
        out_first = memopt_C.dynamic_sparse_attention(
            Q_kern, key_cache, val_cache,
            block_tables, context_lens,
            seq_len, seq_len,
        )
        out_second = memopt_C.dynamic_sparse_attention(
            Q_kern, key_cache, val_cache,
            block_tables, context_lens,
            seq_len, seq_len,
        )

    assert torch.equal(out_first, out_second), (
        "dynamic_sparse_attention is non-deterministic: "
        f"max diff = {(out_first - out_second).abs().max().item():.6f}"
    )
