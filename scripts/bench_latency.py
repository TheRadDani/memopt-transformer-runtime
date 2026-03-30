"""bench_latency.py — Autoregressive generation latency benchmark for MemOpt.

Compares a standard Transformer (nn.MultiheadAttention) against the MemOpt
variant (DynamicAttention + MemoryScheduler) on a fixed 500-token greedy-
decoding loop.  No HuggingFace downloads required — the model is tiny and
fully self-contained.

Usage::

    # CPU run (no GPU required)
    python scripts/bench_latency.py

    # CUDA run with 3 warmup steps
    python scripts/bench_latency.py --device cuda --warmup 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Architecture constants — identical for both models so the comparison is fair
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 256
D_MODEL: int = 256
N_HEADS: int = 4
N_LAYERS: int = 2
D_FF: int = D_MODEL * 4
MAX_SEQ_LEN: int = 1024
TOKENS_TO_GENERATE: int = 500
SEED: int = 42

# ---------------------------------------------------------------------------
# Tiny Transformer decoder
# ---------------------------------------------------------------------------


class _TransformerLayer(nn.Module):
    """Single decoder layer: LayerNorm + attention + LayerNorm + FFN.

    Args:
        attn_module: The attention module to use (MultiheadAttention or
            DynamicAttention).  Must accept ``(x, x, x, need_weights=False)``
            for the standard path or ``(x)`` for the DynamicAttention path.
        d_model: Hidden dimension.
        d_ff: Feed-forward intermediate dimension.
        use_dynamic_attention: When ``True`` the forward call uses the
            DynamicAttention single-argument signature; when ``False`` it
            uses nn.MultiheadAttention's three-argument signature.
    """

    def __init__(
        self,
        attn_module: nn.Module,
        d_model: int,
        d_ff: int,
        use_dynamic_attention: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = attn_module
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self._use_dynamic_attention = use_dynamic_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        Args:
            x: Input tensor of shape ``(B, S, D)``.

        Returns:
            Output tensor of shape ``(B, S, D)``.
        """
        residual = x
        normed = self.norm1(x)
        if self._use_dynamic_attention:
            # DynamicAttention: forward(x) → (B, S, D)
            attn_out = self.attn(normed)
        else:
            # nn.MultiheadAttention: forward(q, k, v) → (attn_out, weights)
            # Input must be (S, B, D) in batch_first=False mode; we use
            # batch_first=True so shape stays (B, S, D).
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = residual + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class _TinyTransformerDecoder(nn.Module):
    """Minimal autoregressive Transformer decoder for benchmarking.

    Architecture: Embedding → N × _TransformerLayer → LayerNorm → LM head.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        d_ff: Feed-forward intermediate size.
        max_seq_len: Maximum sequence length for positional embedding.
        use_dynamic_attention: If ``True``, use DynamicAttention; otherwise
            use nn.MultiheadAttention.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        use_dynamic_attention: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        layers = []
        for _ in range(n_layers):
            if use_dynamic_attention:
                _src = Path(__file__).resolve().parent.parent / "src"
                if _src.is_dir() and str(_src) not in sys.path:
                    sys.path.insert(0, str(_src))
                from memopt import DynamicAttention
                attn: nn.Module = DynamicAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                )
            else:
                attn = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    batch_first=True,
                )
            layers.append(
                _TransformerLayer(
                    attn_module=attn,
                    d_model=d_model,
                    d_ff=d_ff,
                    use_dynamic_attention=use_dynamic_attention,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for the next token at each position.

        Args:
            input_ids: Token indices of shape ``(B, S)``.

        Returns:
            Logits tensor of shape ``(B, S, vocab_size)``.
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


def greedy_generate(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Autoregressive greedy token generation.

    Args:
        model: Decoder model accepting ``(B, S)`` token IDs and returning
            ``(B, S, vocab_size)`` logits.
        prompt_ids: Starting token IDs of shape ``(B, S_prompt)``.
        max_new_tokens: Number of tokens to generate.
        device: Device the model and tensors live on.

    Returns:
        Generated token IDs of shape ``(B, S_prompt + max_new_tokens)``.
    """
    ids = prompt_ids.to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids)           # (B, S, vocab_size)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            ids = torch.cat([ids, next_token], dim=1)
    return ids


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    model_name: str,
    model: nn.Module,
    prompt_ids: torch.Tensor,
    device: torch.device,
    warmup: int,
    use_scheduler: bool = False,
) -> float:
    """Time ``TOKENS_TO_GENERATE`` tokens from ``model``.

    Args:
        model_name: Human-readable label for logging.
        model: The decoder model to benchmark.
        prompt_ids: Initial token IDs of shape ``(1, S_prompt)``.
        device: Target device.
        warmup: Number of warmup generation passes before the timed run.
        use_scheduler: If ``True``, start a MemoryScheduler for this run.

    Returns:
        Wall-clock seconds for generating exactly ``TOKENS_TO_GENERATE`` tokens.
    """
    _src = Path(__file__).resolve().parent.parent / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from memopt import MemoryScheduler

    scheduler: Optional[MemoryScheduler] = None
    if use_scheduler:
        scheduler = MemoryScheduler()
        scheduler.start()

    model.eval()
    model.to(device)

    # Warmup passes
    for _ in range(warmup):
        _ = greedy_generate(model, prompt_ids, max_new_tokens=1, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed run
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    greedy_generate(model, prompt_ids, max_new_tokens=TOKENS_TO_GENERATE, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    if scheduler is not None:
        scheduler.stop()

    elapsed = t_end - t_start
    return elapsed


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------


def print_results_table(
    baseline_secs: float,
    memopt_secs: float,
    tokens: int,
) -> None:
    """Print a human-readable comparison table to stdout.

    Args:
        baseline_secs: Wall-clock seconds for the baseline run.
        memopt_secs: Wall-clock seconds for the MemOpt run.
        tokens: Number of tokens generated by each model.
    """
    baseline_tps = tokens / baseline_secs
    memopt_tps = tokens / memopt_secs
    speedup = memopt_tps / baseline_tps

    col_w = [18, 8, 10, 12, 10]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    header_fmt = "|{:<18}|{:>8}|{:>10}|{:>12}|{:>10}|"
    row_fmt = header_fmt

    print()
    print(sep)
    print(header_fmt.format("Model", "Tokens", "Time (s)", "Tokens/sec", "Speedup"))
    print(sep)
    print(row_fmt.format(
        "Baseline (std attn)",
        str(tokens),
        f"{baseline_secs:.3f}",
        f"{baseline_tps:.2f}",
        "1.00x",
    ))
    print(row_fmt.format(
        "MemOpt (DynAttn)",
        str(tokens),
        f"{memopt_secs:.3f}",
        f"{memopt_tps:.2f}",
        f"{speedup:.2f}x",
    ))
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the latency benchmark."""
    parser = argparse.ArgumentParser(
        description="MemOpt autoregressive generation latency benchmark."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run the benchmark on (default: cpu).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup passes before timing (default: 1).",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda specified but torch.cuda.is_available() is False."
        )

    device = torch.device(args.device)

    torch.manual_seed(SEED)

    # Shared prompt: single sequence of 4 tokens (batch size = 1)
    prompt_ids = torch.zeros(1, 4, dtype=torch.long)

    print(f"Device      : {device}")
    print(f"Warmup steps: {args.warmup}")
    print(f"Tokens gen  : {TOKENS_TO_GENERATE}")
    print(f"Architecture: vocab={VOCAB_SIZE}, d_model={D_MODEL}, "
          f"n_heads={N_HEADS}, n_layers={N_LAYERS}")

    # ------------------------------------------------------------------
    # Build models with identical seeds so weights are initialised
    # identically (for a fair architectural comparison).
    # ------------------------------------------------------------------
    torch.manual_seed(SEED)
    baseline_model = _TinyTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        use_dynamic_attention=False,
    )

    torch.manual_seed(SEED)
    memopt_model = _TinyTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        use_dynamic_attention=True,
    )

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------
    print("\nRunning baseline model ...")
    baseline_secs = run_benchmark(
        model_name="Baseline",
        model=baseline_model,
        prompt_ids=prompt_ids,
        device=device,
        warmup=args.warmup,
        use_scheduler=False,
    )

    print("Running MemOpt model ...")
    memopt_secs = run_benchmark(
        model_name="MemOpt",
        model=memopt_model,
        prompt_ids=prompt_ids,
        device=device,
        warmup=args.warmup,
        use_scheduler=True,
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print_results_table(
        baseline_secs=baseline_secs,
        memopt_secs=memopt_secs,
        tokens=TOKENS_TO_GENERATE,
    )


if __name__ == "__main__":
    main()
