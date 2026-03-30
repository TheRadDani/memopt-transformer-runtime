"""Baseline Transformer model wrappers for MemOpt.

This module provides a thin, unmodified reference implementation of Llama-2
via HuggingFace ``transformers``. It is intentionally free of any custom
memory tricks so that it serves as an accurate baseline for VRAM and accuracy
benchmarking against the optimized MemOpt variants.

Offline usage:
    Set ``TRANSFORMERS_OFFLINE=1`` and ``HF_DATASETS_OFFLINE=1`` in your
    environment to force HuggingFace to load from its local cache without
    making network requests.  Pre-populate the cache by running once with
    internet access, or copy the snapshot directory manually.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "The 'transformers' package is required for BaselineLlama. "
        "Install it with: pip install transformers"
    ) from exc

__all__ = ["BaselineLlama"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_NAME: str = "meta-llama/Llama-2-7b-hf"
_ENV_USE_DUMMY: str = "MEMOPT_USE_DUMMY"


class BaselineLlama(nn.Module):
    """Thin, sterile wrapper around a HuggingFace ``AutoModelForCausalLM``.

    Loads the model and tokenizer from the HuggingFace Hub (or local cache)
    and exposes a PyTorch-idiomatic ``forward`` / ``generate`` interface.
    No MemOpt memory optimizations are applied — this class is the unmodified
    reference implementation used for baseline benchmarking.

    Args:
        model_name: HuggingFace model identifier or path to a local snapshot.
            Defaults to ``"meta-llama/Llama-2-7b-hf"``.
        use_dummy_weights: When ``True`` (or when the ``MEMOPT_USE_DUMMY=1``
            environment variable is set), loads the model with
            ``low_cpu_mem_usage=True`` and ``torch_dtype=torch.float16``.
            Useful for CI environments that need the model graph without
            downloading full FP32 weights.
        device: Target device string (e.g. ``"cpu"``, ``"cuda"``,
            ``"cuda:0"``).  The model and its tokenizer are loaded onto this
            device.

    Attributes:
        model: The underlying ``AutoModelForCausalLM`` instance.
        tokenizer: The corresponding ``AutoTokenizer`` instance with
            ``padding_side="left"`` and ``pad_token`` set to ``eos_token``
            when not already defined.

    Example::

        wrapper = BaselineLlama(use_dummy_weights=True, device="cuda")
        input_ids = wrapper.tokenizer(
            "Hello, world!", return_tensors="pt"
        ).input_ids.to(wrapper.device)
        logits = wrapper(input_ids)  # shape (B, S, vocab_size)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        use_dummy_weights: bool = False,
        device: str = "cpu",
    ) -> None:
        """Initialise BaselineLlama, loading model and tokenizer.

        Args:
            model_name: HuggingFace model identifier or local directory path.
            use_dummy_weights: Load with reduced memory / FP16 dtype.  Also
                activated by ``MEMOPT_USE_DUMMY=1`` environment variable.
            device: Torch device string to load the model onto.

        Raises:
            RuntimeError: If ``device`` starts with ``"cuda"`` but CUDA is
                not available on this machine.
            OSError: If ``model_name`` cannot be resolved either from the Hub
                or from the local HuggingFace cache.
        """
        super().__init__()

        # Honour the env-var override for CI pipelines.
        _use_dummy: bool = use_dummy_weights or (
            os.environ.get(_ENV_USE_DUMMY, "0").strip() == "1"
        )

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device}' was requested but CUDA is not available on "
                "this machine.  Use device='cpu' or ensure a CUDA-capable GPU "
                "is present."
            )

        # ------------------------------------------------------------------
        # Tokenizer
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        load_kwargs: dict = {}
        if _use_dummy:
            load_kwargs["low_cpu_mem_usage"] = True
            load_kwargs["torch_dtype"] = torch.float16

        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )
        self.model.to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Return the device that model parameters reside on.

        Returns:
            The ``torch.device`` of the first model parameter.
        """
        return next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a forward pass and return per-token logits.

        Wraps the underlying HuggingFace model in ``torch.no_grad()`` so that
        no autograd graph is built — appropriate for inference-only usage.

        Args:
            input_ids: Integer token indices of shape ``(B, S)`` where ``B``
                is batch size and ``S`` is sequence length.
            attention_mask: Optional boolean / float mask of shape ``(B, S)``.
                ``1`` (or ``True``) indicates a real token; ``0`` (or
                ``False``) indicates padding.  If ``None``, all positions are
                treated as valid.

        Returns:
            Logits tensor of shape ``(B, S, vocab_size)``.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.logits

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive token generation.

        Delegates directly to ``self.model.generate()`` without modification.

        Args:
            input_ids: Prompt token indices of shape ``(B, S_prompt)``.
            max_new_tokens: Maximum number of new tokens to generate beyond
                the prompt length.
            **kwargs: Additional keyword arguments forwarded verbatim to
                ``AutoModelForCausalLM.generate()`` (e.g. ``temperature``,
                ``do_sample``, ``top_p``).

        Returns:
            Token indices tensor of shape ``(B, S_prompt + S_generated)``
            containing both the prompt and the newly generated tokens.
        """
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        model_name = getattr(
            self.model.config, "_name_or_path", "<unknown>"
        )
        return (
            f"BaselineLlama("
            f"model_name={model_name!r}, "
            f"device={self.device}, "
            f"dtype={next(self.model.parameters()).dtype}"
            f")"
        )
