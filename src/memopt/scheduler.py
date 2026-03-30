"""MemoryScheduler: VRAM-adaptive policy daemon for MemOpt.

This module implements :class:`MemoryScheduler`, a background thread that
polls GPU memory utilization and adjusts the Ada-KV ``retain_ratio`` and
``use_int4`` policy variables in real time.

Policy variables
----------------
``retain_ratio : float``
    Fraction of KV-cache tokens to keep when the C++ eviction kernels run.
    ``1.0`` means no eviction; ``0.75`` evicts the lowest-scoring 25 %.
    Passed directly to ``memopt_C.evict_cache`` / ``memopt_C.evict_tokens``
    at each kernel invocation.

``use_int4 : bool``
    When ``True``, the forward wrapper should quantize the KV cache to INT4
    via ``memopt_C.quantize_kv_cache_int4`` before writing and
    dequantize via ``memopt_C.dequantize_kv_cache_int4`` before reading.

Threshold semantics
-------------------
* VRAM utilization **>= ``vram_high_threshold``** (default 85 %)
  → ``retain_ratio = drop_retain_ratio`` (default 0.75), ``use_int4 = True``
* VRAM utilization **<= ``vram_low_threshold``** (default 50 %)
  → ``retain_ratio = 1.0``, ``use_int4 = False``
* Between the two thresholds → no change (hysteresis dead band).

The 35-percentage-point dead band is intentionally large to prevent rapid
oscillation without requiring an explicit counter or timer.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

__all__ = ["MemoryScheduler", "SchedulerConfig"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_POLL_INTERVAL_SEC: float = 0.1
_DEFAULT_VRAM_HIGH_THRESHOLD: float = 0.85
_DEFAULT_VRAM_LOW_THRESHOLD: float = 0.50
_DEFAULT_DROP_RETAIN_RATIO: float = 0.75
_RETAIN_RATIO_FULL: float = 1.0


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    """Immutable configuration for :class:`MemoryScheduler`.

    Args:
        poll_interval_sec: Seconds between VRAM polls in the background thread.
            Lower values give faster reaction at the cost of more CPU wakeups.
        vram_high_threshold: Fractional VRAM utilization above which the
            scheduler activates token eviction and INT4 quantization.
            Must be in ``(0.0, 1.0]``.
        vram_low_threshold: Fractional VRAM utilization below which the
            scheduler restores full-retention, FP16 mode.
            Must be in ``[0.0, vram_high_threshold)``.
        drop_retain_ratio: The ``retain_ratio`` value applied when VRAM
            exceeds ``vram_high_threshold``.  Must be in ``[0.0, 1.0)``.

    Raises:
        ValueError: If thresholds or retain ratio are out of range.
    """

    poll_interval_sec: float = field(default=_DEFAULT_POLL_INTERVAL_SEC)
    vram_high_threshold: float = field(default=_DEFAULT_VRAM_HIGH_THRESHOLD)
    vram_low_threshold: float = field(default=_DEFAULT_VRAM_LOW_THRESHOLD)
    drop_retain_ratio: float = field(default=_DEFAULT_DROP_RETAIN_RATIO)
    num_sink_tokens: int = field(default=4)

    def __post_init__(self) -> None:  # noqa: D105
        if not (0.0 < self.vram_high_threshold <= 1.0):
            raise ValueError(
                f"vram_high_threshold must be in (0.0, 1.0], got {self.vram_high_threshold}"
            )
        if not (0.0 <= self.vram_low_threshold < self.vram_high_threshold):
            raise ValueError(
                f"vram_low_threshold must be in [0.0, vram_high_threshold), "
                f"got low={self.vram_low_threshold}, high={self.vram_high_threshold}"
            )
        if not (0.0 <= self.drop_retain_ratio < 1.0):
            raise ValueError(
                f"drop_retain_ratio must be in [0.0, 1.0), got {self.drop_retain_ratio}"
            )
        if self.poll_interval_sec <= 0.0:
            raise ValueError(
                f"poll_interval_sec must be positive, got {self.poll_interval_sec}"
            )


# ---------------------------------------------------------------------------
# MemoryScheduler
# ---------------------------------------------------------------------------


class MemoryScheduler:
    """VRAM-adaptive policy scheduler for MemOpt Ada-KV inference.

    Maintains the authoritative ``retain_ratio`` and ``use_int4`` policy
    variables.  The C++ Data Plane reads these values at each kernel
    invocation via the forward wrapper.

    The scheduler can operate in two modes (non-exclusive):

    **Background thread mode** (default on construction)
        A daemon thread polls ``torch.cuda.memory_allocated()`` every
        ``config.poll_interval_sec`` seconds and updates policy when
        utilization crosses the configured thresholds.

    **Forward hook mode**
        :meth:`attach_hook` registers a ``register_forward_pre_hook`` on a
        ``torch.nn.Module`` so that :meth:`adjust_policy` is called
        synchronously before every forward pass.  Useful when GPU memory
        changes rapidly between forward calls.

    Both modes may be active simultaneously; they share the same lock-
    protected state.

    Args:
        config: Scheduler configuration.  Uses defaults when omitted.

    Example::

        scheduler = MemoryScheduler()
        scheduler.start()

        # forward wrapper reads:
        #   retain_ratio = scheduler.retain_ratio
        #   use_int4     = scheduler.use_int4

        scheduler.stop()

    Thread safety
    -------------
    All reads and writes to ``retain_ratio`` and ``use_int4`` are protected
    by ``self._lock`` (a :class:`threading.Lock`).  Callers may also acquire
    this lock directly if they need an atomic snapshot of both values::

        with scheduler._lock:
            rr = scheduler.retain_ratio
            q  = scheduler.use_int4
    """

    def __init__(self, config: Optional[SchedulerConfig] = None) -> None:
        """Initialise the scheduler with the given configuration.

        The background thread is created but not started.  Call :meth:`start`
        to begin polling.

        Args:
            config: :class:`SchedulerConfig` instance.  If ``None``, a
                default config is used.
        """
        self._config: SchedulerConfig = config or SchedulerConfig()

        # ------------------------------------------------------------------
        # Policy state — the authoritative source of truth for the Data Plane.
        # ------------------------------------------------------------------
        self._retain_ratio: float = _RETAIN_RATIO_FULL
        self._use_int4: bool = False

        # ------------------------------------------------------------------
        # Thread machinery
        # ------------------------------------------------------------------
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(
            target=self._poll_loop,
            name="memopt-scheduler",
            daemon=True,
        )

        # ------------------------------------------------------------------
        # Forward hook handle (set by attach_hook / cleared by detach_hook)
        # ------------------------------------------------------------------
        self._hook_handle: Optional[Any] = None  # torch.utils.hooks.RemovableHook

        # ------------------------------------------------------------------
        # CUDA availability guard
        # ------------------------------------------------------------------
        self._cuda_available: bool = torch.cuda.is_available()
        if not self._cuda_available:
            logger.warning(
                "MemoryScheduler: CUDA is not available.  "
                "VRAM polling is disabled; policy variables will remain at defaults."
            )

    # ------------------------------------------------------------------
    # Public policy accessors (thread-safe)
    # ------------------------------------------------------------------

    @property
    def retain_ratio(self) -> float:
        """Current Ada-KV token retention fraction ``[0.0, 1.0]``.

        ``1.0`` means no eviction; lower values drop proportionally more
        low-scoring KV tokens.

        Returns:
            The current ``retain_ratio`` value.
        """
        with self._lock:
            return self._retain_ratio

    @property
    def use_int4(self) -> bool:
        """Whether INT4 KV-cache quantization is currently active.

        When ``True``, the forward wrapper should quantize the KV cache using
        ``memopt_C.quantize_kv_cache_int4`` before writing and dequantize
        with ``memopt_C.dequantize_kv_cache_int4`` before reading.

        Returns:
            The current ``use_int4`` flag.
        """
        with self._lock:
            return self._use_int4

    # ------------------------------------------------------------------
    # Public lifecycle methods
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background VRAM polling thread.

        Safe to call multiple times — subsequent calls are no-ops if the
        thread is already running.

        Raises:
            RuntimeError: If the thread was previously stopped and cannot
                be restarted (Python threads are not restartable).
        """
        if self._thread.is_alive():
            logger.debug("MemoryScheduler.start(): thread already running, ignoring.")
            return
        if self._stop_event.is_set():
            # Thread was stopped; create a fresh one.
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._poll_loop,
                name="memopt-scheduler",
                daemon=True,
            )
        self._thread.start()
        logger.info(
            "MemoryScheduler started (poll_interval=%.3fs, "
            "high_threshold=%.0f%%, low_threshold=%.0f%%, "
            "drop_retain_ratio=%.2f).",
            self._config.poll_interval_sec,
            self._config.vram_high_threshold * 100,
            self._config.vram_low_threshold * 100,
            self._config.drop_retain_ratio,
        )

    def stop(self) -> None:
        """Signal the background thread to stop and block until it exits.

        Safe to call when the thread is not running.
        """
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._config.poll_interval_sec * 10)
            if self._thread.is_alive():
                logger.warning(
                    "MemoryScheduler.stop(): thread did not terminate within timeout."
                )
            else:
                logger.info("MemoryScheduler stopped.")

    def attach_hook(self, model: nn.Module) -> None:
        """Register a forward pre-hook on ``model`` to call :meth:`adjust_policy`.

        The hook is invoked synchronously before each ``model.forward()`` so
        that the policy variables are always current when the forward pass
        reads them.

        Args:
            model: The :class:`torch.nn.Module` to attach the hook to.
                Typically the top-level Transformer or ``BaselineLlama``.

        Raises:
            TypeError: If ``model`` is not a :class:`torch.nn.Module`.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"attach_hook expects a torch.nn.Module, got {type(model).__name__}"
            )
        if self._hook_handle is not None:
            logger.warning(
                "MemoryScheduler.attach_hook(): replacing existing hook on %s.",
                type(model).__name__,
            )
            self._hook_handle.remove()

        def _hook(module: nn.Module, args: tuple) -> None:  # noqa: ARG001
            self.adjust_policy()

        self._hook_handle = model.register_forward_pre_hook(_hook)
        logger.info(
            "MemoryScheduler: forward pre-hook attached to %s.",
            type(model).__name__,
        )

    def detach_hook(self) -> None:
        """Remove the forward hook previously registered by :meth:`attach_hook`.

        No-op if no hook is currently registered.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            logger.info("MemoryScheduler: forward hook detached.")
        else:
            logger.debug("MemoryScheduler.detach_hook(): no hook registered, ignoring.")

    # ------------------------------------------------------------------
    # Core policy logic
    # ------------------------------------------------------------------

    def adjust_policy(self) -> None:
        """Sample VRAM utilization and update policy variables if a threshold is crossed.

        This method is safe to call from any thread.  It is called
        automatically by the background poll loop and by the forward hook.

        When CUDA is unavailable the method returns immediately without
        modifying any state.
        """
        if not self._cuda_available:
            return

        with torch.no_grad():
            utilization = _sample_vram_utilization()

        if utilization is None:
            # Sampling failed (e.g. no device visible); skip silently.
            return

        with self._lock:
            old_ratio = self._retain_ratio
            old_int4 = self._use_int4

            if utilization >= self._config.vram_high_threshold:
                # Pressure: activate eviction + quantization.
                self._retain_ratio = self._config.drop_retain_ratio
                self._use_int4 = True
            elif utilization <= self._config.vram_low_threshold:
                # Relief: restore full-retention, FP16 mode.
                self._retain_ratio = _RETAIN_RATIO_FULL
                self._use_int4 = False
            # else: within dead band — leave state unchanged.

            changed = (
                self._retain_ratio != old_ratio or self._use_int4 != old_int4
            )

        if changed:
            logger.info(
                "MemoryScheduler policy change: vram=%.1f%% "
                "retain_ratio %.2f→%.2f  use_int4 %s→%s",
                utilization * 100,
                old_ratio,
                self._retain_ratio,
                old_int4,
                self._use_int4,
            )

    # ------------------------------------------------------------------
    # Background thread loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background thread target: poll VRAM and adjust policy until stopped.

        Catches all exceptions internally so that transient errors in VRAM
        sampling never crash the host process.
        """
        logger.debug("MemoryScheduler poll loop started.")
        while not self._stop_event.is_set():
            try:
                self.adjust_policy()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "MemoryScheduler: unhandled exception in poll loop; continuing."
                )
            # Use Event.wait so stop() wakes us immediately instead of
            # waiting up to poll_interval_sec.
            self._stop_event.wait(timeout=self._config.poll_interval_sec)
        logger.debug("MemoryScheduler poll loop exited.")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        vram_str = "N/A (no CUDA)"
        if self._cuda_available:
            util = _sample_vram_utilization()
            if util is not None:
                vram_str = f"{util * 100:.1f}%"

        with self._lock:
            rr = self._retain_ratio
            q = self._use_int4

        thread_state = "running" if self._thread.is_alive() else "stopped"
        hook_state = "attached" if self._hook_handle is not None else "none"

        return (
            f"MemoryScheduler("
            f"vram={vram_str}, "
            f"retain_ratio={rr:.2f}, "
            f"use_int4={q}, "
            f"thread={thread_state}, "
            f"hook={hook_state}, "
            f"poll_interval={self._config.poll_interval_sec:.3f}s"
            f")"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_vram_utilization() -> Optional[float]:
    """Sample current VRAM utilization as a fraction of total device memory.

    Uses ``torch.cuda.memory_allocated()`` (bytes currently allocated by
    PyTorch tensors) divided by ``torch.cuda.get_device_properties(0).total_memory``.

    Returns:
        Utilization in ``[0.0, 1.0]``, or ``None`` if sampling fails (e.g.
        no CUDA device is present or the driver reports zero total memory).
    """
    try:
        allocated: int = torch.cuda.memory_allocated()
        total: int = torch.cuda.get_device_properties(0).total_memory
        if total == 0:
            return None
        return allocated / total
    except Exception:  # noqa: BLE001
        return None
