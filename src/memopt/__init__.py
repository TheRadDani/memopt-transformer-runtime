__version__ = "0.1.0"

try:
    import memopt_C
    _C_AVAILABLE = True
except ImportError:
    _C_AVAILABLE = False

from memopt.models import BaselineLlama  # noqa: E402
from memopt.attention import DynamicAttention  # noqa: E402
from memopt.scheduler import MemoryScheduler, SchedulerConfig  # noqa: E402

__all__ = ["BaselineLlama", "DynamicAttention", "MemoryScheduler", "SchedulerConfig", "get_hello"]


def get_hello() -> str:
    """Returns a greeting from the C++ backend if available."""
    if _C_AVAILABLE:
        return memopt_C.get_hello()
    return "C++ backend not loaded."
