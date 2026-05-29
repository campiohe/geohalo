"""Hard memory ceiling so heavy cold builds fail catchably instead of OOM-killing the host.

`RLIMIT_AS` caps *virtual* address space; BLAS/mmap can over-reserve, so the effective
trip point is somewhat conservative. It is chosen as the simplest hard guarantee against
a crash on WSL during the global 0.05 degree resample builds. If it proves too aggressive,
swap to an RSS-polling watchdog thread.
"""

import resource
from collections.abc import Callable

_BYTES_PER_GB = 1024 ** 3


def gb_to_bytes(gb: float) -> int:
    return int(gb * _BYTES_PER_GB)


def set_address_space_limit(limit_gb: float) -> int:
    """Cap this process's virtual address space at `limit_gb` GiB. Returns the byte limit."""
    limit_bytes = gb_to_bytes(limit_gb)
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = limit_bytes if hard == resource.RLIM_INFINITY else min(limit_bytes, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, new_hard))
    return limit_bytes


def run_guarded[T](fn: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run `fn`; return (result, None) on success, (None, "skipped") on MemoryError."""
    try:
        return fn(), None
    except MemoryError:
        return None, "skipped"
