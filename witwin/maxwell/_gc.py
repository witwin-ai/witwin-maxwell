"""Bound cyclic collection during allocation-heavy internal setup and readout."""

from contextlib import contextmanager
import gc


@contextmanager
def paused_cyclic_gc():
    """Pause automatic cyclic collection, preserving the caller's GC state.

    Python's GC enable flag is process-wide, not per thread: while a pause is
    active, automatic cyclic collection is off for every thread. Wrap only
    bounded, allocation-heavy internal sections; never stepping, user
    callbacks or anything that can block. The prior state is restored exactly:
    if collection was already disabled, it stays disabled, and a nested pause
    leaves the outer pause in force. ``gc.collect`` is never called.
    Reference counting still frees acyclic objects immediately; deferred
    cycles are collected at the next automatic pass after the pause ends.
    """
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
