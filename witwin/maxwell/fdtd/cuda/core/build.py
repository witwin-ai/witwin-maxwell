from __future__ import annotations

from functools import lru_cache
import weakref

from .. import build as cuda_build


CORE_ABI_VERSION = cuda_build.CORE.abi


def _handle_type(name, create, ops):
    """A class per C++ launch object, as the former pybind11 classes: construction prepares it once."""
    launch, compile_delayed, release = ops.launch.default, ops.compile_delayed.default, ops.release.default

    def __init__(self, *args):
        self._handle = create(*args)
        weakref.finalize(self, release, self._handle)

    return type(name, (), {
        "__init__": __init__,
        "launch": lambda self: launch(self._handle),
        "compile_delayed": lambda self, delays, output: compile_delayed(self._handle, delays, output),
    })


class _CoreExtension:
    def __init__(self, ops):
        self.__file__ = ops.__file__
        self.CoreLaunch = _handle_type("CoreLaunch", ops.core_launch.default, ops)
        self.CoreStepLaunch = _handle_type("CoreStepLaunch", ops.core_step_launch.default, ops)
        self.CoreObserverGather = _handle_type("CoreObserverGather", ops.core_observer_gather.default, ops)


@lru_cache(maxsize=1)
def build_extension():
    return _CoreExtension(cuda_build.build_extension(library=cuda_build.CORE))
