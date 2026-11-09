"""Explicit initialization for tests that call compiled kernel wrappers directly."""

from hashlib import sha256
from pathlib import Path

import pytest


@pytest.fixture
def native_cuda_extension(request):
    from witwin.maxwell.fdtd.cuda import backend

    # Solver preparation normally loads this module before any kernel launch.
    # Direct wrapper parity tests must establish that precondition themselves.
    extension = backend.get_compiled_extension()
    assert backend._COMPILED_EXTENSION is extension
    assert extension.is_available() is True
    extension.synchronize_noop()
    library = Path(extension.__file__).resolve()
    request.node.user_properties.extend(
        [
            ("native_cuda_extension_path", str(library)),
            ("native_cuda_extension_sha256", sha256(library.read_bytes()).hexdigest()),
        ]
    )
    return extension
