"""Package version of witwin-maxwell.

Kept as a static string so trees without installed distribution metadata (source
checkouts and ``git archive`` exports) report their version. It must equal
``[project] version`` in ``pyproject.toml``; ``tests/api/public/test_version.py``
enforces this.
"""

__version__ = "0.7.0"
