from __future__ import annotations

import pytest

from witwin.maxwell.fdtd.cuda import build as cuda_build


@pytest.mark.parametrize("marker_contents", [None, "stable_abi_v1\n", "garbage\n"])
def test_stale_or_missing_packaged_prebuilt_is_rejected_before_loading(
    monkeypatch,
    tmp_path,
    marker_contents,
):
    library = tmp_path / f"witwin_maxwell_fdtd_cuda{cuda_build.extension_suffix()}"
    library.write_bytes(b"not-loaded")
    marker = cuda_build.prebuilt_abi_marker_path(library)
    if marker_contents is not None:
        marker.write_text(marker_contents, encoding="utf-8")

    load_calls = []
    monkeypatch.setattr(cuda_build, "prebuilt_extension_path", lambda: library)
    monkeypatch.setattr(
        cuda_build,
        "_load_extension_file",
        lambda path: load_calls.append(path),
    )

    assert cuda_build._load_packaged_prebuilt_extension() is None
    assert load_calls == []


def test_matching_packaged_prebuilt_load_failure_is_fail_fast_without_jit(
    monkeypatch,
    tmp_path,
):
    library = tmp_path / f"witwin_maxwell_fdtd_cuda{cuda_build.extension_suffix()}"
    library.write_bytes(b"invalid-v2-binary")
    cuda_build.prebuilt_abi_marker_path(library).write_text(
        f"{cuda_build.STABLE_ABI_VERSION}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cuda_build, "prebuilt_extension_path", lambda: library)
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_SKIP_PREBUILT", raising=False)
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_PREBUILT", raising=False)

    jit_calls = []

    def fail_load(path):
        raise RuntimeError(f"failed to load {path}")

    monkeypatch.setattr(cuda_build, "_load_extension_file", fail_load)
    monkeypatch.setattr(cuda_build, "load", lambda **kwargs: jit_calls.append(kwargs))

    with pytest.raises(RuntimeError, match="failed to load"):
        cuda_build.build_extension()

    assert jit_calls == []
