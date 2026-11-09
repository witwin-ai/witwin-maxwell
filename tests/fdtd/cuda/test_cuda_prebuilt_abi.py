from __future__ import annotations

import pytest

from witwin.maxwell.fdtd.cuda import build as cuda_build


@pytest.mark.parametrize(
    "marker_contents",
    [None, "stable_abi_v1\n", "stable_abi_v2\n", "stable_abi_v3\n", "stable_abi_v4\n", "stable_abi_v5\n", "stable_abi_v6\n", "stable_abi_v7\n", "stable_abi_v8\n", "stable_abi_v9\n", "stable_abi_v10\n", "stable_abi_v11\n", "stable_abi_v12\n", "garbage\n"],
)
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
    library.write_bytes(b"invalid-v4-binary")
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


def test_trilinear_sampling_source_gather_adjoint_and_indexed_point_operators_are_part_of_the_stable_v12_contract():
    assert cuda_build.STABLE_ABI_VERSION == "stable_abi_v13"
    assert {
        "trilinear_border_forward",
        "trilinear_border_entries",
        "trilinear_border_backward",
        "trilinear_border_grid_backward",
        "accumulate_point_observers_indexed",
        "seed_inject_point",
        "accumulate_tfsf_sample_adjoint",
        "sample_wire_emf",
        "update_wire_state",
        "deposit_wire_current",
        "plane_flux_reduce",
        "add_table_time_shifted_source_patch",
        "advance_copper_electric",
        "begin_copper_step",
        "correct_copper_magnetic",
        "accumulate_copper_spectra",
        "thin_sheet_gather",
        "thin_sheet_scatter",
        "thin_sheet_copper_rhs",
        "thin_sheet_chord_forcing",
        "thin_sheet_copper_state",
        "thin_sheet_chord_initialize",
        "thin_sheet_chord_residual",
        "thin_sheet_chord_matvec",
        "thin_sheet_chord_direction",
    }.issubset(cuda_build._REQUIRED_STABLE_OPS)
    assert cuda_build.source_root() / "kernels" / "wire.cu" in cuda_build.extension_sources()
    assert cuda_build.source_root() / "kernels" / "copper.cu" in cuda_build.extension_sources()
    assert cuda_build.source_root() / "kernels" / "thin_sheet.cu" in cuda_build.extension_sources()
    assert cuda_build.source_root() / "kernels" / "trilinear.cu" in cuda_build.extension_sources()


def test_old_v3_packaged_marker_uses_only_explicit_current_prebuilt(monkeypatch,tmp_path):
    packaged=tmp_path/'packaged'/f'witwin_maxwell_fdtd_cuda{cuda_build.extension_suffix()}'
    packaged.parent.mkdir();packaged.write_bytes(b'old-v3-signature-must-not-load')
    cuda_build.prebuilt_abi_marker_path(packaged).write_text('stable_abi_v3\n')
    current_dir=tmp_path/'current-v4';current_dir.mkdir()
    current=current_dir/packaged.name;current.write_bytes(b'explicit-reviewed-v4-binary')
    cuda_build.prebuilt_abi_marker_path(current).write_text(f'{cuda_build.STABLE_ABI_VERSION}\n')
    loaded=[];sentinel=object()
    monkeypatch.setattr(cuda_build,'prebuilt_extension_path',lambda:packaged)
    monkeypatch.setattr(cuda_build,'_load_extension_file',lambda path:loaded.append(path)or sentinel)
    monkeypatch.setattr(cuda_build,'load',lambda **kwargs:pytest.fail('A stale packaged marker must not trigger JIT when explicit prebuilt is required.'))
    monkeypatch.delenv('WITWIN_MAXWELL_FDTD_CUDA_SKIP_PREBUILT',raising=False)
    monkeypatch.setenv('WITWIN_MAXWELL_FDTD_CUDA_PREBUILT','1')
    monkeypatch.setenv('WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR',str(current_dir))
    assert cuda_build.build_extension()is sentinel
    assert loaded==[current]


@pytest.mark.parametrize("marker_contents", [None, "stable_abi_v6\n", "garbage\n"])
def test_explicit_prebuilt_with_stale_or_missing_marker_fails_closed(monkeypatch, tmp_path, marker_contents):
    build_dir = tmp_path / "pinned"
    build_dir.mkdir()
    library = build_dir / f"witwin_maxwell_fdtd_cuda{cuda_build.extension_suffix()}"
    library.write_bytes(b"stale-binary-must-not-load")
    if marker_contents is not None:
        cuda_build.prebuilt_abi_marker_path(library).write_text(marker_contents, encoding="utf-8")
    monkeypatch.setattr(cuda_build, "prebuilt_extension_path", lambda: tmp_path / "absent.so")
    monkeypatch.setattr(cuda_build, "_load_extension_file", lambda path: pytest.fail("stale binary was loaded"))
    monkeypatch.setattr(cuda_build, "load", lambda **kwargs: pytest.fail("explicit prebuilt must not trigger JIT"))
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_SKIP_PREBUILT", raising=False)
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_PREBUILT", "1")
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR", str(build_dir))
    with pytest.raises(ImportError, match=f"expected '{cuda_build.STABLE_ABI_VERSION}'"):
        cuda_build.build_extension()


def test_jit_build_writes_abi_marker_in_overridden_build_dir(monkeypatch, tmp_path):
    build_dir = tmp_path / "override"
    library = build_dir / f"witwin_maxwell_fdtd_cuda{cuda_build.extension_suffix()}"

    def fake_load(**kwargs):
        assert kwargs["build_directory"] == str(build_dir)
        library.write_bytes(b"built")
        return str(library)

    monkeypatch.setattr(cuda_build, "prebuilt_extension_path", lambda: tmp_path / "absent.so")
    monkeypatch.setattr(cuda_build, "load", fake_load)
    monkeypatch.setattr(cuda_build, "_ensure_windows_build_tools_on_path", lambda: None)
    monkeypatch.setattr(cuda_build, "_ensure_cuda_home_from_nvcc", lambda: None)
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_PREBUILT", raising=False)
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR", str(build_dir))
    cuda_build.build_extension()
    assert cuda_build.prebuilt_abi_marker_path(library).read_text(encoding="utf-8").strip() == cuda_build.STABLE_ABI_VERSION
