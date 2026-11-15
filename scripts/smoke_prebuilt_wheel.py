"""Smoke-test an installed witwin-maxwell wheel on a host without a CUDA toolkit.

Run it from outside the checkout, after installing the wheel, with nvcc off PATH and
CUDA_HOME/CUDA_PATH unset. It asserts that:

- the package is imported from the installed wheel, not a source tree;
- all three packaged native libraries (main, core path, accelerated setup) are present with
  matching ABI markers and load from the packaged prebuilt directory;
- with a visible CUDA device, a small default FDTD scene runs on the core path with accelerated
  setup with no warning (no accelerated-setup retry), and no JIT build directory was created.

Without a CUDA device the scene check is skipped unless --require-gpu is given.
"""
from __future__ import annotations

import argparse
import os
from importlib.metadata import version
import shutil
import sysconfig
import warnings
from pathlib import Path


def _scene(mw):
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.6, 0.6),) * 3),
                grid=mw.GridSpec.uniform(0.04),
                boundary=mw.BoundarySpec.pml(num_layers=8),
                structures=(
            mw.Structure(
                name="block",
                geometry=mw.Box(position=(0.1, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
                material=mw.Material(eps_r=4.0),
            ),
            ),
                sources=(
            mw.PointDipole(
                position=(-0.2, 0.0, 0.0), polarization="Ez",
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.5e9),
            ),
            ),
                monitors=(mw.PointMonitor("probe", (0.3, 0.0, 0.0), fields=("Ez",)),),
            )
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    assert shutil.which("nvcc") is None, f"nvcc is on PATH: {shutil.which('nvcc')}"
    for name in ("CUDA_HOME", "CUDA_PATH"):
        assert name not in os.environ, f"{name} is set: {os.environ[name]}"
    jit_directory = os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR")
    assert jit_directory and not Path(jit_directory).exists(), (
        "Set WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR to a missing directory so any JIT build is detected.")

    import torch
    import witwin.maxwell as mw
    from witwin.maxwell.compiler.accel import build as setup_build
    from witwin.maxwell.fdtd.cuda import build as cuda_build
    from witwin.maxwell.fdtd.cuda.core import build as core_build

    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    package = Path(mw.__file__).resolve().parent
    assert site_packages in package.parents, f"{package} is not the installed wheel under {site_packages}"

    for library in cuda_build.LIBRARIES:
        path = cuda_build.prebuilt_extension_path(library)
        marker = cuda_build.prebuilt_abi_marker_path(path)
        assert path.exists(), f"missing packaged library {path}"
        assert marker.read_text(encoding="utf-8").strip() == library.abi, f"stale marker {marker}"

    if torch.cuda.is_available():
        with warnings.catch_warnings():
            # A failed accelerated setup retries with the torch setup and warns.
            warnings.simplefilter("error", mw.MaxwellWarning)
            result = mw.Simulation.fdtd(_scene(mw), frequencies=1e9,
                                        config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=64))).run()
        assert hasattr(result.solver, "_core_runtime"), "the scene did not run on the core path"
        # The run itself loaded both libraries; a failed core load raises.
        assert core_build.build_extension.cache_info().currsize == 1, "core path library was not loaded"
        assert setup_build.build_extension.cache_info().currsize == 1, "accelerated setup did not run"
        value = result.at().monitor("probe").sel(component="Ez").values
        print(f"Core-path scene ran without a toolkit; probe Ez = {value}")
    elif args.require_gpu:
        raise SystemExit("--require-gpu was given but no CUDA device is visible.")
    else:
        print("No CUDA device: scene check skipped; library loading checked only.")

    loaded = {
        cuda_build.MAIN.name: cuda_build.build_extension(),
        cuda_build.CORE.name: core_build.build_extension(),
        cuda_build.SETUP.name: setup_build.build_extension(),
    }
    for library in cuda_build.LIBRARIES:
        expected = cuda_build.prebuilt_extension_path(library).resolve()
        assert Path(loaded[library.name].__file__).resolve() == expected, (library.name, loaded[library.name].__file__)
        print(f"Loaded packaged {library.name} ({library.abi}) from {expected}")
    assert not Path(jit_directory).exists(), f"a JIT build ran in {jit_directory}"
    print(f"witwin-maxwell {version('witwin-maxwell')} wheel smoke passed with torch {torch.__version__}")


if __name__ == "__main__":
    main()
