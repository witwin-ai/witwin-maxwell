from __future__ import annotations

import numpy as np
import torch

import witwin.maxwell as mw


def _scene() -> mw.Scene:
    return mw.Scene(
        # Three cells per axis: the Yee shapes (3, 4, 4), (4, 3, 4), (4, 4, 3) of the stand-in arrays.
        domain=mw.Domain(bounds=((-0.375, 0.375),) * 3),
        grid=mw.GridSpec.uniform(0.25),
        structures=(
            mw.Structure(
                name="target",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
                material=mw.Material(eps_r=3.0),
            ),
        ),
    )


def test_fdfd_smoke_scene_simulation_result(monkeypatch):
    class FakeFDFD:
        def __init__(self, scene, frequency, solver_type, preconditioner, precision, ssor_omega, enable_plot, verbose,
                     device_memory_limit_bytes=None):
            self.scene = scene
            self.frequency = frequency
            self.solver_type = solver_type
            self.enable_plot = enable_plot
            self.verbose = verbose
            self.E_field = (
                torch.ones((3, 4, 4), dtype=torch.complex64),
                torch.ones((4, 3, 4), dtype=torch.complex64) * 2.0,
                torch.ones((4, 4, 3), dtype=torch.complex64) * 3.0,
            )
            self.converged = True
            self.final_residual = 1e-8
            self.solver_info = 0

        def solve(self, max_iter, tol, restart):
            self.solve_args = (max_iter, tol, restart)

    monkeypatch.setattr("witwin.maxwell.simulation.FDFD", FakeFDFD, raising=False)
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_device", lambda scene, method: None)

    sim = mw.Simulation.fdfd(_scene(), frequencies=1e9, config=mw.FDFDConfig(device="cpu"))
    result = sim.run()

    assert result.method == "fdfd"
    assert result.E.z.isel(f=0).values.shape == (4, 4, 3)
    assert result.materials.eps.scalar.shape == result.prepared_scene.permittivity.shape
    assert result.stats()["converged"] is True


def test_fdtd_smoke_scene_simulation_result(monkeypatch):
    class FakeFDTD:
        def __init__(self, scene, frequency, absorber_type, cpml):
            self.scene = scene
            self._compiled_sources = []
            self.frequency = frequency
            self.absorber_type = absorber_type
            self.cpml = cpml
            self._compiled_sources = ()
            self.c = 10.0
            self.dt = 0.25
            self.dft_sample_count = 6
            self.last_solve_elapsed_s = 0.03

        def init_field(self):
            self.initialized = True

        def solve(self, time_steps, dft_frequency, enable_plot, dft_window, full_field_dft, normalize_source=False, shutoff=0.0, shutoff_check_interval=100, use_cuda_graph=False, resume_from=None, stop_step=None):
            self.ex = torch.ones((3, 4, 4), dtype=torch.complex64)
            self.ey = torch.ones((4, 3, 4), dtype=torch.complex64) * 2.0
            self.ez = torch.ones((4, 4, 3), dtype=torch.complex64) * 3.0
            return {
                "observers": {
                    "center": {
                        "kind": "point", "fields": ("Ez",), "components": {"Ez": np.complex64(1.0 + 0.5j)},
                        "field_indices": {"Ez": (1, 1, 1)}, "frequencies": (1e9,), "position": (0.0, 0.0, 0.0),
                    }
                }
            }

    monkeypatch.setattr("witwin.maxwell.simulation._resolve_fdtd_backend", lambda: (FakeFDTD, None))
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_device", lambda scene, method: None)
    # The stand-in solver has no fields to bind the core launches to.
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.core_path.activate_core_path", lambda solver: None)
    sim = mw.Simulation.fdtd(_scene(), frequencies=1e9, config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=12),
        spectral_sampler=mw.fdtd.SpectralSampler(window="none"), device="cpu"))
    result = sim.run()

    assert result.method == "fdtd"
    assert result.time_snapshot["EX"].values.shape == (3, 4, 4)
    assert result.at(freq_index=0).monitor("center").sel(component="Ez").values.item() == 1.0 + 0.5j
    assert result.stats()["time_steps"] == 12
