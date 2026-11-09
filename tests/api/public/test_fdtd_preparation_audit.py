import json
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
def test_preparation_audit_records_actual_grid_bottleneck_and_does_not_qualify():
    axes = (np.array([-.01, -.003, 0, .002, .01]), np.linspace(-.01, .01, 5),
            np.array([-.01, 0, 18e-6, .01]))
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.01, .01),) * 3),
                     grid=mw.GridSpec.custom(*axes), boundary=mw.BoundarySpec.none(),
                     sources=[mw.PointDipole(position=(0, 0, 0), polarization="Ex",
                                             source_time=mw.CW(frequency=2.45e9))], device="cuda")
    prepared = mw.Simulation.fdtd(
        scene, frequency=2.45e9, full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
        run_time=mw.TimeConfig(time_steps=16, transient_cycles=0, steady_cycles=1),
    ).prepare()
    report = prepared.diagnostics
    assert report["status"] == "prepared_execution_not_qualified"
    assert report["grid"]["z"]["minimum_spacing_m"] == pytest.approx(18e-6)
    assert report["grid"]["z"]["limiting_interval_m"] == [0, 18e-6]
    assert report["estimated_steps_for_12ns"] == math.ceil(12e-9 / prepared.solver.dt)
    assert report["pec_constraints"] == {}
    json.dumps(report, allow_nan=False)
    result = prepared.run()
    assert result.solver_stats["steps_run"] == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
def test_audit_reports_actual_grid_quality_and_layer_resolution():
    import numpy as np
    from tests.fdtd.test_pcb_execution_state import simulation

    config = simulation()
    config.scene.add_structure(mw.Structure(mw.Box(position=(0, 0, .005), size=(.03, .03, .01)),
                                            mw.Material(eps_r=2.2), name="layer"))
    prepared = config.prepare()
    report = prepared.diagnostics
    assert sum(report["grid"][axis]["vacuum_cfl_inverse_square_fraction"] for axis in "xyz") == pytest.approx(1.)
    for axis in "xyz":
        widths = np.diff(getattr(prepared.solver.scene, axis + "_nodes64"))
        assert report["grid"][axis]["maximum_spacing_m"] == max(widths)
        assert report["grid"][axis]["maximum_adjacent_ratio"] >= 1
    layer = next(row for row in report["thin_dimension_resolution"] if row["name"] == "layer")
    assert layer["axis"] == "z"
    assert layer["effective_intervals"] == pytest.approx(1., rel=1e-6)
