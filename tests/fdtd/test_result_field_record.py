"""What an FDTD Result records for full-field dataset generation.

Pins the Result.fields guarantee (raw Yee volume-DFT phasors on the full grid, views,
float64 coordinates), the E sample time label, the resolved DFT window, the Bloch wrap
phase, the TFSF H mask, the last-step field ratio and what release_solver() frees.
"""
import gc
import math

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FDTD required")

H = 0.02
PML = 8


def _dipole_scene(*, source_time=None, monitors=()):
    source_time = source_time or mw.GaussianPulse(1e9, fwidth=0.5e9, phase=math.pi / 2)
    return mw.Scene(domain=mw.Domain(bounds=((-0.2, 0.2),) * 3), grid=mw.GridSpec.uniform(H),
                    boundary=mw.BoundarySpec.pml(num_layers=PML),
                    sources=(mw.PointDipole(position=(-0.05, 0.02, 0.0), polarization=(0.3, 0.5, 1.0),
                                            width=H, source_time=source_time),),
                    monitors=tuple(monitors))


def _run(scene, frequencies, steps, **sampler):
    sampler.setdefault("normalize_source", False)
    return mw.Simulation.fdtd(scene, frequencies=frequencies, config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=steps), spectral_sampler=mw.fdtd.SpectralSampler(**sampler),
        full_field_dft=True)).run()


def test_fields_are_raw_full_grid_dft_views_with_float64_nodes():
    frequencies = (0.9e9, 1.1e9)
    result = _run(_dipole_scene(), frequencies, 600, window="none")
    prepared, solver = result.prepared_scene, result.solver
    nodes = [torch.as_tensor(getattr(prepared, axis + "_nodes64")) for axis in "xyz"]
    for axis, axis_nodes in enumerate(nodes):
        # Absorber layers are outside the domain bounds and part of the stored grid.
        assert axis_nodes.numel() == round(0.4 / H) + 2 * PML + 1
        assert float(axis_nodes[PML]) == pytest.approx(-0.2) and float(axis_nodes[-PML - 1]) == pytest.approx(0.2)
    stored = result._fields
    for name, own in (("EX", 0), ("EY", 1), ("EZ", 2)):
        data = result.fields[name]
        shape = tuple(n.numel() - (axis == own) for axis, n in enumerate(nodes))
        assert data.dims == ("f", "x", "y", "z") and tuple(data.values.shape) == (2, *shape)
        for view in (data.values, result.field(name[0] + name[1].lower()).values,
                     getattr(result.E, name[1].lower()).values):
            assert view.untyped_storage().data_ptr() == stored[name].untyped_storage().data_ptr()
        for axis, label in enumerate("xyz"):
            coord = data.coords[label]
            assert coord.dtype == torch.float64
            expected = nodes[axis] if axis != own else nodes[axis][:-1] + 0.5 * torch.diff(nodes[axis])
            assert torch.equal(coord.cpu(), expected)
        assert data.attrs["field_time_offset_steps"] == 0.0
        # Raw Yee DFT: exactly the accumulated sum times the pulse scale dt * f, no averaging.
        for index, entry in enumerate(solver._dft_entries):
            payload = entry["fields"][name[0] + name[1].lower()]
            expected = (payload["real"] + 1j * payload["imag"]) * (solver._spectral_integration_dt * entry["frequency"])
            assert torch.equal(data.values[index], expected)


def test_e_phasor_weights_the_field_after_step_m_at_m_dt():
    position = (0.08, -0.04, 0.06)
    scene = _dipole_scene(monitors=(mw.monitors.FieldTimeMonitor("trace", components=("Ez",), position=position),
                                    mw.PointMonitor("probe", position=position, fields=("Ez",))))
    frequency = 1e9
    result = _run(scene, (frequency,), 900, window="none")
    trace = result.monitor("trace").sel(component="Ez")
    t = trace.coords["t"].to(torch.float64)
    samples = trace.values.to(torch.float64)
    index = tuple(result.raw_monitor("probe")["field_indices"]["Ez"])
    phasor = complex(result.fields["EZ"].values[0][index])
    dt = result.stats()["dt"]
    assert torch.allclose(t, dt * torch.arange(1, t.numel() + 1, dtype=torch.float64, device=t.device))

    def dft(times):
        return complex(dt * frequency * (samples * torch.exp(2j * math.pi * frequency * times)).sum())

    assert abs(phasor - dft(t)) <= 1e-5 * abs(phasor)
    assert abs(phasor - dft(t - 0.5 * dt)) > 1e-2 * abs(phasor)


def test_stats_report_the_resolved_dft_window():
    pulse = _run(_dipole_scene(), (1e9,), 200).stats()
    assert pulse["dft_window"] == "none" and pulse["dft_window_requested"] == "hanning"
    cw = _run(_dipole_scene(source_time=mw.CW(frequency=1e9)), (1e9,), 3000).stats()
    assert cw["dft_window"] == "hanning" and cw["dft_window_requested"] == "hanning"


def test_last_step_field_ratio_tracks_decay():
    short, long = (_run(_dipole_scene(), (0.9e9, 1.1e9), steps, window="none") for steps in (150, 3000))
    for result in (short, long):
        solver = result.solver
        last = math.sqrt(sum(float(getattr(solver, c).double().square().sum()) for c in ("Ex", "Ey", "Ez")))
        expected = []
        for entry in solver._dft_entries:
            total = sum(float(p["real"].double().square().sum() + p["imag"].double().square().sum())
                        for p in entry["fields"].values())
            expected.append(last / (2 * math.sin(math.pi * entry["frequency"] * solver.dt) * math.sqrt(total)))
        assert result.stats()["last_step_field_ratio"] == pytest.approx(expected, rel=1e-9)
    assert min(short.stats()["last_step_field_ratio"]) > 1e-2
    assert max(long.stats()["last_step_field_ratio"]) < 1e-4


def test_release_solver_keeps_only_the_phasors():
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    result = _run(_dipole_scene(), (0.9e9, 1.0e9, 1.1e9), 300, window="none")
    fields = {name: data.values.clone() for name, data in result.fields.items()}
    phasor_bytes = sum(t.numel() * t.element_size() for t in result._fields.values())
    result.release_solver()
    gc.collect()
    torch.cuda.synchronize()
    clone_bytes = sum(t.numel() * t.element_size() for t in fields.values())
    assert torch.cuda.memory_allocated() - before - clone_bytes <= phasor_bytes + 2**20
    assert result.solver is None and all(torch.equal(result.fields[name].values, value)
                                         for name, value in fields.items())


def test_tfsf_mask_marks_the_corrected_h_shell():
    bounds = ((-0.105, 0.105), (-0.085, 0.085), (-0.125, 0.125))
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.2, 0.2),) * 3), grid=mw.GridSpec.uniform(H),
                     boundary=mw.BoundarySpec.pml(num_layers=PML),
                     sources=(mw.PlaneWave(direction=(0, 0, 1), polarization="Ex",
                                           source_time=mw.GaussianPulse(1e9, fwidth=0.5e9, phase=math.pi / 2),
                                           injection=mw.TFSF(bounds)),))
    result = _run(scene, (1e9,), 200)
    record = result.stats()["tfsf"]
    (x0, x1), (y0, y1), (z0, z1) = record["node_indices"]
    x = torch.as_tensor(result.prepared_scene.x_nodes64)
    assert (x0, x1) == (int(torch.argmin((x - bounds[0][0]).abs())), int(torch.argmin((x - bounds[0][1]).abs())))
    # x-polarized along z: tangential incident E is Ex only, so only Hy (z faces) and Hz (y faces) are corrected.
    blocks = {(b["component"], b["start"]) for b in record["h_corrections"]}
    assert {component for component, _ in blocks} == {"Hy", "Hz"}
    assert ("Hy", (x0, y0, z0 - 1)) in blocks and ("Hy", (x0, y0, z1)) in blocks
    assert ("Hz", (x0, y0 - 1, z0)) in blocks and ("Hz", (x0, y1, z0)) in blocks
    masks = mw.fdtd.tfsf_h_mask(result)
    solver = result.solver
    assert {name: tuple(mask.shape) for name, mask in masks.items()} == {
        "HX": tuple(solver.Hx.shape), "HY": tuple(solver.Hy.shape), "HZ": tuple(solver.Hz.shape)}
    assert bool(masks["HX"].all())
    assert int((~masks["HY"]).sum()) == 2 * (x1 - x0) * (y1 - y0 + 1)
    assert not bool(masks["HY"][x0, y0, z0 - 1]) and bool(masks["HY"][x0, y0, z0])
    result.release_solver()
    assert all(torch.equal(masks[name], mask) for name, mask in mw.fdtd.tfsf_h_mask(result).items())
    plain = _run(_dipole_scene(), (1e9,), 50).stats()
    assert plain["tfsf"] is None


def test_bloch_wrap_plane_is_the_phase_shifted_first_plane():
    from benchmark.datagen.scenes import periodic
    from dataclasses import replace

    sample = periodic("bloch", n=12, nz=24, loaded=False, frequency=2.4e9)
    config = replace(sample.fdtd_kwargs["config"], full_field_dft=True)
    result = mw.Simulation.fdtd(sample.scene, frequencies=sample.fdtd_kwargs["frequencies"], config=config).run()
    stats = result.stats()
    kx, ky, kz = stats["bloch_wavevector"]
    lx = 0.8 * 299792458.0 / 2.4e9
    assert kx > 0 and ky == kz == 0
    assert stats["bloch_phase"][0] == complex(math.cos(kx * lx), math.sin(kx * lx))
    assert stats["bloch_phase"][1:] == (1 + 0j, 1 + 0j)
    assert stats["dft_window"] == "hanning"
    phase = stats["bloch_phase"][0]
    for name in ("EY", "EZ"):
        values = result.fields[name].values[0]
        assert float((values[-1] - phase * values[0]).abs().max()) <= 1e-5 * float(values.abs().max())
    for name in ("EX", "EZ"):
        values = result.fields[name].values[0]
        assert float((values[:, -1] - values[:, 0]).abs().max()) <= 1e-6 * float(values.abs().max())
    assert {b["component"] for b in stats["tfsf"]["h_corrections"]} <= {"Hx", "Hy"}
    assert stats["tfsf"]["mode"] == "slab" and stats["tfsf"]["axis"] == "z"
