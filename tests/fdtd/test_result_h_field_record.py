"""The opt-in magnetic volume DFT (``FDTDConfig.volume_dft_components``).

Pins what an FDTD Result records for H next to E: raw Yee volume-DFT phasors on the
full H grids (views, float64 coordinates), the H sample time label (-0.5 steps,
proved against a FieldTimeMonitor trace and by the discrete Faraday/Ampere
relations, each with a wrong-half-step control), equality of the core and existing
paths and determinism, the CopperFoil H correction that H-from-E misses, the Bloch
wrap plane of H, and what release_solver() keeps.
"""
import cmath
import gc
import math

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FDTD required")

H = 0.02
PML = 8
MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12
EH = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
DIPOLE = (-0.05, 0.02, 0.0)


def _scene(*, structures=(), monitors=(), source_time=None):
    source_time = source_time or mw.GaussianPulse(1e9, fwidth=0.5e9, phase=math.pi / 2)
    return mw.Scene(domain=mw.Domain(bounds=((-0.2, 0.2),) * 3), grid=mw.GridSpec.uniform(H),
                    boundary=mw.BoundarySpec.pml(num_layers=PML),
                    sources=(mw.PointDipole(position=DIPOLE, polarization=(0.3, 0.5, 1.0),
                                            width=H, source_time=source_time),),
                    structures=tuple(structures), monitors=tuple(monitors))


def _run(scene, frequencies, steps, *, components=EH, **sampler):
    sampler.setdefault("normalize_source", False)
    sampler.setdefault("window", "none")
    simulation = mw.Simulation.fdtd(scene, frequencies=frequencies, config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=steps), spectral_sampler=mw.fdtd.SpectralSampler(**sampler),
        full_field_dft=True, volume_dft_components=components))
    return simulation.run()


def test_h_fields_are_raw_full_grid_dft_views_and_leave_e_unchanged():
    frequencies = (0.9e9, 1.1e9)
    result = _run(_scene(), frequencies, 600)
    e_only = _run(_scene(), frequencies, 600, components=("Ex", "Ey", "Ez"))
    assert set(result.fields) == {"EX", "EY", "EZ", "HX", "HY", "HZ"} and set(e_only.fields) == {"EX", "EY", "EZ"}
    assert result.solver._core_runtime is not None and e_only.solver._core_runtime is not None
    prepared, solver = result.prepared_scene, result.solver
    nodes = [torch.as_tensor(getattr(prepared, axis + "_nodes64")) for axis in "xyz"]
    midpoints = [n[:-1] + 0.5 * torch.diff(n) for n in nodes]
    stored = result._fields
    for name, own in (("HX", 0), ("HY", 1), ("HZ", 2)):
        data = result.fields[name]
        shape = tuple(n.numel() - (axis != own) for axis, n in enumerate(nodes))
        assert data.dims == ("f", "x", "y", "z") and tuple(data.values.shape) == (2, *shape)
        assert tuple(data.values.shape[1:]) == tuple(getattr(solver, "H" + name[1].lower()).shape)
        for view in (data.values, result.field("H" + name[1].lower()).values, getattr(result.H, name[1].lower()).values):
            assert view.untyped_storage().data_ptr() == stored[name].untyped_storage().data_ptr()
        for axis, label in enumerate("xyz"):
            coord = data.coords[label]
            assert coord.dtype == torch.float64
            assert torch.equal(coord.cpu(), nodes[axis] if axis == own else midpoints[axis])
        assert data.attrs["field_time_offset_steps"] == -0.5
        assert data.attrs["yee_location"] == "face_" + "xyz"[own]
        assert data.attrs["phasor_convention"] == result.fields["EX"].attrs["phasor_convention"]
        assert data.attrs["normalization"] == "raw"
        for index, entry in enumerate(solver._dft_entries):
            payload = entry["fields"]["H" + name[1].lower()]
            expected = (payload["real"] + 1j * payload["imag"]) * (solver._spectral_integration_dt * entry["frequency"])
            assert torch.equal(data.values[index], expected)
    # Recording H changes nothing about E (or the E decay record).
    for name in ("EX", "EY", "EZ"):
        assert torch.equal(result.fields[name].values, e_only.fields[name].values)
        assert result.fields[name].attrs == e_only.fields[name].attrs
    assert result.stats()["last_step_field_ratio"] == e_only.stats()["last_step_field_ratio"]
    # Exact repeatability (no float atomics): a second run is bitwise equal.
    again = _run(_scene(), frequencies, 600)
    assert all(torch.equal(again.fields[name].values, result.fields[name].values) for name in result.fields)


def test_h_phasor_weights_h_after_step_m_at_m_dt_and_it_is_sampled_half_a_step_earlier():
    position = (0.08, -0.04, 0.06)
    scene = _scene(monitors=(mw.monitors.FieldTimeMonitor("trace", components=("Hz", "Hx"), position=position),
                             mw.PointMonitor("probe", position=position, fields=("Hz", "Hx"))))
    frequency = 1e9
    result = _run(scene, (frequency,), 900)
    dt = result.stats()["dt"]
    traces = result.monitor("trace")
    for component in ("Hz", "Hx"):
        trace = traces.sel(component=component)
        assert trace.attrs["field_time_offsets_steps"]["H"] == -0.5
        t = trace.coords["t"].to(torch.float64)
        samples = trace.values.to(torch.float64)
        assert torch.allclose(t, dt * torch.arange(1, t.numel() + 1, dtype=torch.float64, device=t.device))
        index = tuple(result.raw_monitor("probe")["field_indices"][component])
        data = result.fields[component.upper()]
        stored = complex(data.values[0][index])

        def dft(times):
            return complex(dt * frequency * (samples * torch.exp(2j * math.pi * frequency * times)).sum())

        # Stored: the H held after step m (the trace sample at label t = m dt) weighted at m dt.
        assert abs(stored - dft(t)) <= 1e-5 * abs(stored)
        # At its physical time (t - dt/2), from the attrs offset: the true H phasor.
        offset = data.attrs["field_time_offset_steps"]
        physical = stored * cmath.exp(2j * math.pi * frequency * offset * dt)
        assert abs(physical - dft(t - 0.5 * dt)) <= 1e-5 * abs(physical)
        # Wrong half step either way (H at the E time, or a full step early) fails.
        half_step_phase = math.pi * frequency * dt  # omega dt / 2
        for wrong in (0.0, -1.0):
            error = abs(stored * cmath.exp(2j * math.pi * frequency * wrong * dt) - dft(t - 0.5 * dt))
            assert error > 0.5 * half_step_phase * abs(physical)


def _curl_e(fields, inv_h):
    ex, ey, ez = fields
    return ((ez[:, 1:, :] - ez[:, :-1, :] - ey[:, :, 1:] + ey[:, :, :-1]) * inv_h,
            (ex[:, :, 1:] - ex[:, :, :-1] - ez[1:, :, :] + ez[:-1, :, :]) * inv_h,
            (ey[1:, :, :] - ey[:-1, :, :] - ex[:, 1:, :] + ex[:, :-1, :]) * inv_h)


def _curl_h(fields, inv_h):
    """curl H on the interior E edges: Ex[:, 1:-1, 1:-1], Ey[1:-1, :, 1:-1], Ez[1:-1, 1:-1, :]."""
    hx, hy, hz = fields
    return ((hz[:, 1:, 1:-1] - hz[:, :-1, 1:-1] - hy[:, 1:-1, 1:] + hy[:, 1:-1, :-1]) * inv_h,
            (hx[1:-1, :, 1:] - hx[1:-1, :, :-1] - hz[1:, :, 1:-1] + hz[:-1, :, 1:-1]) * inv_h,
            (hy[1:, 1:-1, :] - hy[:-1, 1:-1, :] - hx[1:-1, 1:, :] + hx[1:-1, :-1, :]) * inv_h)


def _interior_e(fields):
    ex, ey, ez = fields
    return ex[:, 1:-1, 1:-1], ey[1:-1, :, 1:-1], ez[1:-1, 1:-1, :]


def _discrete_relations(result, frequency_index=0):
    """Raw discrete Faraday/Ampere residual terms of one DFT frequency (float64).

    With S the stored raw sums (scale s = dt * f) of the N recorded steps, z = exp(i omega dt):
      Faraday: (1/z - 1) S_H + s z^N H^(N+1/2) = -(dt / mu0) curl S_E
      Ampere:  (1 - z) S_E + s z^(N+1) E^N    =  (dt / eps) curl S_H
    H^(N+1/2) is formed from the final solver state (H^(N-1/2) - dt/mu0 curl E^N, exact in vacuum).
    """
    solver = result.solver
    frequency = result.frequencies[frequency_index]
    dt = result.stats()["dt"]
    steps = result.stats()["steps_run"]
    z = cmath.exp(2j * math.pi * frequency * dt)
    scale = dt * frequency
    inv_h = 1.0 / H
    s_e = [result.fields[n].values[frequency_index].to(torch.complex128) for n in ("EX", "EY", "EZ")]
    s_h = [result.fields[n].values[frequency_index].to(torch.complex128) for n in ("HX", "HY", "HZ")]
    e_last = [getattr(solver, n).double() for n in ("Ex", "Ey", "Ez")]
    h_last = [getattr(solver, n).double() for n in ("Hx", "Hy", "Hz")]
    h_next = [h - dt / MU0 * c for h, c in zip(h_last, _curl_e(e_last, inv_h))]
    faraday_lhs = [(1 / z - 1) * s + scale * z**steps * h for s, h in zip(s_h, h_next)]
    faraday_rhs = [-(dt / MU0) * c for c in _curl_e(s_e, inv_h)]
    ampere_lhs = [(1 - z) * s + scale * z ** (steps + 1) * e
                  for s, e in zip(_interior_e(s_e), _interior_e(e_last))]
    ampere_curl = [dt * c for c in _curl_h(s_h, inv_h)]  # divide by eps of the sampled cells
    return faraday_lhs, faraday_rhs, ampere_lhs, ampere_curl, s_e, s_h


def _box_slices(shape, lo, hi):
    return tuple(slice(max(a, 0), min(b, n)) for a, b, n in zip(lo, hi, shape))


@pytest.mark.parametrize("dielectric", [False, True])
def test_e_and_h_phasors_satisfy_the_discrete_curl_relations(dielectric):
    eps_r = 4.0
    lo_box, hi_box = (0.02, -0.12, -0.12), (0.16, 0.12, 0.12)
    structures = ()
    if dielectric:
        structures = (mw.Structure(geometry=mw.Box(position=tuple((a + b) / 2 for a, b in zip(lo_box, hi_box)),
                                                   size=tuple(b - a for a, b in zip(lo_box, hi_box))),
                                   material=mw.Material(eps_r=eps_r), name="slab"),)
    frequency = 1e9
    result = _run(_scene(structures=structures), (frequency,), 1500)
    faraday_lhs, faraday_rhs, ampere_lhs, ampere_curl, _, s_h = _discrete_relations(result)
    nodes = len(result.prepared_scene.x_nodes64)
    interior = (PML + 2, nodes - PML - 3)  # primal node index range clear of the absorber
    # Faraday holds everywhere outside the absorber (the electric dipole drives E only).
    for lhs, rhs, h in zip(faraday_lhs, faraday_rhs, s_h):
        window = _box_slices(lhs.shape, (interior[0],) * 3, (interior[1],) * 3)
        residual = (lhs - rhs)[window].abs().max() / rhs[window].abs().max()
        assert float(residual) < 1e-3, float(residual)
    # Ampere in a uniform medium region away from the dipole and from material interfaces.
    def node(value):
        return round((value + 0.2) / H) + PML
    if dielectric:
        region, eps = ((node(lo_box[0]) + 2, node(lo_box[1]) + 2, node(lo_box[2]) + 2),
                       (node(hi_box[0]) - 2, node(hi_box[1]) - 2, node(hi_box[2]) - 2)), eps_r * EPS0
    else:
        # The dipole's smoothed current reaches about three cells; stay five cells away from it.
        region, eps = ((node(DIPOLE[0]) + 6,) + (interior[0],) * 2, (interior[1],) * 3), EPS0
    for lhs, curl in zip(ampere_lhs, ampere_curl):
        # Interior-edge tensors are offset by one node on their two transverse axes.
        window = _box_slices(lhs.shape, tuple(r - 1 for r in region[0]), tuple(r - 1 for r in region[1]))
        rhs = curl[window] / eps
        residual = (lhs[window] - rhs).abs().max() / rhs.abs().max()
        assert float(residual) < 1e-3, float(residual)
    # Written with the phasors at their own times (E at m dt, H at (m + offset) dt) the Faraday
    # relation is the phase-free i omega~ mu0 H = curl E (+ the record-end term) only for offset
    # -0.5; with H labelled at the E time (offset 0) it misses by about omega dt / 2.
    dt = result.stats()["dt"]
    omega = 2 * math.pi * frequency
    omega_tilde = 2 / dt * math.sin(omega * dt / 2)
    window = _box_slices(s_h[2].shape, (interior[0],) * 3, (interior[1],) * 3)
    curl = -MU0 / dt * faraday_rhs[2][window]
    end_term = MU0 / dt * (faraday_lhs[2] - (cmath.exp(-1j * omega * dt) - 1) * s_h[2])[window]
    for offset in (-0.5, 0.0, -1.0):
        h_true = s_h[2][window] * cmath.exp(1j * omega * offset * dt)
        error = float((1j * omega_tilde * MU0 * h_true - curl - end_term).abs().max() / curl.abs().max())
        if offset == -0.5:
            assert error < 1e-3, error
        else:
            assert error > 0.5 * omega * dt / 2, (offset, error)


def test_existing_path_records_the_same_h_as_the_core_path():
    # The existing-runner record is frozen in addendum w7-row16e (W7 row 16e deleted the runner).
    from tests.support import frozen_oracles
    existing = frozen_oracles.load("tests/fdtd/test_result_h_field_record.py::existing_500")
    frequencies = (0.9e9, 1.1e9)
    core = _run(_scene(), frequencies, 500)
    assert core.solver._core_runtime is not None
    for name in ("EX", "EY", "EZ", "HX", "HY", "HZ"):
        reference = existing[name]
        bound = 1e-4 * float(reference.abs().max())
        assert float((core.fields[name].values.cpu() - reference).abs().max()) <= bound, name
    again = _run(_scene(), frequencies, 500)
    assert all(torch.equal(again.fields[name].values, core.fields[name].values) for name in core.fields)


def test_stored_h_next_to_copper_foil_differs_from_h_derived_from_e():
    foil_x = 0.06
    foil = mw.Structure(geometry=mw.Box(position=(foil_x, 0.0, 0.0), size=(0.0, 0.16, 0.16)),
                        material=mw.media.CopperFoil(thickness=35e-6, frequency_range=(0.5e9, 2e9)), name="foil")
    result = _run(_scene(structures=(foil,)), (1e9,), 1500)
    assert result.solver._copper_foils
    faraday_lhs, faraday_rhs, *_ = _discrete_relations(result)
    nodes = len(result.prepared_scene.x_nodes64)
    foil_plane = round((foil_x + 0.2) / H) + PML  # the foil's primal x plane (= the corrected Hy/Hz cell index)
    footprint = slice(foil_plane - 8, foil_plane + 2), slice(PML + 5, nodes - PML - 5), slice(PML + 5, nodes - PML - 5)
    for name, lhs, rhs in zip(("Hy", "Hz"), faraday_lhs[1:], faraday_rhs[1:]):
        # Per x plane, the stored H minus the plain Faraday step of the stored E relative to that step.
        residual = ((lhs - rhs)[footprint].abs().amax(dim=(1, 2)) / rhs[footprint].abs().amax(dim=(1, 2)))
        on_foil = float(residual[8])
        off_foil = float(torch.cat((residual[:8], residual[9:])).max())
        # Away from the foil, H follows Faraday to float32 accumulation; next to it the update adds the
        # split-trace copper magnetic correction that H derived from E does not have.
        assert off_foil < 5e-5, (name, off_foil)
        assert on_foil > 2e-4 and on_foil > 20 * off_foil, (name, on_foil, off_foil)


def test_bloch_h_wrap_plane_is_the_phase_shifted_first_plane():
    from benchmark.datagen.scenes import periodic
    from dataclasses import replace

    sample = periodic("bloch", n=12, nz=24, loaded=False, frequency=2.4e9)
    config = replace(sample.fdtd_kwargs["config"], full_field_dft=True, volume_dft_components=EH)
    result = mw.Simulation.fdtd(sample.scene, frequencies=sample.fdtd_kwargs["frequencies"], config=config).run()
    phase = result.stats()["bloch_phase"]
    assert phase[0] != 1 and phase[1] == 1
    hx = result.fields["HX"].values[0]  # node along x (Bloch, phase), midpoints along y
    hy = result.fields["HY"].values[0]  # node along y (Bloch at k_y = 0)
    assert float((hx[-1] - phase[0] * hx[0]).abs().max()) <= 1e-5 * float(hx.abs().max())
    assert float((hy[:, -1] - phase[1] * hy[:, 0]).abs().max()) <= 1e-5 * float(hy.abs().max())
    assert float(hx.abs().max()) > 0 and result.fields["HX"].attrs["field_time_offset_steps"] == -0.5


def _footprint(components):
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    result = _run(_scene(), (0.9e9, 1.0e9, 1.1e9), 300, components=components)
    after_run = torch.cuda.memory_allocated() - before
    phasor_bytes = sum(t.numel() * t.element_size() for t in result._fields.values())
    result.release_solver()
    gc.collect()
    torch.cuda.synchronize()
    return result, after_run, torch.cuda.memory_allocated() - before, phasor_bytes


def test_release_solver_keeps_only_the_e_and_h_phasors():
    e_result, e_run, e_kept, e_bytes = _footprint(("Ex", "Ey", "Ez"))
    del e_result
    h_result, h_run, h_kept, h_bytes = _footprint(EH)
    assert h_result.solver is None and set(h_result.fields) == {"EX", "EY", "EZ", "HX", "HY", "HZ"}
    assert e_kept <= e_bytes + 2**20 and h_kept <= h_bytes + 2**20
    # E and H grids have the same sample counts up to one plane per axis; H about doubles the phasors.
    assert 1.8 * e_bytes < h_bytes < 2.2 * e_bytes
    # The live solver adds the H accumulators (two float32 per complex64 phasor sample).
    assert h_run - e_run >= (h_bytes - e_bytes) * 2 - 2**20
