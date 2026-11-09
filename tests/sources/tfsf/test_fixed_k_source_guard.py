"""Broadband fixed-k source preparation and incident-line selection."""

import pytest

import witwin.maxwell as mw


@pytest.fixture(autouse=True)
def host_preparation(monkeypatch):
    # Preparation allocates fields but launches no kernels in these fixtures.
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr("witwin.maxwell.fdtd.solver._require_cuda_scene", lambda scene: None)
    monkeypatch.setattr("witwin.maxwell.fdtd.solver._get_fdtd_module", lambda: object())
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.initialization.require_cuda_scene", lambda scene: None)
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.initialization.get_fdtd_module", lambda: object())


def _simulation(source_time, *, x="bloch", k=2., secondary=False, plane_wave=False, polarization="Ey"):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3),) * 3),
        grid=mw.GridSpec.uniform(0.05),
        boundary=(mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml", num_layers=2,
                                       bloch_wavevector=(k, 0., 0.)) if plane_wave else
                  mw.BoundarySpec.faces(default="pec", x=x, bloch_wavevector=(k, 0., 0.))),
        device="cpu",
    )
    if secondary:
        scene.add_source(mw.PointDipole(position=(0., 0., 0.), polarization="Ez",
                                       source_time=source_time))
    if plane_wave:
        scene.add_source(mw.PlaneWave(direction=(.1, 0., .99), polarization=polarization, source_time=source_time,
                                     injection=mw.TFSF.slab(axis="z", bounds=(-.15, .15))))
    else:
        scene.add_source(mw.PointDipole(position=(0.05, 0., 0.), polarization="Ez", source_time=source_time))
    return mw.Simulation.fdtd(scene, frequencies=(1.e9,), run_time=mw.TimeConfig(time_steps=10))


@pytest.mark.parametrize("source_time", (
    mw.GaussianPulse(frequency=1.e9, fwidth=.2e9),
    mw.RickerWavelet(frequency=1.e9),
    mw.CustomSourceTime(lambda t: t * 0., characteristic_frequency=1.e9),
))
@pytest.mark.parametrize("secondary", (False, True))
@pytest.mark.parametrize("plane_wave", (False, True))
def test_fixed_k_bloch_prepares_broadband_source(source_time, secondary, plane_wave):
    solver = _simulation(source_time, secondary=secondary, plane_wave=plane_wave).prepare().solver
    assert solver.complex_fields_enabled
    if plane_wave:
        assert solver._tfsf_state["provider"] == "plane_wave_fixed_k"
        assert solver._tfsf_state["auxiliary_grid"].electric.is_complex()
        assert solver._source_time_offset == 0.0
        # The shared normalization reference follows the first declared source.
        assert solver._normalization_time_offset == (0.5 if secondary else 0.0)
        assert solver._normalization_label_offset == (0.5 if secondary else 0.0)


def test_fixed_k_incident_reference_uses_physical_yee_labels():
    import torch

    solver = _simulation(mw.GaussianPulse(frequency=1.e9, fwidth=.2e9), plane_wave=True).prepare().solver
    line = solver._tfsf_state["auxiliary_grid"]
    expected = torch.zeros_like(line.spectrum)
    # Reach the entry plane from the soft launch, including a nonzero spectrum.
    for step in range(160):
        line.advance_magnetic()
        line.advance_electric()
        phase = torch.exp(1j * line.omega * ((step + 1) * line.dt))
        expected.add_(phase[:, None] * line.electric[:, line.reference_index])
    reference = line.incident_spectrum(solver)
    scale = line.dt * (line.omega / (2 * torch.pi) if solver._spectral_reference_frequency is None
                       else solver._spectral_reference_frequency)
    assert expected.abs().max() > 0
    torch.testing.assert_close(reference["electric"], expected * scale[:, None], rtol=2.e-13, atol=0.)


@pytest.mark.parametrize("plane_wave", (False, True))
def test_fixed_k_bloch_cw_still_prepares(plane_wave):
    prepared = _simulation(mw.CW(frequency=1.e9), plane_wave=plane_wave).prepare()
    assert prepared.solver.complex_fields_enabled
    assert prepared.solver.resolved_bloch_wavevector == (2., 0., 0.)
    assert prepared.solver.tfsf_enabled if plane_wave else prepared.solver._source_terms


@pytest.mark.parametrize("x,k", (("bloch", 0.), ("periodic", 2.), ("pec", 2.)))
def test_pulsed_zero_k_and_real_fields_still_prepare(x, k):
    prepared = _simulation(mw.GaussianPulse(frequency=1.e9, fwidth=.2e9), x=x, k=k).prepare()
    assert prepared.solver._source_terms


def test_fixed_k_checkpoint_preserves_all_line_state_on_disk(tmp_path):
    import torch
    from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state, restore_checkpoint_state

    pulse = mw.GaussianPulse(frequency=1.e9, fwidth=.2e9)
    solver = _simulation(pulse, plane_wave=True).prepare().solver
    line = solver._tfsf_state["auxiliary_grid"]
    for _ in range(100):
        line.advance_magnetic()
        line.advance_electric()
    state = capture_checkpoint_state(solver, 100)
    names = {"tfsf_aux_" + name for name in line.state_tensors()}
    assert names.issubset(state.tensors)
    checkpoint = mw.FDTDResumeCheckpoint(step=100, total_steps=200, dt=solver.dt,
        physics=state, signature={}, auxiliary={})
    path = tmp_path / "fixed-k.pt"
    checkpoint.save(path)
    restored = _simulation(pulse, plane_wave=True).prepare().solver
    restore_checkpoint_state(restored, mw.FDTDResumeCheckpoint.load(path).physics)
    other = restored._tfsf_state["auxiliary_grid"]
    for _ in range(30):
        for current in (line, other):
            current.advance_magnetic()
            current.advance_electric()
    for name, tensor in line.state_tensors().items():
        assert torch.equal(tensor, other.state_tensors()[name]), name


@pytest.mark.parametrize("polarization", ("Ey", (.99, 0., -.1)))
def test_empty_fixed_k_slab_confines_the_incident_field_on_the_yee_grid(polarization):
    import math
    import torch
    from witwin.maxwell.fdtd.excitation.tfsf_apply import apply_tfsf_e_correction, apply_tfsf_h_correction

    solver = _simulation(mw.GaussianPulse(frequency=1.e9, fwidth=.2e9), plane_wave=True,
                         polarization=polarization).prepare().solver
    line = solver._tfsf_state["auxiliary_grid"]
    fields = {name: torch.zeros_like(getattr(solver, name), dtype=torch.complex64)
              for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    spacing = .05

    def derivative(field, axis, forward):
        if forward:
            return torch.diff(field, dim=axis) / spacing
        shape = list(field.shape)
        shape[axis] += 1
        result = torch.zeros(shape, dtype=field.dtype)
        interior = [slice(None)] * 3
        interior[axis] = slice(1, -1)
        result[tuple(interior)] = torch.diff(field, dim=axis) / spacing
        if axis != 2:
            angle = solver.resolved_bloch_wavevector[axis] * .6
            phase = complex(math.cos(angle), math.sin(angle))
            result.select(axis, 0).copy_((field.select(axis, 0) - field.select(axis, -1) / phase) / spacing)
            result.select(axis, -1).copy_((field.select(axis, 0) * phase - field.select(axis, -1)) / spacing)
        return result

    def advance(target, source, coefficient, forward):
        updates = []
        for i, axis in enumerate("xyz"):
            j, k = (i + 1) % 3, (i + 2) % 3
            updates.append(derivative(fields[source + "xyz"[k]], j, forward) -
                           derivative(fields[source + "xyz"[j]], k, forward))
        for axis, curl in zip("xyz", updates):
            fields[target + axis].add_(curl, alpha=coefficient)

    peak = 0.
    for step in range(700):
        advance("H", "E", -line.ch, True)
        for axis in "xyz":
            getattr(solver, "H" + axis).copy_(fields["H" + axis].real)
            getattr(solver, "H" + axis + "_imag").copy_(fields["H" + axis].imag)
        apply_tfsf_h_correction(solver, step * solver.dt)
        line.advance_magnetic()
        for axis in "xyz":
            fields["H" + axis].copy_(torch.complex(getattr(solver, "H" + axis), getattr(solver, "H" + axis + "_imag")))
        advance("E", "H", line.ce, False)
        for axis in "xyz":
            getattr(solver, "E" + axis).copy_(fields["E" + axis].real)
            getattr(solver, "E" + axis + "_imag").copy_(fields["E" + axis].imag)
        apply_tfsf_e_correction(solver, step * solver.dt)
        line.advance_electric()
        for axis in "xyz":
            fields["E" + axis].copy_(torch.complex(getattr(solver, "E" + axis), getattr(solver, "E" + axis + "_imag")))
        lo, hi = solver._tfsf_state["lower"][2], solver._tfsf_state["upper"][2]
        component = "Ey" if polarization == "Ey" else "Ex"
        peak = max(peak, float(fields[component].abs().max()))
        for name in ("Ex", "Ey", "Ez"):
            stop = hi if name == "Ez" else hi + 1
            outside = torch.cat((fields[name][..., :lo], fields[name][..., stop:]), dim=2)
            assert outside.abs().max() < 2e-5
        positions = solver._component_positions(component, (6, 6, lo), (1, 1, hi - lo + 1), dtype=solver.Ex.dtype)
        expected = line.electric["xyz".index(component[1]), line.padding + lo:line.padding + hi + 1]
        expected = expected * torch.exp(1j * solver.resolved_bloch_wavevector[0] * positions[0, 0, :, 0])
        torch.testing.assert_close(fields[component][6, 6, lo:hi + 1], expected, rtol=0., atol=2e-5)
    assert peak > .1


@pytest.mark.parametrize("resolution,polarization", ((20, "TE"), (40, "TM")))
def test_broadband_acceptance_fixture_prepares_on_host(monkeypatch, resolution, polarization):
    from scripts.dev.fdtd.bloch_floquet_acceptance import common

    scene_type = mw.Scene
    monkeypatch.setattr(common.mw, "Scene", lambda **kwargs: scene_type(**dict(kwargs, device="cpu")))
    simulation = common.make_simulation(resolution, .55 * common.K0, polarization, 12.)
    solver = simulation.prepare().solver
    line = solver._tfsf_state["auxiliary_grid"]
    assert line.frequencies == common.FREQUENCIES
    assert line.ds == pytest.approx(common.LAMBDA0 / resolution)
    assert line.incident_spectrum(solver)["position"] == pytest.approx(-.65 * common.LAMBDA0)
    # The fixed-k line keeps its own reference and opts out of the entry-plane convention.
    assert line.incident_reference is None
    assert "incident_reference" not in solver._tfsf_state



def test_unchanged_path_gpu_probe_prepares_its_scenes_on_host():
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "scripts/dev/fdtd/bloch_floquet_acceptance/broadband_checks.py"
    tree = ast.parse(path.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == "_UNCHANGED_PROBE" for target in node.targets))
    # Only scene construction/preparation runs here; time stepping stays on GPU.
    code = ast.literal_eval(assignment.value).split("    values = capture_checkpoint_state")[0]
    code = code.replace('device="cuda"', 'device="cpu"').replace('.run()', '.prepare()')
    code = "\n".join(line for line in code.splitlines() if not line.startswith("assert Path(mw.__file__)"))
    exec(compile(code, str(path), "exec"), {})


@pytest.mark.parametrize("source_time", (
    mw.GaussianPulse(frequency=1.e9, fwidth=.2e9),
    mw.RickerWavelet(frequency=1.e9),
))
@pytest.mark.parametrize("make_source", ("soft_plane_wave", "gaussian_beam"))
def test_fixed_k_bloch_rejects_non_cw_directional_source_without_slab_line(source_time, make_source):
    # Only TFSF-slab plane waves have an exact fixed-k incident line; beams and soft
    # plane waves still carry one carrier direction across the band.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3),) * 3),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml", num_layers=2,
                                       bloch_wavevector=(2., 0., 0.)),
        device="cpu",
    )
    if make_source == "soft_plane_wave":
        scene.add_source(mw.PlaneWave(direction=(.1, 0., .99), polarization="Ey", source_time=source_time,
                                     injection="soft", injection_axis="z"))
    else:
        scene.add_source(mw.GaussianBeam(direction=(.1, 0., .99), polarization="Ey", beam_waist=.2,
                                        focus=(0., 0., 0.), source_time=source_time, injection_axis="z"))
    with pytest.raises(NotImplementedError, match="Broadband fixed-k injection of beams.*off the carrier"):
        mw.Simulation.fdtd(scene, frequencies=(1.e9,), run_time=mw.TimeConfig(time_steps=10)).prepare()


@pytest.mark.parametrize("source_time", (
    mw.GaussianPulse(frequency=1.e9, fwidth=.2e9),
    mw.RickerWavelet(frequency=1.e9),
))
def test_fixed_k_bloch_admits_broadband_point_source(source_time):
    # The Bloch boundary gives a point source the exact fixed-k phase at every frequency.
    prepared = _simulation(source_time).prepare()
    assert prepared.solver.complex_fields_enabled
    assert prepared.solver._source_terms
