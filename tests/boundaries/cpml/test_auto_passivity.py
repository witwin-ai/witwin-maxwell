"""Host-only boundary resolution and CPML envelope decisions."""
from dataclasses import asdict
import math
import warnings
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.boundary.runtime import (
    closed_transverse_pml, complex_fixed_k, resolve_absorber,
)
from witwin.maxwell.fdtd.runtime.passivity import (
    ENERGY_FLOOR, FIT_WINDOWS, SAMPLE_INTERVAL, WINDOW_SAMPLES,
    FDTDPassivityError, capture_passivity, check_passivity,
    make_passivity_state, preflight_passivity, sample_passivity, total_energy,
)


def _boundary(**kwargs):
    return mw.BoundarySpec.faces(num_layers=kwargs.pop("num_layers", 40), **kwargs)


@pytest.mark.parametrize("faces,expected", [
    (dict(default="periodic"), False),
    (dict(default="pec"), False),
    (dict(default="pml"), False),  # All-PML boxes are open, not vacuously closed.
    (dict(x="periodic", y="periodic", z="pml"), True),
    (dict(x="pec", y="pmc", z="pml"), True),
    (dict(x_low="pec", x_high="pmc", y="pec", z_low="pml", z_high="none"), True),
    (dict(x_low="pec", x_high="none", y="pec", z="pml"), False),
    (dict(x="bloch", y="pec", z="pml"), True),
    (dict(x="mur", y="pec", z="pml"), False),
    (dict(x="pml", y="pml", z="pec"), True),
])
def test_topology_truth_table(faces, expected):
    assert closed_transverse_pml(_boundary(**faces)) is expected


def test_symmetry_replaces_only_its_face():
    boundary = _boundary(x_low="none", x_high="pec", y="pec", z="pml")
    scene = SimpleNamespace(boundary=boundary, symmetry=(("PMC", "low"), None, None))
    assert closed_transverse_pml(scene)
    assert not closed_transverse_pml(boundary)
    scene.boundary = _boundary(x_low="none", x_high="none", y="pec", z="pml")
    assert not closed_transverse_pml(scene)
    # A symmetry plane consumes the last nominal PML face on x.
    scene.boundary = _boundary(x_low="pml", x_high="pec", y="pec", z="pec")
    assert not closed_transverse_pml(scene)


@pytest.mark.parametrize("complex_fields,k,x,expected", [
    (True, (1., 0., 0.), "bloch", True),
    (True, (0., 1., 0.), "bloch", False),
    (True, (0., 0., 0.), "bloch", False),
    (False, (1., 0., 0.), "bloch", False),
    (True, (1., 0., 0.), "periodic", False),
])
def test_complex_fixed_k(complex_fields, k, x, expected):
    solver = SimpleNamespace(scene=SimpleNamespace(boundary=_boundary(x=x, y="pec", z="pml")),
                             complex_fields_enabled=complex_fields, resolved_bloch_wavevector=k)
    assert complex_fixed_k(solver) is expected


@pytest.fixture
def host_preparation(monkeypatch):
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_scene", lambda *a, **kw: None)
    monkeypatch.setattr("witwin.maxwell.fdtd.solver._require_cuda_scene", lambda *a: None)
    monkeypatch.setattr("witwin.maxwell.fdtd.solver._get_fdtd_module", lambda: object())
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.initialization.require_cuda_scene", lambda *a: None)
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.initialization.get_fdtd_module", lambda: object())


def _simulation(boundary, **kwargs):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1, .1),) * 3),
                     grid=mw.GridSpec.uniform(.05), boundary=boundary, device="cpu")
    scene.add_source(mw.PointDipole(position=(0., 0., 0.), polarization="Ez",
                                  source_time=mw.GaussianPulse(frequency=1e9, fwidth=.2e9)))
    return mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=10), **kwargs)


@pytest.mark.parametrize("faces,expected", [
    (dict(x="periodic", y="periodic", z="pml"), "absorber"),
    (dict(x="pec", y="pec", z="pml"), "absorber"),
    (dict(default="pml", num_layers=2), "cpml"),
    (dict(x="none", y="pec", z="pml", num_layers=2), "cpml"),
    (dict(x="bloch", y="bloch", z="pml", num_layers=2, bloch_wavevector=(1., 0., 0.)), "cpml"),
])
def test_auto_prepares(host_preparation, faces, expected):
    simulation = _simulation(_boundary(**faces))
    assert simulation.config.absorber == mw.AbsorberKind.AUTO
    solver = simulation.prepare().solver
    assert solver.active_absorber_type == expected
    assert solver.absorber_resolution_reason.startswith("auto:")


@pytest.mark.parametrize("absorber", ["cpml", "stablepml", "pml", "absorber"])
def test_explicit_overrides_keep_profiles(host_preparation, absorber):
    solver = _simulation(_boundary(x="periodic", y="pec", z="pml", num_layers=2),
                         absorber=absorber).prepare().solver
    assert solver.active_absorber_type == absorber
    assert solver.absorber_resolution_reason == "explicit absorber selection"
    assert solver.uses_cpml == (absorber in ("cpml", "stablepml"))


def test_auto_requires_40_layers(host_preparation):
    with pytest.raises(ValueError, match=r'x\(low=periodic, high=periodic\), y\(low=pec, high=pec\).*z-low=39, z-high=39.*Either pass absorber="cpml".*passivity guard.*>= 40 PML layers'):
        _simulation(_boundary(x="periodic", y="pec", z="pml", num_layers=39)).prepare()


def test_auto_refusal_names_only_effective_pml_faces():
    scene = SimpleNamespace(
        boundary=_boundary(x_low="pml", x_high="pec", y="pmc", z_low="pml", z_high="none",
                           num_layers=12),
        symmetry=(("PMC", "low"), None, None),
    )
    with pytest.raises(ValueError) as error:
        resolve_absorber(scene, "auto", complex_fields=False)
    message = str(error.value)
    assert "x(low=pmc, high=pec), y(low=pmc, high=pmc)" in message
    assert "PML faces with fewer than 40 layers: z-low=12." in message
    assert "x-low=12" not in message and "z-high=" not in message
    assert 'absorber="cpml"' in message and ">= 40 PML layers" in message


def test_distributed_auto_fails_closed():
    scene = SimpleNamespace(boundary=_boundary(x="periodic", y="pec", z="pml"))
    with pytest.raises(NotImplementedError, match="real-field closed-transverse.*distributed"):
        resolve_absorber(scene, "auto", complex_fields=False, distributed=True)
    assert resolve_absorber(scene, "cpml", complex_fields=False, distributed=True)[0] == "cpml"
    scene.boundary = _boundary(default="pml", num_layers=2)
    assert resolve_absorber(scene, "auto", complex_fields=False, distributed=True)[0] == "cpml"


def _history(energy, peak=None):
    steps = torch.arange(len(energy), dtype=torch.float64) * SAMPLE_INTERVAL * WINDOW_SAMPLES
    return torch.stack((steps, energy, torch.full_like(energy, float(energy.max() if peak is None else peak))), 1)


def test_exponential_growth_trips():
    steps = torch.arange(FIT_WINDOWS, dtype=torch.float64) * 2048
    history = _history(torch.exp(2.5e-6 * steps))
    original = history.clone()
    with pytest.raises(FDTDPassivityError, match="real-field closed-transverse PML.*slope=.*40 PML layers") as caught:
        check_passivity(history, dict(topology="real-field closed-transverse PML"))
    assert "Growth slower than 4e-7 per step (about 1.5x energy over 1e6 steps) is not detected" in str(caught.value)
    assert torch.equal(history, original)


def test_complex_fixed_k_growth_warns_once_with_measurements():
    steps = torch.arange(FIT_WINDOWS, dtype=torch.float64) * SAMPLE_INTERVAL * WINDOW_SAMPLES
    history = _history(torch.exp(2.5e-6 * steps))
    original = history.clone()
    state = _sample_state()
    with pytest.warns(RuntimeWarning, match=r"complex fixed-k Bloch.*slope=.*stderr=.*window steps \[0, 30720\]") as caught:
        check_passivity(history, state)
        check_passivity(history, state)
        check_passivity(history * torch.tensor([1., 2., 2.]), state)
    assert len(caught) == 1
    record = state["warning"]
    assert record["slope"] == pytest.approx(2.5e-6)
    assert 0 <= record["stderr"] < 1e-18
    assert record["window"] == (0, int(steps[-1]))
    assert record["step"] == int(steps[-1])
    assert torch.equal(history, original)


@pytest.mark.parametrize("kind", ["decay", "constant", "beating", "noisy", "floor", "slow_growth"])
def test_passive_histories_do_not_trip(kind):
    steps = torch.arange(FIT_WINDOWS, dtype=torch.float64) * 2048
    generator = torch.Generator().manual_seed(123)
    energy = dict(decay=torch.exp(-2.5e-6 * steps), constant=torch.ones_like(steps),
                  beating=1 + .001 * torch.sin(steps / 200),
                  noisy=1 + .001 * torch.randn(FIT_WINDOWS, generator=generator, dtype=torch.float64),
                  floor=ENERGY_FLOOR * .1 * torch.exp(2.5e-6 * steps),
                  slow_growth=torch.exp(1e-7 * steps))[kind]
    check_passivity(_history(energy, peak=1.1 if kind == "floor" else None),
                    dict(topology="complex fixed-k Bloch"))


def test_bounded_beating_sampled_envelopes_do_not_trip():
    # Many field/load exchange cycles in each complete envelope window.
    state = _sample_state()
    for i in range(FIT_WINDOWS * WINDOW_SAMPLES * 2):
        energy = torch.tensor(1 + .4 * math.sin(i * .2), dtype=torch.float64)
        sample_passivity(state, (i + 1) * SAMPLE_INTERVAL, energy)


def _sample_state():
    return dict(topology="complex fixed-k Bloch", start_step=0, sample_count=0, window_count=0, warning=None,
                history=torch.zeros((FIT_WINDOWS, 3), dtype=torch.float64),
                peak=torch.zeros((), dtype=torch.float64), window_max=torch.zeros((), dtype=torch.float64))


def test_window_floor_and_start_are_respected():
    state = _sample_state()
    state["start_step"] = 100000
    sample_passivity(state, 32, torch.tensor(1.))
    assert state["sample_count"] == 0 and state["peak"] == 1
    for i in range(FIT_WINDOWS * WINDOW_SAMPLES):
        step = 100000 + i * SAMPLE_INTERVAL
        sample_passivity(state, step, torch.tensor(1e-12 * math.exp(2.5e-6 * (step - 100000))))
    assert state["window_count"] == FIT_WINDOWS


def test_resume_preserves_partial_window_and_trip_step(tmp_path):
    full, split = _sample_state(), _sample_state()
    energies = torch.exp(torch.arange(FIT_WINDOWS * WINDOW_SAMPLES, dtype=torch.float64) * 32 * 2.5e-6)
    cut = 5 * WINDOW_SAMPLES + 13
    for i in range(cut):
        sample_passivity(full, (i + 1) * 32, energies[i])
        sample_passivity(split, (i + 1) * 32, energies[i])
    from witwin.maxwell.fdtd.resume import _detach_to_cpu
    path = tmp_path / "guard.pt"
    torch.save(_detach_to_cpu(capture_passivity(SimpleNamespace(_passivity_state=split))), path)
    restored = _sample_state()
    copies, updates = [], []
    preflight_passivity(SimpleNamespace(_passivity_state=restored),
                        torch.load(path, weights_only=True), copies, updates)
    for target, source in copies:
        target.copy_(source)
    for target, name, value in updates:
        if isinstance(target, dict):
            target[name] = value
        else:
            setattr(target, name, value)
    for i in range(cut, len(energies)):
        if i == len(energies) - 1:
            for state in (full, restored):
                with pytest.warns(RuntimeWarning, match="complex fixed-k Bloch") as caught:
                    sample_passivity(state, (i + 1) * 32, energies[i])
                assert len(caught) == 1
        else:
            sample_passivity(full, (i + 1) * 32, energies[i])
            sample_passivity(restored, (i + 1) * 32, energies[i])
    assert torch.equal(full["history"], restored["history"])
    assert full["warning"] == restored["warning"]


def test_warned_state_disk_resume_preserves_metadata_without_rewarning(tmp_path, host_preparation):
    solver = _simulation(_boundary(x="bloch", y="pec", z="pml", num_layers=2,
                                   bloch_wavevector=(1., 0., 0.))).prepare().solver
    solver._passivity_state = make_passivity_state(solver)
    state = solver._passivity_state
    steps = torch.arange(FIT_WINDOWS, dtype=torch.float64) * SAMPLE_INTERVAL * WINDOW_SAMPLES
    history = _history(torch.exp(2.5e-6 * steps))
    with pytest.warns(RuntimeWarning, match="complex fixed-k Bloch"):
        check_passivity(history, state)
    record = state["warning"].copy()
    path = tmp_path / "warned.pt"
    captured = capture_passivity(solver)
    torch.save(captured, path)
    state["warning"]["slope"] = -1.  # The captured record owns its host data.
    assert captured["warning"] == record
    solver._passivity_state = make_passivity_state(solver)
    copies, updates = [], []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preflight_passivity(solver, torch.load(path, weights_only=True), copies, updates)
        for target, source in copies:
            target.copy_(source)
        for target, name, value in updates:
            if isinstance(target, dict):
                target[name] = value
            else:
                setattr(target, name, value)
        check_passivity(history, solver._passivity_state)
        for i in range(FIT_WINDOWS * WINDOW_SAMPLES):
            step = (i + 1) * SAMPLE_INTERVAL
            sample_passivity(solver._passivity_state, step, torch.tensor(math.exp(2.5e-6 * step)))
    assert not caught
    assert solver._passivity_state["warning"] == solver.passivity_guard_warning == record
    assert solver.passivity_guard_status == "warned: complex fixed-k growth"


def test_missing_resume_history_fails_closed():
    with pytest.raises(ValueError, match="guard history is missing"):
        preflight_passivity(SimpleNamespace(_passivity_state=_sample_state()), None, [], [])


def test_guard_eligibility_and_energy(host_preparation):
    solver = _simulation(_boundary(x="periodic", y="pec", z="pml"), absorber="cpml").prepare().solver
    state = make_passivity_state(solver)
    assert state["topology"] == "real-field closed-transverse PML"
    assert state["start_step"] == solver._time_plan["latest_source_settling_step"]
    solver.Ex.fill_(2.)
    assert total_energy(solver) > 0
    solver.active_absorber_type = "absorber"
    assert make_passivity_state(solver) is None
    solver = _simulation(_boundary(default="pml", num_layers=2)).prepare().solver
    assert make_passivity_state(solver) is None


@pytest.mark.parametrize("attr,value,label", (
    ("_copper_foils", [object()], "CopperFoil"), ("_thin_sheets", (object(),), "thin sheets"),
    ("_coaxial_element_runtimes", (object(),), "coaxial elements"),
    ("surface_impedance_enabled", True, "SIBC"), ("dispersive_enabled", True, "dispersive/ADE media"),
))
def test_uncounted_storage_skips_guard_with_status(host_preparation, attr, value, label):
    solver = _simulation(_boundary(x="periodic", y="pec", z="pml"), absorber="cpml").prepare().solver
    assert make_passivity_state(solver) is not None
    assert solver.passivity_guard_status == "active"
    setattr(solver, attr, value)
    with pytest.warns(RuntimeWarning, match="passivity guard skipped in a topology with known CPML growth: "
                                            "energy storage not counted"):
        assert make_passivity_state(solver) is None
    assert solver.passivity_guard_status.startswith("skipped: energy storage not counted")
    assert label in solver.passivity_guard_status


def test_copper_and_sheet_scene_is_skipped_without_sampling_storage():
    from tests.boundaries.cpml.test_guard_storage import _mixed_copper_scene

    scene = _mixed_copper_scene(mw.BoundarySpec.faces(x="pec", y="pec", z="pml", num_layers=2))
    simulation = mw.Simulation.fdtd(scene, frequency=2e9, backend="cpu_reference", absorber="cpml",
                                    run_time=mw.TimeConfig(time_steps=1))
    with pytest.warns(RuntimeWarning, match="passivity guard skipped .*CopperFoil, thin sheets"):
        solver = simulation.prepare().solver
        assert make_passivity_state(solver) is None
    assert solver.passivity_guard_status == "skipped: energy storage not counted (CopperFoil, thin sheets)"
    assert not any(hasattr(runtime, "energy") for runtime in solver._thin_sheets)


def test_closed_transverse_core_refused(host_preparation):
    from witwin.maxwell.fdtd.runtime.core_path import validate_core_path_request
    sim = _simulation(_boundary(x="periodic", y="pec", z="pml", num_layers=2), absorber="cpml")
    solver = sim.prepare().solver
    sim.config.fdtd_core_path = True
    with pytest.raises(ValueError, match="closed-transverse CPML \\(passivity guard\\)"):
        validate_core_path_request(sim, solver=solver)


_TRANSVERSE = ("periodic", "pec", "pmc", "bloch", "pml", "none")


@pytest.mark.parametrize("absorber", ("cpml", "stablepml"))
@pytest.mark.parametrize("x", _TRANSVERSE)
@pytest.mark.parametrize("y", _TRANSVERSE)
def test_guard_sampled_configurations_are_core_ineligible(host_preparation, absorber, x, y):
    """The guard samples only on the existing eager loop; the core path never samples.

    Every configuration that builds a guard state must therefore be refused by the
    core-path blockers, which explicit requests and automatic admission share.
    """
    from witwin.maxwell.fdtd.runtime.core_path import core_path_blockers, validate_core_path_request

    bloch = "bloch" in (x, y)
    boundary = mw.BoundarySpec.faces(
        x=x, y=y, z="pml", num_layers=2,
        **({"bloch_wavevector": (1. if x == "bloch" else 0., 1. if y == "bloch" else 0., 0.)} if bloch else {}))
    sim = _simulation(boundary, absorber=absorber, fdtd_core_path="auto")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", mw.FDTDCoreFallbackWarning)
        solver = sim.prepare().solver
    state = make_passivity_state(solver)
    blockers = core_path_blockers(solver)
    guard_blockers = {"closed-transverse CPML (passivity guard)", "Bloch/complex fields"}
    if x == y and x in ("periodic", "pec", "bloch"):
        assert state is not None  # the enumeration covers both guarded topologies
    if state is None:
        return
    assert guard_blockers & set(blockers), (x, y, absorber, blockers)
    # Automatic admission keeps the existing eager loop, where the guard samples.
    assert solver._fdtd_execution_path == "existing"
    assert not hasattr(solver, "_core_runtime")
    assert solver.passivity_guard_status == "active"
    assert solver._fdtd_fallback_reason in blockers
    assert [str(entry.message) for entry in caught if entry.category is mw.FDTDCoreFallbackWarning] == [
        f"FDTD core path fell back to existing execution: {solver._fdtd_fallback_reason}"]
    sim.config.fdtd_core_path = True
    with pytest.raises(ValueError, match="passivity guard|Bloch/complex fields"):
        validate_core_path_request(sim, solver=solver)


def test_nonfinite_energy_fails_closed():
    energy = torch.ones(FIT_WINDOWS, dtype=torch.float64)
    energy[-1] = float("inf")
    with pytest.raises(FDTDPassivityError):
        check_passivity(_history(energy), dict(topology="real-field closed-transverse PML"))
    # A blow-up is not measured slow growth: complex fixed-k cells fail closed too,
    # also after an earlier growth warning.
    state = _sample_state()
    with pytest.raises(FDTDPassivityError, match="complex fixed-k Bloch; non-finite"):
        check_passivity(_history(energy), state)
    assert state["warning"] is None
    state = _sample_state()
    state["warning"] = dict(slope=5e-7, stderr=1e-8, window=(0, 1), step=1)
    with pytest.raises(FDTDPassivityError, match="non-finite"):
        check_passivity(_history(energy), state)


def test_upper_three_stderr_and_rising_endpoint_required():
    steps = torch.arange(FIT_WINDOWS, dtype=torch.float64) * 2048
    # Slope alone is below the limit; deterministic residuals make its upper
    # three-standard-error bound exceed it, with a rising endpoint.
    perturbation = .02 * torch.cos(torch.arange(FIT_WINDOWS, dtype=torch.float64) * math.pi)
    energy = torch.exp(3e-7 * steps + perturbation)
    energy[-1] *= math.exp(.05)
    state = _sample_state()
    with pytest.warns(RuntimeWarning, match="complex fixed-k Bloch"):
        check_passivity(_history(energy), state)
    assert state["warning"]["slope"] + 3 * state["warning"]["stderr"] > 4e-7
    energy[-1] = energy[0]
    state = _sample_state()
    check_passivity(_history(energy), state)
    assert state["warning"] is None


def test_cw_uses_transient_even_without_spectral_window(host_preparation):
    scene = _simulation(_boundary(x="bloch", y="pec", z="pml", num_layers=2,
                                   bloch_wavevector=(1., 0., 0.))).scene
    scene.sources = [mw.PointDipole(position=(0., 0., 0.), source_time=mw.CW(frequency=1e9))]
    sim = mw.Simulation.fdtd(scene, frequency=1e9,
        spectral_sampler=mw.SpectralSampler(window="none"),
        run_time=mw.TimeConfig(time_steps=10, transient_cycles=100))
    solver = sim.prepare().solver
    state = make_passivity_state(solver)
    assert state["start_step"] >= math.ceil(100 / 1e9 / solver.dt)
    assert state["start_step"] > solver._time_plan["spectral_start_step"]


def test_total_energy_includes_both_lc_banks_once(host_preparation):
    solver = _simulation(_boundary(x="bloch", y="pec", z="pml", num_layers=2,
                                   bloch_wavevector=(1., 0., 0.))).prepare().solver
    make_passivity_state(solver)
    load = SimpleNamespace(inductance=2., capacitance=3.,
        inductor_current=torch.tensor(4.), capacitor_voltage=torch.tensor(5.),
        inductor_current_imag=torch.tensor(6.), capacitor_voltage_imag=torch.tensor(7.))
    solver._port_runtimes = [SimpleNamespace(lumped=load, embedded_network_name=None, field_name="Ex")]
    assert float(total_energy(solver)) == .5 * (2 * (4**2 + 6**2) + 3 * (5**2 + 7**2))


def test_cpu_reference_auto_resolves_before_capability_check():
    sim = _simulation(_boundary(default="none", num_layers=0), backend="cpu_reference")
    assert sim.prepare().solver.active_absorber_type == "none"
    graded = _simulation(_boundary(x="pec", y="pec", z="pml"), backend="cpu_reference")
    with pytest.raises(NotImplementedError, match="CPML absorber only"):
        graded.prepare()


def test_main_loop_raises_and_disk_resume_keeps_trip_step(monkeypatch, tmp_path):
    # Exercise the real solve/checkpoint lifecycle on the existing CPU reference
    # profile. Only the sampled energy and window lengths are controlled here.
    from witwin.maxwell.fdtd.runtime import passivity
    monkeypatch.setattr(passivity, "SAMPLE_INTERVAL", 2)
    monkeypatch.setattr(passivity, "WINDOW_SAMPLES", 4)
    monkeypatch.setattr(passivity, "FIT_WINDOWS", 4)

    def synthetic_energy(solver):
        state = solver._passivity_state
        sample = state["window_count"] * 4 + state["sample_count"] + 1
        return torch.tensor(math.exp(2.5e-6 * sample * 2), dtype=torch.float64)

    monkeypatch.setattr(passivity, "total_energy", synthetic_energy)

    def prepared():
        scene = mw.Scene(domain=mw.Domain(bounds=((- .1, .1),) * 3),
            grid=mw.GridSpec.uniform(.05), device="cpu",
            boundary=_boundary(x="pec", y="pec", z="pml", num_layers=2))
        return mw.Simulation.fdtd(scene, frequency=1e9, absorber="cpml", backend="cpu_reference",
            full_field_dft=False, cuda_graph=False, run_time=mw.TimeConfig(time_steps=40)).prepare()

    direct = prepared()
    with pytest.raises(FDTDPassivityError):
        direct.run()
    split = prepared()
    checkpoint = split.run_until(13)  # Six samples: one full and one partial window.
    assert checkpoint.auxiliary["passivity"]["sample_count"] == 2
    path = tmp_path / "main-loop.pt"
    checkpoint.save(path)
    resumed = prepared()
    with pytest.raises(FDTDPassivityError):
        resumed.run(resume_from=mw.FDTDResumeCheckpoint.load(path))
    assert torch.equal(direct.solver._passivity_state["history"], resumed.solver._passivity_state["history"])
    assert direct.solver._passivity_state["history"][-1, 0] == 32


def test_main_loop_warning_metadata_survives_disk_resume(monkeypatch, tmp_path):
    # Isolate warning lifecycle/Result plumbing on the CPU reference runtime;
    # the growing complex-field physics is covered by the CUDA scene gate.
    from witwin.maxwell.fdtd.runtime import passivity
    monkeypatch.setattr(passivity, "SAMPLE_INTERVAL", 2)
    monkeypatch.setattr(passivity, "WINDOW_SAMPLES", 4)
    monkeypatch.setattr(passivity, "FIT_WINDOWS", 4)
    make_state = passivity.make_passivity_state

    def synthetic_state(solver):
        state = make_state(solver)
        state["topology"] = "complex fixed-k Bloch"
        return state

    def synthetic_energy(solver):
        state = solver._passivity_state
        sample = state["window_count"] * 4 + state["sample_count"] + 1
        return torch.tensor(math.exp(2.5e-6 * sample * 2), dtype=torch.float64)

    monkeypatch.setattr(passivity, "make_passivity_state", synthetic_state)
    monkeypatch.setattr(passivity, "total_energy", synthetic_energy)

    def prepared():
        scene = mw.Scene(domain=mw.Domain(bounds=((- .1, .1),) * 3),
            grid=mw.GridSpec.uniform(.05), device="cpu",
            boundary=_boundary(x="pec", y="pec", z="pml", num_layers=2))
        return mw.Simulation.fdtd(scene, frequency=1e9, absorber="cpml", backend="cpu_reference",
            full_field_dft=True, cuda_graph=False, run_time=mw.TimeConfig(time_steps=80)).prepare()

    direct = prepared()
    with pytest.warns(RuntimeWarning, match="complex fixed-k Bloch") as caught:
        result = direct.run()
    assert len(caught) == 1
    assert result.metadata["passivity_guard"] == "warned: complex fixed-k growth"
    record = result.metadata["passivity_guard_warning"]
    assert record == direct.solver.passivity_guard_warning == result.solver_stats["passivity_guard_warning"]
    assert record["slope"] == pytest.approx(2.5e-6)
    assert record["window"] == (8, 32) and record["step"] == 32
    split = prepared()
    with pytest.warns(RuntimeWarning, match="complex fixed-k Bloch") as caught:
        checkpoint = split.run_until(45)
    assert len(caught) == 1
    path = tmp_path / "warned-main-loop.pt"
    checkpoint.save(path)
    resumed = prepared()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = resumed.run(resume_from=mw.FDTDResumeCheckpoint.load(path))
    assert not caught
    assert restored.metadata["passivity_guard"] == result.metadata["passivity_guard"]
    assert restored.metadata["passivity_guard_warning"] == record
    assert restored.solver_stats["passivity_guard_warning"] == record
    assert torch.equal(direct.solver._passivity_state["history"], resumed.solver._passivity_state["history"])


@pytest.mark.parametrize("fixed_k", (False, True))
def test_auto_cpml_coefficients_equal_explicit_cpml(host_preparation, fixed_k):
    boundary = (_boundary(x="bloch", y="pec", z="pml", num_layers=2, bloch_wavevector=(1., 0., 0.))
                if fixed_k else _boundary(default="pml", num_layers=2))
    automatic = _simulation(boundary).prepare().solver
    explicit = _simulation(boundary, absorber="cpml").prepare().solver
    assert automatic.cpml_config == explicit.cpml_config
    if fixed_k:
        assert automatic.cpml_config["alpha_max"] > 0
    for name, value in vars(explicit).items():
        if name.startswith("cpml_") and isinstance(value, torch.Tensor):
            assert torch.equal(getattr(automatic, name), value), name


def test_result_records_resolved_absorber_and_reason():
    result = _simulation(_boundary(default="pml", num_layers=2), backend="cpu_reference",
                         full_field_dft=True).run()
    assert result.metadata["absorber"] == "cpml"
    assert result.metadata["absorber_resolution_reason"] == "auto: complex fields or open PML topology"


@pytest.mark.parametrize("slope", (0., 1e-7, 3.99e-7))
def test_growth_below_resolution_is_not_detected(slope):
    steps = torch.arange(FIT_WINDOWS, dtype=torch.float64) * SAMPLE_INTERVAL * WINDOW_SAMPLES
    check_passivity(_history(torch.exp(slope * steps)), dict(topology="complex fixed-k Bloch"))


def test_cw_benchmark_fixture_and_horizon_are_sufficient():
    from tests.boundaries.cpml.test_passivity_gpu import (
        CW_GROWTH_HORIZON, CW_REPORTED_ONSET, CW_TRANSIENT, _cw_modulated_slab,
    )
    from benchmark.scenes import SCENARIOS, build_scene

    simulation = _cw_modulated_slab(device="cpu")
    benchmark = build_scene("modulated_slab")
    assert simulation.config.absorber == mw.AbsorberKind.CPML
    assert asdict(simulation.scene.grid) == asdict(benchmark.grid)
    assert simulation.scene.boundary == benchmark.boundary
    assert all(source.source_time.kind == "cw" for source in simulation.scene.sources)
    assert simulation.scene.structures[0].material.modulation is not None
    assert closed_transverse_pml(simulation.scene)
    assert simulation.config.run_time.transient_cycles == math.ceil(
        CW_TRANSIENT * min(SCENARIOS["modulated_slab"].frequencies))
    # The archived benchmark step is 16.667 ps. This check sizes the record;
    # the GPU case repeats it with the actual prepared dt.
    duration = FIT_WINDOWS * WINDOW_SAMPLES * SAMPLE_INTERVAL * (1 / 60e9)
    assert CW_TRANSIENT + duration < CW_REPORTED_ONSET
    assert CW_REPORTED_ONSET + duration < CW_GROWTH_HORIZON


def test_cw_trip_report_records_onset_and_trip_step():
    from tests.boundaries.cpml.test_passivity_gpu import _cw_growth_report

    step = SAMPLE_INTERVAL * WINDOW_SAMPLES
    windows = {i * step: [i * step, 1., 1.] for i in range(1, FIT_WINDOWS + 1)}
    windows[(FIT_WINDOWS + 1) * step] = [(FIT_WINDOWS + 1) * step, 1.002, 1.002]
    windows[(FIT_WINDOWS + 2) * step] = [(FIT_WINDOWS + 2) * step, 1.01, 1.01]
    trip = (FIT_WINDOWS + 2) * step
    report = _cw_growth_report(windows, dt=1e-11, start_step=0,
                               planned_steps=100000, trip_step=trip, error="test")
    assert report["onset_step"] == (FIT_WINDOWS + 1) * step
    assert report["trip_step"] == trip
    assert report["onset_time_s"] == report["onset_step"] * 1e-11
    assert report["trip_time_s"] == trip * 1e-11
    assert len(report["windows"]) == FIT_WINDOWS + 2
