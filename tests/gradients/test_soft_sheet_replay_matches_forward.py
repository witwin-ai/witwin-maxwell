"""Adjoint soft-sheet rebuilds must preserve the forward launch exactly."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.adjoint import core
from witwin.maxwell.fdtd.runtime.stepping import electric_update_support


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _prepare_solver(source_kind, face, waveform, device):
    boundary = mw.BoundarySpec.pml(num_layers=4)
    if face == "pec":
        boundary = boundary.with_faces(y_low="pec")
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2),) * 3),
        grid=mw.GridSpec.uniform(0.025),
        boundary=boundary,
        subpixel_samples=mw.SubpixelSpec(samples=1, averaging="arithmetic"),
        device=device,
    )
    # CPU preparation is forward-only. The native cases also exercise preparation
    # with a trainable scene; both devices rebuild using trainable epsilon leaves.
    density = torch.full((1, 1, 1), 0.4, device=device, requires_grad=device == "cuda")
    scene.add_material_region(mw.MaterialRegion(
        name="design",
        geometry=mw.Box(position=(0.03, 0.0, 0.0), size=(0.10, 0.12, 0.12)),
        density=density,
        eps_bounds=(1.0, 4.0),
    ))
    source_time = (mw.CW(frequency=1e9, phase=0.37) if waveform == "cw" else
                   mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, phase=0.37))
    kwargs = dict(
        direction=(1, 0, 0), polarization=(0, 0.6, 0.8), injection="soft",
        injection_axis="x", source_time=source_time,
    )
    if source_kind == "plane_wave":
        source = mw.PlaneWave(**kwargs)
    elif source_kind == "gaussian_beam":
        source = mw.GaussianBeam(beam_waist=0.16, focus=(0.03, 0.02, -0.01), **kwargs)
    else:
        source = mw.AstigmaticGaussianBeam(
            beam_waist=(0.16, 0.22), focus=(0.03, 0.02, -0.01),
            focus_u=0.02, focus_v=-0.03, **kwargs,
        )
    scene.add_source(source)
    simulation = mw.Simulation.fdtd(
        scene, frequency=1e9, backend="cpu_reference" if device == "cpu" else "native",
        cuda_graph=False, full_field_dft=False, run_time=mw.TimeConfig(time_steps=2),
    )
    # prepare() initializes source terms through init_field(), exactly
    # as the forward launch does; no separate test-only source initialization.
    solver = simulation.prepare().solver
    assert simulation.has_trainable_parameters == (device == "cuda")
    assert solver._compiled_sources[0]["kind"] == source_kind
    assert source_kind in core._SOURCE_REPLAY_REBUILD_KINDS
    assert solver._compiled_sources[0]["injection"]["kind"] == "soft"
    assert not solver.tfsf_enabled
    return solver


def _resolve_replay(solver):
    # The adjoint supplies epsilon leaves with the frozen forward values. Its
    # checkpoint replay enters this same resolver under no_grad.
    epsilon = tuple(getattr(solver, "eps_" + component).detach().clone().requires_grad_()
                    for component in ("Ex", "Ey", "Ez"))
    with torch.no_grad():
        return core._resolved_source_term_lists(solver, *epsilon)


def _assert_terms_equal(forward_terms, replay_terms, waveform):
    # Mixed transverse polarization keeps both components, making order observable.
    assert len(forward_terms) == len(replay_terms) == 2
    for forward, replay in zip(forward_terms, replay_terms):
        for key in ("field_name", "offsets", "source_index", "source_time", "omega",
                    "phase_real", "phase_imag"):
            assert replay[key] == forward[key], key
        for key in ("patch", "cw_cos_patch", "cw_sin_patch", "delay_patch",
                    "activation_delay_patch"):
            expected, actual = forward[key], replay[key]
            if expected is None:
                assert actual is None, key
            else:
                assert actual is not None, key
                assert actual.shape == expected.shape, key
                assert actual.dtype == expected.dtype, key
                assert torch.equal(actual, expected), key
        if waveform == "cw":
            assert replay["cw_cos_patch"] is not None
            assert replay["cw_sin_patch"] is not None
            assert replay["patch"] is None
        else:
            assert replay["patch"] is not None
            assert replay["delay_patch"] is not None
            assert replay["cw_cos_patch"] is replay["cw_sin_patch"] is None


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs CUDA for trainable FDTD preparation")),
])
@pytest.mark.parametrize("waveform", ("cw", "pulse"))
@pytest.mark.parametrize("face", ("pml", "pec"), ids=("all-pml", "transverse-pec"))
@pytest.mark.parametrize("source_kind", ("plane_wave", "gaussian_beam", "astigmatic_gaussian_beam"))
def test_soft_sheet_replay_matches_forward(source_kind, face, waveform, device, monkeypatch):
    solver = _prepare_solver(source_kind, face, waveform, device)
    namespaces = []
    build = core._build_source_replay_solver

    def capture_namespace(*args):
        replay = build(*args)
        namespaces.append(replay)
        return replay

    monkeypatch.setattr(core, "_build_source_replay_solver", capture_namespace)
    source_terms, electric_terms, magnetic_terms = _resolve_replay(solver)
    # Prove the adjoint resolver rebuilt and initialized a fresh namespace rather
    # than taking its fallback that reuses forward terms.
    assert len(namespaces) == 1
    replay = namespaces[0]
    assert source_terms == tuple(solver._source_terms) == ()
    for family, terms in (("electric", electric_terms), ("magnetic", magnetic_terms)):
        forward_terms = getattr(solver, f"_{family}_source_terms")
        assert all(term is rebuilt for term, rebuilt in
                   zip(terms, getattr(replay, f"_{family}_source_terms")))
        assert all(term is not original for term, original in zip(terms, forward_terms))
        _assert_terms_equal(forward_terms, terms, waveform)

    for axis in "xyz":
        for side in ("low", "high"):
            name = f"boundary_{axis}_{side}_code"
            assert getattr(replay, name) == getattr(solver, name)
    for component in ("Ex", "Ey", "Ez"):
        assert electric_update_support(replay, component) == electric_update_support(solver, component)
    for term in electric_terms:
        patch = term["cw_cos_patch"] if waveform == "cw" else term["patch"]
        support = electric_update_support(solver, term["field_name"])
        for axis, (start, size, (lo, hi)) in enumerate(zip(term["offsets"], patch.shape, support)):
            assert lo <= start < start + size <= hi
            if axis != 0:
                assert (start, start + size) == (lo, hi)


def test_soft_sheet_replay_requires_boundary_codes(monkeypatch):
    solver = _prepare_solver("plane_wave", "pec", "cw", "cpu")
    namespace = core.SimpleNamespace

    def missing_boundary_code(**kwargs):
        del kwargs["boundary_y_low_code"]
        return namespace(**kwargs)

    # Remove the code from the actual replay namespace, after forward preparation.
    monkeypatch.setattr(core, "SimpleNamespace", missing_boundary_code)
    with pytest.raises(AttributeError, match="boundary_y_low_code"):
        _resolve_replay(solver)
