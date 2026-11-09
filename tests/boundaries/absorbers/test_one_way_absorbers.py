"""Second-order Mur, mode-matched ABC faces, broadband (Higdon) mode ABC and
internal absorber planes.

Tier A checks the declarative API and the closed-form coefficient algebra on the
host. Tier B measures reflections on GPU by reference subtraction:

* Mur faces: a dipole radiates toward one absorbing x-high face; the reference
  is the same scene with that face terminated by CPML (whose own reflection is
  ~3e-5 here, two orders below the Mur levels being compared). Measured
  (2026-10-29) at 600 MHz: first-order 3.2e-2 (0 deg) / 7.3e-2 (45 deg),
  second-order (Higdon product of two Mur operators) 6.2e-3 / 1.2e-2.
* Guided mode: an eps_r = 4 rectangular guide is terminated 0.205 m after a mode
  monitor; the reference guide continues 1.8 m so its far-end echo does not
  return within the window. R = |b - b_ref| / |a_ref|. Measured at
  1.6/2.0/2.4 GHz: CPML 4.5e-5/4.7e-5/1.6e-4, vacuum Mur 0.36/0.39/0.43,
  ModeABC@2 GHz 3.6e-2/1.1e-4/4.7e-2, broadband ModeABC (order 2)
  3.1e-4/1.3e-3/5.0e-4, InternalAbsorber identical to the matching face.
* Late-time stability: opposite one-way faces on a dielectric guide must not
  grow (an undamped transverse-stencil second-order Mur and all-pass mode ABC
  were measured to diverge in exactly this configuration).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.boundary.abc import (
    HIGDON_DAMPING,
    chebyshev_band_frequencies,
    higdon_coefficients,
    one_way_mode_coefficient,
)

_INF = math.inf
_PLANE = (0.0, _INF, _INF)


# --------------------------------------------------------------------------- #
# Tier A: declarative API and algebra (host)                                  #
# --------------------------------------------------------------------------- #


def test_mur_abc_order_and_permittivity_validation():
    assert mw.MurABC().order == 1
    assert mw.MurABC(order=2, permittivity=2.25).permittivity == 2.25
    for kwargs in ({"order": 3}, {"order": True}, {"permittivity": 0.5}, {"permittivity": math.nan}):
        with pytest.raises(ValueError):
            mw.MurABC(**kwargs)


def test_mode_abc_validation():
    spec = mw.ModeABC(frequency_range=(1e9, 2e9), order=2)
    assert spec.is_broadband and spec.frequency is None
    with pytest.raises(ValueError):
        mw.ModeABC(frequency=1e9, frequency_range=(1e9, 2e9))
    with pytest.raises(ValueError):
        mw.ModeABC(frequency_range=(2e9, 1e9))
    with pytest.raises(ValueError):
        mw.ModeABC(mode_index=-1)
    with pytest.raises(ValueError):
        mw.ModeABC(frequency_range=(1e9, 2e9), order=3)


def test_boundary_spec_normalizes_abc_parameters():
    second = mw.BoundarySpec.mur(order=2, permittivity=1.5)
    assert second.kind == "mur" and second.mur_abc == mw.MurABC(order=2, permittivity=1.5)
    # ABC parameters only survive when a face uses them.
    pml = mw.BoundarySpec.faces(default="pml", num_layers=8, mur_abc=mw.MurABC(order=2), mode_abc=mw.ModeABC())
    assert pml.mur_abc == mw.MurABC() and pml.mode_abc is None
    assert pml.kind == "pml"
    faced = mw.BoundarySpec.faces(default="pml", num_layers=8, x_high="mode_abc")
    assert faced.face_kind("x", "high") == "mode_abc" and faced.mode_abc == mw.ModeABC()
    assert faced.with_faces(x_low="mur").mode_abc == mw.ModeABC()
    with pytest.raises(ValueError):
        mw.BoundarySpec.faces(default="mode_abc_typo")


def test_set_abc_parameters_enter_the_resume_fingerprint():
    import hashlib

    from witwin.maxwell.fdtd.resume import _hash_value

    def digest(boundary):
        hasher = hashlib.sha256()
        _hash_value(hasher, "boundary", boundary)
        return hasher.hexdigest()

    variants = [
        mw.BoundarySpec.mur(),
        mw.BoundarySpec.mur(order=2),
        mw.BoundarySpec.mur(permittivity=2.0),
        mw.BoundarySpec.faces(default="pml", num_layers=8, x_high="mode_abc", mode_abc=mw.ModeABC(frequency=1e9)),
        mw.BoundarySpec.faces(default="pml", num_layers=8, x_high="mode_abc", mode_abc=mw.ModeABC(frequency=2e9)),
    ]
    assert len({digest(boundary) for boundary in variants}) == len(variants)
    # Unset ABC fields are omitted, so the default spec hashes as before they existed.
    assert digest(mw.BoundarySpec.mur()) == digest(mw.BoundarySpec.faces(default="mur"))


def test_internal_absorber_declaration():
    absorber = mw.InternalAbsorber((0.1, 0.0, 0.0), _PLANE, "-", boundary=mw.ModeABC(frequency=1e9), name="sink")
    assert absorber.normal_axis == "x" and absorber.direction_sign == -1
    with pytest.raises(ValueError):
        mw.InternalAbsorber((0, 0, 0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        mw.InternalAbsorber((0, 0, 0), _PLANE, "up")
    assert mw.InternalAbsorber((0, 0, 0), _PLANE, boundary=mw.MurABC(order=2)).boundary.order == 2
    scene = mw.Scene(device="cpu", internal_absorbers=[absorber])
    assert scene.clone().internal_absorbers == [absorber]
    with pytest.raises(ValueError):
        scene.add_internal_absorber(mw.InternalAbsorber((0.2, 0, 0), _PLANE, name="sink"))


def test_one_way_mode_coefficient_limits_and_annihilation():
    c_dt, delta = 0.4, 1.0
    omega_dt = 1e-3
    beta_delta = omega_dt * delta / c_dt
    mur = (c_dt - delta) / (c_dt + delta)
    assert one_way_mode_coefficient(omega_dt, beta_delta) == pytest.approx(mur, rel=1e-6)
    # The coefficient annihilates the outgoing discrete wave exactly:
    # E_b(n+1) - E_a(n) - p (E_a(n+1) - E_b(n)) = 0 with E_b = E_a exp(-j beta d).
    omega_dt, beta_delta = 0.31, 0.97
    p = one_way_mode_coefficient(omega_dt, beta_delta)
    z, shift = np.exp(1j * omega_dt), np.exp(-1j * beta_delta)
    assert abs(shift * z - 1.0 - p * (z - shift)) < 1e-14


def test_higdon_product_reduces_to_mur_and_multiplies_reflections():
    p = -0.4
    single = higdon_coefficients([p])
    # E0(n+1) = E1(n) + p (E1(n+1) - E0(n)): c[1,0]=1, c[1,1]=-p, c[0,0]=p, c[0,1]=-1.
    np.testing.assert_allclose(single, [[p, -1.0], [1.0, -p]])
    p1, p2 = -0.3, -0.6
    table = higdon_coefficients([p1, p2])
    assert table.shape == (3, 3) and table[2, 0] == 1.0
    # The product polynomial factorizes: evaluate at an arbitrary (Z, S).
    z, shift = 0.7 + 0.2j, 1.3 - 0.4j
    product = ((z + p1) - (1 + p1 * z) * shift) * ((z + p2) - HIGDON_DAMPING * (1 + p2 * z) * shift)
    value = sum(table[m, l] * z ** m * shift ** l for m in range(3) for l in range(3))
    assert abs(value - product) < 1e-12
    # Undamped factors vanish for a static uniform field; the damped product
    # keeps only a simple root there.
    assert abs(sum(table[m, l] for m in range(3) for l in range(3))) < 1e-12
    assert abs(sum(m * table[m, l] for m in range(3) for l in range(3))) > 1e-4


def test_chebyshev_band_frequencies():
    nodes = chebyshev_band_frequencies(1.0, 3.0, 2)
    np.testing.assert_allclose(nodes, [2.0 - math.sqrt(0.5), 2.0 + math.sqrt(0.5)])
    np.testing.assert_allclose(chebyshev_band_frequencies(1.0, 3.0, 1), [2.0])


# --------------------------------------------------------------------------- #
# Tier B: GPU physics                                                         #
# --------------------------------------------------------------------------- #

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_FACE_FREQ = 6e8
_FACE_DX = 0.025


def _face_scene(boundary):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 0.5), (-1.0, 1.0), (-1.0, 1.0))),
        grid=mw.GridSpec.uniform(_FACE_DX),
        boundary=boundary,
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=_FACE_DX,
            source_time=mw.GaussianPulse(frequency=_FACE_FREQ, fwidth=0.6 * _FACE_FREQ),
        )
    )
    for name, y in (("deg0", 0.0), ("deg45", 0.35)):
        scene.add_monitor(mw.FieldTimeMonitor(name, components=("Ez",), position=(0.35, y, 0.0125)))
    return scene


def _face_traces(boundary):
    result = mw.Simulation.fdtd(
        _face_scene(boundary),
        frequencies=[_FACE_FREQ],
        run_time=mw.TimeConfig(time_steps=520),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()
    dt = float(result.solver.dt)
    return {
        name: np.asarray(result.monitor(name)["field"].detach().cpu(), dtype=np.float64).ravel()
        for name in ("deg0", "deg45")
    }, dt


def _spectral_error(trace, reference, dt, frequency):
    phase = np.exp(-2j * np.pi * frequency * np.arange(reference.size) * dt)
    return abs(np.sum((trace - reference) * phase)) / abs(np.sum(reference * phase))


@cuda
def test_second_order_mur_reflects_less_than_first_order():
    def faces(**kwargs):
        return mw.BoundarySpec.faces(default="pml", num_layers=10, **kwargs)

    reference, dt = _face_traces(faces())
    first, _ = _face_traces(faces(x_high="mur"))
    second, _ = _face_traces(faces(x_high="mur", mur_abc=mw.MurABC(order=2)))
    for name in ("deg0", "deg45"):
        assert np.all(np.isfinite(second[name]))
        r1 = _spectral_error(first[name], reference[name], dt, _FACE_FREQ)
        r2 = _spectral_error(second[name], reference[name], dt, _FACE_FREQ)
        assert r1 > 1e-2
        # Second order cuts the face reflection by >= 2x at normal and oblique incidence.
        assert r2 < 0.5 * r1, (name, r1, r2)


_GUIDE_FREQS = (1.6e9, 2.0e9, 2.4e9)
_GUIDE_DX = 0.0125
_GUIDE_X_TEST = 0.205
_GUIDE_X_REF = 2.005


def _guide_scene(boundary, x_high, absorbers=()):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.32, x_high), (-0.32, 0.32), (-0.32, 0.32))),
        grid=mw.GridSpec.uniform(_GUIDE_DX),
        boundary=boundary,
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized"),
        device="cuda",
        internal_absorbers=absorbers,
    )
    scene.add_structure(
        mw.Structure(geometry=mw.Box(position=(0, 0, 0), size=(20.0, 0.20, 0.30)), material=mw.Material(eps_r=4.0))
    )
    scene.add_source(
        mw.ModeSource(
            position=(-0.25, 0, 0),
            size=(0, 0.5, 0.5),
            polarization="Ez",
            source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=0.5e9),
        )
    )
    scene.add_monitor(
        mw.ModeMonitor("m", position=(0.0, 0, 0), size=(0, 0.5, 0.5), polarization="Ez", frequencies=_GUIDE_FREQS)
    )
    return scene


def _guide_amplitudes(scene):
    from witwin.maxwell.postprocess import compute_mode_overlap

    result = mw.Simulation.fdtd(
        scene,
        frequencies=list(_GUIDE_FREQS),
        run_time=mw.TimeConfig(time_steps=1300),
        spectral_sampler=mw.SpectralSampler(normalize_source=True),
        full_field_dft=False,
    ).run()
    forward, backward = [], []
    for index in range(len(_GUIDE_FREQS)):
        overlap = compute_mode_overlap(result, "m", direction="+", freq_index=index)
        forward.append(complex(overlap["amplitude_forward"].detach().cpu().item()))
        backward.append(complex(overlap["amplitude_backward"].detach().cpu().item()))
    return np.asarray(forward), np.asarray(backward), result.solver


def _pml_faces(**kwargs):
    return mw.BoundarySpec.faces(default="pml", num_layers=12, **kwargs)


@cuda
def test_mode_abc_and_internal_absorber_terminate_guide_like_cpml():
    a_ref, b_ref, _ = _guide_amplitudes(_guide_scene(_pml_faces(), _GUIDE_X_REF))

    def reflection(scene):
        _, backward, solver = _guide_amplitudes(scene)
        return np.abs(backward - b_ref) / np.abs(a_ref), solver

    cpml, _ = reflection(_guide_scene(_pml_faces(), _GUIDE_X_TEST))
    single_spec = mw.ModeABC(polarization="Ez", frequency=2.0e9)
    single, solver = reflection(_guide_scene(_pml_faces(x_high="mode_abc", mode_abc=single_spec), _GUIDE_X_TEST))
    info = solver._mode_abc_info[0]
    assert info["face"] == ("x", "high") and len(info["coefficients"]) == 1
    assert 1.9 < info["effective_index"] < 2.2
    broadband_spec = mw.ModeABC(polarization="Ez", frequency_range=(1.5e9, 2.5e9), order=2)
    broadband, solver = reflection(
        _guide_scene(_pml_faces(x_high="mode_abc", mode_abc=broadband_spec), _GUIDE_X_TEST)
    )
    assert solver._abc_higdon_state and solver._mode_abc_info[0]["order"] == 2
    internal, solver = reflection(
        _guide_scene(
            _pml_faces(),
            _GUIDE_X_REF,
            (mw.InternalAbsorber((_GUIDE_X_TEST, 0, 0), _PLANE, "+", boundary=single_spec),),
        )
    )
    plane_index = solver._internal_absorber_info[0]["plane_index"]
    assert float(solver.scene.x_nodes64[plane_index]) == pytest.approx(_GUIDE_X_TEST, abs=1e-9)

    assert np.max(cpml) < 1e-3
    # Single-frequency ModeABC: CPML-level at its design frequency only.
    assert single[1] < 1e-3
    assert np.min(single[[0, 2]]) > 1e-2
    # The order-2 Higdon product holds ~1e-3 across the band (30x below the
    # single-frequency band edges).
    assert np.max(broadband) < 3e-3
    assert np.max(broadband) < 0.1 * np.min(single[[0, 2]])
    # An internal plane absorbs exactly like the matching outer face: nothing
    # behind it reaches the upstream region.
    np.testing.assert_allclose(internal, single, rtol=2e-2, atol=2e-5)


@cuda
def test_extended_one_way_boundaries_reject_adjoint():
    density = torch.full((1, 1, 1), 0.5, device="cuda", dtype=torch.float64, requires_grad=True)
    scene = mw.Scene(
        subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
        domain=mw.Domain(bounds=((-0.45, 0.45),) * 3),
        grid=mw.GridSpec.uniform(0.15),
        boundary=mw.BoundarySpec.mur(order=2),
        device="cuda",
    )
    scene.add_material_region(
        mw.MaterialRegion(name="design", geometry=mw.Box(position=(0, 0, 0), size=(0.15,) * 3),
                          density=density, eps_bounds=(1.0, 4.0))
    )
    scene.add_source(mw.PointDipole(position=(-0.15, 0, 0), polarization="Ez", width=0.12,
                                    source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9)))
    scene.add_monitor(mw.PointMonitor("probe", (0.15, 0.0, 0.0), fields=("Ez",)))
    simulation = mw.Simulation.fdtd(scene, frequencies=[1e9], run_time=mw.TimeConfig(time_steps=40),
                                    spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False)
    with pytest.raises(NotImplementedError, match="second-order Mur"):
        result = simulation.run()
        result.monitor("probe")["data"].abs().square().sum().backward()


def _small_guide_scene(boundary, *, material=None, absorbers=(), design=None):
    scene = mw.Scene(
        subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=boundary,
        device="cuda",
        internal_absorbers=absorbers,
    )
    scene.add_structure(
        mw.Structure(geometry=mw.Box(position=(0, 0, 0), size=(2.0, 0.12, 0.16)),
                     material=material or mw.Material(eps_r=4.0))
    )
    if design is not None:
        scene.add_material_region(design)
    scene.add_source(mw.PointDipole(position=(-0.1, 0, 0), polarization="Ez", width=0.02,
                                    source_time=mw.GaussianPulse(frequency=2e9, fwidth=0.5e9)))
    return scene


def _prepare(scene):
    return mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=4)).prepare()


@cuda
def test_mode_abc_rejects_lossy_mode():
    boundary = mw.BoundarySpec.faces(default="pml", num_layers=6, x_high="mode_abc",
                                     mode_abc=mw.ModeABC(polarization="Ez", frequency=2e9))
    with pytest.raises(NotImplementedError, match="lossless guided mode"):
        _prepare(_small_guide_scene(boundary, material=mw.Material(eps_r=4.0, sigma_e=0.5)))


@cuda
def test_mode_abc_rejects_trainable_material_on_its_cross_section():
    density = torch.full((2, 2, 2), 0.5, device="cuda", dtype=torch.float64, requires_grad=True)
    design = mw.MaterialRegion(name="design", geometry=mw.Box(position=(0.17, 0, 0), size=(0.04, 0.04, 0.04)),
                               density=density, eps_bounds=(1.0, 4.0))
    boundary = mw.BoundarySpec.faces(default="pml", num_layers=6, x_high="mode_abc",
                                     mode_abc=mw.ModeABC(polarization="Ez", frequency=2e9))
    with pytest.raises(NotImplementedError, match="trainable materials"):
        _prepare(_small_guide_scene(boundary, design=design))


@cuda
def test_internal_absorber_rejects_finite_aperture():
    absorber = mw.InternalAbsorber((0.1, 0, 0), (0.0, 0.2, _INF), "+", name="partial")
    with pytest.raises(NotImplementedError, match="full domain cross-section"):
        _prepare(_small_guide_scene(mw.BoundarySpec.pml(num_layers=6), absorbers=(absorber,)))


@cuda
def test_internal_absorber_and_mode_faces_join_the_one_way_pass():
    absorber = mw.InternalAbsorber((0.1, 0, 0), _PLANE, "+", boundary=mw.MurABC(permittivity=3.0), grid_shift=-1)
    boundary = mw.BoundarySpec.faces(default="pml", num_layers=6, x_low="mode_abc",
                                     mode_abc=mw.ModeABC(polarization="Ez", frequency=2e9))
    solver = _prepare(_small_guide_scene(boundary, absorbers=(absorber,))).solver
    assert solver.has_mur_faces and solver.mode_abc_faces == (("x", "low"),)
    sources = [entry["source"] for entry in solver._mur_state]
    assert sources == [("mode_abc", "x", "low")] * 2 + [("internal_absorber", 0)] * 2
    plane = solver._internal_absorber_info[0]["plane_index"]
    assert plane == int(round((0.1 + 0.2) / 0.02)) - 1
    internal = [entry for entry in solver._mur_state if entry["source"][0] == "internal_absorber"]
    assert {entry["field"] for entry in internal} == {"Ey", "Ez"}
    assert all(entry["boundary_index"] == plane and entry["adjacent_index"] == plane - 1 for entry in internal)
    speed_dt = 299792458.0 * float(solver.dt) / math.sqrt(3.0)
    delta = float(solver.scene.dx_primal64[plane - 1])
    assert internal[0]["coef"] == pytest.approx((speed_dt - delta) / (speed_dt + delta))


@cuda
def test_extended_one_way_histories_survive_disk_resume(tmp_path):
    boundary = mw.BoundarySpec.faces(
        default="mur",
        mur_abc=mw.MurABC(order=2),
        x_high="mode_abc",
        mode_abc=mw.ModeABC(polarization="Ez", frequency_range=(1.4e9, 2.0e9), order=2),
    )
    absorber = mw.InternalAbsorber((-0.12, 0, 0), _PLANE, "-")
    scene = _small_guide_scene(boundary, absorbers=(absorber,))
    scene.add_monitor(mw.PointMonitor("probe", (0.06, 0.0, 0.0), fields=("Ez",)))

    def simulation():
        return mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=300),
                                  spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False)

    expected = simulation().prepare()
    expected.run()
    sources = {entry["source"][0] for entry in expected.solver._abc_higdon_state}
    assert sources == {"mur", "mode_abc"} and expected.solver._mur_state
    checkpoint = simulation().prepare().run_until(131)
    path = tmp_path / "abc.pt"
    checkpoint.save(path)
    restored = mw.FDTDResumeCheckpoint.load(path, map_location="cpu")
    actual = simulation().prepare()
    actual.simulation._execute_fdtd_solve(actual.solver, resume_from=restored)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert float(getattr(expected.solver, name).abs().max()) > 0.0
        torch.testing.assert_close(getattr(actual.solver, name), getattr(expected.solver, name), rtol=0, atol=0)


def _field_energy(solver):
    electric = sum(float((getattr(solver, name).double() ** 2).sum()) for name in ("Ex", "Ey", "Ez"))
    magnetic = sum(float((getattr(solver, name).double() ** 2).sum()) for name in ("Hx", "Hy", "Hz"))
    return solver.eps0 * electric + solver.mu0 * magnetic


@cuda
@pytest.mark.parametrize(
    "boundary",
    [
        mw.BoundarySpec.faces(default="pml", num_layers=8, x="mur", mur_abc=mw.MurABC(order=2)),
        mw.BoundarySpec.faces(default="pml", num_layers=8, x="mode_abc",
                              mode_abc=mw.ModeABC(polarization="Ez", frequency_range=(1.4e9, 2.0e9), order=2)),
        mw.BoundarySpec.mur(order=2),
    ],
    ids=["mur2_opposite_faces", "broadband_mode_abc_opposite_faces", "mur2_closed_box"],
)
def test_opposite_higdon_faces_on_dielectric_guide_stay_passive(boundary):
    energies = []
    for steps in (1000, 9000):
        scene = _small_guide_scene(boundary)
        scene.add_monitor(mw.PointMonitor("probe", (0.06, 0.0, 0.0), fields=("Ez",)))
        prepared = mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=steps),
                                      spectral_sampler=mw.SpectralSampler(window="none"),
                                      full_field_dft=False).prepare()
        prepared.run()
        assert prepared.solver._abc_higdon_state
        energies.append(_field_energy(prepared.solver))
    assert all(math.isfinite(value) for value in energies)
    assert energies[1] < 0.1 * energies[0]
