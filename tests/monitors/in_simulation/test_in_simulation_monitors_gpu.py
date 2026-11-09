"""Small FDTD physics checks of the in-simulation monitors (CUDA).

Tolerances are stated per test; every scene stays below ~0.2 GB of device memory.
"""

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0, EPSILON_0, MU_0
from witwin.maxwell.postprocess import FieldProjector

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")

_F = 1.0e9


def _dipole_scene(spacing=0.04):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64),) * 3),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0), polarization="Ez", width=0.05,
            source_time=mw.GaussianPulse(frequency=_F, fwidth=0.5e9), name="src",
        )
    )
    return scene


def _run(scene, frequencies=(_F,), **kwargs):
    return mw.Simulation.fdtd(
        scene,
        frequencies=list(frequencies),
        run_time=kwargs.pop("run_time", mw.TimeConfig.auto(steady_cycles=4, transient_cycles=10)),
        full_field_dft=False,
        **kwargs,
    ).run()


# ---------------------------------------------------------------------------
# Projection monitors
# ---------------------------------------------------------------------------


def test_projection_monitors_reproduce_field_projector_on_the_same_run():
    """Exact (torch.equal) agreement with FieldProjector on identical surface DFTs."""
    theta = torch.linspace(0.2, 2.9, 9).tolist()
    phi = [0.0, 1.0]
    scene = _dipole_scene()
    scene.add_monitor(mw.AngleProjectionMonitor("angle", size=(0.6, 0.6, 0.6), theta=theta, phi=phi, radius=5.0))
    scene.add_monitor(mw.CartesianProjectionMonitor(
        "cart", size=(0.6, 0.6, 0.6), x=[-0.5, 0.0, 0.5], y=[0.0, 0.2], axis="x", distance=2.0))
    scene.add_monitor(mw.KSpaceProjectionMonitor(
        "ks", size=(0.6, 0.6, 0.6), ux=[-0.5, 0.0, 0.5, 0.95], uy=[0.0, 0.9], axis="x"))
    scene.add_monitor(mw.ClosedSurfaceMonitor.box("huygens", size=(0.6, 0.6, 0.6)))
    scene.add_monitor(mw.AngleProjectionMonitor(
        "planar", position=(0.0, 0.0, 0.3), size=(0.6, 0.6, 0.0), theta=[0.2, 0.6], phi=[0.0], radius=5.0,
        origin=(0.0, 0.0, 0.0)))
    scene.add_monitor(mw.FinitePlaneMonitor(
        "planar_ref", position=(0.0, 0.0, 0.3), size=(0.6, 0.6, 0.0), fields=("Ex", "Ey", "Hx", "Hy")))
    scene.add_monitor(mw.AuxFieldTimeMonitor("aux_vacuum", components=("Pz",), position=(0.1, 0.0, 0.0)))
    result = _run(scene, frequencies=(_F, 1.2e9))

    projector = FieldProjector.from_monitor(result, "huygens")
    angle = result.monitor("angle")
    expected = projector.angles(theta, phi, radius=5.0)
    assert angle.E.shape == (2, 9, 2, 3) and angle.E.is_cuda
    assert torch.equal(angle.E, expected.E) and torch.equal(angle.H, expected.H)
    assert torch.equal(result.monitor("cart").E, projector.cartesian([-0.5, 0.0, 0.5], [0.0, 0.2], axis="x", distance=2.0).E)
    ks = result.monitor("ks")
    expected_ks = projector.kspace([-0.5, 0.0, 0.5, 0.95], [0.0, 0.9], axis="x")
    assert torch.equal(ks.E, expected_ks.E) and torch.equal(ks.mask, expected_ks.mask)
    assert torch.equal(result.monitor("angle", freq_index=1).E[0], angle.E[1])

    # A planar surface radiates through the single-face Huygens path; in vacuum
    # it equals the plain plane-monitor projection to float tolerance.
    planar = result.monitor("planar")
    planar_ref = FieldProjector.from_monitor(result, "planar_ref").angles([0.2, 0.6], [0.0], radius=5.0)
    torch.testing.assert_close(planar.E, planar_ref.E, rtol=1e-5, atol=1e-6 * float(planar_ref.E.abs().max()))

    # Physics: a short z dipole radiates |E_theta| ~ sin(theta); tolerance 7% on
    # this coarse (lambda/7.5) staircased Huygens box.
    pattern = torch.abs(angle.E_theta[0, :, 0]).cpu()
    reference = torch.sin(torch.tensor(theta))
    ratio = pattern / reference
    assert float(torch.max(torch.abs(ratio / ratio.mean() - 1.0))) < 0.07
    assert float(torch.max(torch.abs(angle.E_phi[0]))) < 0.02 * float(pattern.max())

    aux = result.monitor("aux_vacuum")
    assert aux["present"] == {"Pz": False}
    assert float(torch.max(torch.abs(aux["data"]))) == 0.0


# ---------------------------------------------------------------------------
# Gaussian-overlap monitors
# ---------------------------------------------------------------------------


def _gaussian_parameter(waist, distance, wavelength):
    """Complex transverse exponent a = 1/w(z)^2 - i k / (2 R(z)) of exp(-a r^2)."""
    k = 2.0 * math.pi / wavelength
    z_rayleigh = math.pi * waist**2 / wavelength
    width = waist * math.sqrt(1.0 + (distance / z_rayleigh) ** 2)
    curvature = math.inf if distance == 0 else distance * (1.0 + (z_rayleigh / distance) ** 2)
    return complex(1.0 / width**2, -k / (2.0 * curvature))


def _gaussian_power_overlap(a_u1, a_v1, a_u2, a_v2):
    def one_axis(a1, a2):
        return 2.0 * math.sqrt(a1.real * a2.real) / abs(a1 + a2.conjugate())

    return one_axis(a_u1, a_u2) * one_axis(a_v1, a_v2)


def _beam_scene(source):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.9, 0.9), (-0.9, 0.9), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.03),
        boundary=mw.BoundarySpec.pml(num_layers=10, strength=1.0),
        device="cuda",
    )
    scene.add_source(source)
    return scene


def _beam_run(scene):
    return _run(
        scene,
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="none"),
    )


def test_gaussian_overlap_decomposes_an_injected_gaussian_beam():
    """Matching profile ~1 (>= 0.985), orthogonal/shifted ~0, mismatched waist = analytic (+/-0.02)."""
    waist, wavelength = 0.3, C_0 / _F
    focus = (0.0, 0.0, -0.3)
    scene = _beam_scene(mw.GaussianBeam(
        direction=(0, 0, 1), polarization=(1, 0, 0), beam_waist=waist, focus=focus,
        source_time=mw.CW(frequency=_F, amplitude=1.0), name="beam"))
    plane = dict(position=(0.0, 0.0, 0.35), size=(1.6, 1.6, 0.0))
    scene.add_monitor(mw.GaussianOverlapMonitor("match", **plane, beam_waist=waist, focus=focus))
    scene.add_monitor(mw.GaussianOverlapMonitor("orthogonal", **plane, beam_waist=waist, focus=focus, polarization=(0, 1, 0)))
    scene.add_monitor(mw.GaussianOverlapMonitor("shifted", **plane, beam_waist=waist, focus=(2.5 * waist, 0.0, -0.3)))
    scene.add_monitor(mw.GaussianOverlapMonitor("wide", **plane, beam_waist=1.6 * waist, focus=focus))
    scene.add_monitor(mw.AstigmaticGaussianOverlapMonitor("round_astigmatic", **plane, beam_waist=(waist, waist), focus=focus))
    scene.add_monitor(mw.GaussianOverlapMonitor(
        "reverse", position=(0.0, 0.0, -0.3), size=(1.6, 1.6, 0.0), beam_waist=waist, focus=focus, direction=(0, 0, -1)))
    result = _beam_run(scene)

    match = result.monitor("match")
    assert match["amplitude_forward"].shape == (1,) and match["amplitude_forward"].is_complex()
    assert float(match["power_fraction_forward"][0]) >= 0.985
    assert float(match["power_fraction_backward"][0]) < 5e-3
    torch.testing.assert_close(match["power_forward"], torch.abs(match["amplitude_forward"]) ** 2)
    assert float(result.monitor("orthogonal")["power_fraction_forward"][0]) < 1e-6
    assert float(result.monitor("shifted")["power_fraction_forward"][0]) < 1e-2

    distance = 0.35 - focus[2]
    a_beam = _gaussian_parameter(waist, distance, wavelength)
    a_wide = _gaussian_parameter(1.6 * waist, distance, wavelength)
    expected = _gaussian_power_overlap(a_beam, a_beam, a_wide, a_wide)
    assert float(result.monitor("wide")["power_fraction_forward"][0]) == pytest.approx(expected, abs=0.02)

    round_astigmatic = result.monitor("round_astigmatic")
    torch.testing.assert_close(round_astigmatic["amplitude_forward"], match["amplitude_forward"])

    # At the waist plane a -z-declared monitor sees the +z beam as its backward wave.
    reverse = result.monitor("reverse")
    assert float(reverse["power_fraction_backward"][0]) >= 0.985
    assert float(reverse["power_fraction_forward"][0]) < 5e-3


def test_astigmatic_gaussian_overlap_matches_an_astigmatic_beam():
    """Matching astigmatic profile >= 0.98; a round profile recovers the analytic mismatch (+/-0.02)."""
    waists, wavelength = (0.3, 0.45), C_0 / _F
    focus = (0.0, 0.0, -0.2)
    scene = _beam_scene(mw.AstigmaticGaussianBeam(
        direction=(0, 0, 1), polarization=(1, 0, 0), beam_waist=waists, focus=focus, focus_v=0.2,
        source_time=mw.CW(frequency=_F, amplitude=1.0), name="beam"))
    plane = dict(position=(0.0, 0.0, 0.3), size=(1.6, 1.6, 0.0))
    scene.add_monitor(mw.AstigmaticGaussianOverlapMonitor("match", **plane, beam_waist=waists, focus=focus, focus_v=0.2))
    scene.add_monitor(mw.GaussianOverlapMonitor("round", **plane, beam_waist=waists[0], focus=focus))
    result = _beam_run(scene)
    assert float(result.monitor("match")["power_fraction_forward"][0]) >= 0.98
    a_u = _gaussian_parameter(waists[0], 0.5, wavelength)
    a_v = _gaussian_parameter(waists[1], 0.3, wavelength)
    expected = _gaussian_power_overlap(a_u, a_v, a_u, a_u)
    assert float(result.monitor("round")["power_fraction_forward"][0]) == pytest.approx(expected, abs=0.02)


# ---------------------------------------------------------------------------
# Surface field-time monitor
# ---------------------------------------------------------------------------


def test_surface_field_time_faces_equal_full_plane_field_time_traces():
    """Each face is the matching sub-block of a native full-plane FieldTimeMonitor (torch.equal)."""
    from witwin.maxwell.fdtd.observers import get_component_coords

    scene = _dipole_scene(spacing=0.08)
    start, stop, interval = 7, 120, 3
    # Faces at +/-0.26 avoid equidistant Yee planes, so the nearest native plane is unique.
    scene.add_monitor(mw.SurfaceFieldTimeMonitor(
        "surface", size=(0.52, 0.52, 0.52), components=("Ez", "Hx"), start=start, stop=stop, interval=interval))
    scene.add_monitor(mw.FieldTimeMonitor(
        "zplane", components=("Ez", "Hx"), position=(0.0, 0.0, 0.26), size=(1.28, 1.28, 0.0),
        start=start, stop=stop, interval=interval))
    scene.add_monitor(mw.FieldTimeMonitor(
        "point", components=("Ez",), position=(0.26, 0.0, 0.0), start=start, stop=stop, interval=interval))
    result = _run(scene, run_time=mw.TimeConfig(time_steps=160))

    surface = result.monitor("surface")
    assert surface["kind"] == "surface_field_time" and surface["differentiable"] is False
    steps = torch.round(surface["t"] / result.solver.dt).to(torch.int64).cpu()
    # A sample taken after update n carries the physical post-update label (n + 1) * dt.
    assert steps.tolist() == [n + 1 for n in range(start, stop, interval)]
    assert surface["face_labels"] == ("x_neg", "x_pos", "y_neg", "y_pos", "z_neg", "z_pos")

    plane = result.monitor("zplane")
    face = surface["faces"]["z_pos"]
    for component in ("Ez", "Hx"):
        x_coords, y_coords, _ = (np.asarray(value) for value in get_component_coords(result.prepared_scene, component))
        face_x, face_y = (np.asarray(value) for value in face["coords"][component])
        ix = np.nonzero(np.isclose(x_coords[:, None], face_x[None, :]).any(axis=1))[0]
        iy = np.nonzero(np.isclose(y_coords[:, None], face_y[None, :]).any(axis=1))[0]
        assert ix.size == face_x.size and iy.size == face_y.size
        expected = plane["components"][component][:, ix][:, :, iy]
        assert torch.equal(face["components"][component], expected)

    # A point on the x_pos face reproduces the point FieldTimeMonitor trace.
    face = surface["faces"]["x_pos"]
    face_y, face_z = (np.asarray(value) for value in face["coords"]["Ez"])
    iy = int(np.argmin(np.abs(face_y)))
    iz = int(np.argmin(np.abs(face_z)))
    assert torch.equal(face["components"]["Ez"][:, iy, iz], result.monitor("point")["data"])


# ---------------------------------------------------------------------------
# Auxiliary field-time monitor
# ---------------------------------------------------------------------------


def test_aux_field_time_records_native_ade_state():
    """ADE state obeys its discrete recurrence (rel 1e-5) and the continuous susceptibility (rel 5%)."""
    omega = 2.0 * math.pi * _F
    tau = 1.0 / omega
    debye = mw.DebyePole(delta_eps=2.0, tau=tau)
    drude = mw.DrudePole(plasma_frequency=0.5e9, gamma=1.0e9)
    magnetic = mw.DebyePole(delta_eps=0.5, tau=tau)
    material = mw.Material(eps_r=1.0, debye_poles=(debye,), drude_poles=(drude,), mu_debye_poles=(magnetic,))
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4),) * 3),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
    )
    scene.add_structure(mw.Structure(mw.Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), material, name="block"))
    scene.add_source(mw.PointDipole(position=(0, 0, 0), polarization="Ez", width=0.03,
                                    source_time=mw.CW(frequency=_F, amplitude=1.0), name="src"))
    box = dict(position=(0.06, 0.0, 0.0), size=(0.04, 0.04, 0.04))
    with pytest.warns(UserWarning, match="volume time series"):
        scene.add_monitor(mw.FieldTimeMonitor("fields", components=("Ez", "Hy"), **box))
        scene.add_monitor(mw.AuxFieldTimeMonitor("aux", components=("Pz", "Jpz", "My"), **box))
        result = _run(scene, run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=10))

    fields, aux = result.monitor("fields"), result.monitor("aux")
    assert aux["present"] == {"Pz": True, "Jpz": True, "My": True}
    assert aux["components"]["Pz"].shape == fields["components"]["Ez"].shape
    t = fields["t"].cpu().numpy()
    dt = float(result.solver.dt)
    window = int(round(4.0 / _F / dt))

    def spectrum(trace):
        values = trace.detach().cpu().numpy()[-window:]
        return np.tensordot(np.exp(1j * omega * t[-window:]), values, axes=(0, 0))

    e, h = spectrum(fields["components"]["Ez"]), spectrum(fields["components"]["Hy"])
    p, j, m = (spectrum(aux["components"][name]) for name in ("Pz", "Jpz", "My"))
    e_mask, h_mask = np.abs(e) > 0.2 * np.abs(e).max(), np.abs(h) > 0.2 * np.abs(h).max()

    # Discrete ADE: X_n = d X_{n-1} + b F_{n-1} with the sample delay z = exp(i w dt).
    z = np.exp(1j * omega * dt)
    decay = (2 * tau - dt) / (2 * tau + dt)
    p_discrete = 2 * EPSILON_0 * debye.delta_eps * dt / (2 * tau + dt) * z / (1 - decay * z)
    wp, gamma = 2 * math.pi * drude.plasma_frequency, 2 * math.pi * drude.gamma
    drude_decay = (2 - gamma * dt) / (2 + gamma * dt)
    j_discrete = p_discrete * (1 - z) / dt + 2 * EPSILON_0 * wp * wp * dt / (2 + gamma * dt) * z / (1 - drude_decay * z)
    m_discrete = 2 * MU_0 * magnetic.delta_eps * dt / (2 * tau + dt) * z / (1 - decay * z)
    for ratio, expected, mask in (
        (p / e, p_discrete, e_mask),
        (j / e, j_discrete, e_mask),
        (m / h, m_discrete, h_mask),
    ):
        assert float(np.max(np.abs(ratio[mask] / expected - 1.0))) < 1e-5

    # Continuous constitutive relations, up to the O((w dt)^2) ADE discretization.
    for ratio, expected, mask in (
        (p / e, EPSILON_0 * debye.susceptibility(omega), e_mask),
        (j / e, -1j * omega * EPSILON_0 * (debye.susceptibility(omega) + drude.susceptibility(omega)), e_mask),
        (m / h, MU_0 * magnetic.susceptibility(omega), h_mask),
    ):
        assert float(np.max(np.abs(ratio[mask] / expected - 1.0))) < 0.05


# ---------------------------------------------------------------------------
# Differentiable runs
# ---------------------------------------------------------------------------


class _DesignScene(mw.SceneModule):
    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def to_scene(self):
        scene = mw.Scene(
            subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
            domain=mw.Domain(bounds=((-0.36, 0.36),) * 3),
            grid=mw.GridSpec.uniform(0.06),
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(mw.MaterialRegion(
            name="design", geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
            density=torch.sigmoid(self.logits), eps_bounds=(1.0, 6.0), mu_bounds=(1.0, 1.0)))
        scene.add_source(mw.PointDipole(
            position=(0.0, 0.0, -0.06), polarization="Ez", width=0.04,
            source_time=mw.GaussianPulse(frequency=_F, fwidth=0.25e9, amplitude=50.0)))
        scene.add_monitor(mw.AngleProjectionMonitor(
            "proj", size=(0.36, 0.36, 0.36), theta=[0.6, 1.2, 1.8], phi=[0.0, 0.8], radius=4.0))
        scene.add_monitor(mw.GaussianOverlapMonitor(
            "overlap", position=(0.0, 0.0, 0.18), size=(0.36, 0.36, 0.0), beam_waist=0.12, polarization=(0, 1, 0),
            focus=(0.06, 0.06, 0.18)))  # off-axis: a centred profile is orthogonal by symmetry
        scene.add_monitor(mw.AuxFieldTimeMonitor("aux", components=("Pz",), position=(0.0, 0.0, 0.06)))
        scene.add_monitor(mw.SurfaceFieldTimeMonitor("trace", size=(0.24, 0.24, 0.0), components=("Ez",), interval=4))
        return scene


def _objectives(model):
    result = mw.Simulation.fdtd(
        model, frequencies=[_F], run_time=mw.TimeConfig(time_steps=60),
        spectral_sampler=mw.SpectralSampler(window="none"), full_field_dft=False,
    ).run()
    projection = result.monitor("proj").E_intensity.sum()
    overlap = torch.abs(result.monitor("overlap")["amplitude_forward"]).square().sum()
    return result, projection, overlap


def test_projection_and_overlap_objectives_are_differentiable_and_time_traces_reject_gradients():
    """Adjoint gradients match central differences (rel 1%, measured ~3e-4); time traces refuse backward."""
    model = _DesignScene().cuda()
    result, projection, overlap = _objectives(model)
    (gradient_projection,) = torch.autograd.grad(projection, model.logits, retain_graph=True)
    (gradient_overlap,) = torch.autograd.grad(overlap, model.logits, retain_graph=True)

    step = 0.05
    saved = model.logits.detach().clone()
    model.requires_grad_(False)
    with torch.no_grad():
        model.logits.copy_(saved + step)
    _, plus_projection, plus_overlap = _objectives(model)
    with torch.no_grad():
        model.logits.copy_(saved - step)
    _, minus_projection, minus_overlap = _objectives(model)
    with torch.no_grad():
        model.logits.copy_(saved)
    model.requires_grad_(True)
    fd_projection = float(plus_projection - minus_projection) / (2 * step)
    fd_overlap = float(plus_overlap - minus_overlap) / (2 * step)
    assert abs(fd_projection) > 0.0 and abs(fd_overlap) > 0.0
    assert float(gradient_projection) == pytest.approx(fd_projection, rel=1e-2)
    assert float(gradient_overlap) == pytest.approx(fd_overlap, rel=1e-2)

    aux = result.monitor("aux")
    trace = result.monitor("trace")
    assert aux["differentiable"] is False and aux["data"].requires_grad
    with pytest.raises(RuntimeError, match="detached time-domain traces"):
        (aux["data"].square().sum() + projection).backward()
    with pytest.raises(RuntimeError, match="detached time-domain traces"):
        trace["faces"]["plane"]["components"]["Ez"].sum().backward()
