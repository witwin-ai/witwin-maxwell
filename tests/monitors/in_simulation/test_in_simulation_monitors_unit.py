"""CPU contracts of the in-simulation projection, Gaussian-overlap and time-trace monitors."""

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.monitors import compile_fdtd_observers, compile_fdtd_time_observers
from witwin.maxwell.postprocess import FieldProjector
from witwin.maxwell.postprocess.gaussian_overlap import compute_gaussian_overlap


def _scene(device="cpu"):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        device=device,
    )


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def test_projection_monitors_are_public_and_resolve_to_named_faces():
    monitor = mw.AngleProjectionMonitor(
        "proj", size=(0.4, 0.4, 0.4), theta=np.linspace(0.1, 3.0, 5), phi=(0.0, 1.0),
        frequencies=(1e9, 2e9), exclude_surfaces=("z_neg",),
    )
    assert monitor.surface_kind == "box"
    assert monitor.face_monitor_names == tuple(
        f"proj::{label}" for label in ("x_neg", "x_pos", "y_neg", "y_pos", "z_pos")
    )
    assert all(face.surface_name == "proj" and face.frequencies == (1e9, 2e9) for face in monitor.faces)
    assert monitor.origin == (0.0, 0.0, 0.0)
    assert monitor.theta == pytest.approx(tuple(np.linspace(0.1, 3.0, 5)))
    scene = _scene().add_monitor(monitor)
    assert tuple(face.name for face in scene.resolved_monitors()) == monitor.face_monitor_names

    planar = mw.CartesianProjectionMonitor(
        "cart", position=(0.0, 0.0, 0.2), size=(0.4, 0.4, 0.0), x=(-1.0, 1.0), y=(0.0,),
        distance=3.0, normal_direction="-", window_size=(0.2, 0.1), origin=(0.0, 0.0, 0.0),
    )
    assert planar.surface_kind == "planar"
    assert planar.faces[0].normal_direction == "-" and planar.faces[0].name == "cart::plane"
    assert planar.origin == (0.0, 0.0, 0.0)

    kspace = mw.KSpaceProjectionMonitor("ks", size=(0.4, 0.0, 0.4), ux=(0.0,), uy=(0.0,), axis="y", normal_direction="-")
    assert kspace.hemisphere == "-"
    assert mw.KSpaceProjectionMonitor("ks2", size=(0.4, 0.4, 0.4), ux=(0.0,), uy=(0.0,)).hemisphere == "+"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(size=(0.4, 0.0, 0.0)), "plane patch"),
        (dict(size=(0.4, 0.4, 0.4), normal_direction="+"), "planar surfaces only"),
        (dict(size=(0.4, 0.4, 0.0), exclude_surfaces=("z_neg",)), "box surfaces only"),
        (dict(size=(0.4, 0.4, 0.4), exclude_surfaces=("top",)), "face labels"),
        (dict(size=(0.4, 0.4, 0.4), exclude_surfaces=("x_neg", "x_pos", "y_neg", "y_pos", "z_neg", "z_pos")), "at least one face"),
        (dict(size=(0.4, 0.4, 0.4), window_size=(0.1, 0.1)), "window_size"),
        (dict(size=(0.4, 0.4, 0.0), window_size=(1.5, 0.1)), r"\[0, 1\]"),
        (dict(size=(0.4, 0.4, 0.4), radius=0.0), "radius"),
        (dict(size=(0.4, 0.4, 0.4), theta=()), "theta must not be empty"),
    ],
)
def test_projection_monitor_rejects_invalid_declarations(kwargs, message):
    arguments = dict(theta=(0.1,), phi=(0.0,))
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        mw.AngleProjectionMonitor("bad", **arguments)


def test_projection_monitor_rejects_trainable_grids():
    with pytest.raises(NotImplementedError, match="Trainable"):
        mw.AngleProjectionMonitor("bad", size=(0.4, 0.4, 0.4), theta=torch.ones(2, requires_grad=True), phi=(0.0,))


def test_gaussian_overlap_monitors_mirror_beam_vocabulary():
    monitor = mw.GaussianOverlapMonitor("g", position=(0.0, 0.0, 0.3), size=(1.0, 1.0, 0.0), beam_waist=0.2)
    assert monitor.direction == (0.0, 0.0, 1.0) and monitor.normal_direction == "+"
    assert monitor.focus == (0.0, 0.0, 0.3)
    assert monitor.resolved_monitors()[0].name == "g"
    assert monitor.beam_waists == (0.2, 0.2) and monitor.focus_offsets == (0.0, 0.0)
    backward = mw.GaussianOverlapMonitor("b", size=(0.0, 1.0, 1.0), beam_waist=0.2, direction=(-1, 0, 0), polarization=(0, 0, 1))
    assert backward.axis == "x" and backward.normal_direction == "-"
    astig = mw.AstigmaticGaussianOverlapMonitor(
        "a", size=(1.0, 1.0, 0.0), beam_waist=(0.2, 0.3), focus=(0, 0, -0.1), focus_u=0.05, focus_v=-0.05
    )
    assert astig.beam_waists == (0.2, 0.3) and astig.focus_offsets == (0.05, -0.05)
    with pytest.raises(ValueError, match="orthogonal"):
        mw.GaussianOverlapMonitor("x", size=(1, 1, 0), beam_waist=0.2, polarization=(0, 0, 1))
    with pytest.raises(ValueError, match="cross the monitor plane"):
        mw.GaussianOverlapMonitor("x", size=(1, 1, 0), beam_waist=0.2, direction=(1, 0, 0), polarization=(0, 0, 1))
    with pytest.raises(ValueError, match="beam_waist"):
        mw.GaussianOverlapMonitor("x", size=(1, 1, 0), beam_waist=0.0)
    with pytest.raises(ValueError, match="zero-thickness"):
        mw.GaussianOverlapMonitor("x", size=(1, 0, 0), beam_waist=0.2)


def test_time_trace_monitors_validate_components_and_schedule():
    surface = mw.SurfaceFieldTimeMonitor("s", size=(0.4, 0.4, 0.4), components=("ex", "Hz"), start=3, stop=40, interval=4,
                                         exclude_surfaces=("z_neg",))
    assert surface.components == ("Ex", "Hz")
    assert surface.face_labels == ("x_neg", "x_pos", "y_neg", "y_pos", "z_pos")
    assert mw.SurfaceFieldTimeMonitor("p", size=(0.4, 0.0, 0.4)).face_labels == ("plane",)
    with pytest.raises(ValueError, match="stop must be greater"):
        mw.SurfaceFieldTimeMonitor("s", size=(0.4, 0.4, 0.4), start=5, stop=5)
    aux = mw.AuxFieldTimeMonitor("a", components=("px", "JPZ", "My"))
    assert aux.components == ("Px", "Jpz", "My") and aux.region_kind == "point"
    assert mw.AuxFieldTimeMonitor("v", size=(0.1, 0.1, 0.1)).region_kind == "volume"
    with pytest.raises(ValueError, match="components must be drawn"):
        mw.AuxFieldTimeMonitor("a", components=("Nfx",))
    with pytest.raises(ValueError, match="unique"):
        mw.AuxFieldTimeMonitor("a", components=("Px", "px"))


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def test_monitor_compiler_routes_new_monitors():
    scene = _scene()
    scene.add_monitor(mw.AngleProjectionMonitor("proj", size=(0.4, 0.4, 0.4), theta=(0.1,), phi=(0.0,)))
    scene.add_monitor(mw.GaussianOverlapMonitor("g", position=(0, 0, 0.2), size=(0.6, 0.6, 0.0), beam_waist=0.2))
    scene.add_monitor(mw.SurfaceFieldTimeMonitor("s", size=(0.4, 0.4, 0.0), components=("Ez",), interval=2))
    scene.add_monitor(mw.AuxFieldTimeMonitor("a", components=("Pz", "Jmx"), size=(0.2, 0.2, 0.2)))

    spectral = compile_fdtd_observers(scene)
    monitor_names = {record["monitor_name"] for record in spectral}
    assert monitor_names == {f"proj::{label}" for label in ("x_neg", "x_pos", "y_neg", "y_pos", "z_neg", "z_pos")} | {"g"}
    assert {record["component"] for record in spectral if record["monitor_name"] == "g"} == {"ex", "ey", "hx", "hy"}

    time_records = {record["name"]: record for record in compile_fdtd_time_observers(scene)}
    assert set(time_records) == {"s", "a"}
    assert time_records["s"]["kind"] == "surface_field_time"
    assert [face["label"] for face in time_records["s"]["faces"]] == ["plane"]
    assert time_records["a"]["kind"] == "aux_field_time"
    assert time_records["a"]["component_sources"] == {
        "Pz": ("electric", "Ez", "polarization"),
        "Jmx": ("magnetic", "Hx", "current"),
    }


def test_non_fdtd_simulations_reject_in_simulation_monitors():
    scene = _scene(device="cuda" if torch.cuda.is_available() else "cpu")
    scene.add_source(mw.PointDipole(position=(0, 0, 0), polarization="Ez", source_time=mw.CW(frequency=1e9)))
    scene.add_monitor(mw.AuxFieldTimeMonitor("a"))
    with pytest.raises(NotImplementedError, match="not supported by FDFD"):
        mw.Simulation.fdfd(scene, frequencies=[1e9])


# ---------------------------------------------------------------------------
# Result-time evaluation on synthetic surface data
# ---------------------------------------------------------------------------


def _synthetic_surface_payloads(faces, frequencies, coords):
    monitors = {}
    generator = torch.Generator().manual_seed(7)
    for face in faces:
        names = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[face.axis]
        payload = {
            "kind": "plane",
            "fields": face.fields,
            "components": {},
            "samples": 8,
            "frequencies": frequencies,
            "axis": face.axis,
            "position": face.plane_position,
            "compute_flux": False,
            "normal_direction": face.normal_direction,
        }
        for component in face.fields:
            data = torch.complex(
                torch.randn((len(frequencies), coords.numel(), coords.numel()), generator=generator, dtype=torch.float64),
                torch.randn((len(frequencies), coords.numel(), coords.numel()), generator=generator, dtype=torch.float64),
            )
            payload["components"][component] = {
                "data": data,
                "coords": (coords, coords),
                "plane_index": 0,
                "plane_indices": (0,),
                "plane_weights": (1.0,),
                "plane_positions": (face.plane_position,),
            }
            payload[component] = data
        payload[names[0]] = coords
        payload[names[1]] = coords
        payload["coords"] = (coords, coords)
        monitors[face.name] = payload
    return monitors


def test_projection_monitors_match_field_projector_on_identical_surface_data():
    frequencies = (1.0e9, 1.5e9)
    coords = torch.linspace(-0.2, 0.2, 5, dtype=torch.float64)
    theta, phi = (0.3, 1.2, 2.5), (0.0, 1.7)
    projections = (
        mw.AngleProjectionMonitor("angle", size=(0.4, 0.4, 0.4), theta=theta, phi=phi, radius=8.0, origin=(0.01, 0.0, 0.0)),
        mw.CartesianProjectionMonitor("cart", size=(0.4, 0.4, 0.4), x=(-0.3, 0.4), y=(0.0, 0.2), axis="y", distance=1.5),
        mw.KSpaceProjectionMonitor("ks", size=(0.4, 0.4, 0.4), ux=(-0.5, 0.0, 0.9), uy=(0.0, 0.6), axis="x"),
    )
    reference = mw.ClosedSurfaceMonitor.box("huygens", size=(0.4, 0.4, 0.4))
    scene = _scene()
    monitors = {}
    for monitor in (*projections, reference):
        scene.add_monitor(monitor)
    reference_payloads = _synthetic_surface_payloads(reference.faces, frequencies, coords)
    monitors.update(reference_payloads)
    for monitor in projections:
        for face, reference_face in zip(monitor.faces, reference.faces):
            monitors[face.name] = reference_payloads[reference_face.name]
    result = mw.Result(method="fdtd", scene=scene, frequencies=frequencies, monitors=monitors)

    projector = FieldProjector.from_monitor(result, "huygens", origin=(0.01, 0.0, 0.0), device="cpu")
    angle = result.monitor("angle")
    expected = projector.angles(theta, phi, radius=8.0)
    assert angle.kind == "angle" and angle.E.shape == (2, 3, 2, 3)
    assert torch.equal(angle.E, expected.E) and torch.equal(angle.H, expected.H)

    projector = FieldProjector.from_monitor(result, "huygens", device="cpu")
    cart = result.monitor("cart")
    expected = projector.cartesian((-0.3, 0.4), (0.0, 0.2), axis="y", distance=1.5)
    assert torch.equal(cart.E, expected.E) and torch.equal(cart.points, expected.points)
    ks = result.monitor("ks")
    expected = projector.kspace((-0.5, 0.0, 0.9), (0.0, 0.6), axis="x")
    assert torch.equal(ks.E, expected.E) and torch.equal(ks.mask, expected.mask)
    assert not bool(ks.mask[2, 1])

    single = result.monitor("angle", frequency=1.5e9)
    assert single.E.shape[0] == 1 and torch.equal(single.E[0], angle.E[1])
    raw = result.raw_monitor("angle", frequency=1.0e9)
    assert raw["kind"] == "closed_surface"
    assert set(raw["faces"]) == {face.face_label for face in reference.faces}


def test_gaussian_overlap_of_its_own_profile_is_unit_forward_power():
    """A plane carrying exactly the reference profile decomposes into a pure forward beam."""
    from witwin.maxwell.constants import C_0, EPSILON_0, MU_0
    from witwin.maxwell.fdtd.excitation.spatial import _beam_profile_core

    frequency = 1.0e9
    eta = math.sqrt(MU_0 / EPSILON_0)
    coords = torch.linspace(-0.6, 0.6, 61, dtype=torch.float64)
    monitor = mw.GaussianOverlapMonitor("g", position=(0, 0, 0.25), size=(1.2, 1.2, 0.0), beam_waist=0.15, focus=(0, 0, 0))
    shifted = mw.GaussianOverlapMonitor("shift", position=(0, 0, 0.25), size=(1.2, 1.2, 0.0), beam_waist=0.15, focus=(0.3, 0, 0))
    crossed = mw.GaussianOverlapMonitor("cross", position=(0, 0, 0.25), size=(1.2, 1.2, 0.0), beam_waist=0.15,
                                        focus=(0, 0, 0), polarization=(0, 1, 0))
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack((xx, yy, torch.full_like(xx, 0.25)), dim=-1)
    amplitude, delay = _beam_profile_core(
        points, direction=(0, 0, 1), polarization=(1, 0, 0), beam_waist_u=0.15, beam_waist_v=0.15,
        focus=(0, 0, 0), focus_offset_u=0.0, focus_offset_v=0.0, frequency=frequency, propagation_speed=C_0,
    )
    phasor = amplitude * torch.exp(1j * 2 * math.pi * frequency * delay)
    payload = {
        "kind": "plane", "axis": "z", "position": 0.25, "frequency": frequency, "frequencies": (frequency,),
        "normal_direction": "+", "x": coords, "y": coords,
        "Ex": 2.0 * phasor, "Ey": torch.zeros_like(phasor), "Hx": torch.zeros_like(phasor), "Hy": 2.0 * phasor / eta,
    }

    class _Result:
        frequencies = (frequency,)
        prepared_scene = None

        def raw_monitor(self, name, frequency=None, freq_index=None):
            return payload

    matched = compute_gaussian_overlap(_Result(), monitor)
    torch.testing.assert_close(matched["power_fraction_forward"], torch.ones(1, dtype=torch.float64), rtol=1e-9, atol=0)
    assert float(matched["power_fraction_backward"][0]) < 1e-20
    total = float(matched["total_power"][0])
    torch.testing.assert_close(matched["power_forward"], torch.tensor([total], dtype=torch.float64), rtol=1e-9, atol=0)
    off = compute_gaussian_overlap(_Result(), shifted)
    # Equal beams exp(-a x^2), a = 1/w^2 - i k/(2R), displaced by d overlap with
    # |c|^2 = exp(-d^2 |a|^2 / Re a) (the curvature mismatch adds a tilt).
    wavelength = C_0 / frequency
    k = 2 * math.pi / wavelength
    z_rayleigh = math.pi * 0.15**2 / wavelength
    waist_z = 0.15 * math.sqrt(1 + (0.25 / z_rayleigh) ** 2)
    curvature = 0.25 * (1 + (z_rayleigh / 0.25) ** 2)
    alpha, beta = 1 / waist_z**2, k / (2 * curvature)
    expected = math.exp(-(0.3**2) * (alpha**2 + beta**2) / alpha)
    assert float(off["power_fraction_forward"][0]) == pytest.approx(expected, rel=1e-2)
    assert float(compute_gaussian_overlap(_Result(), crossed)["power_fraction_forward"][0]) < 1e-20
