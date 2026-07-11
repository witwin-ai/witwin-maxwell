"""Tests for the Tidy3D adapter (maxwell/adapters/tidy3d.py).

These tests verify the conversion logic without requiring tidy3d to be installed.
They mock the tidy3d module to check that the correct Tidy3D constructors are
called with the right arguments.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import math
import sys
import types
from unittest.mock import MagicMock

import pytest


@contextlib.contextmanager
def _real_tidy3d():
    """Yield the genuine tidy3d module, bypassing the mock in sys.modules.

    The autouse ``inject_mock_tidy3d`` fixture replaces ``sys.modules['tidy3d']``
    for every test in this module. A handful of physical-convention checks need
    the real package, so temporarily drop the mock, import the real module, and
    restore the mock afterwards.
    """
    saved = sys.modules.pop("tidy3d", None)
    try:
        real_td = importlib.import_module("tidy3d")
        yield real_td
    finally:
        if saved is not None:
            sys.modules["tidy3d"] = saved
        else:
            sys.modules.pop("tidy3d", None)


# ---------------------------------------------------------------------------
# Build a mock tidy3d module so the adapter can import it.
# ---------------------------------------------------------------------------

def _make_mock_tidy3d():
    """Create a mock tidy3d module with the classes the adapter uses."""
    td = types.ModuleType("tidy3d")

    class _Recorder:
        """Stores kwargs for inspection."""
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __repr__(self):
            return f"{type(self).__name__}({self._kwargs})"

    class Medium(_Recorder): pass
    class PECMedium(_Recorder): pass
    class Drude(_Recorder): pass
    class Lorentz(_Recorder): pass
    class Debye(_Recorder): pass
    class PoleResidue(_Recorder): pass
    class Box(_Recorder): pass
    class Sphere(_Recorder): pass
    class Cylinder(_Recorder): pass
    class Structure(_Recorder): pass
    class PointDipole(_Recorder): pass
    class PlaneWave(_Recorder): pass
    class GaussianBeam(_Recorder): pass
    class FieldMonitor(_Recorder): pass
    class FluxMonitor(_Recorder): pass
    class ContinuousWave(_Recorder): pass
    class GaussianPulse(_Recorder): pass
    class PML(_Recorder): pass
    class Periodic(_Recorder): pass
    class PECBoundary(_Recorder): pass
    class PMCBoundary(_Recorder): pass
    class BlochBoundary(_Recorder): pass
    class Boundary(_Recorder): pass
    class UniformGrid(_Recorder): pass
    class Simulation(_Recorder): pass

    class BoundarySpec(_Recorder):
        @classmethod
        def all_sides(cls, boundary):
            return cls(boundary=boundary)

    class GridSpec(_Recorder):
        @classmethod
        def uniform(cls, dl):
            return cls(dl=dl)

    td.Medium = Medium
    td.PECMedium = PECMedium
    td.Drude = Drude
    td.Lorentz = Lorentz
    td.Debye = Debye
    td.PoleResidue = PoleResidue
    td.Box = Box
    td.Sphere = Sphere
    td.Cylinder = Cylinder
    td.Structure = Structure
    td.PointDipole = PointDipole
    td.PlaneWave = PlaneWave
    td.GaussianBeam = GaussianBeam
    td.FieldMonitor = FieldMonitor
    td.FluxMonitor = FluxMonitor
    td.ContinuousWave = ContinuousWave
    td.GaussianPulse = GaussianPulse
    td.PML = PML
    td.Periodic = Periodic
    td.PECBoundary = PECBoundary
    td.PMCBoundary = PMCBoundary
    td.BlochBoundary = BlochBoundary
    td.Boundary = Boundary
    td.BoundarySpec = BoundarySpec
    td.GridSpec = GridSpec
    td.UniformGrid = UniformGrid
    td.Simulation = Simulation
    td.inf = float("inf")
    return td


@pytest.fixture(autouse=True)
def inject_mock_tidy3d():
    """Inject mock tidy3d into sys.modules for every test."""
    mock_td = _make_mock_tidy3d()
    old = sys.modules.get("tidy3d")
    sys.modules["tidy3d"] = mock_td
    yield mock_td
    if old is None:
        sys.modules.pop("tidy3d", None)
    else:
        sys.modules["tidy3d"] = old


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import witwin.maxwell as mw
from witwin.maxwell.adapters.tidy3d import (
    _convert_boundary,
    _convert_geometry,
    _convert_material,
    _convert_monitor,
    _convert_source,
    _convert_source_time,
    _domain_to_center_size,
    scene_to_tidy3d,
)


class TestDomainConversion:
    def test_center_size(self):
        domain = mw.Domain(bounds=((-1, 1), (-2, 2), (-3, 3)))
        center, size = _domain_to_center_size(domain, 1.0)
        assert center == pytest.approx((0.0, 0.0, 0.0))
        assert size == pytest.approx((2.0, 4.0, 6.0))

    def test_offset_domain(self):
        domain = mw.Domain(bounds=((1, 3), (2, 6), (0, 10)))
        center, size = _domain_to_center_size(domain, 1.0)
        assert center == pytest.approx((2.0, 4.0, 5.0))
        assert size == pytest.approx((2.0, 4.0, 10.0))


class TestGridConversion:
    def test_uniform(self, inject_mock_tidy3d):
        from witwin.maxwell.adapters.tidy3d import _convert_grid
        td = inject_mock_tidy3d
        grid = mw.GridSpec.uniform(0.05)
        result = _convert_grid(grid, td, 1.0)
        assert result.dl == 0.05

    def test_anisotropic(self, inject_mock_tidy3d):
        from witwin.maxwell.adapters.tidy3d import _convert_grid
        td = inject_mock_tidy3d
        grid = mw.GridSpec.anisotropic(0.01, 0.02, 0.03)
        result = _convert_grid(grid, td, 1.0)
        assert hasattr(result, "grid_x")

    def test_custom_rejected(self, inject_mock_tidy3d):
        from witwin.maxwell.adapters.tidy3d import _convert_grid
        td = inject_mock_tidy3d
        nodes = [0.0, 0.1, 0.25, 0.45]
        grid = mw.GridSpec.custom(nodes, nodes, nodes)
        with pytest.raises(NotImplementedError, match="Tidy3D export does not support nonuniform"):
            _convert_grid(grid, td, 1.0)

    def test_auto_rejected(self, inject_mock_tidy3d):
        from witwin.maxwell.adapters.tidy3d import _convert_grid
        td = inject_mock_tidy3d
        grid = mw.GridSpec.auto(wavelength=0.3)
        with pytest.raises(NotImplementedError, match="GridSpec.auto"):
            _convert_grid(grid, td, 1.0)


class TestBoundaryConversion:
    def test_pml(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        boundary = mw.BoundarySpec.pml(num_layers=12)
        result = _convert_boundary(boundary, td)
        assert isinstance(result.boundary, td.PML)
        assert result.boundary.num_layers == 12

    def test_periodic(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        boundary = mw.BoundarySpec.periodic()
        result = _convert_boundary(boundary, td)
        assert isinstance(result.boundary, td.Periodic)

    def test_pec(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        boundary = mw.BoundarySpec.pec()
        result = _convert_boundary(boundary, td)
        assert isinstance(result.boundary, td.PECBoundary)

    def test_mixed_boundary(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        boundary = mw.BoundarySpec.faces(
            default="pml",
            num_layers=8,
            y="periodic",
            z=("pec", "pmc"),
        )
        result = _convert_boundary(boundary, td)
        assert isinstance(result.x.minus, td.PML)
        assert isinstance(result.x.plus, td.PML)
        assert result.x.minus.num_layers == 8
        assert isinstance(result.y.minus, td.Periodic)
        assert isinstance(result.y.plus, td.Periodic)
        assert isinstance(result.z.minus, td.PECBoundary)
        assert isinstance(result.z.plus, td.PMCBoundary)

    def test_auto_bloch_boundary_requires_solver_resolution(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        boundary = mw.BoundarySpec.bloch("auto")

        with pytest.raises(ValueError, match="Simulation.prepare"):
            _convert_boundary(boundary, td)

    def test_mixed_auto_bloch_boundary_requires_solver_resolution(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        boundary = mw.BoundarySpec.faces(
            default="pml",
            num_layers=8,
            x="bloch",
            z="periodic",
            bloch_wavevector="auto",
        )

        with pytest.raises(ValueError, match="Simulation.prepare"):
            _convert_boundary(boundary, td)


class TestMediumConversion:
    def test_simple_medium(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Material(eps_r=2.25)
        result = _convert_material(medium, td)
        assert isinstance(result, td.Medium)
        assert result.permittivity == 2.25

    def test_drude_medium(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Material.drude(eps_inf=1.0, plasma_frequency=2e15, gamma=1e13)
        result = _convert_material(medium, td)
        assert isinstance(result, td.Drude)
        assert result.eps_inf == 1.0
        assert len(result.coeffs) == 1
        assert result.coeffs[0] == (2e15, 1e13)

    def test_lorentz_medium(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Material.lorentz(
            eps_inf=1.0, delta_eps=2.0, resonance_frequency=5e14, gamma=1e13
        )
        result = _convert_material(medium, td)
        assert isinstance(result, td.Lorentz)
        assert result.coeffs[0] == (2.0, 5e14, 1e13)

    def test_debye_medium(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Material.debye(eps_inf=2.0, delta_eps=1.0, tau=1e-12)
        result = _convert_material(medium, td)
        assert isinstance(result, td.Debye)
        assert result.coeffs[0] == (1.0, 1e-12)

    def test_static_magnetic_medium_is_rejected(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Material(eps_r=2.0, mu_r=1.2)
        with pytest.raises(NotImplementedError, match="mu_r = 1"):
            _convert_material(medium, td)

    def test_magnetic_dispersive_medium_is_rejected(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Material(
            eps_r=2.0,
            mu_lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=5e14, gamma=1e13),),
        )
        with pytest.raises(NotImplementedError, match="magnetic dispersive"):
            _convert_material(medium, td)

    def test_kerr_medium_is_rejected(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Material(eps_r=2.0, kerr_chi3=1.0e-10)
        with pytest.raises(NotImplementedError, match="Kerr nonlinear"):
            _convert_material(medium, td)

    def test_pec_medium_exports_as_pec_not_vacuum(self, inject_mock_tidy3d):
        # Regression: Material.pec() carries eps_r == 1.0 by construction, so
        # without a dedicated PEC branch it fell through to the non-dispersive
        # td.Medium(permittivity=1.0) path and silently exported as vacuum.
        td = inject_mock_tidy3d
        medium = mw.Material.pec()
        assert medium.is_pec
        result = _convert_material(medium, td)
        # Must use Tidy3D's dedicated perfect-conductor construct, never the
        # finite-dielectric Medium (which for eps_r=1.0 would be vacuum).
        assert isinstance(result, td.PECMedium)
        assert not isinstance(result, td.Medium)
        assert not hasattr(result, "permittivity")

    def test_pec_structure_scene_exports_pec_medium(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = (
            mw.Scene(
                domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
                grid=mw.GridSpec.uniform(0.05),
                boundary=mw.BoundarySpec.pml(num_layers=10),
                device="cpu",
            )
            .add_structure(
                mw.Structure(mw.Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), mw.Material.pec())
            )
            .add_source(mw.PointDipole(position=(0.4, 0, 0), polarization="Ez", source_time=mw.CW(frequency=1e9)))
        )
        sim = scene.to_tidy3d(frequencies=1e9)
        assert len(sim.structures) == 1
        assert isinstance(sim.structures[0].medium, td.PECMedium)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the eps_model convention check",
    )
    def test_pec_export_is_conductor_not_vacuum_real_tidy3d(self):
        # Physical-convention check against the real tidy3d: the exported PEC
        # medium must model a perfect conductor (eps_model -> large negative),
        # emphatically NOT the vacuum value of 1.0 the buggy path produced.
        with _real_tidy3d() as real_td:
            medium = mw.Material.pec()
            result = _convert_material(medium, real_td)
            assert isinstance(result, real_td.PECMedium)
            freq = 3e14
            eps = complex(result.eps_model(freq))
            assert eps.real < -1.0
            assert not math.isclose(eps.real, 1.0, rel_tol=0.0, abs_tol=1e-6)


class TestGeometryConversion:
    def test_box(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        geom = mw.Box(position=(0, 0, 0), size=(1, 2, 3))
        result = _convert_geometry(geom, td, 1.0)
        assert isinstance(result, td.Box)
        assert result.center == (0, 0, 0)
        assert result.size == (1.0, 2.0, 3.0)

    def test_sphere(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        geom = mw.Sphere(position=(1, 2, 3), radius=0.5)
        result = _convert_geometry(geom, td, 1.0)
        assert isinstance(result, td.Sphere)
        assert result.radius == 0.5

    def test_cylinder(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        geom = mw.Cylinder(position=(0, 0, 0), radius=1.0, height=2.0, axis="y")
        result = _convert_geometry(geom, td, 1.0)
        assert isinstance(result, td.Cylinder)
        assert result.axis == 1
        assert result.length == 2.0

    def test_unsupported_geometry(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        geom = mw.Torus(position=(0, 0, 0), major_radius=1.0, minor_radius=0.3)
        with pytest.raises(NotImplementedError, match="torus"):
            _convert_geometry(geom, td, 1.0)


class TestSourceConversion:
    def test_point_dipole(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))), device="cpu")
        src = mw.PointDipole(
            position=(0, 0, 0),
            polarization="Ez",
            source_time=mw.CW(frequency=1e9),
        )
        result = _convert_source(src, scene, td, 1.0)
        assert isinstance(result, td.PointDipole)
        assert result.center == (0, 0, 0)
        assert result.polarization == "Ez"

    def test_plane_wave(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-5, 5), (-5, 5), (-5, 5))), device="cpu")
        src = mw.PlaneWave(
            direction=(0, 0, 1),
            polarization=(1, 0, 0),
            source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13),
        )
        result = _convert_source(src, scene, td, 1.0)
        assert isinstance(result, td.PlaneWave)
        assert result.direction == "+"
        # Source plane should be zero-thickness along z.
        assert result.size[2] == 0.0


class TestSourceTimeConversion:
    def test_cw(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        st = mw.CW(frequency=1e9, amplitude=2.0, phase=0.5)
        result = _convert_source_time(st, td)
        assert isinstance(result, td.ContinuousWave)
        assert result.freq0 == 1e9
        assert result.amplitude == 2.0
        assert result.phase == 0.5

    def test_gaussian_pulse(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        st = mw.GaussianPulse(frequency=3e14, fwidth=1e13)
        result = _convert_source_time(st, td)
        assert isinstance(result, td.GaussianPulse)
        assert result.freq0 == 3e14
        assert result.fwidth == 1e13
        assert result.offset == pytest.approx(st.delay / st.sigma_t)
        assert result.phase == pytest.approx(st.phase + 2.0 * math.pi * st.frequency * st.delay)


class TestMonitorConversion:
    def test_point_monitor(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.PointMonitor(name="probe", position=(0, 0, 0))
        bounds = ((-1, 1), (-1, 1), (-1, 1))
        result = _convert_monitor(mon, bounds, (1e9,), td, 1.0)
        assert isinstance(result, td.FieldMonitor)
        assert result.size == (0.0, 0.0, 0.0)
        assert result.name == "probe"

    def test_plane_monitor(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.PlaneMonitor(name="field_xy", axis="z", position=0.5)
        bounds = ((-2, 2), (-2, 2), (-2, 2))
        result = _convert_monitor(mon, bounds, (1e9,), td, 1.0)
        assert isinstance(result, td.FieldMonitor)
        assert result.center[2] == 0.5
        assert result.size[2] == 0.0

    def test_finite_plane_monitor(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.FinitePlaneMonitor(
            name="finite_xy",
            position=(0.0, 0.0, 0.5),
            size=(1.2, 1.4, 0.0),
            frequencies=(1e9,),
        )
        bounds = ((-2, 2), (-2, 2), (-2, 2))
        result = _convert_monitor(mon, bounds, None, td, 1.0)
        assert isinstance(result, td.FieldMonitor)
        assert result.center == pytest.approx((0.0, 0.0, 0.5))
        assert result.size == pytest.approx((1.2, 1.4, 0.0))

    def test_flux_monitor(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.FluxMonitor(name="flux_z", axis="z", position=1.0, frequencies=(1e9, 2e9))
        bounds = ((-3, 3), (-3, 3), (-3, 3))
        result = _convert_monitor(mon, bounds, None, td, 1.0)
        assert isinstance(result, td.FluxMonitor)
        assert result.freqs == [1e9, 2e9]

    def test_mode_monitor_exports_as_field_monitor(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.ModeMonitor(
            name="port0",
            position=(0.0, 0.0, 0.5),
            size=(1.0, 1.0, 0.0),
            polarization="Ey",
            frequencies=(1e9,),
        )
        bounds = ((-2, 2), (-2, 2), (-2, 2))
        result = _convert_monitor(mon, bounds, None, td, 1.0)
        assert isinstance(result, td.FieldMonitor)
        assert result.center[2] == 0.5
        assert result.size[2] == 0.0

    def test_monitor_without_frequencies_raises(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.PointMonitor(name="probe", position=(0, 0, 0))
        bounds = ((-1, 1), (-1, 1), (-1, 1))
        with pytest.raises(ValueError, match="no frequencies"):
            _convert_monitor(mon, bounds, None, td, 1.0)


class TestFullSceneConversion:
    def test_basic_scene(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = (
            mw.Scene(
                domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
                grid=mw.GridSpec.uniform(0.05),
                boundary=mw.BoundarySpec.pml(num_layers=10),
                device="cpu",
            )
            .add_structure(mw.Structure(mw.Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), mw.Material(eps_r=4.0)))
            .add_source(mw.PointDipole(position=(0, 0, 0), polarization="Ez", source_time=mw.CW(frequency=1e9)))
            .add_monitor(mw.PlaneMonitor(name="field_xy", axis="z", position=0.0))
        )
        sim = scene.to_tidy3d(frequencies=1e9)
        assert isinstance(sim, td.Simulation)
        assert sim.center == pytest.approx((0.0, 0.0, 0.0))
        assert sim.size == pytest.approx((2.0e6, 2.0e6, 2.0e6))
        assert len(sim.structures) == 1
        assert len(sim.sources) == 1
        assert len(sim.monitors) == 1

    def test_closed_surface_monitor_exports_as_resolved_finite_faces(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = (
            mw.Scene(
                domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
                grid=mw.GridSpec.uniform(0.05),
                boundary=mw.BoundarySpec.pml(num_layers=10),
                device="cpu",
            )
            .add_source(mw.PointDipole(position=(0, 0, 0), polarization="Ez", source_time=mw.CW(frequency=1e9)))
            .add_monitor(
                mw.ClosedSurfaceMonitor.box(
                    "huygens",
                    position=(0.0, 0.0, 0.0),
                    size=(0.8, 0.6, 0.4),
                    frequencies=(1e9,),
                )
            )
        )
        sim = scene.to_tidy3d(frequencies=1e9)
        assert isinstance(sim, td.Simulation)
        assert len(sim.monitors) == 6
        names = {monitor.name for monitor in sim.monitors}
        assert names == {
            "huygens::x_neg",
            "huygens::x_pos",
            "huygens::y_neg",
            "huygens::y_pos",
            "huygens::z_neg",
            "huygens::z_pos",
        }

    def test_symmetry_mapping(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
            grid=mw.GridSpec.uniform(0.1),
            boundary=mw.BoundarySpec.pml(),
            sources=[mw.PointDipole(position=(0, 0, 0), source_time=mw.CW(frequency=1e9))],
            symmetry=("PEC", None, "PMC"),
            device="cpu",
        )
        sim = scene.to_tidy3d(frequencies=1e9)
        assert sim.symmetry == (-1, 0, 1)

    def test_custom_run_time(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
            sources=[mw.PointDipole(position=(0, 0, 0), source_time=mw.CW(frequency=1e9))],
            device="cpu",
        )
        sim = scene.to_tidy3d(frequencies=1e9, run_time=1e-9)
        assert sim.run_time == 1e-9

    def test_mode_port_without_excitation_exports_as_monitor_only(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
            grid=mw.GridSpec.uniform(0.05),
            device="cpu",
        ).add_port(
            mw.ModePort(
                "port0",
                position=(0.0, 0.0, 0.0),
                size=(0.0, 1.0, 1.0),
                polarization="Ez",
                frequencies=(1e9,),
            )
        )
        sim = scene.to_tidy3d(frequencies=1e9)
        assert isinstance(sim, td.Simulation)
        assert len(sim.sources) == 0
        assert len(sim.monitors) == 1
