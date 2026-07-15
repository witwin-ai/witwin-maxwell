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

import numpy as np
import pytest
import torch


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
    class Medium2D(_Recorder): pass
    class LossyMetalMedium(_Recorder): pass
    class AnisotropicMedium(_Recorder): pass
    class FullyAnisotropicMedium(_Recorder): pass
    class Drude(_Recorder): pass
    class Lorentz(_Recorder): pass
    class Debye(_Recorder): pass
    class PoleResidue(_Recorder): pass
    class NonlinearSpec(_Recorder): pass
    class NonlinearSusceptibility(_Recorder): pass
    class TwoPhotonAbsorption(_Recorder): pass
    class ModulationSpec(_Recorder): pass
    class SpaceTimeModulation(_Recorder): pass
    class SpaceModulation(_Recorder): pass
    class ContinuousWaveTimeModulation(_Recorder): pass
    class Box(_Recorder): pass
    class Sphere(_Recorder): pass
    class Cylinder(_Recorder): pass
    class PolySlab(_Recorder): pass
    class Structure(_Recorder): pass
    class PointDipole(_Recorder): pass
    class PlaneWave(_Recorder): pass
    class GaussianBeam(_Recorder): pass
    class AstigmaticGaussianBeam(_Recorder): pass
    class ModeSource(_Recorder): pass
    class ModeSpec(_Recorder): pass
    class ModeSortSpec(_Recorder): pass
    class TFSF(_Recorder): pass
    class UniformCurrentSource(_Recorder): pass
    class CustomCurrentSource(_Recorder): pass
    class CustomFieldSource(_Recorder): pass
    class FieldDataset(_Recorder): pass
    class FieldMonitor(_Recorder): pass
    class FluxMonitor(_Recorder): pass
    class ModeMonitor(_Recorder): pass
    class DiffractionMonitor(_Recorder): pass
    class PermittivityMonitor(_Recorder): pass
    class FieldTimeMonitor(_Recorder): pass
    class FluxTimeMonitor(_Recorder): pass
    class ContinuousWave(_Recorder): pass
    class GaussianPulse(_Recorder): pass

    class ScalarFieldDataArray:
        def __init__(self, data, coords=None):
            self.data = data
            self.coords = coords

    class TriangleMesh(_Recorder):
        @classmethod
        def from_vertices_faces(cls, vertices, faces):
            return cls(vertices=vertices, faces=faces)
    class PML(_Recorder): pass
    class Periodic(_Recorder): pass
    class PECBoundary(_Recorder): pass
    class PMCBoundary(_Recorder): pass
    class BlochBoundary(_Recorder): pass
    class Boundary(_Recorder): pass
    class UniformGrid(_Recorder): pass
    class CustomGridBoundaries(_Recorder): pass
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
    td.Medium2D = Medium2D
    td.LossyMetalMedium = LossyMetalMedium
    td.AnisotropicMedium = AnisotropicMedium
    td.FullyAnisotropicMedium = FullyAnisotropicMedium
    td.Drude = Drude
    td.Lorentz = Lorentz
    td.Debye = Debye
    td.PoleResidue = PoleResidue
    td.NonlinearSpec = NonlinearSpec
    td.NonlinearSusceptibility = NonlinearSusceptibility
    td.TwoPhotonAbsorption = TwoPhotonAbsorption
    td.ModulationSpec = ModulationSpec
    td.SpaceTimeModulation = SpaceTimeModulation
    td.SpaceModulation = SpaceModulation
    td.ContinuousWaveTimeModulation = ContinuousWaveTimeModulation
    td.Box = Box
    td.Sphere = Sphere
    td.Cylinder = Cylinder
    td.PolySlab = PolySlab
    td.TriangleMesh = TriangleMesh
    td.Structure = Structure
    td.PointDipole = PointDipole
    td.PlaneWave = PlaneWave
    td.GaussianBeam = GaussianBeam
    td.AstigmaticGaussianBeam = AstigmaticGaussianBeam
    td.ModeSource = ModeSource
    td.ModeSpec = ModeSpec
    td.ModeSortSpec = ModeSortSpec
    td.TFSF = TFSF
    td.UniformCurrentSource = UniformCurrentSource
    td.CustomCurrentSource = CustomCurrentSource
    td.CustomFieldSource = CustomFieldSource
    td.FieldDataset = FieldDataset
    td.ScalarFieldDataArray = ScalarFieldDataArray
    td.FieldMonitor = FieldMonitor
    td.FluxMonitor = FluxMonitor
    td.ModeMonitor = ModeMonitor
    td.DiffractionMonitor = DiffractionMonitor
    td.PermittivityMonitor = PermittivityMonitor
    td.FieldTimeMonitor = FieldTimeMonitor
    td.FluxTimeMonitor = FluxTimeMonitor
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
    td.CustomGridBoundaries = CustomGridBoundaries
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
        scene = types.SimpleNamespace(grid=mw.GridSpec.uniform(0.05))
        result = _convert_grid(scene, td, 1.0)
        assert result.dl == 0.05

    def test_anisotropic(self, inject_mock_tidy3d):
        from witwin.maxwell.adapters.tidy3d import _convert_grid
        td = inject_mock_tidy3d
        scene = types.SimpleNamespace(grid=mw.GridSpec.anisotropic(0.01, 0.02, 0.03))
        result = _convert_grid(scene, td, 2.0)
        assert isinstance(result.grid_x, td.UniformGrid)
        assert (result.grid_x.dl, result.grid_y.dl, result.grid_z.dl) == (0.02, 0.04, 0.06)

    def test_custom_exports_boundary_coords(self, inject_mock_tidy3d):
        # A nonuniform (GridSpec.custom) grid exports as per-axis CustomGridBoundaries
        # carrying the node coordinates scaled by the metre->Tidy3D length factor.
        import numpy as np

        from witwin.maxwell.adapters.tidy3d import _convert_grid
        td = inject_mock_tidy3d
        xn = [0.0, 0.1, 0.25, 0.45]
        yn = [0.0, 0.2, 0.5]
        zn = [0.0, 0.3, 0.55, 0.7, 1.0]
        scene = types.SimpleNamespace(grid=mw.GridSpec.custom(xn, yn, zn))
        result = _convert_grid(scene, td, 2.0)
        assert isinstance(result.grid_x, td.CustomGridBoundaries)
        assert np.allclose(result.grid_x.coords, np.asarray(xn) * 2.0)
        assert np.allclose(result.grid_y.coords, np.asarray(yn) * 2.0)
        assert np.allclose(result.grid_z.coords, np.asarray(zn) * 2.0)

    def test_auto_exports_maxwell_resolved_boundary_coords(self, inject_mock_tidy3d):
        # A GridSpec.auto grid is resolved through maxwell's own mesher and exported as
        # the exact resolved node coordinates, so Tidy3D reuses maxwell's mesh rather
        # than running its own AutoGrid.
        import numpy as np

        from witwin.maxwell.adapters.tidy3d import _convert_grid
        from witwin.maxwell.fdtd.meshing import resolve_auto_grid
        td = inject_mock_tidy3d
        scene = (
            mw.Scene(
                domain=mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
                grid=mw.GridSpec.auto(min_steps_per_wavelength=10, wavelength=0.5),
                device="cpu",
            )
            .add_structure(
                mw.Structure(mw.Box(position=(0, 0, 0), size=(0.4, 0.4, 0.4)), mw.Material(eps_r=12.0))
            )
        )
        expected = resolve_auto_grid(scene)
        result = _convert_grid(scene, td, 3.0)
        for axis, nodes in zip(("grid_x", "grid_y", "grid_z"), expected):
            component = getattr(result, axis)
            assert isinstance(component, td.CustomGridBoundaries)
            assert np.allclose(component.coords, np.asarray(nodes) * 3.0)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package to build the discretized Yee grid",
    )
    def test_custom_grid_builds_identical_yee_boundaries_in_real_tidy3d(self):
        # "Correct, not merely accepted": the exported CustomGridBoundaries must make the
        # real Tidy3D Simulation discretize on the SAME in-domain Yee boundaries maxwell
        # uses. Tidy3D adds PML cells outside the domain, so compare only the interior.
        import numpy as np

        graded = np.array([0.0, 0.12, 0.28, 0.5, 0.68, 0.82, 1.0], dtype=np.float64)
        with _real_tidy3d():
            scene = (
                mw.Scene(
                    domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
                    grid=mw.GridSpec.custom(graded, graded, graded),
                    boundary=mw.BoundarySpec.pml(num_layers=6),
                    device="cpu",
                )
                .add_source(
                    mw.PointDipole(
                        position=(0.5, 0.5, 0.5),
                        polarization="Ez",
                        source_time=mw.GaussianPulse(frequency=1e14, fwidth=1e13),
                    )
                )
            )
            sim = scene.to_tidy3d()
            s = 1.0e6
            for axis in ("x", "y", "z"):
                boundaries = np.asarray(getattr(sim.grid.boundaries, axis))
                lo, hi = graded[0] * s, graded[-1] * s
                interior = boundaries[(boundaries >= lo - 1e-3) & (boundaries <= hi + 1e-3)]
                assert np.allclose(interior, graded * s)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package to build the discretized Yee grid",
    )
    def test_auto_grid_builds_maxwell_resolved_mesh_in_real_tidy3d(self):
        # The resolved auto mesh must survive round-trip into a real Tidy3D Simulation:
        # the in-domain Yee boundaries equal maxwell's own resolve_auto_grid nodes.
        import numpy as np

        from witwin.maxwell.fdtd.meshing import resolve_auto_grid
        with _real_tidy3d():
            scene = (
                mw.Scene(
                    domain=mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
                    grid=mw.GridSpec.auto(min_steps_per_wavelength=10, wavelength=0.5),
                    boundary=mw.BoundarySpec.pml(num_layers=6),
                    device="cpu",
                )
                .add_structure(
                    mw.Structure(mw.Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)), mw.Material(eps_r=9.0))
                )
                .add_source(
                    mw.PointDipole(
                        position=(0, 0, 0),
                        polarization="Ez",
                        source_time=mw.GaussianPulse(frequency=6e14, fwidth=6e13),
                    )
                )
            )
            expected = resolve_auto_grid(scene)
            sim = scene.to_tidy3d()
            s = 1.0e6
            for axis, nodes in zip(("x", "y", "z"), expected):
                nodes = np.asarray(nodes)
                boundaries = np.asarray(getattr(sim.grid.boundaries, axis))
                lo, hi = nodes[0] * s, nodes[-1] * s
                interior = boundaries[(boundaries >= lo - 1e-3) & (boundaries <= hi + 1e-3)]
                assert interior.size == nodes.size
                assert np.allclose(interior, nodes * s)


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

    def test_bloch_wavevector_is_normalized_by_physical_period(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        boundary = mw.BoundarySpec.faces(
            default="pml",
            num_layers=8,
            x="bloch",
            bloch_wavevector=(3.0 * np.pi, 0.0, 0.0),
        )

        result = _convert_boundary(
            boundary,
            td,
            ((-0.25, 0.25), (-1.0, 1.0), (-2.0, 2.0)),
        )

        assert result.x.minus.bloch_vec == pytest.approx(0.75)
        assert result.x.plus.bloch_vec == pytest.approx(0.75)

    def test_explicit_bloch_export_requires_domain_bounds(self, inject_mock_tidy3d):
        boundary = mw.BoundarySpec.bloch((1.0, 0.0, 0.0))

        with pytest.raises(ValueError, match="physical domain bounds"):
            _convert_boundary(boundary, inject_mock_tidy3d)

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
        assert result.coeffs[0] == (2.0, 5e14, 0.5e13)

    def test_debye_medium(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Material.debye(eps_inf=2.0, delta_eps=1.0, tau=1e-12)
        result = _convert_material(medium, td)
        assert isinstance(result, td.Debye)
        assert result.coeffs[0] == pytest.approx((1.0, 2.0 * math.pi * 1e-12))

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the dispersive convention check",
    )
    @pytest.mark.parametrize(
        "material",
        [
            mw.Material.debye(eps_inf=2.0, delta_eps=1.5, tau=1e-10),
            mw.Material.lorentz(
                eps_inf=2.0,
                delta_eps=1.5,
                resonance_frequency=3e9,
                gamma=0.2e9,
            ),
        ],
        ids=("debye", "lorentz"),
    )
    def test_dispersive_medium_matches_maxwell_permittivity_real_tidy3d(self, material):
        with _real_tidy3d() as real_td:
            exported = _convert_material(material, real_td)
            for frequency in (1.0e9, 2.0e9, 4.0e9):
                assert complex(exported.eps_model(frequency)) == pytest.approx(
                    material.relative_permittivity(frequency), rel=1e-12
                )

    def test_static_magnetic_medium_is_rejected(self, inject_mock_tidy3d):
        # Tidy3D's FDTD solver fixes mu_r = 1: a genuinely magnetic medium
        # (mu_r != 1) has no equivalent construct, so the guard must reject it
        # while stating that physical reason -- not a deferral phrase.
        td = inject_mock_tidy3d
        medium = mw.Material(eps_r=2.0, mu_r=1.2)
        # The material really is magnetic at the export frequency (mu != mu0),
        # which is exactly why Tidy3D cannot represent it.
        assert medium.relative_permeability(5e14) == pytest.approx(1.2)
        with pytest.raises(NotImplementedError, match="mu_r = 1") as info:
            _convert_material(medium, td)
        message = str(info.value)
        assert "no magnetic-material model" in message
        assert "not implemented yet" not in message

    def test_magnetic_dispersive_medium_is_rejected(self, inject_mock_tidy3d):
        # A mu-pole (dispersive permeability) medium has no Tidy3D permeability
        # pole construct; the guard rejects it with the physical reason.
        td = inject_mock_tidy3d
        medium = mw.Material(
            eps_r=2.0,
            mu_lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=5e14, gamma=1e13),),
        )
        assert medium.is_magnetic_dispersive
        # The permeability is genuinely frequency-dispersive (mu(omega) != 1 off
        # resonance), so no static Tidy3D medium can carry it.
        assert medium.relative_permeability(1e14) != pytest.approx(1.0)
        with pytest.raises(NotImplementedError, match="magnetic dispersive") as info:
            _convert_material(medium, td)
        message = str(info.value)
        assert "no magnetic-material model" in message
        assert "not implemented yet" not in message

    def test_kerr_chi3_exports_nonlinear_susceptibility(self, inject_mock_tidy3d):
        # A Kerr (chi3) Material exports a non-dispersive Medium carrying a
        # NonlinearSpec whose single NonlinearSusceptibility uses the length-scaled
        # chi3 (maxwell [m^2/V^2] -> Tidy3D [um^2/V^2] = x length_scale^2).
        td = inject_mock_tidy3d
        chi3_si = 2.0e-20
        length_scale = 1.0e6
        medium = mw.Material(eps_r=4.0, kerr_chi3=chi3_si)
        result = _convert_material(medium, td, length_scale)
        assert isinstance(result, td.Medium)
        assert result.permittivity == 4.0
        spec = result.nonlinear_spec
        assert isinstance(spec, td.NonlinearSpec)
        assert len(spec.models) == 1
        model = spec.models[0]
        assert isinstance(model, td.NonlinearSusceptibility)
        assert model.chi3 == pytest.approx(chi3_si * length_scale ** 2)

    def test_two_photon_absorption_exports_with_scaled_beta_and_explicit_n0(self, inject_mock_tidy3d):
        # TPA exports a TwoPhotonAbsorption model: beta [m/W] -> [um/W] scales by
        # length_scale, and n0 defaults to sqrt(eps_r) and must be forwarded so
        # Tidy3D does not silently infer it from the source frequencies.
        td = inject_mock_tidy3d
        beta_si = 1.0e-11
        length_scale = 1.0e6
        medium = mw.Material(eps_r=9.0, nonlinearity=(mw.TwoPhotonAbsorption(beta=beta_si),))
        result = _convert_material(medium, td, length_scale)
        spec = result.nonlinear_spec
        assert len(spec.models) == 1
        tpa = spec.models[0]
        assert isinstance(tpa, td.TwoPhotonAbsorption)
        assert tpa.beta == pytest.approx(beta_si * length_scale)
        assert tpa.n0 == pytest.approx(math.sqrt(9.0))

    def test_tpa_explicit_n0_is_forwarded(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Material(eps_r=9.0, nonlinearity=(mw.TwoPhotonAbsorption(beta=1.0e-11, n0=3.4),))
        result = _convert_material(medium, td, 1.0e6)
        assert result.nonlinear_spec.models[0].n0 == pytest.approx(3.4)

    def test_chi3_and_tpa_export_as_two_additive_models(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Material(
            eps_r=4.0,
            nonlinearity=(
                mw.NonlinearSusceptibility(chi3=1.0e-20),
                mw.TwoPhotonAbsorption(beta=2.0e-11, n0=2.0),
            ),
        )
        result = _convert_material(medium, td, 1.0e6)
        kinds = {type(m).__name__ for m in result.nonlinear_spec.models}
        assert kinds == {"NonlinearSusceptibility", "TwoPhotonAbsorption"}

    def test_chi2_medium_is_rejected_as_no_tidy3d_equivalent(self, inject_mock_tidy3d):
        # Tidy3D's public nonlinear API is the chi3/Kerr/TPA family only; a
        # second-order (chi2 / SHG) susceptibility has no equivalent and must be
        # rejected with a physics-worded message (not a Kerr blanket rejection).
        td = inject_mock_tidy3d
        medium = mw.Material(eps_r=4.0, nonlinearity=(mw.NonlinearSusceptibility(chi2=5.0e-12),))
        with pytest.raises(NotImplementedError, match="second-order"):
            _convert_material(medium, td)

    def test_kerr_plus_dispersion_exports_dispersive_medium_with_nonlinear_spec(self, inject_mock_tidy3d):
        # nonlinear + dispersive (same material) rides the Kerr NonlinearSpec on the
        # dispersive Tidy3D medium rather than being rejected.
        td = inject_mock_tidy3d
        medium = mw.Material(
            eps_r=1.0,
            lorentz_poles=(mw.LorentzPole(delta_eps=2.0, resonance_frequency=5e14, gamma=1e13),),
            kerr_chi3=1.0e-20,
        )
        result = _convert_material(medium, td, 1.0e6)
        assert isinstance(result, td.Lorentz)
        assert isinstance(result.nonlinear_spec, td.NonlinearSpec)
        assert isinstance(result.nonlinear_spec.models[0], td.NonlinearSusceptibility)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the chi3 unit-convention check",
    )
    def test_kerr_chi3_reproduces_physical_n2_real_tidy3d(self):
        # Physical-convention check: the exported chi3 must reproduce the same
        # nonlinear refractive index n2 the maxwell material encodes. Tidy3D's own
        # relation n2 = 3/(4 n0^2 eps0 c0) chi3 (in its um unit system) applied to
        # the exported chi3 must equal the SI n2 = 3 chi3_SI/(4 n0^2 eps0 c0)
        # expressed in Tidy3D units [um^2/W] (i.e. x length_scale^2). A wrong scale
        # (e.g. exporting chi3_SI unscaled) fails this by 1e12.
        import scipy.constants as sc

        chi3_si = 3.0e-20
        n0 = 2.0
        length_scale = 1.0e6
        with _real_tidy3d() as real_td:
            from tidy3d.constants import EPSILON_0 as TD_EPS0, C_0 as TD_C0

            medium = mw.Material(eps_r=n0 ** 2, kerr_chi3=chi3_si)
            exported = _convert_material(medium, real_td, length_scale)
            chi3_td = exported.nonlinear_spec.models[0].chi3
            n2_td = 3.0 / (4.0 * n0 ** 2 * TD_EPS0 * TD_C0) * chi3_td
            n2_si = 3.0 * chi3_si / (4.0 * n0 ** 2 * sc.epsilon_0 * sc.c)
            assert n2_td == pytest.approx(n2_si * length_scale ** 2, rel=1e-6)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the TPA construction/convention check",
    )
    def test_two_photon_absorption_builds_in_real_tidy3d(self):
        # The exported TPA model must satisfy Tidy3D's own validators (real beta,
        # real positive n0) and carry the physical beta [um/W] and the forwarded
        # linear index n0 = sqrt(eps_r).
        beta_si = 1.0e-11
        length_scale = 1.0e6
        with _real_tidy3d() as real_td:
            medium = mw.Material(eps_r=9.0, nonlinearity=(mw.TwoPhotonAbsorption(beta=beta_si),))
            exported = _convert_material(medium, real_td, length_scale)
            assert isinstance(exported, real_td.Medium)
            (tpa,) = exported.nonlinear_spec.models
            assert isinstance(tpa, real_td.TwoPhotonAbsorption)
            assert tpa.beta == pytest.approx(beta_si * length_scale)
            assert tpa.n0 == pytest.approx(3.0)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the nonlinear-spec construction check",
    )
    def test_kerr_plus_sigma_e_dispersive_structure_real_tidy3d(self):
        # End-to-end: a Kerr + dispersive + conductive material exports a single
        # PoleResidue that carries both the folded conductivity pole and the Kerr
        # NonlinearSpec, and the real tidy3d Simulation validators accept it.
        from witwin.maxwell.adapters.tidy3d import _convert_structure

        with _real_tidy3d() as real_td:
            structure = mw.Structure(
                mw.Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)),
                mw.Material(
                    eps_r=1.0,
                    drude_poles=(mw.DrudePole(plasma_frequency=2e15, gamma=1e13),),
                    sigma_e=0.5,
                    kerr_chi3=1.0e-20,
                ),
            )
            td_structure = _convert_structure(structure, real_td, 1e6)
            assert isinstance(td_structure.medium, real_td.PoleResidue)
            assert td_structure.medium.nonlinear_spec is not None

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

    def test_diagonal_anisotropic_epsilon_exports_anisotropic_medium(self, inject_mock_tidy3d):
        # Axis-aligned DiagonalTensor3 permittivity -> AnisotropicMedium of three
        # per-axis isotropic Media whose permittivities are the tensor diagonal.
        td = inject_mock_tidy3d
        medium = mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0))
        result = _convert_material(medium, td)
        assert isinstance(result, td.AnisotropicMedium)
        assert result.xx.permittivity == pytest.approx(2.0)
        assert result.yy.permittivity == pytest.approx(3.0)
        assert result.zz.permittivity == pytest.approx(4.0)

    def test_diagonal_anisotropic_conductivity_is_scaled_per_axis(self, inject_mock_tidy3d):
        # A DiagonalTensor3 electric conductivity exports as per-axis Medium.conductivity,
        # each scaled by the metre->um length factor exactly as the isotropic sigma path.
        td = inject_mock_tidy3d
        length_scale = 1.0e6
        medium = mw.Material(
            epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0),
            sigma_e_tensor=mw.DiagonalTensor3(0.5, 0.1, 0.0),
        )
        result = _convert_material(medium, td, length_scale)
        assert result.xx.conductivity == pytest.approx(0.5 / length_scale)
        assert result.yy.conductivity == pytest.approx(0.1 / length_scale)
        # sigma == 0 axis carries no conductivity kwarg (plain non-lossy Medium).
        assert not hasattr(result.zz, "conductivity")

    def test_diagonal_anisotropic_dispersion_is_carried_per_axis(self, inject_mock_tidy3d):
        # Electric dispersion enters each axis isotropically, so every component of a
        # dispersive diagonal-anisotropic material is the same dispersive Tidy3D medium
        # with a per-axis eps_inf background.
        td = inject_mock_tidy3d
        medium = mw.Material(
            epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0),
            lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=3e14, gamma=1e13),),
        )
        result = _convert_material(medium, td)
        assert isinstance(result, td.AnisotropicMedium)
        assert isinstance(result.xx, td.Lorentz)
        assert result.xx.eps_inf == pytest.approx(2.0)
        assert result.zz.eps_inf == pytest.approx(4.0)
        # Same isotropic pole set on every axis.
        assert result.xx.coeffs == result.zz.coeffs

    def test_full_offdiagonal_tensor_exports_fully_anisotropic_medium(self, inject_mock_tidy3d):
        # A full off-diagonal Tensor3x3 permittivity maps row-for-row onto a Tidy3D
        # FullyAnisotropicMedium; a scalar sigma_e becomes the scaled diagonal
        # conductivity tensor (sigma*I commutes with the permittivity).
        td = inject_mock_tidy3d
        length_scale = 1.0e6
        rows = ((2.0, 0.3, 0.0), (0.3, 3.0, 0.0), (0.0, 0.0, 4.0))
        medium = mw.Material(epsilon_tensor=mw.Tensor3x3(rows), sigma_e=0.5)
        result = _convert_material(medium, td, length_scale)
        assert isinstance(result, td.FullyAnisotropicMedium)
        assert result.permittivity == [[2.0, 0.3, 0.0], [0.3, 3.0, 0.0], [0.0, 0.0, 4.0]]
        assert result.conductivity[0][0] == pytest.approx(0.5 / length_scale)
        assert result.conductivity[1][1] == pytest.approx(0.5 / length_scale)
        assert result.conductivity[0][1] == 0.0

    def test_magnetic_anisotropy_is_rejected(self, inject_mock_tidy3d):
        # Tidy3D anisotropic media are electric-only (mu_r = 1); a mu_tensor has no
        # counterpart, so it must raise rather than silently drop the magnetic tensor.
        td = inject_mock_tidy3d
        medium = mw.Material(mu_tensor=mw.DiagonalTensor3(1.1, 1.2, 1.3))
        with pytest.raises(NotImplementedError, match="anisotropic magnetic medium"):
            _convert_material(medium, td)

    def test_dispersive_full_tensor_is_rejected(self, inject_mock_tidy3d):
        # FullyAnisotropicMedium is strictly non-dispersive: a full off-diagonal tensor
        # combined with poles has no Tidy3D equivalent and must raise, not drop the poles.
        td = inject_mock_tidy3d
        medium = mw.Material(
            epsilon_tensor=mw.Tensor3x3(((2.0, 0.3, 0.0), (0.3, 3.0, 0.0), (0.0, 0.0, 4.0))),
            lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=3e14, gamma=1e13),),
        )
        with pytest.raises(NotImplementedError, match="non-dispersive"):
            _convert_material(medium, td)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the eps_diagonal convention check",
    )
    def test_diagonal_anisotropic_eps_matches_per_axis_physics_real_tidy3d(self):
        # Physical-convention check: the exported AnisotropicMedium must reproduce the
        # per-axis complex permittivity eps_axis + i*sigma_axis/(w*eps0). A wrong axis
        # mapping or an unscaled conductivity (1e6x too lossy) fails this.
        from witwin.core.material import VACUUM_PERMITTIVITY

        freq = 2.0e14
        omega = 2.0 * math.pi * freq
        eps_diag = (2.0, 3.0, 4.0)
        sigma_diag = (0.5, 0.1, 0.0)
        with _real_tidy3d() as real_td:
            material = mw.Material(
                epsilon_tensor=mw.DiagonalTensor3(*eps_diag),
                sigma_e_tensor=mw.DiagonalTensor3(*sigma_diag),
            )
            medium = _convert_material(material, real_td)  # default length_scale = 1e6
            assert isinstance(medium, real_td.AnisotropicMedium)
            eps_axes = medium.eps_diagonal(freq)
            for axis in range(3):
                eps = complex(eps_axes[axis])
                assert eps.real == pytest.approx(eps_diag[axis])
                assert eps.imag == pytest.approx(
                    sigma_diag[axis] / (omega * VACUUM_PERMITTIVITY), rel=1e-6, abs=1e-9
                )

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

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the eps_model convention check",
    )
    def test_nondispersive_sigma_e_conductivity_units_real_tidy3d(self):
        # A static electric conductivity must export with the physically correct
        # imaginary permittivity eps'' = sigma / (w * eps0). Tidy3D is a micrometre
        # solver whose Medium.conductivity is [S/um], so the SI value [S/m] must be
        # divided by the metre->um length scale; passing it unscaled (the pre-fix
        # path) made the exported medium 1e6x too lossy.
        from witwin.core.material import VACUUM_PERMITTIVITY

        freq = 2.0e14
        omega = 2.0 * math.pi * freq
        sigma = 0.5
        with _real_tidy3d() as real_td:
            material = mw.Material(eps_r=3.0, sigma_e=sigma)
            medium = _convert_material(material, real_td)  # default length_scale = 1e6
            assert isinstance(medium, real_td.Medium)
            eps = complex(medium.eps_model(freq))
            assert eps.real == pytest.approx(3.0)
            # Loss (positive imaginary part under e^{-i w t}) of the physical size.
            assert eps.imag == pytest.approx(sigma / (omega * VACUUM_PERMITTIVITY), rel=1e-6)
            assert eps == pytest.approx(material.relative_permittivity(freq), rel=1e-6)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the eps_model convention check",
    )
    def test_drude_plus_sigma_e_matches_physical_permittivity_real_tidy3d(self):
        # A dispersive pole model combined with a static conductivity must export to
        # a single Tidy3D PoleResidue whose eps_model reproduces the physical
        # e^{-i w t} permittivity eps_disp(w) + i*sigma/(w*eps0). Drude is the pole
        # family whose Tidy3D convention coincides with maxwell's, so the full
        # complex value is asserted against Material.relative_permittivity.
        from witwin.core.material import VACUUM_PERMITTIVITY

        freq = 2.0e14
        omega = 2.0 * math.pi * freq
        sigma = 0.5
        with _real_tidy3d() as real_td:
            conductive = mw.Material.drude(
                eps_inf=1.0, plasma_frequency=2e15, gamma=1e13, sigma_e=sigma
            )
            lossless = mw.Material.drude(eps_inf=1.0, plasma_frequency=2e15, gamma=1e13)
            medium = _convert_material(conductive, real_td)
            # Specialized Drude/Lorentz/Debye media cannot carry a conductivity, so
            # the combination must fold into a PoleResidue.
            assert isinstance(medium, real_td.PoleResidue)
            eps = complex(medium.eps_model(freq))
            reference = conductive.relative_permittivity(freq)
            assert eps.real == pytest.approx(reference.real, rel=1e-6)
            assert eps.imag == pytest.approx(reference.imag, rel=1e-6)
            # Isolate the conductivity term against the sigma_e = 0 export: the
            # difference must be exactly +i*sigma/(w*eps0), independent of length
            # scale, confirming the pole sign (loss) and magnitude (SI units).
            lossless_eps = complex(_convert_material(lossless, real_td).eps_model(freq))
            delta = eps - lossless_eps
            assert delta.real == pytest.approx(0.0, abs=1e-6 * abs(eps.real))
            assert delta.imag == pytest.approx(sigma / (omega * VACUUM_PERMITTIVITY), rel=1e-6)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the eps_model convention check",
    )
    @pytest.mark.parametrize(
        "make_pair",
        [
            pytest.param(
                lambda s: (
                    mw.Material.lorentz(
                        eps_inf=2.0, delta_eps=1.5, resonance_frequency=5e14, gamma=1e13, sigma_e=s
                    ),
                    mw.Material.lorentz(
                        eps_inf=2.0, delta_eps=1.5, resonance_frequency=5e14, gamma=1e13
                    ),
                ),
                id="lorentz",
            ),
            pytest.param(
                lambda s: (
                    mw.Material.debye(eps_inf=2.0, delta_eps=1.0, tau=1e-12, sigma_e=s),
                    mw.Material.debye(eps_inf=2.0, delta_eps=1.0, tau=1e-12),
                ),
                id="debye",
            ),
        ],
    )
    def test_dispersive_plus_sigma_e_isolates_conductivity_term_real_tidy3d(self, make_pair):
        # The exported mixed PoleResidue must preserve both the Maxwell dispersive
        # response and the analytic +i*sigma/(w*eps0) conductivity contribution.
        from witwin.core.material import VACUUM_PERMITTIVITY

        freq = 2.0e14
        omega = 2.0 * math.pi * freq
        sigma = 0.5
        with _real_tidy3d() as real_td:
            conductive, lossless = make_pair(sigma)
            medium = _convert_material(conductive, real_td)
            assert isinstance(medium, real_td.PoleResidue)
            eps = complex(medium.eps_model(freq))
            assert eps == pytest.approx(conductive.relative_permittivity(freq), rel=1e-6)
            lossless_eps = complex(_convert_material(lossless, real_td).eps_model(freq))
            delta = eps - lossless_eps
            assert delta.real == pytest.approx(0.0, abs=1e-6 * max(abs(eps.real), 1.0))
            assert delta.imag == pytest.approx(sigma / (omega * VACUUM_PERMITTIVITY), rel=1e-6)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the eps_model convention check",
    )
    def test_dispersive_plus_sigma_e_structure_exports_pole_residue_real_tidy3d(self):
        # End-to-end through _convert_structure: a conductive dispersive structure
        # exports a single PoleResidue medium carrying both the dispersion and the
        # conductivity.
        with _real_tidy3d() as real_td:
            structure = mw.Structure(
                mw.Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)),
                mw.Material.drude(eps_inf=1.0, plasma_frequency=2e15, gamma=1e13, sigma_e=0.5),
            )
            from witwin.maxwell.adapters.tidy3d import _convert_structure

            td_structure = _convert_structure(structure, real_td, 1e6)
            assert isinstance(td_structure.medium, real_td.PoleResidue)


class TestModulationConversion:
    def test_scalar_modulation_exports_medium_with_modulation_spec(self, inject_mock_tidy3d):
        # A time-modulated non-dispersive Material exports as a Medium carrying a
        # ModulationSpec on its permittivity (eps_inf); the base permittivity stays eps_r.
        td = inject_mock_tidy3d
        medium = mw.Material(eps_r=4.0, modulation=mw.ModulationSpec(frequency=2e13, amplitude=0.3, phase=0.7))
        result = _convert_material(medium, td)
        assert isinstance(result, td.Medium)
        assert result.permittivity == pytest.approx(4.0)
        spec = result.modulation_spec
        assert isinstance(spec, td.ModulationSpec)
        stm = spec.permittivity
        assert isinstance(stm, td.SpaceTimeModulation)
        # Time part carries the modulation frequency; magnitude/phase live in space.
        assert stm.time_modulation.freq0 == pytest.approx(2e13)
        assert stm.time_modulation.amplitude == pytest.approx(1.0)
        assert stm.time_modulation.phase == pytest.approx(0.0)

    def test_modulation_amplitude_is_absolute_permittivity_deviation(self, inject_mock_tidy3d):
        # Tidy3D's SpaceModulation amplitude is an ABSOLUTE eps deviation, so the
        # dimensionless maxwell depth must be multiplied by eps_static (= eps_r here):
        # depth 0.2 on eps_r 9 -> absolute amplitude 1.8.
        td = inject_mock_tidy3d
        medium = mw.Material(eps_r=9.0, modulation=mw.ModulationSpec(frequency=1e13, amplitude=0.2, phase=0.0))
        space = _convert_material(medium, td).modulation_spec.permittivity.space_modulation
        assert isinstance(space, td.SpaceModulation)
        assert space.amplitude == pytest.approx(9.0 * 0.2)

    def test_modulation_phase_sign_is_flipped(self, inject_mock_tidy3d):
        # Tidy3D's e^{-i w t} time factor is the conjugate of maxwell's +phase
        # convention, so the exported space phase must be -phase to reproduce
        # cos(2*pi*f*t + phase).
        td = inject_mock_tidy3d
        medium = mw.Material(eps_r=2.0, modulation=mw.ModulationSpec(frequency=1e13, amplitude=0.3, phase=0.9))
        space = _convert_material(medium, td).modulation_spec.permittivity.space_modulation
        assert space.phase == pytest.approx(-0.9)

    def test_modulated_dispersive_is_rejected_as_no_equivalent(self, inject_mock_tidy3d):
        # Tidy3D modulates only the non-dispersive eps_inf; maxwell folds the same
        # per-step factor through the ADE polarization current, so a modulated +
        # dispersive material has no equivalent and must raise (not silently drop the
        # pole modulation).
        td = inject_mock_tidy3d
        medium = mw.Material(
            eps_r=2.0,
            lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=5e14, gamma=1e13),),
            modulation=mw.ModulationSpec(frequency=1e13, amplitude=0.2),
        )
        with pytest.raises(NotImplementedError, match="ModulationSpec modulates only"):
            _convert_material(medium, td)

    def test_modulated_nonlinear_is_rejected_as_no_equivalent(self, inject_mock_tidy3d):
        # Tidy3D's nonlinear polarization is unmodulated, whereas maxwell folds the
        # modulation factor through the instantaneous Kerr coefficient, so the two
        # models disagree and a modulated + Kerr material must raise.
        td = inject_mock_tidy3d
        medium = mw.Material(eps_r=4.0, kerr_chi3=1.0e-20, modulation=mw.ModulationSpec(frequency=1e13, amplitude=0.2))
        with pytest.raises(NotImplementedError, match="ModulationSpec modulates only"):
            _convert_material(medium, td)

    def test_spatially_varying_modulation_is_rejected(self, inject_mock_tidy3d):
        # A 3D-tensor (spatially varying) amplitude/phase profile is defined relative
        # to the owning structure's Box, which the material-level export cannot resolve
        # into Tidy3D absolute coordinates, so it must raise.
        import torch

        td = inject_mock_tidy3d
        amp = torch.full((4, 4, 4), 0.3)
        medium = mw.Material(eps_r=3.0, modulation=mw.ModulationSpec(frequency=1e13, amplitude=amp))
        with pytest.raises(NotImplementedError, match="spatially-varying modulation"):
            _convert_material(medium, td)

    def test_modulated_structure_scene_exports_medium(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = (
            mw.Scene(
                domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
                grid=mw.GridSpec.uniform(0.05),
                boundary=mw.BoundarySpec.pml(num_layers=10),
                device="cpu",
            )
            .add_structure(
                mw.Structure(
                    mw.Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)),
                    mw.Material(eps_r=4.0, modulation=mw.ModulationSpec(frequency=2e13, amplitude=0.3)),
                )
            )
            .add_source(mw.PointDipole(position=(0, 0, 0.4), polarization="Ex", source_time=mw.CW(frequency=1e13)))
        )
        sim = scene.to_tidy3d(frequencies=1e13)
        assert len(sim.structures) == 1
        medium = sim.structures[0].medium
        assert isinstance(medium, td.Medium)
        assert medium.modulation_spec is not None

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the modulation-convention check",
    )
    def test_modulation_reproduces_maxwell_delta_eps_real_tidy3d(self):
        # Physical-convention check: the exported ModulationSpec must reproduce
        # maxwell's absolute permittivity deviation
        # delta_eps(t) = eps_static * depth * cos(2*pi*f*t + phase) at every time,
        # including the phase sign flip. The exported medium must also satisfy Tidy3D's
        # own non-negative-permittivity modulation validator (it builds without error).
        import cmath

        eps_static, depth, phase, freq = 4.0, 0.3, 0.7, 2.0e13
        omega = 2.0 * math.pi * freq
        with _real_tidy3d() as real_td:
            material = mw.Material(
                eps_r=eps_static,
                modulation=mw.ModulationSpec(frequency=freq, amplitude=depth, phase=phase),
            )
            exported = _convert_material(material, real_td)
            assert isinstance(exported, real_td.Medium)
            assert exported.permittivity == pytest.approx(eps_static)
            stm = exported.modulation_spec.permittivity
            assert stm.time_modulation.freq0 == pytest.approx(freq)
            amp_space = complex(stm.space_modulation.amplitude) * cmath.exp(
                1j * float(stm.space_modulation.phase)
            )
            for t in (0.0, 1.0e-15, 7.3e-15, 2.1e-14):
                delta_td = (complex(stm.time_modulation.amp_time(t)) * amp_space).real
                delta_mx = eps_static * depth * math.cos(omega * t + phase)
                assert delta_td == pytest.approx(delta_mx, rel=1e-6, abs=1e-9)
            # The minimum modulated permittivity stays strictly positive (depth < 0.5),
            # which is exactly what let the real Medium validator accept the export.
            assert eps_static - eps_static * depth > 0.0

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the modulation-sign control check",
    )
    def test_modulation_phase_sign_flip_is_load_bearing_real_tidy3d(self):
        # Control: dropping the phase sign flip (exporting +phase instead of -phase)
        # would reproduce cos(2*pi*f*t - phase), which differs from maxwell's
        # cos(2*pi*f*t + phase) at t=0 for a nonzero phase. Confirm the exported value
        # matches the correct convention and NOT the wrong-sign one.
        import cmath

        eps_static, depth, phase, freq = 3.0, 0.25, 1.1, 1.0e13
        with _real_tidy3d() as real_td:
            material = mw.Material(
                eps_r=eps_static,
                modulation=mw.ModulationSpec(frequency=freq, amplitude=depth, phase=phase),
            )
            stm = _convert_material(material, real_td).modulation_spec.permittivity
            amp_space = complex(stm.space_modulation.amplitude) * cmath.exp(
                1j * float(stm.space_modulation.phase)
            )
            delta_t0 = (complex(stm.time_modulation.amp_time(0.0)) * amp_space).real
            correct = eps_static * depth * math.cos(phase)
            wrong_sign = eps_static * depth * math.cos(-phase)  # identical at t=0 (cos even)
            # At t=0 both signs coincide; use a quarter modulation period to separate them.
            quarter = 0.25 / freq
            delta_q = (complex(stm.time_modulation.amp_time(quarter)) * amp_space).real
            correct_q = eps_static * depth * math.cos(2.0 * math.pi * freq * quarter + phase)
            wrong_q = eps_static * depth * math.cos(2.0 * math.pi * freq * quarter - phase)
            assert delta_t0 == pytest.approx(correct)
            assert delta_q == pytest.approx(correct_q)
            assert abs(delta_q - wrong_q) > 1e-3  # the wrong sign is genuinely different here


class TestSheetAndMetalConversion:
    def test_static_medium2d_exports_medium2d(self, inject_mock_tidy3d):
        # A static Medium2D sheet exports as a Tidy3D Medium2D whose two tangential
        # surface media are the same isotropic Medium carrying the sheet conductance.
        td = inject_mock_tidy3d
        medium = mw.Medium2D(sigma_s=0.002)
        result = _convert_material(medium, td)
        assert isinstance(result, td.Medium2D)
        assert isinstance(result.ss, td.Medium)
        assert isinstance(result.tt, td.Medium)
        assert result.ss.conductivity == pytest.approx(0.002)
        assert result.tt.conductivity == pytest.approx(0.002)

    def test_static_medium2d_conductance_is_not_length_scaled(self, inject_mock_tidy3d):
        # Sheet conductivity is a surface conductance [S] (unit-system independent),
        # so unlike the volumetric sigma_e path it must NOT be divided by length_scale.
        td = inject_mock_tidy3d
        medium = mw.Medium2D(sigma_s=0.002)
        result = _convert_material(medium, td, 1.0e6)
        assert result.ss.conductivity == pytest.approx(0.002)

    def test_zero_conductivity_medium2d_carries_no_conductivity_kwarg(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.Medium2D(sigma_s=0.0)
        result = _convert_material(medium, td)
        assert isinstance(result, td.Medium2D)
        assert not hasattr(result.ss, "conductivity")

    def test_graphene_intraband_exports_medium2d_with_drude_sheet(self, inject_mock_tidy3d):
        # Graphene's intraband Kubo channel is a single Drude sheet term, so the
        # tangential surface medium is a Tidy3D Drude with one pole.
        td = inject_mock_tidy3d
        medium = mw.Graphene(chemical_potential=0.4, scattering_time=1.0e-13, include_interband=False)
        result = _convert_material(medium, td)
        assert isinstance(result, td.Medium2D)
        assert isinstance(result.ss, td.Drude)
        assert len(result.ss.coeffs) == 1
        # tangential-isotropic: both faces share the same surface medium.
        assert result.tt is result.ss

    def test_lossy_metal_exports_lossy_metal_medium(self, inject_mock_tidy3d):
        # LossyMetalMedium exports as a Tidy3D LossyMetalMedium; the SI bulk
        # conductivity [S/m] becomes [S/um] via the length scale, exactly like the
        # volumetric sigma_e path.
        td = inject_mock_tidy3d
        length_scale = 1.0e6
        medium = mw.LossyMetalMedium(conductivity=1.0e7)
        result = _convert_material(medium, td, length_scale, frequencies=(1.0e10,))
        assert isinstance(result, td.LossyMetalMedium)
        assert result.conductivity == pytest.approx(1.0e7 / length_scale)
        # A single operating frequency is widened into a non-degenerate fit band
        # that still contains it.
        lo, hi = result.frequency_range
        assert lo < 1.0e10 < hi

    def test_lossy_metal_frequency_range_spans_export_frequencies(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.LossyMetalMedium(conductivity=1.0e7)
        result = _convert_material(medium, td, 1.0e6, frequencies=(8.0e9, 1.2e10))
        assert result.frequency_range == pytest.approx((8.0e9, 1.2e10))

    def test_lossy_metal_without_frequencies_raises(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        medium = mw.LossyMetalMedium(conductivity=1.0e7)
        with pytest.raises(ValueError, match="frequenc"):
            _convert_material(medium, td)

    def test_medium2d_structure_scene_exports_medium2d(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = (
            mw.Scene(
                domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
                grid=mw.GridSpec.uniform(0.05),
                boundary=mw.BoundarySpec.pml(num_layers=10),
                device="cpu",
            )
            .add_structure(
                mw.Structure(mw.Box(position=(0, 0, 0), size=(1.0, 1.0, 0.0)), mw.Medium2D(sigma_s=0.002))
            )
            .add_source(mw.PointDipole(position=(0, 0, 0.3), polarization="Ex", source_time=mw.CW(frequency=1e13)))
        )
        sim = scene.to_tidy3d(frequencies=1e13)
        assert len(sim.structures) == 1
        assert isinstance(sim.structures[0].medium, td.Medium2D)

    def test_lossy_metal_structure_scene_exports_lossy_metal(self, inject_mock_tidy3d):
        # The scene-level export threads the frequencies to the LossyMetal surface-
        # impedance fit; without them the metal export would raise.
        td = inject_mock_tidy3d
        scene = (
            mw.Scene(
                domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
                grid=mw.GridSpec.uniform(0.05),
                boundary=mw.BoundarySpec.pml(num_layers=10),
                device="cpu",
            )
            .add_structure(
                mw.Structure(mw.Box(position=(0, 0, -0.9), size=(2.0, 2.0, 0.2)), mw.LossyMetalMedium(conductivity=1.0e7))
            )
            .add_source(mw.PointDipole(position=(0, 0, 0.3), polarization="Ex", source_time=mw.CW(frequency=1e10)))
        )
        sim = scene.to_tidy3d(frequencies=1e10)
        assert len(sim.structures) == 1
        assert isinstance(sim.structures[0].medium, td.LossyMetalMedium)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the sheet-conductivity convention check",
    )
    def test_static_medium2d_sheet_conductivity_matches_real_tidy3d(self):
        # Physical-convention check: the exported Medium2D's complex surface
        # conductivity sigma_model(omega) must equal maxwell's own sheet_conductivity
        # at every frequency (a surface conductance in siemens, no length scaling).
        with _real_tidy3d() as real_td:
            medium = mw.Medium2D(sigma_s=0.002)
            result = _convert_material(medium, real_td)
            assert isinstance(result, real_td.Medium2D)
            for freq in (1.0e12, 5.0e12):
                got = complex(result.sigma_model(freq))
                ref = complex(medium.sheet_conductivity_at_freq(freq))
                assert got.real == pytest.approx(ref.real, rel=1e-6, abs=1e-12)
                assert got.imag == pytest.approx(ref.imag, rel=1e-6, abs=1e-12)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the graphene intraband convention check",
    )
    def test_graphene_intraband_sheet_conductivity_matches_real_tidy3d(self):
        # The intraband Drude sheet term must reproduce maxwell's frequency-dependent
        # sheet conductivity (magnitude AND the inductive Im(sigma) > 0 reactance).
        with _real_tidy3d() as real_td:
            medium = mw.Graphene(chemical_potential=0.4, scattering_time=1.0e-13, include_interband=False)
            result = _convert_material(medium, real_td)
            for freq in (1.0e12, 5.0e12, 2.0e13):
                got = complex(result.sigma_model(freq))
                ref = complex(medium.sheet_conductivity_at_freq(freq))
                assert got.real == pytest.approx(ref.real, rel=1e-6)
                assert got.imag == pytest.approx(ref.imag, rel=1e-6)
            # graphene intraband is inductive: positive imaginary conductivity.
            assert complex(result.sigma_model(5.0e12)).imag > 0.0

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the graphene interband convention check",
    )
    def test_graphene_interband_sheet_conductivity_matches_real_tidy3d(self):
        # The fitted interband Lorentz sheet terms fold into a single PoleResidue whose
        # sigma_model reproduces maxwell's total (intraband + interband) sheet
        # conductivity across the optical band, including the below-edge capacitive
        # (Im(sigma) < 0) reactance a Drude term cannot represent.
        import math as _math

        with _real_tidy3d() as real_td:
            medium = mw.Graphene(
                chemical_potential=0.4,
                scattering_time=1.0e-13,
                temperature=300.0,
                include_interband=True,
            )
            result = _convert_material(medium, real_td)
            assert isinstance(result, real_td.Medium2D)
            edge = 2.0 * 0.4 * 1.602176634e-19 / 1.054571817e-34 / (2.0 * _math.pi)
            for freq in (0.2 * edge, 0.5 * edge, 0.9 * edge):
                got = complex(result.sigma_model(freq))
                ref = complex(medium.sheet_conductivity_at_freq(freq))
                assert got.real == pytest.approx(ref.real, rel=1e-6)
                assert got.imag == pytest.approx(ref.imag, rel=1e-6)
            # Below the edge the interband channel makes the reactance capacitive.
            assert complex(result.sigma_model(0.9 * edge)).imag < 0.0

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the surface-impedance convention check",
    )
    def test_lossy_metal_surface_impedance_matches_real_tidy3d(self):
        # Physical-convention check: the exported LossyMetalMedium must reproduce
        # maxwell's Leontovich surface impedance Z_s = (1 - i)*sqrt(w*mu0/(2*sigma))
        # [ohm] (ohms are unit-system independent). The imaginary part must be
        # negative (inductive loss under e^{-i w t}), not positive (which would be
        # gain), confirming the conductivity is not misscaled into a different regime.
        import numpy as _np

        freq = 1.0e10
        with _real_tidy3d() as real_td:
            medium = mw.LossyMetalMedium(conductivity=1.0e7)
            result = _convert_material(medium, real_td, frequencies=(freq,))
            assert isinstance(result, real_td.LossyMetalMedium)
            z_td = complex(result.surface_impedance(_np.array([freq]))[0])
            z_mx = complex(medium.surface_impedance_at_freq(freq))
            assert z_td.real == pytest.approx(z_mx.real, rel=1e-3)
            assert z_td.imag == pytest.approx(z_mx.imag, rel=1e-3)
            assert z_td.real > 0.0 and z_td.imag < 0.0


class TestCustomPoleAndPerturbationConversion:
    def test_uniform_custom_lorentz_pole_exports_as_scalar_lorentz(self, inject_mock_tidy3d):
        # A spatially-uniform CustomLorentzPole lowers to its scalar reference pole
        # (peak == the uniform value), so the export is the ordinary td.Lorentz.
        import torch

        td = inject_mock_tidy3d
        delta = torch.full((3, 3, 3), 2.0)
        medium = mw.Material(
            eps_r=1.5,
            lorentz_poles=(mw.CustomLorentzPole(delta_eps=delta, resonance_frequency=5e14, gamma=1e13),),
        )
        result = _convert_material(medium, td)
        assert isinstance(result, td.Lorentz)
        assert result.eps_inf == pytest.approx(1.5)
        assert result.coeffs[0] == pytest.approx((2.0, 5e14, 0.5e13))

    def test_uniform_custom_drude_pole_exports_as_scalar_drude(self, inject_mock_tidy3d):
        import torch

        td = inject_mock_tidy3d
        plasma = torch.full((4, 4, 4), 2e15)
        medium = mw.Material(drude_poles=(mw.CustomDrudePole(plasma_frequency=plasma, gamma=1e13),))
        result = _convert_material(medium, td)
        assert isinstance(result, td.Drude)
        assert result.coeffs[0] == pytest.approx((2e15, 1e13))

    def test_uniform_custom_debye_pole_exports_as_scalar_debye(self, inject_mock_tidy3d):
        import torch

        td = inject_mock_tidy3d
        delta = torch.full((2, 2, 2), 1.0)
        medium = mw.Material(eps_r=2.0, debye_poles=(mw.CustomDebyePole(delta_eps=delta, tau=1e-12),))
        result = _convert_material(medium, td)
        assert isinstance(result, td.Debye)
        assert result.coeffs[0] == pytest.approx((1.0, 2.0 * math.pi * 1e-12))

    def test_spatially_varying_custom_pole_is_rejected(self, inject_mock_tidy3d):
        # A per-cell strength profile is defined relative to the owning structure's
        # Box, which the material-level export cannot resolve into Tidy3D absolute
        # coordinates, so it must raise rather than silently collapse to the peak.
        import torch

        td = inject_mock_tidy3d
        delta = torch.linspace(1.0, 2.0, 8).reshape(2, 2, 2)
        medium = mw.Material(
            lorentz_poles=(mw.CustomLorentzPole(delta_eps=delta, resonance_frequency=5e14, gamma=1e13),),
        )
        with pytest.raises(NotImplementedError, match="spatially-varying custom dispersive pole"):
            _convert_material(medium, td)

    def test_uniform_perturbation_exports_shifted_medium(self, inject_mock_tidy3d):
        # A spatially-uniform perturbation is a constant background shift
        # eps_base + eps_sensitivity * value = 2.0 + 2.0 * 0.5 = 3.0.
        import torch

        td = inject_mock_tidy3d
        pert = torch.full((3, 3, 3), 0.5)
        medium = mw.PerturbationMedium(mw.Material(eps_r=2.0), perturbation=pert, eps_sensitivity=2.0)
        result = _convert_material(medium, td)
        assert isinstance(result, td.Medium)
        assert result.permittivity == pytest.approx(3.0)

    def test_uniform_perturbation_on_diagonal_base_shifts_each_axis(self, inject_mock_tidy3d):
        # The isotropic shift adds to every principal axis of a DiagonalTensor3 base.
        import torch

        td = inject_mock_tidy3d
        pert = torch.full((3, 3, 3), 1.0)
        base = mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0))
        medium = mw.PerturbationMedium(base, perturbation=pert, eps_sensitivity=0.5)
        result = _convert_material(medium, td)
        assert isinstance(result, td.AnisotropicMedium)
        assert result.xx.permittivity == pytest.approx(2.5)
        assert result.yy.permittivity == pytest.approx(3.5)
        assert result.zz.permittivity == pytest.approx(4.5)

    def test_uniform_perturbation_preserves_base_dispersion(self, inject_mock_tidy3d):
        # The perturbation shifts only eps_inf; the base poles ride through unchanged.
        import torch

        td = inject_mock_tidy3d
        pert = torch.full((2, 2, 2), 1.0)
        base = mw.Material.lorentz(eps_inf=2.0, delta_eps=1.5, resonance_frequency=5e14, gamma=1e13)
        medium = mw.PerturbationMedium(base, perturbation=pert, eps_sensitivity=0.5)
        result = _convert_material(medium, td)
        assert isinstance(result, td.Lorentz)
        assert result.eps_inf == pytest.approx(2.5)
        assert result.coeffs[0] == pytest.approx((1.5, 5e14, 0.5e13))

    def test_spatially_varying_perturbation_is_rejected(self, inject_mock_tidy3d):
        import torch

        td = inject_mock_tidy3d
        pert = torch.linspace(0.1, 1.0, 8).reshape(2, 2, 2)
        medium = mw.PerturbationMedium(mw.Material(eps_r=2.0), perturbation=pert)
        with pytest.raises(NotImplementedError, match="spatially-varying PerturbationMedium"):
            _convert_material(medium, td)

    def test_perturbation_structure_scene_exports_shifted_medium(self, inject_mock_tidy3d):
        import torch

        td = inject_mock_tidy3d
        pert = torch.full((3, 3, 3), 0.5)
        scene = (
            mw.Scene(
                domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
                grid=mw.GridSpec.uniform(0.05),
                boundary=mw.BoundarySpec.pml(num_layers=10),
                device="cpu",
            )
            .add_structure(
                mw.Structure(
                    mw.Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)),
                    mw.PerturbationMedium(mw.Material(eps_r=2.0), perturbation=pert, eps_sensitivity=2.0),
                )
            )
            .add_source(mw.PointDipole(position=(0, 0, 0.4), polarization="Ex", source_time=mw.CW(frequency=1e13)))
        )
        sim = scene.to_tidy3d(frequencies=1e13)
        assert len(sim.structures) == 1
        medium = sim.structures[0].medium
        assert isinstance(medium, td.Medium)
        assert medium.permittivity == pytest.approx(3.0)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the custom-pole convention check",
    )
    def test_uniform_custom_drude_pole_matches_scalar_permittivity_real_tidy3d(self):
        # Physical-convention check: the uniform custom pole exports to the same
        # td.Drude as the equivalent scalar pole, so its eps_model reproduces the
        # scalar Material.relative_permittivity (Drude's Tidy3D convention coincides
        # with maxwell's, so the full complex value is asserted). The custom material's
        # own relative_permittivity is undefined, so the scalar twin is the reference.
        import torch

        freq = 2.0e14
        with _real_tidy3d() as real_td:
            plasma = torch.full((3, 3, 3), 2e15)
            custom = mw.Material(eps_r=1.0, drude_poles=(mw.CustomDrudePole(plasma_frequency=plasma, gamma=1e13),))
            scalar = mw.Material.drude(eps_inf=1.0, plasma_frequency=2e15, gamma=1e13)
            medium = _convert_material(custom, real_td)
            assert isinstance(medium, real_td.Drude)
            eps = complex(medium.eps_model(freq))
            ref = scalar.relative_permittivity(freq)
            assert eps.real == pytest.approx(ref.real, rel=1e-6)
            assert eps.imag == pytest.approx(ref.imag, rel=1e-6)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the perturbation-shift convention check",
    )
    def test_uniform_perturbation_reproduces_shifted_permittivity_real_tidy3d(self):
        # Physical-convention check: the exported permittivity is exactly
        # eps_base + eps_sensitivity * value, matching Material(eps_r=shifted). The
        # difference from the unperturbed base is exactly eps_sensitivity * value, so
        # the perturbation is provably not dropped.
        import torch

        freq = 2.0e14
        base_eps, sensitivity, value = 2.0, 2.0, 0.5
        shifted = base_eps + sensitivity * value
        with _real_tidy3d() as real_td:
            pert = torch.full((3, 3, 3), value)
            medium = mw.PerturbationMedium(mw.Material(eps_r=base_eps), perturbation=pert, eps_sensitivity=sensitivity)
            exported = _convert_material(medium, real_td)
            assert isinstance(exported, real_td.Medium)
            eps = complex(exported.eps_model(freq))
            ref = mw.Material(eps_r=shifted).relative_permittivity(freq)
            assert eps.real == pytest.approx(shifted)
            assert eps.real == pytest.approx(ref.real, rel=1e-6)
            base_eps_model = complex(_convert_material(mw.Material(eps_r=base_eps), real_td).eps_model(freq))
            assert eps.real - base_eps_model.real == pytest.approx(sensitivity * value)


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

    @pytest.mark.parametrize(
        ("axis", "axis_idx", "expected_center"),
        [
            ("x", 0, (12.0, 4.0, 6.0)),
            ("y", 1, (2.0, 14.0, 6.0)),
            ("z", 2, (2.0, 4.0, 16.0)),
        ],
    )
    def test_cone_preserves_apex_and_axis(self, inject_mock_tidy3d, axis, axis_idx, expected_center):
        td = inject_mock_tidy3d
        geom = mw.Cone(position=(1.0, 2.0, 3.0), radius=0.5, height=10.0, axis=axis)

        result = _convert_geometry(geom, td, 2.0)

        assert isinstance(result, td.Cylinder)
        assert result.center == pytest.approx(expected_center)
        assert result.radius == pytest.approx(1.0)
        assert result.length == pytest.approx(20.0)
        assert result.axis == axis_idx
        assert result.sidewall_angle == pytest.approx(-math.atan2(0.5, 10.0))
        assert result.reference_plane == "top"

    def test_rotated_cone_exports_as_triangle_mesh(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        geom = mw.Cone(
            position=(0.0, 0.0, 0.0),
            radius=0.5,
            height=1.0,
            axis="z",
            rotation=(0.9238795, 0.0, 0.3826834, 0.0),
        )

        result = _convert_geometry(geom, td, 1.0)

        assert isinstance(result, td.TriangleMesh)

    def test_torus_exports_as_triangle_mesh(self, inject_mock_tidy3d):
        # A torus has no analytic Tidy3D primitive, so it tessellates to a TriangleMesh
        # built from the primitive's own surface mesh (vertices scaled to Tidy3D units).
        td = inject_mock_tidy3d
        geom = mw.Torus(position=(0, 0, 0), major_radius=1.0, minor_radius=0.3)
        result = _convert_geometry(geom, td, 2.0)
        assert isinstance(result, td.TriangleMesh)
        # Faces are an integer connectivity table (never length-scaled); vertices are
        # scaled by the metre->Tidy3D length factor.
        assert result.faces.shape[1] == 3
        expected_vertices, _ = geom.to_mesh()
        assert result.vertices.shape == tuple(expected_vertices.shape)
        assert result.vertices.max() == pytest.approx(float(expected_vertices.max()) * 2.0, rel=1e-5)

    @pytest.mark.parametrize(
        "geom",
        [
            mw.Pyramid(position=(0, 0, 0), base_size=0.6, height=0.5),
            mw.HollowBox(position=(0, 0, 0), outer_size=(0.6, 0.6, 0.6), inner_size=(0.4, 0.4, 0.4)),
            mw.Prism(position=(0, 0, 0), radius=0.3, height=0.4, num_sides=6),
            mw.Ellipsoid(position=(0, 0, 0), radii=(0.3, 0.2, 0.15)),
        ],
    )
    def test_mesh_only_primitives_export_as_triangle_mesh(self, inject_mock_tidy3d, geom):
        td = inject_mock_tidy3d
        result = _convert_geometry(geom, td, 1.0)
        assert isinstance(result, td.TriangleMesh)

    def test_isotropic_ellipsoid_stays_analytic_sphere(self, inject_mock_tidy3d):
        # Radii equal on all axes: prefer Tidy3D's exact analytic Sphere over a mesh.
        td = inject_mock_tidy3d
        geom = mw.Ellipsoid(position=(1, 0, 0), radii=(0.5, 0.5, 0.5))
        result = _convert_geometry(geom, td, 1.0)
        assert isinstance(result, td.Sphere)
        assert result.radius == pytest.approx(0.5)

    def test_simple_poly_slab_exports_as_poly_slab(self, inject_mock_tidy3d):
        # An un-rotated, origin-centred, vertical PolySlab maps to the faithful Tidy3D
        # PolySlab primitive: polygon vertices and axial slab_bounds scale by length_scale.
        td = inject_mock_tidy3d
        geom = mw.PolySlab(
            vertices=[(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)],
            bounds=(-0.1, 0.2),
            axis="z",
        )
        result = _convert_geometry(geom, td, 10.0)
        assert isinstance(result, td.PolySlab)
        assert result.axis == 2
        assert result.slab_bounds == pytest.approx((-1.0, 2.0))
        assert result.vertices[0] == pytest.approx((-3.0, -3.0))
        assert result.reference_plane == "middle"

    def test_tapered_poly_slab_falls_back_to_triangle_mesh(self, inject_mock_tidy3d):
        # A sidewall taper (or rotation/offset) is baked into a TriangleMesh because Tidy3D's
        # PolySlab primitive only maps exactly for the vertical, untransformed case.
        td = inject_mock_tidy3d
        geom = mw.PolySlab(
            vertices=[(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3)],
            bounds=(-0.1, 0.1),
            axis="z",
            sidewall_angle=0.15,
        )
        result = _convert_geometry(geom, td, 1.0)
        assert isinstance(result, td.TriangleMesh)


class TestSourceConversion:
    def test_point_dipole(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))), device="cpu")
        src = mw.PointDipole(
            position=(0, 0, 0),
            polarization="Ez",
            source_time=mw.CW(frequency=1e9),
        )
        result = _convert_source(src, scene, td, 2.0)
        assert isinstance(result, td.PointDipole)
        assert result.center == (0, 0, 0)
        assert result.polarization == "Ez"
        assert result.source_time.amplitude == pytest.approx(2.0)

    def test_plane_wave(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-5, 5), (-5, 5), (-5, 5))), device="cpu")
        src = mw.PlaneWave(
            direction=(0, 0, 1),
            polarization=(1, 0, 0),
            source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13),
        )
        result = _convert_source(src, scene, td, 2.0)
        assert isinstance(result, td.PlaneWave)
        assert result.direction == "+"
        assert result.source_time.amplitude == pytest.approx(1.0)
        # Source plane should be zero-thickness along z.
        assert result.size[2] == 0.0

    def test_plane_wave_exports_requested_polarization_angle(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-5, 5), (-5, 5), (-5, 5))), device="cpu")
        polarization = (0.9436283192, 0.3310069414, 0.0)
        src = mw.PlaneWave(
            direction=(0, 0, 1),
            polarization=polarization,
            source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13),
        )

        result = _convert_source(src, scene, td, 1.0)

        assert result.pol_angle == pytest.approx(math.atan2(polarization[1], polarization[0]))
        assert result.angle_theta == pytest.approx(0.0)
        assert result.angle_phi == pytest.approx(0.0)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for source-vector convention checks",
    )
    @pytest.mark.parametrize(
        ("direction", "polarization"),
        (
            ((0.0, 0.0, 1.0), (0.9436283192, 0.3310069414, 0.0)),
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((0.3, -0.9, 0.3), (2.0**-0.5, 0.0, -(2.0**-0.5))),
        ),
    )
    def test_plane_wave_vectors_match_real_tidy3d(self, direction, polarization):
        with _real_tidy3d() as real_td:
            scene = mw.Scene(
                domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))),
                device="cpu",
            )
            source = mw.PlaneWave(
                direction=direction,
                polarization=polarization,
                source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13),
            )

            result = _convert_source(source, scene, real_td, 1.0)

            np.testing.assert_allclose(result._dir_vector, source.direction, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                result._pol_vector, source.polarization, rtol=1e-12, atol=1e-12
            )

    def test_mode_source(self, inject_mock_tidy3d):
        # A ModeSource exports as a Tidy3D ModeSource carrying both polarization
        # candidates through the requested order; the plane is zero-thickness normally.
        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))), device="cpu")
        src = mw.ModeSource(
            position=(0, 0, 0.5), size=(1.0, 1.0, 0.0), mode_index=1, direction="-",
            source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13), name="p0",
        )
        result = _convert_source(src, scene, td, 2.0)
        assert isinstance(result, td.ModeSource)
        assert result.direction == "-"
        assert result.mode_index == 1
        assert isinstance(result.mode_spec, td.ModeSpec)
        assert result.mode_spec.num_modes == 10
        assert result.mode_spec.sort_spec.filter_key == "TM_fraction"
        assert result.mode_spec.sort_spec.filter_reference == pytest.approx(0.5)
        assert result.mode_spec.sort_spec.filter_order == "over"
        assert result.center == pytest.approx((0.0, 0.0, 1.0))
        assert result.size[2] == 0.0

        te_result = _convert_source(
            mw.ModeSource(
                position=(0, 0, 0.5),
                size=(1.0, 1.0, 0.0),
                polarization="Ex",
                source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13),
            ),
            scene,
            td,
            2.0,
        )
        assert te_result.mode_spec.sort_spec.filter_key == "TE_fraction"

    def test_astigmatic_gaussian_beam(self, inject_mock_tidy3d):
        # AstigmaticGaussianBeam maps to Tidy3D's AstigmaticGaussianBeam with per-axis
        # waist sizes and per-axis waist distances, all length-scaled.
        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))), device="cpu")
        src = mw.AstigmaticGaussianBeam(
            direction=(0, 0, 1), polarization=(1, 0, 0), beam_waist=(0.5, 0.6),
            focus=(0, 0, 0.2), focus_u=0.1, focus_v=0.15,
            source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13),
        )
        result = _convert_source(src, scene, td, 2.0)
        assert isinstance(result, td.AstigmaticGaussianBeam)
        assert result.source_time.amplitude == pytest.approx(1.0)
        assert result.waist_sizes == pytest.approx((1.0, 1.2))
        source_axis_parameter = result.center[2] / 2.0 - src.focus[2]
        assert result.waist_distances == pytest.approx(
            ((source_axis_parameter - src.focus_u) * 2.0, (source_axis_parameter - src.focus_v) * 2.0)
        )
        assert result.center[2] < 0.0
        assert result.size[2] == 0.0

    def test_uniform_current_source(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))), device="cpu")
        src = mw.UniformCurrentSource(
            size=(0.4, 0.4, 0.4), polarization="Ez", center=(0.1, 0.0, 0.0),
            source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13),
        )
        result = _convert_source(src, scene, td, 2.0)
        assert isinstance(result, td.UniformCurrentSource)
        assert result.polarization == "Ez"
        assert result.center == pytest.approx((0.2, 0.0, 0.0))
        assert result.size == pytest.approx((0.8, 0.8, 0.8))
        assert result.source_time.amplitude == pytest.approx(0.25)

    def test_tfsf_box_injection_exports_tfsf(self, inject_mock_tidy3d):
        # A plane wave injected as a TFSF box exports a Tidy3D TFSF volume source whose box
        # is the total-field region and whose injection axis is the propagation axis.
        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))), device="cpu")
        src = mw.PlaneWave(
            direction=(0, 0, 1), polarization=(1, 0, 0),
            source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13),
            injection=mw.TFSF(((-1, 1), (-1, 1), (-0.5, 0.5))),
        )
        result = _convert_source(src, scene, td, 2.0)
        assert isinstance(result, td.TFSF)
        assert result.injection_axis == 2
        assert result.direction == "+"
        expected_amplitude = 1.0 / (
            2.0 * math.sqrt(2.0 / (299_792_458.0 * 8.8541878128e-12))
        )
        assert result.source_time.amplitude == pytest.approx(expected_amplitude)
        assert result.center == pytest.approx((0.0, 0.0, 0.0))
        assert result.size == pytest.approx((4.0, 4.0, 2.0))

    def test_tfsf_slab_injection_is_infinite_in_transverse(self, inject_mock_tidy3d):
        # A slab-mode TFSF bounds only the propagation axis; the transverse extent is td.inf.
        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))), device="cpu")
        src = mw.PlaneWave(
            direction=(0, 0, -1), polarization=(1, 0, 0),
            source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13),
            injection=mw.TFSF.slab(axis="z", bounds=(-1.0, 1.0)),
        )
        result = _convert_source(src, scene, td, 1.0)
        assert isinstance(result, td.TFSF)
        assert result.injection_axis == 2
        assert result.direction == "-"
        expected_amplitude = 1.0 / (
            math.sqrt(2.0 / (299_792_458.0 * 8.8541878128e-12))
        )
        assert result.source_time.amplitude == pytest.approx(expected_amplitude)
        assert result.size[0] == td.inf and result.size[1] == td.inf
        assert result.size[2] == pytest.approx(2.0)

    def test_custom_field_source_dataset_is_relative_to_center(self, inject_mock_tidy3d):
        # Tidy3D custom-source datasets carry coordinates RELATIVE to the source center,
        # a single injection frequency, and identity E/H component names.
        import numpy as _np

        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))), device="cpu")
        x = _np.array([-0.5, 0.0, 0.5]); y = _np.array([-0.5, 0.5]); z = _np.array([0.2])
        ds = mw.FieldDataset((x, y, z), {"Ex": _np.ones((3, 2, 1)), "Hy": _np.full((3, 2, 1), 0.3)})
        src = mw.CustomFieldSource(ds, source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13))
        result = _convert_source(src, scene, td, 2.0)
        assert isinstance(result, td.CustomFieldSource)
        # Center is the midpoint of each coordinate span, scaled to Tidy3D units.
        assert result.center == pytest.approx((0.0, 0.0, 0.4))
        # Spans: x,y are 1.0 wide (x2 = 2.0 um); z is a single sample (zero-thickness plane).
        assert result.size == pytest.approx((2.0, 2.0, 0.0))
        field = result.field_dataset
        assert set(field._kwargs) == {"Ex", "Hy"}
        # x-coordinates are (coord - center) * length_scale = [-1, 0, 1] um.
        assert list(field.Ex.coords["x"]) == pytest.approx([-1.0, 0.0, 1.0])
        assert list(field.Ex.coords["f"]) == pytest.approx([3e14])
        assert result.source_time.amplitude == pytest.approx(0.5)

    def test_custom_current_source_maps_j_and_m_to_e_and_h(self, inject_mock_tidy3d):
        # Tidy3D's CustomCurrentSource reuses a FieldDataset whose E/H slots carry the
        # electric (J) / magnetic (M) currents: Jz -> Ez, Mx -> Hx.
        import numpy as _np

        td = inject_mock_tidy3d
        scene = mw.Scene(domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))), device="cpu")
        x = _np.array([-0.5, 0.5]); y = _np.array([-0.5, 0.5]); z = _np.array([-0.1, 0.1])
        ds = mw.CurrentDataset((x, y, z), {"Jz": _np.ones((2, 2, 2)), "Mx": _np.full((2, 2, 2), 0.3)})
        src = mw.CustomCurrentSource(ds, source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13))
        result = _convert_source(src, scene, td, 2.0)
        assert isinstance(result, td.CustomCurrentSource)
        assert set(result.current_dataset._kwargs) == {"Ez", "Hx"}
        assert result.current_dataset.Ez.data.reshape(-1)[0].real == pytest.approx(1.0)
        assert result.current_dataset.Hx.data.reshape(-1)[0].real == pytest.approx(0.3)
        assert result.source_time.amplitude == pytest.approx(0.25)


class TestSourceTimeConversion:
    def test_cw(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        st = mw.CW(frequency=1e9, amplitude=2.0, phase=0.5)
        result = _convert_source_time(st, td)
        assert isinstance(result, td.ContinuousWave)
        assert result.freq0 == 1e9
        assert result.fwidth == pytest.approx(1e8)
        assert result.amplitude == 2.0
        assert result.phase == 0.5

    def test_gaussian_pulse(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        st = mw.GaussianPulse(frequency=3e14, fwidth=1e13, phase=0.37)
        result = _convert_source_time(st, td)
        assert isinstance(result, td.GaussianPulse)
        assert result.freq0 == 3e14
        assert result.fwidth == 1e13
        assert result.offset == pytest.approx(st.delay / st.sigma_t)
        expected_phase = math.remainder(
            2.0 * math.pi * st.frequency * st.delay - st.phase - 0.5 * math.pi,
            2.0 * math.pi,
        )
        assert result.phase == pytest.approx(expected_phase)
        assert result.remove_dc_component is False

        times = np.linspace(st.delay - 3.0 * st.sigma_t, st.delay + 3.0 * st.sigma_t, 101)
        envelope = np.exp(-0.5 * ((times - st.delay) / st.sigma_t) ** 2)
        tidy_real = np.real(
            1j
            * np.exp(1j * result.phase)
            * np.exp(-1j * 2.0 * np.pi * st.frequency * times)
        ) * envelope
        maxwell_real = np.asarray([st.evaluate(time) for time in times])
        np.testing.assert_allclose(tidy_real, maxwell_real, rtol=1e-11, atol=1e-11)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="tidy3d is not installed",
    )
    def test_gaussian_pulse_matches_real_tidy3d_time_function(self):
        st = mw.GaussianPulse(
            frequency=2.0e9,
            fwidth=0.4e9,
            amplitude=1.7,
            phase=0.37,
        )
        times = np.linspace(st.delay - 4.0 * st.sigma_t, st.delay + 4.0 * st.sigma_t, 257)

        with _real_tidy3d() as real_td:
            exported = _convert_source_time(st, real_td)
            tidy3d_real = np.real(np.asarray(exported.amp_time(times)))

        maxwell_real = np.asarray([st.evaluate(time) for time in times])
        np.testing.assert_allclose(tidy3d_real, maxwell_real, rtol=2e-12, atol=2e-12)


class TestMonitorConversion:
    def test_point_monitor(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.PointMonitor(name="probe", position=(0, 0, 0), fields=("Hz",))
        bounds = ((-1, 1), (-1, 1), (-1, 1))
        result = _convert_monitor(mon, bounds, (1e9,), td, 1.0)
        assert isinstance(result, td.FieldMonitor)
        assert result.size == (0.0, 0.0, 0.0)
        assert result.name == "probe"
        assert result.fields == ["Hz"]

    def test_plane_monitor(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.PlaneMonitor(name="field_xy", axis="z", position=0.5, fields=("Ey", "Hz"))
        bounds = ((-2, 2), (-2, 2), (-2, 2))
        result = _convert_monitor(mon, bounds, (1e9,), td, 1.0)
        assert isinstance(result, td.FieldMonitor)
        assert result.center[2] == 0.5
        assert result.size[2] == 0.0
        assert result.fields == ["Ey", "Hz"]
        assert result.colocate is True

    def test_finite_plane_monitor(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.FinitePlaneMonitor(
            name="finite_xy",
            position=(0.0, 0.0, 0.5),
            size=(1.2, 1.4, 0.0),
            fields=("Hx",),
            frequencies=(1e9,),
        )
        bounds = ((-2, 2), (-2, 2), (-2, 2))
        result = _convert_monitor(mon, bounds, None, td, 1.0)
        assert isinstance(result, td.FieldMonitor)
        assert result.center == pytest.approx((0.0, 0.0, 0.5))
        assert result.size == pytest.approx((1.2, 1.4, 0.0))
        assert result.fields == ["Hx"]
        assert result.colocate is True

    def test_flux_monitor(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.FluxMonitor(name="flux_z", axis="z", position=1.0, frequencies=(1e9, 2e9))
        bounds = ((-3, 3), (-3, 3), (-3, 3))
        result = _convert_monitor(mon, bounds, None, td, 1.0)
        assert isinstance(result, td.FluxMonitor)
        assert result.freqs == [1e9, 2e9]

    def test_mode_monitor_exports_as_mode_monitor_with_mode_spec(self, inject_mock_tidy3d):
        # A modal monitor must export as a Tidy3D ModeMonitor carrying a ModeSpec, not a
        # raw FieldMonitor: the modal decomposition is what distinguishes it, and the mode
        # solver must resolve both polarization candidates through mode_index.
        td = inject_mock_tidy3d
        mon = mw.ModeMonitor(
            name="port0",
            position=(0.0, 0.0, 0.5),
            size=(1.0, 1.0, 0.0),
            mode_index=2,
            polarization="Ey",
            frequencies=(1e9,),
        )
        bounds = ((-2, 2), (-2, 2), (-2, 2))
        result = _convert_monitor(mon, bounds, None, td, 1.0)
        assert isinstance(result, td.ModeMonitor)
        assert not isinstance(result, td.FieldMonitor)
        assert result.center[2] == 0.5
        assert result.size[2] == 0.0
        assert isinstance(result.mode_spec, td.ModeSpec)
        assert result.mode_spec.num_modes == 12
        assert result.mode_spec.sort_spec.filter_key == "TM_fraction"
        assert result.freqs == [1e9]

    def test_diffraction_monitor_exports_with_infinite_transverse_size(self, inject_mock_tidy3d):
        # Tidy3D decomposes orders over the whole periodic cell, so the transverse size
        # must be td.inf; the normal axis stays zero-thickness at the plane position.
        td = inject_mock_tidy3d
        mon = mw.DiffractionMonitor(
            name="orders", position=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.0), frequencies=(1e9,), normal_direction="+"
        )
        bounds = ((-2, 2), (-2, 2), (-2, 2))
        result = _convert_monitor(mon, bounds, None, td, 1.0)
        assert isinstance(result, td.DiffractionMonitor)
        assert result.size == (td.inf, td.inf, 0.0)
        assert result.center[2] == 0.5
        assert result.normal_dir == "+"

    def test_permittivity_monitor_exports(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.PermittivityMonitor(name="eps", position=(0.1, 0.2, 0.3), size=(1.0, 2.0, 3.0), frequencies=(1e9,))
        bounds = ((-2, 2), (-2, 2), (-2, 2))
        result = _convert_monitor(mon, bounds, None, td, 2.0)
        assert isinstance(result, td.PermittivityMonitor)
        assert result.center == pytest.approx((0.2, 0.4, 0.6))
        assert result.size == pytest.approx((2.0, 4.0, 6.0))

    def test_field_time_monitor_exports_without_frequencies(self, inject_mock_tidy3d):
        # A time-domain monitor samples the running field, so it must convert even when no
        # frequencies are supplied (unlike frequency-domain monitors).
        td = inject_mock_tidy3d
        mon = mw.FieldTimeMonitor(
            name="probe_t", components=("Ex", "Hz"), position=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
            start=10, stop=100, interval=2,
        )
        bounds = ((-2, 2), (-2, 2), (-2, 2))
        result = _convert_monitor(mon, bounds, None, td, 1.0)
        assert isinstance(result, td.FieldTimeMonitor)
        assert result.start == 10
        assert result.stop == 100
        assert result.interval == 2
        assert list(result.fields) == ["Ex", "Hz"]

    def test_flux_time_monitor_exports_as_plane(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        mon = mw.FluxTimeMonitor(name="flux_t", axis="z", position=1.0, start=0, interval=1, normal_direction="-")
        bounds = ((-2, 2), (-2, 2), (-2, 2))
        result = _convert_monitor(mon, bounds, None, td, 1.0)
        assert isinstance(result, td.FluxTimeMonitor)
        assert result.center[2] == 1.0
        assert result.size[2] == 0.0
        assert result.normal_dir == "-"

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

    def test_uniform_material_region_exports_as_homogeneous_structure(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
            grid=mw.GridSpec.uniform(0.1),
            boundary=mw.BoundarySpec.pml(),
            device="cpu",
        ).add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)),
                density=torch.full((2, 2, 2), 0.25),
                eps_bounds=(2.0, 6.0),
            )
        )

        sim = scene.to_tidy3d(frequencies=1e9)

        assert len(sim.structures) == 1
        assert isinstance(sim.structures[0].medium, td.Medium)
        assert sim.structures[0].medium.permittivity == pytest.approx(3.0)

    def test_spatial_material_region_export_fails_loudly(self):
        scene = mw.Scene(device="cpu").add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0, 0, 0), size=(0.5, 0.5, 0.5)),
                density=torch.arange(8, dtype=torch.float64).reshape(2, 2, 2),
                bounds=(0.0, 7.0),
                eps_bounds=(1.0, 4.0),
            )
        )

        with pytest.raises(NotImplementedError, match="spatially uniform"):
            scene.to_tidy3d(frequencies=1e9)

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
            sources=[
                mw.PlaneWave(
                    direction=(0, 0, 1),
                    polarization="Ex",
                    source_time=mw.CW(frequency=1e9),
                )
            ],
            symmetry=("PEC", "PMC", None),
            device="cpu",
        )
        sim = scene.to_tidy3d(frequencies=1e9)
        assert sim.symmetry == (-1, 1, 0)

    def test_localized_face_symmetry_is_rejected_as_not_center_equivalent(self):
        scene = mw.Scene(
            sources=[
                mw.PointDipole(
                    position=(0, 0, 0),
                    source_time=mw.CW(frequency=1e9),
                )
            ],
            symmetry=("PEC", None, None),
            device="cpu",
        )

        with pytest.raises(NotImplementedError, match="center-based"):
            scene.to_tidy3d(frequencies=1e9)

    def test_custom_run_time(self, inject_mock_tidy3d):
        td = inject_mock_tidy3d
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
            sources=[mw.PointDipole(position=(0, 0, 0), source_time=mw.CW(frequency=1e9))],
            device="cpu",
        )
        sim = scene.to_tidy3d(frequencies=1e9, run_time=1e-9)
        assert sim.run_time == 1e-9

    def test_scene_time_monitor_steps_are_exported_as_seconds(self):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
            grid=mw.GridSpec.uniform(0.1),
            boundary=mw.BoundarySpec.periodic(),
            sources=[
                mw.PointDipole(
                    position=(0, 0, 0),
                    source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.2e9),
                )
            ],
            monitors=[
                mw.FieldTimeMonitor(
                    "trace",
                    components=("Ez",),
                    start=2,
                    stop=6,
                    interval=2,
                )
            ],
            device="cpu",
        )

        sim = scene.to_tidy3d(frequencies=1e9, courant=0.5)

        dt = 0.5 / (299_792_458.0 * math.sqrt(3.0 / 0.1**2))
        assert sim.monitors[0].start == pytest.approx(2 * dt)
        assert sim.monitors[0].stop == pytest.approx(6 * dt)
        assert sim.monitors[0].interval == 2

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


class TestGeometrySourceMonitorRealTidy3d:
    """Correctness checks against the genuine tidy3d validators (not the mock).

    These assert the exported objects both satisfy Tidy3D's own field validators AND
    reproduce the maxwell geometry/placement, so a merely-accepted-but-wrong conversion
    (wrong axis, unscaled coordinate, absolute-instead-of-relative dataset) is caught.
    """

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the tapered Cylinder geometry check",
    )
    def test_cone_bounds_and_inside_match_real_tidy3d(self):
        from witwin.maxwell.adapters.tidy3d import _convert_geometry

        length_scale = 1.0e6
        position = (0.1, -0.2, -0.3)
        radius = 0.4
        height = 0.8
        geom = mw.Cone(position=position, radius=radius, height=height, axis="z")
        points = np.asarray(
            [
                (0.1, -0.2, 0.3),
                (0.2, -0.2, 0.3),
                (0.3, -0.2, -0.22),
                (0.1, -0.2, -0.4),
            ]
        )
        maxwell_inside = np.asarray(
            [
                bool(geom.signed_distance(*(torch.tensor(value) for value in point)) <= 0)
                for point in points
            ]
        )

        with _real_tidy3d() as real_td:
            cone = _convert_geometry(geom, real_td, length_scale)
            assert isinstance(cone, real_td.Cylinder)
            assert cone.bounds[0] == pytest.approx(
                ((position[0] - radius) * length_scale, (position[1] - radius) * length_scale, position[2] * length_scale)
            )
            assert cone.bounds[1] == pytest.approx(
                ((position[0] + radius) * length_scale, (position[1] + radius) * length_scale, (position[2] + height) * length_scale)
            )
            tidy_inside = cone.inside(*(points[:, axis] * length_scale for axis in range(3)))

        assert np.array_equal(tidy_inside, maxwell_inside)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the TriangleMesh geometry check",
    )
    def test_torus_triangle_mesh_bounds_match_scaled_extent_real_tidy3d(self):
        # A torus tessellates to a real TriangleMesh; its axial half-extent equals the
        # minor radius and its in-plane half-extent the (major + minor) radius, each scaled
        # to Tidy3D units. A wrong (unscaled) vertex conversion fails by 1e6.
        from witwin.maxwell.adapters.tidy3d import _convert_geometry

        length_scale = 1.0e6
        major, minor = 0.5, 0.15
        with _real_tidy3d() as real_td:
            geom = mw.Torus(position=(0, 0, 0), major_radius=major, minor_radius=minor, axis="z")
            mesh = _convert_geometry(geom, real_td, length_scale)
            assert isinstance(mesh, real_td.TriangleMesh)
            (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
            assert xmax == pytest.approx((major + minor) * length_scale, rel=1e-4)
            assert xmin == pytest.approx(-(major + minor) * length_scale, rel=1e-4)
            assert zmax == pytest.approx(minor * length_scale, rel=1e-4)
            assert zmin == pytest.approx(-minor * length_scale, rel=1e-4)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the PolySlab geometry check",
    )
    def test_simple_poly_slab_inside_matches_real_tidy3d(self):
        # The exported real PolySlab reproduces the maxwell cross-section: a point inside
        # the polygon (and slab bounds) is inside, one outside the polygon is outside.
        import numpy as _np

        from witwin.maxwell.adapters.tidy3d import _convert_geometry

        length_scale = 1.0e6
        with _real_tidy3d() as real_td:
            geom = mw.PolySlab(
                vertices=[(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)],
                bounds=(-0.1, 0.1),
                axis="z",
            )
            slab = _convert_geometry(geom, real_td, length_scale)
            assert isinstance(slab, real_td.PolySlab)
            inside = slab.inside(_np.array([0.0]), _np.array([0.0]), _np.array([0.0]))
            outside = slab.inside(_np.array([0.4 * length_scale]), _np.array([0.0]), _np.array([0.0]))
            assert bool(inside[0]) is True
            assert bool(outside[0]) is False

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the ModeSource/ModeMonitor check",
    )
    def test_mode_source_and_monitor_build_in_real_tidy3d(self):
        # ModeSource/ModeMonitor build under Tidy3D's own validators with a ModeSpec whose
        # num_modes covers the requested order, and the plane is normal to its zero axis.
        from witwin.maxwell.adapters.tidy3d import _convert_monitor, _convert_source

        with _real_tidy3d() as real_td:
            scene = mw.Scene(domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))), device="cpu")
            src = _convert_source(
                mw.ModeSource(
                    position=(0, 0, 0.5), size=(1.0, 1.0, 0.0), mode_index=2, direction="+",
                    source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13),
                ),
                scene, real_td, 1.0e6,
            )
            assert isinstance(src, real_td.ModeSource)
            assert src.mode_spec.num_modes == 12
            assert src.mode_spec.sort_spec.filter_key == "TM_fraction"
            assert src.mode_index == 2
            mon = _convert_monitor(
                mw.ModeMonitor(name="mm", position=(0, 0, 0.5), size=(1.0, 1.0, 0.0), mode_index=1, frequencies=(3e14,)),
                ((-2, 2), (-2, 2), (-2, 2)), None, real_td, 1.0e6,
            )
            assert isinstance(mon, real_td.ModeMonitor)
            assert mon.mode_spec.num_modes == 10
            assert mon.mode_spec.sort_spec.filter_key == "TM_fraction"

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the DiffractionMonitor validator check",
    )
    def test_diffraction_monitor_builds_in_real_tidy3d(self):
        # Tidy3D's DiffractionMonitor validator REQUIRES td.inf transverse size; the export
        # building successfully proves the transverse extent was set to inf (a finite size
        # raises inside tidy3d).
        from witwin.maxwell.adapters.tidy3d import _convert_monitor

        with _real_tidy3d() as real_td:
            mon = _convert_monitor(
                mw.DiffractionMonitor(name="d", position=(0, 0, 0.5), size=(1.0, 1.0, 0.0), frequencies=(3e14,)),
                ((-2, 2), (-2, 2), (-2, 2)), None, real_td, 1.0e6,
            )
            assert isinstance(mon, real_td.DiffractionMonitor)
            assert mon.size[0] == real_td.inf and mon.size[1] == real_td.inf
            assert mon.size[2] == 0.0

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the custom-source dataset convention check",
    )
    def test_custom_field_source_relative_coordinates_real_tidy3d(self):
        # Tidy3D custom-source datasets are RELATIVE to the source center. Adding the center
        # back to the exported relative coordinates must recover the maxwell absolute
        # coordinates scaled to um, and the tidy3d validators (single freq, interpolatable)
        # must accept the source.
        import numpy as _np

        from witwin.maxwell.adapters.tidy3d import _convert_source

        length_scale = 1.0e6
        with _real_tidy3d() as real_td:
            scene = mw.Scene(domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))), device="cpu")
            x = _np.array([-0.5, 0.0, 0.5]); y = _np.array([-0.5, 0.5]); z = _np.array([0.2])
            ds = mw.FieldDataset((x, y, z), {"Ex": _np.ones((3, 2, 1)), "Hy": _np.full((3, 2, 1), 0.3)})
            src = _convert_source(
                mw.CustomFieldSource(ds, source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13)),
                scene, real_td, length_scale,
            )
            assert isinstance(src, real_td.CustomFieldSource)
            ex = src.field_dataset.Ex
            abs_x = _np.asarray(ex.coords["x"]) + src.center[0]
            assert list(abs_x) == pytest.approx(list(x * length_scale))
            assert list(_np.asarray(ex.coords["f"])) == pytest.approx([3e14])
            # Center is the midpoint of the (single-sample) z span, scaled to um.
            assert src.center[2] == pytest.approx(0.2 * length_scale)

    @pytest.mark.skipif(
        importlib.util.find_spec("tidy3d") is None,
        reason="requires the real tidy3d package for the current-source J/M convention check",
    )
    def test_custom_current_source_j_m_map_to_e_h_real_tidy3d(self):
        # Tidy3D's CustomCurrentSource injects a FieldDataset whose E/H slots ARE the
        # electric (J) and magnetic (M) currents. The real source must carry Jz in Ez and
        # Mx in Hx with the physical dataset amplitudes, while source_time carries the
        # SI current-density conversion, and pass Tidy3D's validators.
        import numpy as _np

        from witwin.maxwell.adapters.tidy3d import _convert_source

        with _real_tidy3d() as real_td:
            scene = mw.Scene(domain=mw.Domain(bounds=((-2, 2), (-2, 2), (-2, 2))), device="cpu")
            x = _np.array([-0.5, 0.5]); y = _np.array([-0.5, 0.5]); z = _np.array([-0.1, 0.1])
            ds = mw.CurrentDataset((x, y, z), {"Jz": _np.ones((2, 2, 2)), "Mx": _np.full((2, 2, 2), 0.3)})
            src = _convert_source(
                mw.CustomCurrentSource(ds, source_time=mw.GaussianPulse(frequency=3e14, fwidth=1e13)),
                scene, real_td, 1.0e6,
            )
            assert isinstance(src, real_td.CustomCurrentSource)
            assert src.current_dataset.Ez is not None
            assert src.current_dataset.Hx is not None
            assert complex(src.current_dataset.Ez.values.reshape(-1)[0]).real == pytest.approx(1.0)
            assert complex(src.current_dataset.Hx.values.reshape(-1)[0]).real == pytest.approx(0.3)
            assert src.source_time.amplitude == pytest.approx(1.0e-12)
