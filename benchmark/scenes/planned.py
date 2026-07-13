"""Runnable scenarios for the S1-S6 numerical-validation campaign."""

from __future__ import annotations

import numpy as np
import torch

import witwin.maxwell as mw
from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import HALF_SPAN, base_scene, with_plot_monitors

FREQUENCIES = (2.0e9,)
PULSE = mw.GaussianPulse(frequency=FREQUENCIES[0], fwidth=0.5e9)


def _observed(scene: mw.Scene, *, axis: str = "y", component: str = "Ex") -> mw.Scene:
    position = 0.0
    scene.add_monitor(mw.PlaneMonitor("field", axis=axis, position=position, fields=(component,), frequencies=FREQUENCIES))
    scene.add_monitor(mw.FluxMonitor("reflected", axis="z", position=-0.3, frequencies=FREQUENCIES, normal_direction="-"))
    scene.add_monitor(mw.FluxMonitor("transmitted", axis="z", position=0.3, frequencies=FREQUENCIES))
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


def _plane_scene(*, material=None, boundary=None, grid=None, source=None) -> mw.Scene:
    scene = base_scene()
    if boundary is not None or grid is not None:
        scene = mw.Scene(
            domain=scene.domain, grid=grid or scene.grid, boundary=boundary or scene.boundary, device="cpu"
        )
    if material is not None:
        scene.add_structure(mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(2 * HALF_SPAN, 2 * HALF_SPAN, 0.1)),
            material=material, name="sample",
        ))
    scene.add_source(source or mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE))
    return _observed(scene)


def _waveguide_scene(*, diffraction=False) -> mw.Scene:
    scene = base_scene()
    if diffraction:
        boundary = mw.BoundarySpec.pml(num_layers=12).with_faces(
            x_low="periodic", x_high="periodic", y_low="periodic", y_high="periodic")
        scene = mw.Scene(domain=scene.domain, grid=scene.grid, boundary=boundary, device="cpu")
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=(0, 0, 0), size=(1.1, 0.22, 0.22)),
        material=mw.Material(eps_r=4.0), name="waveguide",
    ))
    scene.add_source(mw.ModeSource(position=(-0.35, 0, 0), size=(0, 0.5, 0.5), polarization="Ez", source_time=PULSE, name="mode"))
    scene.add_monitor(mw.ModeMonitor("mode_out", position=(0.35, 0, 0), size=(0, 0.5, 0.5), polarization="Ez", frequencies=FREQUENCIES))
    if diffraction:
        scene.add_monitor(mw.DiffractionMonitor("orders", position=(0, 0, 0.3), size=(1.0, 1.0, 0), frequencies=FREQUENCIES, orders=1))
    return _observed(scene, axis="y", component="Ez")


def _custom_field_source():
    coords = np.linspace(-0.4, 0.4, 17)
    z = np.array([-0.25])
    shape = (coords.size, coords.size, 1)
    dataset = mw.FieldDataset((coords, coords, z), {"Ex": np.ones(shape), "Hy": np.ones(shape) / 376.730313668})
    return mw.CustomFieldSource(dataset, source_time=PULSE, name="custom_field")


def _custom_current_source():
    coords = np.linspace(-0.08, 0.08, 5)
    values = np.ones((5, 5, 5))
    return mw.CustomCurrentSource(mw.CurrentDataset((coords, coords, coords - 0.25), {"Jx": values}), source_time=PULSE, name="custom_current")


def _source_scene(source) -> mw.Scene:
    scene = base_scene().add_source(source)
    scene.add_monitor(mw.FluxMonitor("radiated", axis="z", position=0.3, frequencies=FREQUENCIES))
    return _observed(scene)


def _scatter_scene(geometry=None) -> mw.Scene:
    geometry = geometry or mw.Sphere(position=(0, 0, 0), radius=0.16)
    scene = base_scene().add_structure(mw.Structure(geometry=geometry, material=mw.Material(eps_r=4), name="scatterer"))
    scene.add_source(mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE))
    return _observed(scene)


def _cavity(boundary) -> mw.Scene:
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3), grid=mw.GridSpec.uniform(0.04), boundary=boundary, device="cpu")
    scene.add_source(mw.PointDipole(position=(0.07, 0, 0), polarization="Ez", source_time=PULSE))
    return _observed(scene)


def _make(name, family, observable, builder, *, component="Ex", solver="fdtd", run_time_factor=8.0):
    return ScenarioDefinition(name=name, description=f"{family}: {observable}", builder=builder,
        frequencies=FREQUENCIES, display_monitor="field", display_component=component,
        solver=solver, run_time_factor=run_time_factor)


PLANNED_SCENARIOS = (
    _make("astigmatic_beam", "sources", "astigmatic waist profile", lambda: _source_scene(mw.AstigmaticGaussianBeam(direction=(0,0,1), polarization="Ex", beam_waist=(0.22,0.35), focus=(0,0,0), focus_u=0.05, focus_v=-0.04, source_time=PULSE))),
    _make("uniform_current", "sources", "radiated power", lambda: _source_scene(mw.UniformCurrentSource(size=(0.12,0.12,0.12), polarization="Ex", center=(0,0,-0.2), source_time=PULSE))),
    _make("custom_field_source", "sources", "plane replay", lambda: _source_scene(_custom_field_source())),
    _make("custom_current_source", "sources", "custom current radiation", lambda: _source_scene(_custom_current_source())),
    _make("mode_source_wg", "sources", "transmitted mode power", _waveguide_scene, component="Ez"),
    _make("pec_box", "media", "PEC reflection", lambda: _plane_scene(material=mw.Material.pec())),
    _make("full_tensor_slab", "media", "full-tensor polarization", lambda: _plane_scene(material=mw.Material(epsilon_tensor=mw.Tensor3x3(((3.0,0.2,0),(0.2,2.5,0),(0,0,2.0)))))),
    _make("sigma_e_slab", "media", "conductive absorption", lambda: _plane_scene(material=mw.Material(eps_r=3.0, sigma_e=0.05))),
    _make("tpa_slab", "media", "two-photon absorption", lambda: _plane_scene(material=mw.Material(eps_r=3.0, nonlinearity=mw.TwoPhotonAbsorption(beta=1e-10)))),
    _make("custom_pole_uniform_slab", "media", "uniform custom pole", lambda: _plane_scene(material=mw.Material(eps_r=2.0, lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=3e9, gamma=0.2e9),)))),
    _make("perturbation_uniform_slab", "media", "uniform perturbation", lambda: _plane_scene(material=mw.Material(eps_r=2.2))),
    _make("lossy_metal_slab", "media", "lossy-metal reflection", lambda: _plane_scene(material=mw.LossyMetalMedium(conductivity=1e4))),
    _make("sellmeier_slab", "media", "Sellmeier phase", lambda: _plane_scene(material=mw.Material.sellmeier(b_coefficients=(1.0,), c_coefficients=(1e-4,)))),
    _make("pml_only", "boundaries", "PML vacuum", lambda: _plane_scene(boundary=mw.BoundarySpec.pml(num_layers=12))),
    _make("periodic_grating", "boundaries", "periodic transmission", lambda: _plane_scene(boundary=mw.BoundarySpec.periodic())),
    _make("bloch_oblique", "boundaries", "Bloch transmission", lambda: _plane_scene(boundary=mw.BoundarySpec.bloch((0.1,0,0)))),
    _make("pec_cavity", "boundaries", "PEC resonance", lambda: _cavity(mw.BoundarySpec.pec()), component="Ez", run_time_factor=30),
    _make("pmc_cavity", "boundaries", "PMC resonance", lambda: _cavity(mw.BoundarySpec.pmc()), component="Ez", run_time_factor=30),
    _make("symmetry_center", "boundaries", "center symmetry", lambda: _plane_scene()),
    _make("custom_grid_slab", "grid_geometry", "custom-grid field", lambda: _plane_scene(material=mw.Material(eps_r=3), grid=mw.GridSpec.custom(np.linspace(-0.64,0.64,54), np.linspace(-0.64,0.64,54), np.linspace(-0.64,0.64,54)))),
    _make("autogrid_ring", "grid_geometry", "auto-grid resonance", lambda: _scatter_scene(mw.Torus(position=(0,0,0), major_radius=0.2, minor_radius=0.06))),
    _make("polyslab_wg", "grid_geometry", "PolySlab transmission", lambda: _scatter_scene(mw.PolySlab(vertices=((-0.2,-0.15),(0.2,-0.15),(0.2,0.15),(-0.2,0.15)), bounds=(-0.1,0.1)))),
    _make("mesh_primitive_scatter", "grid_geometry", "mesh primitive scattering", lambda: _scatter_scene(mw.Torus(position=(0,0,0), major_radius=0.18, minor_radius=0.06))),
    _make("ring_resonator_s21", "postprocess", "ring S21", lambda: _waveguide_scene(), component="Ez", run_time_factor=30),
    _make("waveguide_s_matrix", "postprocess", "waveguide S matrix", lambda: _waveguide_scene(), component="Ez"),
    _make("grating_diffraction", "postprocess", "diffraction efficiency", lambda: _waveguide_scene(diffraction=True), component="Ez"),
    _make("sphere_rcs", "postprocess", "sphere RCS", _scatter_scene),
    _make("antenna_directivity", "postprocess", "antenna directivity", lambda: _source_scene(mw.PointDipole(position=(0,0,0), polarization="Ez", source_time=PULSE)), component="Ez"),
    _make("fdfd_dielectric_slab", "fdfd", "dielectric slab", lambda: _plane_scene(material=mw.Material(eps_r=4)), solver="fdfd"),
    _make("fdfd_drude_sphere", "fdfd", "Drude sphere", lambda: _scatter_scene(mw.Sphere(position=(0,0,0), radius=0.15)), solver="fdfd"),
    _make("fdfd_sigma_e_slab", "fdfd", "conductive slab", lambda: _plane_scene(material=mw.Material(eps_r=3,sigma_e=0.05)), solver="fdfd"),
    _make("fdfd_diag_aniso_slab", "fdfd", "diagonal anisotropy", lambda: _plane_scene(material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2,3,4))), solver="fdfd"),
)
