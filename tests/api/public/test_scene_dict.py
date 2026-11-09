"""Acceptance contracts for fixed declarative scene dictionaries."""

from dataclasses import fields
from inspect import signature
import json
import math

import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.core.geometry.serialize import _GEOMETRY_TYPES

from witwin.maxwell.constants import C_0, EPSILON_0
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.scene_dict import _Reader, _TYPES, _Writer


FREQUENCY = 1e9
WAVELENGTH = C_0 / FREQUENCY


def _monitors():
    return [
        mw.PointMonitor("point", (0., 0., .01)),
        mw.PlaneMonitor("plane", axis="x", position=.01, fields=("Ey", "Hz")),
        mw.FinitePlaneMonitor("finite", position=(0., 0., .01), size=(.02, .02, 0.)),
        mw.FluxMonitor("flux", axis="z", position=.015),
        mw.ClosedSurfaceMonitor.box("closed", size=(.04,) * 3),
        mw.AngleProjectionMonitor("angle", size=(.04,) * 3, theta=(.4,), phi=(.2,), radius=1.),
        mw.CartesianProjectionMonitor("cartesian", size=(.04,) * 3, x=(0.,), y=(0.,), distance=1.),
        mw.KSpaceProjectionMonitor("kspace", size=(.04,) * 3, ux=(0.,), uy=(0.,)),
        mw.PermittivityMonitor("eps", size=(.02,) * 3),
    ]


def _scene_a():
    lam = WAVELENGTH
    return mw.Scene(
        domain=mw.Domain(((-1.5 * lam, 1.5 * lam),) * 3),
        grid=mw.GridSpec.uniform(lam / 16), boundary=mw.BoundarySpec.pml(num_layers=8),
        structures=[
            mw.Structure(mw.Sphere(position=(.3 * lam, .1 * lam, .1 * lam), radius=.4 * lam, device="cuda"),
                         mw.Material(eps_r=4)),
            mw.Structure(mw.Box(position=(-.5 * lam, .5 * lam, .2 * lam),
                                size=(.44 * lam, .6 * lam, .36 * lam), rotation=(.21, -.32, .43), device="cuda"),
                         mw.Material(eps_r=6, sigma_e=.2 * 2 * math.pi * FREQUENCY * EPSILON_0)),
        ],
        sources=[mw.PointDipole(tuple(v * lam for v in (-.75, -.625, -.46875)),
                                polarization="Ez", profile="ideal",
                                source_time=mw.CW(FREQUENCY, amplitude=1.7, phase=.37))],
        monitors=[mw.PermittivityMonitor("eps", size=(2 * lam,) * 3)],
        subpixel_samples=mw.SubpixelSpec(samples=4, averaging="polarized"),
    )


def _scene_b():
    nodes = np.array([-.03, -.025, -.02, -.015, -.01, -.005, 0., .005, .01, .015, .02, .026, .03])
    dielectric = mw.Material(eps_r=3.2, sigma_e=.01)
    foil = mw.CopperFoil(thickness=35e-6, frequency_range=(.8e9, 1.2e9), mode_count=4)
    polygon = [(-.01, -.01), (.01, -.01), (.01, .01), (-.01, .01)]
    hole = [(-.003, -.003), (.003, -.003), (.003, .003), (-.003, .003)]
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.),
        structures=[
            mw.Structure(mw.PolySlab(polygon, bounds=(-.01, -.005)), dielectric, name="board1"),
            mw.Structure(mw.Box(position=(0., 0., .0025), size=(.02, .02, .005)),
                         dielectric, name="board2"),
            mw.Structure(mw.Box(position=(0., 0., -.005), size=(.01, .01, 0.)), foil, name="trace1"),
            mw.Structure(mw.PolySlab(polygon, bounds=(.005, .005)), foil, name="trace2"),
            mw.Structure(mw.PlanarPEC(polygon, holes=(hole,), coordinate=.015),
                         mw.Material.pec(), name="perforated"),
            mw.Structure(mw.SurfaceMesh([(-.01, -.01, .02), (.01, -.01, .02),
                                         (.01, .01, .02), (-.01, .01, .02)],
                                        [(0, 1, 2), (0, 2, 3)]), mw.Material.pec(), name="surface"),
        ],
        ports=[mw.LumpedPort("feed", positive=(0., 0., -.015), negative=(0., 0., -.02),
                             voltage_path=mw.AxisPath("z"),
                             current_surface=mw.Box(position=(0., 0., -.0175), size=(.01, .01, 0.)),
                             termination=mw.SeriesRLC(r=50., l=1e-9, c=1e-12))],
        lumped_elements=[mw.Resistor("resistor", (.01, 0., -.015), (.01, 0., -.02), 75.)],
        sources=[mw.PointDipole((-.015, 0., 0.), profile="ideal", source_time=mw.CW(FREQUENCY))],
        metadata={"case": "pcb", "nested": {"values": [1, True, None]}},
    )


def _scene_c():
    material = mw.Material(
        eps_r=2., drude_poles=(mw.DrudePole(plasma_frequency=.2e9, gamma=.1e9),),
        lorentz_poles=(mw.LorentzPole(delta_eps=.4, resonance_frequency=2e9, gamma=.1e9),),
    )
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml", num_layers=2,
                                       strength=1., bloch_wavevector="auto"),
        structures=[mw.Structure(mw.Box(size=(.01, .01, .01)), material)],
        sources=[mw.PlaneWave(direction=(.3, .4, math.sqrt(.75)), polarization=(.8, -.6, 0.),
                               source_time=mw.CW(FREQUENCY, phase=.2),
                               injection=mw.TFSF.slab(axis="z", bounds=(-.02, .02)))],
        ports=[mw.LumpedPort("loaded", positive=(0., 0., .015), negative=(0., 0., .01),
                             voltage_path=mw.AxisPath("z"),
                             current_surface=mw.Box(position=(0., 0., .0125), size=(.01, .01, 0.)),
                             termination=mw.ParallelRLC(r=100., l=2e-9, c=2e-12))],
    )


def _scene_d():
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.faces(x="pec", y="pmc", z_low="pec", z_high="pmc"),
        structures=[mw.Structure(mw.Difference(mw.Box(size=(.01,) * 3), mw.Sphere(radius=.003)),
                                 mw.Material(eps_r=2.5))],
        thin_wires=[mw.ThinWire("dipole", [(-.015, 0., -.01), (-.015, 0., .01)],
                                radius=.0001, conductor=mw.WireConductor.pec(), snap="nearest")],
        sources=[mw.PointDipole((.015, 0., 0.), profile="gaussian", width=.005,
                                source_time=mw.GaussianPulse(FREQUENCY, fwidth=.2e9))],
        monitors=_monitors(),
    )


def _scene_e():
    lam = WAVELENGTH
    return mw.Scene(
        domain=mw.Domain(((-lam, lam),) * 3), grid=mw.GridSpec.uniform(lam / 16),
        boundary=mw.BoundarySpec.faces(
            default="pml", num_layers=4, z_high="mode_abc",
            mode_abc=mw.ModeABC(order=2, frequency_range=(.8 * FREQUENCY, 1.2 * FREQUENCY)),
        ),
        structures=[mw.Structure(mw.Box(size=(.6 * lam, 1.2 * lam, 2 * lam)),
                                 mw.Material(eps_r=4))],
        sources=[mw.PointDipole((0., 0., -.5 * lam), polarization="Ey", profile="ideal",
                                source_time=mw.CW(FREQUENCY))],
    )


@pytest.mark.parametrize("factory,fdfd", [
    pytest.param(_scene_a, True, id="a-dielectrics"),
    pytest.param(_scene_b, False, id="b-pcb"),
    pytest.param(_scene_c, False, id="c-periodic"),
    pytest.param(_scene_d, False, id="d-wire"),
    pytest.param(_scene_e, False, id="e-mode-abc"),
])
def test_round_trip_compiles_identically(factory, fdfd, record_property):
    scene = factory()
    original = scene.to_dict()
    assert original["background"] == {"material": None}
    restored = mw.Scene.from_dict(json.loads(json.dumps(original)))
    assert restored.background is None
    assert restored.boundary.mode_abc == scene.boundary.mode_abc
    assert restored.to_dict() == original
    actual = prepare_scene(restored).compile_material_tensors(frequency=FREQUENCY)
    expected = prepare_scene(scene).compile_material_tensors(frequency=FREQUENCY)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)
    record_property("compile_entry_point", "prepare_scene(...).compile_material_tensors")
    if fdfd:
        source_vectors = []
        for item in (scene, restored):
            solver = mw.Simulation.fdfd(item, frequency=FREQUENCY).prepare().solver
            source_vectors.append(torch.from_dlpack(solver._build_source_vector_yee()).clone())
        assert torch.equal(*source_vectors)
    arrays = {}
    with_arrays = scene.to_dict(arrays=arrays)
    assert mw.Scene.from_dict(json.loads(json.dumps(with_arrays)), arrays=arrays).to_dict() == original
    assert torch.cuda.max_memory_allocated() < 10 * 1024**3


def test_every_constructor_parameter_is_serialized():
    from witwin.maxwell.absorbers import ModeABC, MurABC

    box, sphere = mw.Box(), mw.Sphere()
    wire_binding = mw.WirePortBinding.nodes(negative=mw.WireNodeRef("wire", 0),
                                           positive=mw.WireNodeRef("wire", 1))
    samples = [
        mw.Domain(((-1., 1.),) * 3), mw.GridSpec.uniform(.1),
        mw.GridSpec.auto(wavelength=.3, override_structures=(mw.MeshOverrideStructure(box, .01),),
                         layer_refinement=mw.LayerRefinementSpec(axis="z", bounds=(-.01, .01),
                             bounds_refinement=mw.GridRefinement(dl=.001),
                             corner_finder=mw.CornerFinderSpec(),
                             corner_refinement=mw.GridRefinement(refinement_factor=2))),
        mw.GridSpec.custom(np.linspace(-1., 1., 4097), [-1., 1.], [-1., 1.]),
        mw.BoundarySpec.faces(x="bloch", y="pml", z_low="pec", z_high="pmc",
                              num_layers=3, strength=2., bloch_wavevector=(.1, .2, 0.)),
        MurABC(), ModeABC(mode_index=1, polarization="Ey", frequency_range=(1e9, 2e9), order=1),
        mw.MeshOverrideStructure(box, .01), mw.LayerRefinementSpec(),
        mw.GridRefinement(), mw.CornerFinderSpec(),
        mw.Material(sigma_m=.1, mass_density=1000., debye_poles=(mw.DebyePole(1., 1e-10),),
                    mu_debye_poles=(mw.DebyePole(.1, 1e-10),),
                    mu_drude_poles=(mw.DrudePole(.1e9, .1e9),),
                    mu_lorentz_poles=(mw.LorentzPole(.1, 2e9, .1e9),)),
        mw.Material.pec(), mw.Material.pmc(),
        mw.CopperFoil(thickness=35e-6, frequency_range=(.8e9, 1.2e9)),
        mw.LossyMetalMedium(conductivity=5.8e7), mw.DebyePole(1., 1e-10),
        mw.DrudePole(.1e9, .1e9), mw.LorentzPole(1., 2e9, .1e9),
        mw.Structure(box, mw.Material(eps_r=2.), tags=("tag",), metadata={"type": "user"}),
        mw.ThinWire("wire", [(0., 0., 0.), (0., 0., .01)], .0001, mw.WireConductor.pec()),
        mw.ThinWire("loop", [(0., 0., 0.), (.01, 0., 0.), (.01, .01, 0.), (0., 0., 0.)],
                    .0001, mw.WireConductor.finite(5.8e7)),
        mw.WireConductor.pec(), mw.WireEnd.open(), mw.PointDipole((0., 0., 0.)), mw.PlaneWave(),
        mw.CW(FREQUENCY), mw.GaussianPulse(FREQUENCY, .2e9), mw.RickerWavelet(FREQUENCY),
        mw.TFSF(((-.01, .01),) * 3), mw.TFSF.slab(axis="z", bounds=(-.01, .01)),
        mw.LumpedPort("p", wire_binding=wire_binding, reference_impedance=50. + 2j,
                       termination=mw.SeriesRLC(r=torch.tensor(50., device="cuda", dtype=torch.float64))),
        mw.TerminalPort("terminal", mw.TerminalRef("positive"), mw.TerminalRef("negative"),
                         mw.AxisPath("z"), reference_plane=0.,
                         current_surface=mw.Box(size=(.01, .01, 0.))),
        mw.TerminalRef("positive"), mw.AxisPath("z"), mw.WireNodeRef("wire", 0), wire_binding,
        mw.Resistor("r", (0., 0., 0.), (0., 0., .01), 50.),
        mw.Capacitor("c", (0., 0., 0.), (0., 0., .01), 1e-12),
        mw.Inductor("l", (0., 0., 0.), (0., 0., .01), 1e-9),
        mw.SeriesRLC(r=50., l=1e-9, c=1e-12), mw.ParallelRLC(r=50., l=1e-9, c=1e-12),
        mw.SubpixelSpec(averaging="arithmetic", samples=1), *_monitors(),
        mw.PolySlab([(0., 0.), (.01, 0.), (0., .01)], (0., .01)),
        mw.PlanarPEC([(0., 0.), (.01, 0.), (0., .01)]),
        mw.SurfaceMesh([(0., 0., 0.), (.01, 0., 0.), (0., .01, 0.)], [(0, 1, 2)]),
        box, sphere, mw.Cylinder(), mw.Ellipsoid(), mw.Cone(), mw.Pyramid(), mw.Prism(),
        mw.Torus(), mw.HollowBox(), mw.Union(box, sphere), mw.Intersection(box, sphere),
        mw.Difference(box, sphere),
        mw.Mesh([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
                 [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)], recenter=False),
    ]
    assert set(_TYPES) | set(_GEOMETRY_TYPES) | {mw.PolySlab, mw.PlanarPEC, mw.SurfaceMesh} == {
        type(sample) for sample in samples
    }
    for sample in samples:
        arrays = {}
        writer = _Writer(arrays)
        encoded = writer.encode(sample, "sample")
        parameters = set(signature(type(sample)).parameters) - {"device"}
        if isinstance(sample, mw.TFSF):
            parameters |= {field.name for field in fields(sample)}
        assert parameters <= encoded.keys(), type(sample).__name__
        reader = _Reader(arrays, "cuda")
        reader.materials = {key: reader.decode(value) for key, value in writer.materials.items()}
        restored = reader.decode(json.loads(json.dumps(encoded)))
        if isinstance(sample, mw.GridSpec) and sample.is_custom:
            assert encoded["x_coords"] == {"array": "sample/x_coords"}
            assert torch.equal(arrays["sample/x_coords"], torch.from_numpy(sample.x_coords))
            assert np.array_equal(restored.x_coords, sample.x_coords)
        second_writer = _Writer({})
        assert second_writer.encode(restored, "sample") == encoded, type(sample).__name__
    shared = mw.Material(eps_r=3.)
    scene = mw.Scene(background=shared, structures=[mw.Structure(box, shared), mw.Structure(sphere, shared)])
    restored = mw.Scene.from_dict(scene.to_dict())
    assert len(scene.to_dict()["materials"]) == 1
    assert restored.background is restored.structures[0].material is restored.structures[1].material


@pytest.mark.parametrize("case", ["mode_port", "material_region", "symmetry", "anisotropic"])
def test_out_of_scope_raises(case):
    scene = mw.Scene()
    if case == "mode_port":
        scene.add_port(mw.ModePort("mode", position=(0., 0., 0.), size=(.01, .01, 0.)))
    elif case == "material_region":
        scene.add_material_region(mw.MaterialRegion("density", mw.Box(), torch.ones((2, 2, 2), device="cuda")))
    elif case == "symmetry":
        scene = mw.Scene(symmetry=("PEC", None, None))
    else:
        scene.add_structure(mw.Structure(mw.Box(), mw.Material(epsilon_tensor=mw.DiagonalTensor3(2., 3., 4.))))
    with pytest.raises(TypeError, match="not in witwin.scene/1:"):
        scene.to_dict()
