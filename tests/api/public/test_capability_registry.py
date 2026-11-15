"""Capability registry: per-method tables, default-deny admission and the aggregated refusal.

Every refusal row is observed on a minimal CPU scene that declares its key, and
every table decides every key the extractor can emit. The scenes are built
only to be classified; nothing here compiles or runs a solver.
"""
from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest
import torch

import witwin.core as core
import witwin.maxwell as mw
from witwin.core import PhaseScreen
from witwin.maxwell.capability import Support, check_scene
from witwin.maxwell.capability.keys import (
    BOUNDARY_KINDS, IN_SIMULATION_MONITOR_TYPES, MOM_CONFIG_KEYS, MONITOR_TYPES, PORT_TYPES, SCENE_KEYS,
    SOURCE_TYPES,
)
from witwin.maxwell.capability.tables import TABLES

from tests.support.public_api import public_objects

pytestmark = pytest.mark.host_validation

ROOT = Path(__file__).resolve().parents[3]
F = 1.0e9
PUBLIC = public_objects()
_FOLDED = (("PMC", "low"), None, None)


def _scene(**kwargs):
    kwargs.setdefault("domain", mw.Domain(bounds=((-0.5, 0.5),) * 3))
    kwargs.setdefault("grid", mw.GridSpec.uniform(0.1))
    return mw.Scene(**kwargs)


def _body(material=None, *, geometry=None, **kwargs):
    geometry = mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)) if geometry is None else geometry
    material = mw.Material(eps_r=2.0) if material is None else material
    return _scene(structures=[mw.Structure(name="body", geometry=geometry, material=material, **kwargs)])


def _stand_in(cls, name="x", **attributes):
    """A bare instance: the key extractor reads only an object's type, name and named flags."""
    value = object.__new__(cls)
    for key, item in {"name": name, **attributes}.items():
        object.__setattr__(value, key, item)
    return value


def _appended(collection, *values, **kwargs):
    # Stand-ins bypass the constructor checks on purpose: the registry must refuse them itself.
    scene = _scene(**kwargs)
    object.__setattr__(scene, collection, (*getattr(scene, collection), *values))
    return scene


def _plane_wave(**kwargs):
    return mw.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0), source_time=mw.CW(frequency=F), **kwargs)


def _plane_modes(trainable=False):
    k = 2 * math.pi * F / 299792458.0
    electric = torch.tensor([[1.0 + 0j, 0, 0]], dtype=torch.complex128, requires_grad=trainable)
    return mw.mom.MaxwellPlaneModes(F, torch.tensor([[0, 0, k + 0j]], dtype=torch.complex128), electric,
                                torch.tensor([[0, 1.0 / 376.73 + 0j, 0]], dtype=torch.complex128))


def _terminal(name="a", **kwargs):
    kwargs = kwargs or {"potential": 1.0}
    return mw.electrostatic.ElectrostaticTerminal(name, geometry=mw.Box(position=(-0.25, 0, 0), size=(0.2, 0.2, 0.2)), **kwargs)


def _full_tensor():
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    rotation = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)
    tensor = rotation @ torch.diag(torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64)) @ rotation.T
    return mw.Material(epsilon_tensor=mw.media.Tensor3x3(tensor.tolist()))


def _circuit():
    circuit = mw.circuits.Circuit("rc")
    circuit.add(mw.rf.Resistor("R1", circuit.node("a"), circuit.ground, 50.0))
    return circuit


def _folded(monitor, **kwargs):
    return _scene(boundary=mw.BoundarySpec.pml(num_layers=2), symmetry=_FOLDED, monitors=[monitor], **kwargs)


def _boundary(kind):
    if kind == "bloch":
        return mw.BoundarySpec.bloch((0.1, 0.0, 0.0))
    if kind == "mur":
        return mw.BoundarySpec.mur()
    if kind == "mode_abc":
        return mw.BoundarySpec.faces(default="none", z="mode_abc")
    return {"none": mw.BoundarySpec.none, "pml": mw.BoundarySpec.pml, "periodic": mw.BoundarySpec.periodic,
            "pec": mw.BoundarySpec.pec, "pmc": mw.BoundarySpec.pmc}[kind]()


def _surface_impedance():
    impedance = mw.media.LossyMetalMedium(conductivity=5.8e7).rational_surface_impedance((1e9, 4e9))
    return mw.media.SurfaceImpedanceMedium(impedance=impedance)


def _tilted_sheet():
    vertices = torch.tensor([[-0.2, -0.2, -0.05], [0.2, -0.2, 0.05], [0.0, 0.2, 0.0]], dtype=torch.float64)
    return mw.geometry.SurfaceMesh(vertices, torch.tensor([[0, 1, 2]]))


def _layer_stack():
    return mw.mom.LayerStack((mw.mom.Layer("air", 1e-3, 1.0),), mw.mom.HalfSpace(), mw.mom.HalfSpace())


# One minimal CPU scene per refusable scene key: (scene, solve frequencies).
CASES = {
    "background.homogeneous": lambda: _scene(background=mw.Material()),
    "background.layer_stack": lambda: _scene(background=_layer_stack()),
    "coordinates": lambda: _scene(coordinates=mw.coordinates.BendCoordinates(
        curvature=0.5, origin=(0.2, -0.1, 0.3), frame=((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))),
    "symmetry": lambda: _scene(symmetry=_FOLDED),
    "symmetry.projection_monitor": lambda: _folded(mw.monitors.AngleProjectionMonitor(
        "far", position=(0.2, 0.2, 0.0), size=(0.3, 0.3, 0.3), theta=(0.5,), phi=(0.0,))),
    "symmetry.closed_surface_auto_grid": lambda: _folded(
        mw.monitors.ClosedSurfaceMonitor.box("box", position=(0.2, 0.2, 0.0), size=(0.3, 0.3, 0.3)),
        grid=mw.GridSpec.auto(wavelength=0.3)),
    "symmetry.closed_surface_image_plane": lambda: _folded(
        mw.monitors.ClosedSurfaceMonitor.box("box", position=(-0.35, 0.2, 0.0), size=(0.3, 0.3, 0.3))),
    "symmetry.diffraction_folded_axis": lambda: _folded(
        mw.monitors.DiffractionMonitor("orders", position=(0.3, 0.3, 0.3), size=(0.6, 0.6, 0.0))),
    "symmetry.mode_plane_image_plane": lambda: _folded(mw.ModeMonitor(
        "mode", position=(-0.3, 0.2, 0.0), size=(0.4, 0.4, 0.0), polarization="Ey", frequencies=(1e9,))),
    "symmetry.finite_plane_image_plane": lambda: _folded(
        mw.monitors.FinitePlaneMonitor("rect", position=(-0.45, 0.2, 0.0), size=(0.3, 0.3, 0.0), compute_flux=True,
                                       fields=("Ex", "Ey", "Hx", "Hy"))),
    "subpixel.pec_conformal": lambda: _scene(subpixel=mw.SubpixelSpec(pec="conformal")),
    **{f"boundary.{kind}": (lambda kind=kind: _scene(boundary=_boundary(kind))) for kind in BOUNDARY_KINDS},
    "structure.optical_assignment": lambda: _body(phase_screen=PhaseScreen(height=torch.rand(4, 4),
                                                                           height_scale_m=0.05)),
    "structure.trainable_conductor_loss": lambda: _body(mw.media.LossyMetalMedium(
        conductivity=torch.tensor(5.8e7, dtype=torch.float64, requires_grad=True))),
    "geometry.surface_mesh": lambda: _body(mw.Material(pec=True), geometry=_tilted_sheet()),
    "geometry.planar_pec": lambda: _body(mw.Material.pec(), geometry=mw.geometry.PlanarPEC(
        [(0., 0.), (1e-4, 0.), (1e-4, 2e-5), (0., 2e-5)], normal_axis="z", coordinate=0.0)),
    "material.lossy_metal": lambda: _body(mw.media.LossyMetalMedium(conductivity=5.8e7)),
    "material.surface_impedance": lambda: _body(_surface_impedance()),
    "material.copper_foil": lambda: _body(mw.media.CopperFoil(thickness=2e-6, frequency_range=(0.5e9, 4e9))),
    "material.medium2d": lambda: _body(mw.media.Medium2D(sigma_s=1e-3)),
    "material.gyromagnetic": lambda: _body(mw.media.GyromagneticFerrite(
        eps_r=14.5, saturation_magnetization=1.40e5, bias_field=(0.0, 0.0, 1.75e5), gilbert_damping=2.0e-3)),
    "material.nonlinear": lambda: _body(mw.Material(eps_r=2.0, kerr_chi3=1e-10)),
    "material.full_tensor": lambda: _body(_full_tensor()),
    "material.modulation": lambda: _body(mw.Material(eps_r=2.0, modulation=mw.media.ModulationSpec(frequency=1e8,
                                                                                               amplitude=0.1))),
    "material.pmc": lambda: _body(mw.Material(pmc=True)),
    "material.sigma_e": lambda: _body(mw.Material(eps_r=4.0, sigma_e=1e7)),
    "material.core_surface_fields": lambda: _body(core.PhysicalMaterial(eps_r=4.0, conductor_model="perfect")),
    "material.breakdown": lambda: _body(mw.Material(eps_r=2.0, breakdown=mw.breakdown.DielectricBreakdown(
        critical_field=1e6, post_breakdown_conductivity=1.0))),
    "material.perturbation": lambda: _body(mw.media.PerturbationMedium(mw.Material(eps_r=2.0),
                                                                 perturbation=torch.ones(2, 2, 2))),
    "material.trainable_mu_r": lambda: _body(mw.Material(eps_r=4.0, mu_r=torch.tensor(2.0, requires_grad=True))),
    "source": lambda: _scene(sources=[_plane_wave()]),
    "monitor": lambda: _scene(monitors=[mw.PointMonitor("m", (0, 0, 0), ("Ez",))]),
    "port": lambda: _scene(ports=[mw.LumpedPort("p", positive=(0.05, 0, 0), negative=(-0.05, 0, 0),
                                                voltage_path=mw.AxisPath("x"),
                                                current_surface=mw.Box(position=(0, 0, 0), size=(0.1, 0.1, 0.1)))]),
    "lumped_element": lambda: _scene(lumped_elements=[mw.rf.Resistor("r1", (0, 0, 0.1), (0, 0, -0.1), 50.0)]),
    "network": lambda: _appended("networks", object()),
    "circuit": lambda: _scene(circuits=(_circuit(),)),
    "thin_wire": lambda: _scene(thin_wires=[mw.rf.ThinWire(name="w", points=((-0.1, 0.3, 0), (0.1, 0.3, 0)),
                                                        radius=1e-3, conductor=mw.rf.WireConductor.pec())]),
    "internal_absorber": lambda: _scene(internal_absorbers=[mw.fdtd.InternalAbsorber(position=(0, 0, 0.3),
                                                                                 size=(1.0, 1.0, 0.0))]),
    "electrostatic_terminal": lambda: _scene(electrostatic_terminals=[_terminal()]),
    "charge_density": lambda: _scene(charge_densities=[mw.electrostatic.ChargeDensity(mw.Box(position=(0, 0, 0),
                                                                               size=(0.2, 0.2, 0.2)), 1e-6)]),
    **{f"source.{name}": (lambda name=name: _appended("sources", _stand_in(PUBLIC[name])))
       for name in SOURCE_TYPES if name != "MaxwellPlaneModes"},
    "source.MaxwellPlaneModes": lambda: _scene(sources=[_plane_modes()]),
    "source.MaxwellPlaneModes.trainable": lambda: _scene(sources=[_plane_modes(trainable=True)]),
    "source.injection_tfsf": lambda: _scene(sources=[_plane_wave(injection=mw.TFSF(bounds=((-0.1, 0.1),) * 3))]),
    "source.injection_axis": lambda: _scene(sources=[_plane_wave(injection_axis="z")]),
    "source.cw_frequency_mismatch": lambda: _scene(sources=[mw.PointDipole(
        position=(0, 0, 0), polarization="Ez", source_time=mw.CW(frequency=3 * F), name="d")]),
    **{f"monitor.{name}": (lambda name=name: _appended("monitors", _stand_in(PUBLIC[name])))
       for name in MONITOR_TYPES},
    **{f"port.{name}": (lambda name=name: _appended("ports", _stand_in(PUBLIC[name]))) for name in PORT_TYPES},
    "port.termination": lambda: _appended("ports", _stand_in(mw.LumpedPort, termination=mw.rf.SeriesRLC(r=50.0))),
    "port.wire_binding": lambda: _appended("ports", _stand_in(mw.LumpedPort, wire_binding=object())),
    "electrostatic_terminal.trainable_geometry": lambda: _scene(electrostatic_terminals=[mw.electrostatic.ElectrostaticTerminal(
        "a", geometry=mw.Box(position=torch.tensor((-0.25, 0.0, 0.0), requires_grad=True), size=(0.2, 0.2, 0.2)),
        potential=1.0)]),
    "electrostatic_terminal.floating": lambda: _scene(electrostatic_terminals=[_terminal("f", charge=1e-12)]),
}

REFUSED_ROWS = sorted((method, key) for method, rows in TABLES.items() for key, row in rows.items()
                      if row.support is not Support.SUPPORTED and key not in MOM_CONFIG_KEYS)


@pytest.mark.parametrize("method, key", REFUSED_ROWS, ids=[f"{method}:{key}" for method, key in REFUSED_ROWS])
def test_refused_rows_are_reported(method, key):
    report = check_scene(CASES[key](), method, frequencies=(F,))
    refused = {item.key for item in report.unsupported}
    assert key in refused, (key, [str(item) for item in report.unsupported])


@pytest.mark.parametrize("method", tuple(TABLES))
def test_every_table_decides_every_scene_key(method):
    rows = TABLES[method]

    def decided(key):
        parts = key.split(".")
        prefixes = (".".join(parts[:end]) for end in range(1, len(parts)))
        return key in rows or any(prefix in rows and rows[prefix].support is not Support.SUPPORTED
                                  for prefix in prefixes)

    assert [key for key in SCENE_KEYS if not decided(key)] == []
    assert set(rows) - set(SCENE_KEYS) - set(MOM_CONFIG_KEYS) == set()
    assert all(row.key == key for key, row in rows.items())
    if method.startswith("mom"):
        assert all(key in rows for key in MOM_CONFIG_KEYS)


def test_supported_rows_admit_their_minimal_scene():
    for method, rows in TABLES.items():
        for key, row in rows.items():
            if row.support is Support.SUPPORTED and key in CASES:
                report = check_scene(CASES[key](), method, frequencies=(F,))
                assert key not in {item.key for item in report.unsupported}, (method, key)


def _test_functions(path):
    tree = ast.parse((ROOT / path).read_text())
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_every_refusal_row_names_existing_negative_tests():
    for method, rows in TABLES.items():
        for key, row in rows.items():
            if row.support is Support.SUPPORTED:
                assert row.negative_tests == (), (method, key)
                continue
            assert row.reason and row.negative_tests, (method, key)
            for reference in row.negative_tests:
                path, function = reference.split("::")
                assert function in _test_functions(path), (method, key, reference)


def _public_classes(module, *, exclude=()):
    return {name for name, value in inspect.getmembers(module, inspect.isclass)
            if value.__module__ == module.__name__ and not name.startswith("_") and name not in exclude}


def test_type_vocabulary_covers_every_public_scene_object():
    from witwin.maxwell import esd, fdfd_response, monitors, ports, sources

    source_helpers = {"CW", "GaussianPulse", "RickerWavelet", "CustomSourceTime", "BroadbandPulse",
                      "BasebandGaussianPulse", "BasebandRectangularPulse", "TFSF", "FieldDataset", "CurrentDataset",
                      "DipoleProfile"}
    assert _public_classes(sources, exclude=source_helpers) | {"ESDCurrentSource", "ESDVoltageSource",
                                                               "LumpedCurrentSource"} == set(SOURCE_TYPES)
    assert {"ESDCurrentSource", "ESDVoltageSource"} <= _public_classes(esd)
    assert "LumpedCurrentSource" in _public_classes(fdfd_response)
    exported_monitors = {name for name in _public_classes(monitors) if name in PUBLIC}
    assert exported_monitors == set(MONITOR_TYPES)
    assert {type_.__name__ for type_ in monitors.IN_SIMULATION_MONITOR_TYPES} == set(IN_SIMULATION_MONITOR_TYPES)
    port_helpers = {"AxisPath", "WireNodeRef", "WirePortBinding", "TerminalRef", "WaveModeSpec", "PECFrame"}
    assert {name for name in _public_classes(ports, exclude=port_helpers) if name in PUBLIC} == set(PORT_TYPES)


def test_unknown_object_type_is_refused_by_default():
    class Probe:
        name = "probe"

    report = check_scene(_appended("monitors", Probe()), "fdtd")
    assert [(item.key, item.owner) for item in report.unsupported] == [("monitor.Probe", "Monitor 'probe' (Probe)")]
    assert "monitor.Probe is not declared for fdtd" in str(report.unsupported[0])
    with pytest.raises(mw.UnsupportedFeatureError, match="not declared for fdtd") as error:
        report.raise_if_unsupported()
    assert error.value.feature == "monitor.Probe" and error.value.method == "fdtd"


def test_subclass_inherits_its_base_row():
    class PlainMonitor(mw.PointMonitor):
        pass

    monitor = PlainMonitor("m", (0, 0, 0), ("Ez",))
    assert check_scene(_scene(monitors=[monitor]), "fdtd").unsupported == ()
    (item,) = check_scene(_scene(monitors=[monitor]), "fdfd", frequencies=(F,)).unsupported
    assert item.key == "monitor.PlainMonitor" and "not supported by FDFD" in str(item)


def test_refused_collection_subsumes_its_members():
    scene = _scene(sources=[_plane_wave(injection_axis="z")], symmetry=_FOLDED)
    scene = scene.updated(monitors=(
        *scene.monitors,
        mw.monitors.AngleProjectionMonitor("far", position=(0.2, 0.2, 0.0), size=(0.3, 0.3, 0.3),
                                  theta=(0.5,), phi=(0.0,)),
    ))
    keys = [item.key for item in check_scene(scene, "electrostatic").unsupported]
    assert keys == ["symmetry", "source", "monitor"]
    keys = [item.key for item in check_scene(scene, "fdfd", frequencies=(F,)).unsupported]
    assert keys == ["symmetry", "monitor.AngleProjectionMonitor"]


def test_capabilities_is_a_read_only_table_per_method():
    assert set(TABLES) == {"fdtd", "fdfd", "mom", "mom.layered", "electrostatic", "capacitance"}
    table = mw.capabilities(mw.SimulationMethod.FDFD)
    assert table is mw.capabilities("fdfd")
    row = table["material.lossy_metal"]
    assert row.support is Support.UNSUPPORTED and row.alternative.startswith("use Simulation.fdtd")
    assert table["monitor.PermittivityMonitor"].support is Support.SUPPORTED
    assert mw.capabilities("electrostatic")["source"].support is Support.INHERENT
    with pytest.raises(TypeError):
        table["material.lossy_metal"] = row
    with pytest.raises(ValueError, match="Unknown simulation method 'fdfd2'"):
        mw.capabilities("fdfd2")


def test_simulation_check_reports_without_raising():
    scene = _scene(sources=[mw.PointDipole(position=(0, 0, 0), polarization="Ez", source_time=mw.CW(frequency=F),
                                           name="d")], monitors=[mw.monitors.PermittivityMonitor("eps")])
    simulation = mw.Simulation.fdfd(scene, frequencies=F)
    report = simulation.check()
    assert report.method == "fdfd" and report.unsupported == ()
    assert report.supported == ("boundary.none", "source", "monitor", "source.PointDipole",
                                "monitor.PermittivityMonitor")
    # A scene swapped in after construction is re-checked (scenes are immutable).
    simulation.scene_input = simulation.scene = scene.updated(lumped_elements=(
        mw.rf.Resistor("r1", positive=(0, 0, 0.1), negative=(0, 0, -0.1), resistance=50.0),
    ))
    assert [item.key for item in simulation.check().unsupported] == ["lumped_element"]
    with pytest.raises(mw.UnsupportedFeatureError, match="Lumped elements"):
        simulation.prepare()


def test_construction_raises_once_with_every_item():
    scene = _body(mw.Material(pmc=True), phase_screen=PhaseScreen(height=torch.rand(4, 4), height_scale_m=0.05))
    scene = scene.updated(sources=(*scene.sources, _plane_wave(injection_axis="z")), monitors=(*scene.monitors, mw.monitors.AuxFieldTimeMonitor("aux")))
    with pytest.raises(mw.UnsupportedFeatureError) as error:
        mw.Simulation.mom(scene, frequencies=F)
    items = error.value.items
    assert [item.key for item in items] == ["structure.optical_assignment", "material.pmc",
                                            "source.injection_axis", "monitor.AuxFieldTimeMonitor"]
    assert all(item.method == "mom" for item in items)
    assert str(error.value).startswith("Simulation.mom(...) cannot represent 4 scene feature(s):\n- Structure 'body'")
    assert error.value.feature == "structure.optical_assignment" and error.value.alternative is None
    assert isinstance(error.value, NotImplementedError)


def test_thin_wires_outside_fdtd_raise_the_unsupported_feature_error():
    scene = CASES["thin_wire"]()
    for build in (lambda: mw.Simulation.fdfd(scene, frequencies=F), lambda: mw.Simulation.mom(scene, frequencies=F)):
        with pytest.raises(mw.UnsupportedFeatureError, match="ThinWire is an FDTD-only subgrid model") as error:
            build()
        assert error.value.feature == "thin_wire" and error.value.alternative == "use Simulation.fdtd(...)"


def test_layered_mom_is_its_own_method():
    scene = _scene(background=_layer_stack(), structures=[mw.Structure(mw.geometry.PlanarPEC(
        [(0., 0.), (1e-4, 0.), (1e-4, 2e-5), (0., 2e-5)], normal_axis="z", coordinate=0.0), mw.Material.pec())])
    assert check_scene(scene, "mom.layered").unsupported == ()
    assert [item.key for item in check_scene(scene, "mom").unsupported] == ["background.layer_stack",
                                                                            "geometry.planar_pec"]


def test_mom_config_refuses_two_runtime_selectors():
    config = mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5,
                                                          edge_cell_m=1e-5))
    # Presence is what selects a runtime in Simulation.prepare(); the sub-config values are not read here.
    object.__setattr__(config, "harmonic_balance", object())
    scene = _scene(background=_layer_stack(), structures=[mw.Structure(mw.geometry.PlanarPEC(
        [(0., 0.), (1e-4, 0.), (1e-4, 2e-5), (0., 2e-5)], normal_axis="z", coordinate=0.0), mw.Material.pec())])
    with pytest.raises(mw.UnsupportedFeatureError, match=r"selects more than one MoM runtime \(layered, "
                                                          r"harmonic_balance\); prepare\(\) would run only the "
                                                          r"layered runtime") as error:
        mw.Simulation.mom(scene, frequencies=(F,), config=config)
    assert error.value.feature == "mom_config.runtime_selectors" and error.value.method == "mom.layered"
    single = mw.mom.MoMConfig(layered=config.layered)
    assert check_scene(scene, "mom.layered", config=single).unsupported == ()
