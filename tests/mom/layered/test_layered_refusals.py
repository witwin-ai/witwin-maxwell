"""Layered MoM refuses every declared input it would not apply (host-only unless marked).

Each case adds one input to a valid two-sheet, one-port layered scene and checks
that compile_layered_scene, which LayeredNetworkRuntime runs first, names it.
Inputs with a capability-registry key are also refused at Simulation.mom(...)
construction by the mom.layered table.
"""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.runtime import compile_layered_scene, layered_scene_signature

Z = 1e-4


def stack():
    return mw.mom.LayerStack((mw.mom.Layer("air", 1e-3),), mw.mom.HalfSpace(), mw.mom.HalfSpace())


def config():
    return mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5))


def sheets():
    return [mw.Structure(mw.geometry.PlanarPEC([(0., 0.), (4e-5, 0.), (4e-5, 2e-5), (0., 2e-5)], coordinate=Z),
                         mw.Material.pec()),
            mw.Structure(mw.geometry.PlanarPEC([(6e-5, 0.), (1e-4, 0.), (1e-4, 2e-5), (6e-5, 2e-5)], coordinate=Z),
                         mw.Material.pec())]


def port(name="p", **fields):
    return mw.LumpedPort(name, positive=(7e-5, 1e-5, Z), negative=(3e-5, 1e-5, Z),
                         voltage_path=mw.AxisPath("x"), reference_impedance=50.0, **fields)


def scene(*, ports=None, **fields):
    return mw.Scene(background=stack(), structures=sheets(), ports=(port(),) if ports is None else ports,
                    **fields)


def box():
    return mw.Box(position=(5e-5, 1e-5, Z), size=(1e-5, 1e-5, 1e-5))


def _wire():
    return mw.rf.ThinWire(name="w", points=((0., 0., 5e-4), (1e-4, 0., 5e-4)), radius=1e-6,
                       conductor=mw.rf.WireConductor.pec())


def _network():
    from tests.rf.network.test_network_block_contract import _block
    return _block(connections={1: "p"})


def _circuit():
    circuit = mw.circuits.Circuit("c")
    circuit.add(mw.rf.Resistor("R1", circuit.node("signal"), circuit.ground, 50.0))
    return circuit


def _terminal_port():
    return mw.rf.TerminalPort("t", wire_binding=mw.rf.WirePortBinding.nodes(
        negative=mw.rf.WireNodeRef("wire0", 0), positive=mw.rf.WireNodeRef("wire0", 1)))


def _wire_port():
    return mw.LumpedPort("w", wire_binding=mw.rf.WirePortBinding.nodes(
        negative=mw.rf.WireNodeRef("wire0", 0), positive=mw.rf.WireNodeRef("wire0", 1)))


# (id, Scene factory, expected item text)
CASES = (
    ("termination_series", lambda: scene(ports=[port(termination=mw.rf.SeriesRLC(r=10.))]),
     "LumpedPort 'p'.termination = SeriesRLC"),
    ("termination_parallel", lambda: scene(ports=[port(termination=mw.rf.ParallelRLC(r=50., c=1e-12))]),
     "LumpedPort 'p'.termination = ParallelRLC"),
    ("reference_plane", lambda: scene(ports=[port(reference_plane=1e-5)]), "LumpedPort 'p'.reference_plane = float"),
    ("current_surface", lambda: scene(ports=[port(current_surface=box())]), "LumpedPort 'p'.current_surface = Box"),
    ("wire_binding", lambda: scene(ports=[port(), _wire_port()]), "LumpedPort 'w'.wire_binding = WirePortBinding"),
    ("terminal_port", lambda: scene(ports=[port(), _terminal_port()]), "Scene.ports[1] = TerminalPort 't'"),
    ("sources", lambda: scene(sources=[mw.PointDipole(position=(0., 0., Z), polarization=(1., 0., 0.))]),
     "Scene.sources = [PointDipole"),
    ("monitors", lambda: scene(monitors=[mw.PointMonitor("m", (0., 0., Z))]), "Scene.monitors = [PointMonitor 'm']"),
    ("thin_wires", lambda: scene(thin_wires=[_wire()]), "Scene.thin_wires = [ThinWire 'w']"),
    ("networks", lambda: scene(networks=[_network()]), "Scene.networks = [NetworkBlock 'matching']"),
    ("lumped_elements", lambda: scene(lumped_elements=[mw.rf.Resistor("r1", (7e-5, 1e-5, Z), (3e-5, 1e-5, Z), 50.)]),
     "Scene.lumped_elements = [Resistor 'r1']"),
    ("circuits", lambda: scene(circuits=[_circuit()]), "Scene.circuits = [Circuit 'c']"),
    ("material_regions", lambda: scene(material_regions=[mw.MaterialRegion(
        "d", mw.Box(position=(5e-5, 1e-5, 2e-4), size=(1e-4, 1e-4, 1e-4)), torch.ones(4, 4, 4))]),
     "Scene.material_regions = [MaterialRegion 'd']"),
    ("electrostatic_terminals", lambda: scene(electrostatic_terminals=[mw.electrostatic.ElectrostaticTerminal(
        "t", geometry=mw.Box(position=(0., 0., .5), size=(.1, .1, .1)), potential=1.)]),
     "Scene.electrostatic_terminals = [ElectrostaticTerminal 't']"),
    ("charge_densities", lambda: scene(charge_densities=[mw.electrostatic.ChargeDensity(
        mw.Box(position=(0., 0., .5), size=(.1, .1, .1)), 1e-6)]), "Scene.charge_densities = [ChargeDensity"),
    ("internal_absorbers", lambda: scene(internal_absorbers=[mw.fdtd.InternalAbsorber(
        position=(0., 0., .5), size=(float("inf"), float("inf"), 0.))]), "Scene.internal_absorbers = [InternalAbsorber"),
    ("symmetry", lambda: scene(symmetry=("PEC", None, None)), "Scene.symmetry = (('PEC', 'low'), None, None)"),
    ("boundary_periodic", lambda: scene(boundary=mw.BoundarySpec.periodic()), "Scene.boundary face kinds ['periodic']"),
    ("boundary_pec", lambda: scene(boundary=mw.BoundarySpec.pec()), "Scene.boundary face kinds ['pec']"),
    ("boundary_pml", lambda: scene(boundary=mw.BoundarySpec.pml(num_layers=8)), "Scene.boundary face kinds ['pml']"),
    ("coordinates", lambda: scene(coordinates=mw.coordinates.BendCoordinates(curvature=.1)),
     "Scene.coordinates = BendCoordinates"),
)


@pytest.mark.parametrize("build,item", [case[1:] for case in CASES], ids=[case[0] for case in CASES])
def test_unsupported_layered_input_is_refused_by_name(build, item):
    with pytest.raises(NotImplementedError, match="Layered MoM does not apply these declared inputs") as error:
        compile_layered_scene(build())
    assert f"- {item}" in str(error.value)


# Registry key of each case the mom.layered capability table refuses at construction. The remaining
# cases (material regions and the LumpedPort reference plane, voltage aperture and current surface)
# have no registry key and are refused by the layered compiler at prepare().
REGISTRY_KEYS = {
    "termination_series": "port.termination", "termination_parallel": "port.termination",
    "wire_binding": "port.wire_binding", "terminal_port": "port.TerminalPort", "sources": "source",
    "monitors": "monitor", "thin_wires": "thin_wire", "lumped_elements": "lumped_element", "circuits": "circuit",
    "electrostatic_terminals": "electrostatic_terminal", "charge_densities": "charge_density",
    "internal_absorbers": "internal_absorber", "symmetry": "symmetry", "boundary_periodic": "boundary.periodic",
    "boundary_pec": "boundary.pec", "boundary_pml": "boundary.pml", "coordinates": "coordinates",
}
_BUILD = {case[0]: case[1] for case in CASES}


@pytest.mark.parametrize("case", tuple(REGISTRY_KEYS))
def test_public_construction_refuses_through_the_registry(case):
    with pytest.raises(mw.UnsupportedFeatureError) as error:
        mw.Simulation.mom(_BUILD[case](), frequencies=[1e9], config=config())
    assert error.value.method == "mom.layered"
    assert REGISTRY_KEYS[case] in {item.key for item in error.value.items}


def test_public_termination_refusal_names_the_network_alternative():
    with pytest.raises(mw.UnsupportedFeatureError, match=r"Port 'p' \(LumpedPort\): the layered delta-gap port does "
                                                         r"not apply a termination; Result.network is the open N-port; "
                                                         r"close it with Result.network.terminate"):
        mw.Simulation.mom(scene(ports=[port(termination=mw.rf.SeriesRLC(r=50.))]), frequencies=[1e9], config=config())


def test_public_prepare_refuses_before_any_cuda_work():
    # Inputs without a registry key reach prepare(); LayeredNetworkRuntime compiles the Scene before
    # meshing, so the refusal is host-visible.
    simulation = mw.Simulation.mom(_BUILD["current_surface"](), frequencies=[1e9], config=config())
    with pytest.raises(NotImplementedError, match=r"LumpedPort 'p'.current_surface = Box: the port current is the "
                                                   r"delta-gap MoM current; omit current_surface"):
        simulation.prepare()


def test_every_unsupported_input_is_listed_in_one_error():
    combined = scene(ports=[port(termination=mw.rf.SeriesRLC(r=10.), reference_plane=1e-5, current_surface=box()),
                            _terminal_port()],
                     lumped_elements=[mw.rf.Resistor("r1", (7e-5, 1e-5, Z), (3e-5, 1e-5, Z), 50.)],
                     symmetry=("PEC", None, None), boundary=mw.BoundarySpec.periodic(),
                     monitors=[mw.PointMonitor("m", (0., 0., Z))])
    with pytest.raises(NotImplementedError) as error:
        compile_layered_scene(combined)
    lines = str(error.value).splitlines()
    assert lines[0] == "Layered MoM does not apply these declared inputs; remove them or use the stated alternative:"
    assert [line.split(" = ")[0].split(":")[0] for line in lines[1:]] == [
        "- Scene.monitors", "- Scene.lumped_elements", "- Scene.symmetry",
        "- Scene.boundary face kinds ['periodic']", "- LumpedPort 'p'.termination",
        "- LumpedPort 'p'.reference_plane", "- LumpedPort 'p'.current_surface", "- Scene.ports[1]"]


@pytest.mark.parametrize("field", ["thickness", "eps_r", "sigma"])
def test_trainable_layer_fields_are_refused(field):
    values = dict(name="sub", thickness=1e-3, eps_r=4.0, sigma=0.1)
    values[field] = torch.tensor(values[field], requires_grad=True)
    with pytest.raises(NotImplementedError, match=rf"^Trainable Layer\.{field} is not supported"):
        mw.mom.Layer(**values)


@pytest.mark.parametrize("field", ["eps_r", "sigma"])
def test_trainable_half_space_fields_are_refused(field):
    with pytest.raises(NotImplementedError, match=rf"^Trainable HalfSpace\.{field} is not supported"):
        mw.mom.HalfSpace(**{field: torch.tensor(2.0, requires_grad=True)})


def APERTURE():
    return mw.Box(position=(5e-5, 1e-5, Z), size=(4e-5, 2e-5, 0.))


def test_voltage_aperture_is_admitted_as_an_edge_port():
    # The layered meaning (the delta-gap bridge spans the aperture width) matches FDTD's
    # (voltage distributed over parallel paths); the remaining port fields stay refused.
    compiled = compile_layered_scene(scene(ports=[port(voltage_aperture=APERTURE())]))
    assert compiled[2][0].voltage_aperture is not None
    with pytest.raises(NotImplementedError, match=r"LumpedPort 'p'.reference_plane = float") as error:
        compile_layered_scene(scene(ports=[port(voltage_aperture=APERTURE(), reference_plane=1e-5)]))
    assert "voltage_aperture" not in str(error.value)


def test_valid_layered_scene_still_compiles_and_omits_current_surface():
    fixed = mw.mom.LayerStack((mw.mom.Layer("sub", torch.tensor(1e-3), torch.tensor(4.0), torch.tensor(0.1)),),
                          mw.mom.HalfSpace(torch.tensor(2.0)), "pec")
    valid = mw.Scene(background=fixed, structures=sheets(), ports=(port(),))
    assert valid.ports[0].current_surface is None
    simulation = mw.Simulation.mom(valid, frequencies=[1e9, 2e9], config=config())
    stack_, rectangles, ports, materials, sheets_, vias, via_materials = compile_layered_scene(simulation.scene)
    assert stack_ is fixed and len(rectangles) == 2 and ports == (valid.ports[0],)
    assert materials == [(None, 0.0, None)] * 2 and vias == [] and via_materials == []


def _signature(built):
    return layered_scene_signature(compile_layered_scene(built), "cpu")


def test_scene_signature_is_deterministic_and_tracks_compiled_inputs():
    base = _signature(scene())
    assert _signature(scene()) == base
    moved = scene(ports=[mw.LumpedPort("p", positive=(8e-5, 1e-5, Z), negative=(3e-5, 1e-5, Z),
                                       voltage_path=mw.AxisPath("x"), reference_impedance=50.0)])
    renamed = scene(ports=[port("q")])
    loaded = scene(ports=[mw.LumpedPort(
        "p", positive=(7e-5, 1e-5, Z), negative=(3e-5, 1e-5, Z), voltage_path=mw.AxisPath("x"),
        reference_impedance=75.0)])
    layered = scene()
    layered = layered.updated(background=mw.mom.LayerStack((mw.mom.Layer("air", 1e-3, 2.0),), mw.mom.HalfSpace(), mw.mom.HalfSpace()))
    edge = scene(ports=[port(voltage_aperture=APERTURE())])
    narrower = scene(ports=[port(voltage_aperture=mw.Box(position=(5e-5, 1e-5, Z), size=(4e-5, 1e-5, 0.)))])
    assert _signature(edge) != _signature(narrower)
    for changed in (moved, renamed, loaded, layered, edge):
        assert _signature(changed) != base
    # An in-place update of a trainable Scene tensor (an optimizer step) changes the signature.
    sigma = torch.tensor(5.8e7, requires_grad=True)
    lossy = scene()
    lossy = lossy.updated(structures=(mw.Structure(mw.geometry.PlanarPEC([(0., 0.), (4e-5, 0.), (4e-5, 2e-5), (0., 2e-5)], coordinate=Z,
                                                    physical_thickness=1e-6),
                                       mw.media.LossyMetalMedium(conductivity=sigma)), *lossy.structures[1:]))
    before = _signature(lossy)
    with torch.no_grad():
        sigma.mul_(2)
    assert _signature(lossy) != before


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
def test_run_refuses_a_scene_changed_after_prepare():
    cuda = mw.Scene(background=stack(), structures=sheets(), ports=(port(),))
    simulation = mw.Simulation.mom(cuda, frequencies=[1e9], config=config())
    simulation.prepare().run()
    prepared = simulation.prepare()
    # Scenes are immutable; rebinding the simulation's scene stands in for an edit after preparation.
    simulation.scene = cuda.updated(ports=(mw.LumpedPort("p", positive=(8e-5, 1e-5, Z), negative=(3e-5, 1e-5, Z),
                                  voltage_path=mw.AxisPath("x"), reference_impedance=50.0), *cuda.ports[1:]))
    with pytest.raises(ValueError, match="The Scene changed after layered MoM preparation"):
        prepared.run()
    prepared = simulation.prepare()
    simulation.scene = cuda.updated(monitors=(*cuda.monitors, mw.PointMonitor("m", (0., 0., Z))))
    with pytest.raises(NotImplementedError, match=r"- Scene.monitors = \[PointMonitor 'm'\]"):
        prepared.run()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
@pytest.mark.parametrize("model,modelled", [("sheet", False), ("closed", True)])
def test_metadata_records_whether_thickness_is_modelled_geometrically(model, modelled):
    # A PEC sheet keeps its declared thickness: "sheet" solves it infinitely thin, "closed" extrudes it.
    thick = mw.Scene(background=stack(), structures=[mw.Structure(mw.geometry.PlanarPEC(
                geometry.vertices, coordinate=Z, physical_thickness=2e-6), mw.Material.pec())
                for geometry in (structure.geometry for structure in sheets())],
                     ports=(port(),))
    options = mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5, conductor_model=model)
    result = mw.Simulation.mom(thick, frequencies=[1e9], config=mw.mom.MoMConfig(layered=options)).run()
    assert result.metadata["conductor_model"] == result.network.metadata["conductor_model"] == model
    assert result.metadata["thickness_modelled_geometrically"] is modelled
    assert result.network.metadata["thickness_modelled_geometrically"] is modelled
