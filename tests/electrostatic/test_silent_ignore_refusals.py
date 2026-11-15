"""Electrostatics/capacitance refuse every declaration they would otherwise drop.

All checks run at ``Simulation.electrostatic/capacitance(...)`` construction on
CPU scenes; the trainable-permittivity gradient is checked against central
finite differences and its forward values against the fixed-value solve.
"""
from __future__ import annotations

import pytest
import torch

import witwin.core as core
import witwin.maxwell as mw
from witwin.maxwell.capability import check_scene
from witwin.maxwell.compiler.electrostatic import compile_electrostatics

pytestmark = pytest.mark.host_validation

DOMAIN = ((-0.5, 0.5),) * 3


def _terminals():
    return [
        mw.electrostatic.ElectrostaticTerminal("a", geometry=mw.Box(position=(-0.25, 0, 0), size=(0.2, 0.2, 0.2)), potential=1.0),
        mw.electrostatic.ElectrostaticTerminal("b", geometry=mw.Box(position=(0.25, 0, 0), size=(0.2, 0.2, 0.2)), grounded=True),
    ]


def _scene(material=None, *, terminals=None, **kwargs):
    structures = [] if material is None else [
        mw.Structure(name="slab", geometry=mw.Box(position=(0.0, 0, 0), size=(0.2, 0.6, 0.6)), material=material)]
    return mw.Scene(domain=mw.Domain(bounds=DOMAIN), grid=mw.GridSpec.uniform(0.1), structures=structures, electrostatic_terminals=_terminals() if terminals is None else terminals,
                    **kwargs)


def _owners(scene):
    return [item.owner for item in check_scene(scene, "electrostatic").unsupported]


def _refused(scene, *fragments, capacitance=False):
    entry = mw.Simulation.capacitance if capacitance else mw.Simulation.electrostatic
    with pytest.raises(mw.UnsupportedFeatureError, match=r"cannot represent \d+ scene feature") as error:
        entry(scene)
    for fragment in fragments:
        assert fragment in str(error.value), (fragment, str(error.value))


@pytest.mark.parametrize("capacitance", [False, True])
def test_full_wave_collections_are_refused(capacitance):
    scene = _scene(
        sources=[mw.PlaneWave(source_time=mw.CW(frequency=1e9))],
        monitors=[mw.PointMonitor("m", (0, 0, 0), ("Ez",))],
        ports=[mw.LumpedPort("p", positive=(0.05, 0, 0), negative=(-0.05, 0, 0), voltage_path=mw.AxisPath("x"),
                             current_surface=mw.Box(position=(0, 0, 0), size=(0.1, 0.1, 0.1)))],
        lumped_elements=[mw.rf.Resistor("r", (-0.25, 0, 0), (0.25, 0, 0), 50.0)],
        internal_absorbers=[mw.fdtd.InternalAbsorber(position=(0, 0, 0.3), size=(float("inf"), float("inf"), 0))],
    )
    _refused(scene, "Scene.sources (1)", "Scene.monitors (1)", "Scene.ports (1)", "Scene.lumped_elements (1)",
             "Scene.internal_absorbers (1)", capacitance=capacitance)


@pytest.mark.parametrize("capacitance", [False, True])
def test_full_wave_boundary_is_refused(capacitance):
    scene = _scene(boundary=mw.BoundarySpec.faces(default="none", z="pml", num_layers=4))
    _refused(scene, "Boundary faces ['pml'] of Scene.boundary", "BoundarySpec.none()", capacitance=capacitance)
    report = check_scene(_scene(boundary=mw.BoundarySpec.pec()), "capacitance")
    assert [item.key for item in report.unsupported] == ["boundary.pec"]
    with pytest.raises(mw.UnsupportedFeatureError, match="Boundary faces \\['periodic'\\] of Scene.boundary"):
        compile_electrostatics(_scene(boundary=mw.BoundarySpec.periodic()), device="cpu")


def test_circuits_symmetry_and_coordinates_are_refused():
    circuit = mw.circuits.Circuit("rc")
    node = circuit.node("a")
    circuit.add(mw.rf.Resistor("R1", node, circuit.ground, 50.0))
    _refused(_scene(circuits=(circuit,)), "Scene.circuits (1)")
    _refused(_scene(symmetry=("PEC", None, None)), "Scene.symmetry")
    bend = mw.coordinates.BendCoordinates(curvature=0.5, origin=(0.2, -0.1, 0.3),
                              frame=((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
    assert "Scene.coordinates" in _owners(_scene(coordinates=bend))


def test_network_and_wire_collections_are_refused():
    scene = _scene()
    scene = scene.updated(thin_wires=(
        *scene.thin_wires,
        mw.rf.ThinWire(name="w", points=((-0.1, 0.3, 0), (0.1, 0.3, 0)), radius=1e-3,
                    conductor=mw.rf.WireConductor.pec()),
    ))
    _refused(scene, "Scene.thin_wires (1)")
    scene = _scene()
    # Any declared network block, never read by the solve; the stand-in bypasses the constructor check.
    object.__setattr__(scene, "networks", (object(),))
    assert "Scene.networks (1)" in _owners(scene)


@pytest.mark.parametrize("material", [
    mw.media.LossyMetalMedium(conductivity=5.8e7),
    mw.media.Medium2D(sigma_s=1e-3),
    mw.media.CopperFoil(thickness=2e-6, frequency_range=(0.5e9, 4e9)),
    mw.media.SurfaceImpedanceMedium(impedance=mw.media.LossyMetalMedium(conductivity=5.8e7).rational_surface_impedance((1e9, 4e9))),
], ids=["lossy_metal", "medium2d", "copper_foil", "surface_impedance"])
@pytest.mark.parametrize("capacitance", [False, True])
def test_conductor_media_are_refused(material, capacitance):
    _refused(_scene(material), f"{type(material).__name__} is a conductor model",
             "ElectrostaticTerminal", capacitance=capacitance)


def test_conductivity_is_refused():
    _refused(_scene(mw.Material(eps_r=4.0, sigma_e=1e7)), "Structure 'slab': sigma_e")
    _refused(_scene(mw.Material(eps_r=4.0, sigma_e_tensor=mw.DiagonalTensor3(1.0, 1.0, 1.0))), "sigma_e")
    mw.Simulation.electrostatic(_scene(mw.Material(eps_r=4.0)), config=mw.electrostatic.ElectrostaticConfig(device="cpu"))


@pytest.mark.parametrize("fields,expected", [
    (dict(conductor_model="perfect"), "conductor_model"),
    (dict(dispersion=core.PowerLawDispersion(reference_frequency_hz=1e9, eps_r_exponent=-0.5)), "dispersion"),
    (dict(thickness_m=0.02, geometry_mode="volumetric"), "thickness_m, geometry_mode"),
])
def test_core_surface_fields_are_refused(fields, expected):
    _refused(_scene(core.PhysicalMaterial(eps_r=4.0, **fields)), f"core material field(s) {expected}")


_POLES = {"debye": mw.DebyePole(delta_eps=1.0, tau=1e-9), "drude": mw.DrudePole(plasma_frequency=1e9, gamma=1e8),
          "lorentz": mw.LorentzPole(delta_eps=1.0, resonance_frequency=1e9, gamma=1e8)}


def _perturbed(pole=None, **fields):
    """A PerturbationMedium on a plain dielectric, or on a one-pole base whose pole strength it shifts."""
    base = mw.Material(eps_r=4.0, **({} if pole is None else {f"{pole}_poles": (_POLES[pole],)}))
    if pole is not None:
        fields[f"{pole}_sensitivities"] = (0.1,)
    return mw.media.PerturbationMedium(base, perturbation=torch.full((2, 2, 2), 0.5), **fields)


_PERTURBATION = "Structure 'slab': PerturbationMedium; the static solve evaluates only the base permittivity"
_NONLINEAR = "Structure 'slab': nonlinear susceptibility (kerr_chi3 or nonlinearity)"


# One case per field the static solves previously admitted and dropped (capacitance was bit-identical with and
# without them).
@pytest.mark.parametrize("material,expected", [
    (mw.Material(eps_r=4.0, kerr_chi3=1e-10), _NONLINEAR),
    (mw.Material(eps_r=4.0, nonlinearity=(mw.media.NonlinearSusceptibility(chi2=1e-12),)), _NONLINEAR),
    (mw.Material(eps_r=4.0, nonlinearity=(mw.media.TwoPhotonAbsorption(beta=1e-11),)), _NONLINEAR),
    (mw.Material(eps_r=4.0, breakdown=mw.breakdown.DielectricBreakdown(critical_field=1e6, post_breakdown_conductivity=1.0)),
     "Structure 'slab': DielectricBreakdown is a time-domain feedback model"),
    (_perturbed(), _PERTURBATION),
    (_perturbed(eps_sensitivity=2.0), _PERTURBATION),
    (_perturbed("debye"), _PERTURBATION),
    (_perturbed("drude"), _PERTURBATION),
    (_perturbed("lorentz"), _PERTURBATION),
], ids=["kerr_chi3", "nonlinearity_chi2", "nonlinearity_tpa", "breakdown", "perturbation", "eps_sensitivity",
        "debye_sensitivities", "drude_sensitivities", "lorentz_sensitivities"])
@pytest.mark.parametrize("capacitance", [False, True])
def test_dropped_material_fields_are_refused(material, expected, capacitance):
    _refused(_scene(material), expected, capacitance=capacitance)


def test_trainable_terminal_geometry_is_refused():
    position = torch.tensor((-0.25, 0.0, 0.0), requires_grad=True)
    terminals = [mw.electrostatic.ElectrostaticTerminal("a", geometry=mw.Box(position=position, size=(0.2, 0.2, 0.2)), potential=1.0)]
    _refused(_scene(terminals=terminals), "Terminal 'a': trainable geometry")


def test_trainable_terminal_and_boundary_values_are_refused():
    box = mw.Box(position=(0, 0, 0), size=(0.2, 0.2, 0.2))
    value = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(NotImplementedError, match="Trainable ElectrostaticTerminal 'a' potential"):
        mw.electrostatic.ElectrostaticTerminal("a", geometry=box, potential=value)
    with pytest.raises(NotImplementedError, match="Trainable ElectrostaticTerminal 'a' charge"):
        mw.electrostatic.ElectrostaticTerminal("a", geometry=box, charge=value)
    with pytest.raises(NotImplementedError, match="Trainable ElectrostaticBoundarySpec z_low value"):
        mw.electrostatic.ElectrostaticBoundarySpec(default="neumann", z_low=("dirichlet", value))
    with pytest.raises(NotImplementedError, match="Trainable ElectrostaticBoundarySpec.dirichlet value"):
        mw.electrostatic.ElectrostaticBoundarySpec.dirichlet(value)


def test_capacitance_refuses_floating_terminals_and_free_charge():
    floating = mw.electrostatic.ElectrostaticTerminal("f", geometry=mw.Box(position=(0.0, 0.3, 0), size=(0.2, 0.1, 0.2)), charge=1e-12)
    scene = _scene(terminals=[*_terminals(), floating],
                   charge_densities=[mw.electrostatic.ChargeDensity(mw.Box(position=(0, 0, 0), size=(0.2, 0.2, 0.2)), 1e-6)])
    _refused(scene, "Terminal 'f': floating (charge=) conductor", "Scene.charge_densities (1)", capacitance=True)
    # The electrostatic solve handles both.
    mw.Simulation.electrostatic(scene, config=mw.electrostatic.ElectrostaticConfig(device="cpu"))


def test_one_error_lists_every_item_and_compile_refuses_too():
    scene = _scene(mw.media.LossyMetalMedium(conductivity=5.8e7), sources=[mw.PlaneWave(source_time=mw.CW(frequency=1e9))],
                   symmetry=("PEC", None, None))
    _refused(scene, "Scene.sources (1)", "Scene.symmetry", "LossyMetalMedium")
    with pytest.raises(mw.UnsupportedFeatureError, match=r"Simulation\.electrostatic\(\.\.\.\) cannot represent 3"):
        compile_electrostatics(scene, device="cpu")


def _capacitance(eps):
    solver = mw.electrostatic.ElectrostaticSolverConfig(tolerance=1e-13)
    cap = mw.electrostatic.capacitance(mw.Simulation.capacitance(_scene(mw.Material(eps_r=eps)), config=mw.electrostatic.CapacitanceConfig(solver=solver, device="cpu")).run())
    es = mw.electrostatic.electrostatic(mw.Simulation.electrostatic(_scene(mw.Material(eps_r=eps)), config=mw.electrostatic.ElectrostaticConfig(solver=solver, device="cpu")).run())
    return torch.cat((cap.matrix.reshape(-1), es.terminal_charge("a").reshape(1), es.energy.reshape(1))) * 1e11


def test_trainable_permittivity_gradient_matches_finite_differences():
    eps = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)
    values = _capacitance(eps)
    assert values.requires_grad
    # Same forward values as the fixed-value solve of the same float64 permittivity.
    assert torch.equal(values.detach(), _capacitance(eps.detach()))
    adjoint = torch.stack([torch.autograd.grad(value, eps, retain_graph=True)[0] for value in values])
    step = 1e-4
    with torch.no_grad():
        central = (_capacitance(torch.tensor(3.7 + step, dtype=torch.float64))
                   - _capacitance(torch.tensor(3.7 - step, dtype=torch.float64))) / (2 * step)
    torch.testing.assert_close(adjoint, central, rtol=1e-7, atol=0)


def test_trainable_permittivity_under_a_tensor_dielectric_fails_closed():
    eps = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)
    scene = _scene(mw.Material(eps_r=eps))
    scene = scene.updated(structures=(
        *scene.structures,
        mw.Structure(geometry=mw.Box(position=(0, 0.35, 0), size=(0.6, 0.1, 0.6)),
                     material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0))),
    ))
    with pytest.raises(NotImplementedError, match="tensor"):
        mw.Simulation.capacitance(scene, config=mw.electrostatic.CapacitanceConfig(device="cpu")).run()
