"""Every type ``witwin.scene/2`` registers round-trips exactly through strict JSON text.

One representative instance per registered type (CPU). An instance is exact when the rebuilt object has
the same structural fingerprint (class, every attribute, tensors by dtype, shape and bytes:
``benchmark.core_path.fingerprint.canonical``) and re-serializes to the same document.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import witwin.core as core
import witwin.maxwell as mw

from benchmark.core_path.fingerprint import canonical
from witwin.maxwell.io.scene_dict import _Reader, _Writer, _codecs


def _f64(*values):
    return torch.tensor(values, dtype=torch.float64)


def _rational_model():
    return mw.rf.rational.RationalModel(poles=torch.tensor([-1.0e9, -3.0e9], dtype=torch.float64),
                                        residues=torch.tensor([[[2.0e9, 1.0e9]]], dtype=torch.float64),
                                        direct=torch.tensor([[0.5]], dtype=torch.float64))


def _network_data():
    frequencies = torch.tensor([1.0e9, 2.0e9, 3.0e9], dtype=torch.float64)
    s = torch.tensor([[[0.1 + 0.05j, 0.8], [0.8, 0.1 - 0.02j]]] * 3, dtype=torch.complex128)
    return mw.rf.NetworkData(frequencies=frequencies, s=s, z0=50.0, port_names=("a", "b"),
                             metadata={"source": "unit", "order": (1, 2)})


def _state_space():
    return mw.rf.rational.StateSpaceNetwork(
        A=torch.tensor([[-1.0e9]], dtype=torch.float64), B=torch.tensor([[1.0, 0.5]], dtype=torch.float64),
        C=torch.tensor([[1.0e7], [2.0e6]], dtype=torch.float64),
        D=torch.tensor([[0.02, 0.0], [0.0, 0.02]], dtype=torch.float64), representation="Y", port_order=("a", "b"))


def _fit_report(cls):
    return cls(rms_error=1e-3, max_error=2e-3, relative_rms_error=1e-4, relative_max_error=2e-4,
               frequency_band=(1e9, 3e9), order=4, iterations=6, condition_numbers=(1.5, 2.5),
               warnings=("note",))


def _circuit():
    circuit = mw.circuits.Circuit("lumped", parameters={"r0": 50.0}, metadata={"case": "unit"})
    a, b = circuit.node("a"), circuit.node("b")
    ground = circuit.ground
    circuit.add(mw.circuits.VoltageSource("V1", a, ground, voltage=1.0,
                                          waveform=mw.circuits.SineWaveform(0.0, 1.0, 1e9)))
    circuit.add(mw.rf.Resistor("R1", a, b, 50.0))
    circuit.add(mw.rf.Inductor("L1", b, ground, 1e-9))
    circuit.add(mw.rf.Inductor("L2", a, ground, 2e-9))
    circuit.add(mw.circuits.MutualInductor("K1", "L1", "L2", 0.5))
    circuit.add(mw.circuits.Diode("D1", b, ground, saturation_current=1e-14))
    circuit.bind_port("feed", positive=a, negative=ground)
    circuit.set_initial_condition(b, 0.25, constraint=False)
    return circuit


def _samples():
    box, sphere = mw.Box(position=_f64(0.1, 0.0, 1 / 3), size=(0.2, 0.3, 0.4)), mw.Sphere(radius=0.05)
    polygon = [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)]
    hole = [(-0.003, -0.003), (0.003, -0.003), (0.003, 0.003), (-0.003, 0.003)]
    material = mw.Material(eps_r=2.5, sigma_e=0.01)
    pole_d, pole_dr, pole_l = mw.DebyePole(1.0, 1e-10), mw.DrudePole(0.1e9, 0.1e9), mw.LorentzPole(1.0, 2e9, 0.1e9)
    model = _rational_model()
    impedance = mw.media.RationalSurfaceImpedance(
        torch.tensor([-2.0e9], dtype=torch.float64), torch.tensor([1.0e9], dtype=torch.float64), 0.02,
        frequency_range=(1.0e9, 5.0e9), representation="Z", certificate_samples=48)
    ports_box = mw.Box(position=(0.0, 0.0, 0.5e-3), size=(5e-3, 5e-3, 0.0))
    coords = ([-0.01, 0.0, 0.01],) * 3
    currents = {"Jx": torch.ones((3, 3, 3), dtype=torch.complex64), "Jz": torch.zeros((3, 3, 3))}
    fields = {"Ey": torch.ones((1, 3, 3)), "Hz": torch.full((1, 3, 3), 0.5)}
    segment = mw.rf.AxisSegment((0.0, 0.0, 0.0), (0.0, 0.0, 1e-3))
    loop = mw.rf.RectangleLoop(center=(0.0, 0.0, 0.0), size=(0.0, 1e-3, 1e-3))
    wire = mw.rf.ThinWire("wire", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.01)], 1e-4, mw.rf.WireConductor.finite(5.8e7))
    waveform = mw.esd.ESDWaveform.iec_61000_4_2(4000.0)
    node_a, ground = mw.circuits.CircuitNode("a"), mw.circuits.CircuitNode("0")
    perturbation = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).reshape(2, 2, 2)
    return [
        # geometry
        box, sphere, mw.Cylinder(radius=0.1, height=0.2, axis="x", rotation=(0.1, 0.2, 0.3)),
        mw.geometry.Ellipsoid(radii=(0.1, 0.2, 0.3)), mw.geometry.Cone(), mw.geometry.Pyramid(),
        mw.geometry.Prism(num_sides=5), mw.geometry.Torus(),
        mw.geometry.HollowBox(outer_size=(1.0, 1.0, 1.0), inner_size=(0.5, 0.5, 0.5)),
        mw.geometry.ExtrudedPolygon([(0., 0.), (.01, 0.), (.01, .004), (.004, .004), (0., .01)], height=.002),
        mw.PolySlab(polygon, bounds=(-0.01, 0.0), sidewall_angle=0.1),
        mw.geometry.ComplexPolySlab(loops=[polygon, hole], bounds=(-0.001, 0.001)),
        mw.Mesh([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
                [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)], recenter=False),
        mw.geometry.Union(box, sphere), mw.geometry.Intersection(box, sphere, smoothness=0.01),
        mw.geometry.Difference(box, sphere, mw.Cylinder()),
        mw.geometry.PlanarPEC(polygon, holes=(hole,), coordinate=0.015),
        mw.geometry.SurfaceMesh([(-.01, -.01, .02), (.01, -.01, .02), (.01, .01, .02), (-.01, .01, .02)],
                                [(0, 1, 2), (0, 2, 3)]),
        mw.Structure(box, material, name="body", priority=2, tags=("a",), metadata={"type": "user", "n": [1, 2]}),
        mw.Structure(mw.Box(rotation=_f64(0.1, -0.2, 0.3)), material,
                     phase_screen=core.PhaseScreen(height=torch.rand(4, 4), height_scale_m=0.05,
                                                   correlation=core.SurfaceRoughness(rms_height_m=1e-6, correlation_length_y_m=2e-5,
                                                                                     correlation_length_x_m=1e-5))),
        mw.Mesh(torch.tensor([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)], dtype=torch.float64),
                torch.tensor([(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)], dtype=torch.int32), recenter=False,
                topology_diagnostics=True, rotation=(0.1, 0.2, 0.3)),
        mw.Mesh(np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]),
                np.array([(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]), position=(0.1, 0.0, 0.0)),
        core.PhysicalMaterial(eps_r=4.0, sigma_e=0.01, dispersion=core.PowerLawDispersion(
            reference_frequency_hz=1e8, eps_r_exponent=-0.5, sigma_e_exponent=1.0)),
        core.PhysicalMaterial(layers=(core.MaterialLayer(thickness_m=0.01, eps_r=2.0),), thickness_m=0.02),
        mw.mom.MaxwellPlaneModes(1e9, torch.tensor([[0, 0, 20.9 + 0j]], dtype=torch.complex128),
                                 torch.tensor([[1.0 + 0j, 0, 0]], dtype=torch.complex128),
                                 torch.tensor([[0, 1.0 / 376.73 + 0j, 0]], dtype=torch.complex128)),
        # scene specs
        mw.Domain(((-1.0, 1.0), (-2.0, 2.0), (-0.5, 0.5))),
        mw.GridSpec.custom(np.linspace(-1.0, 1.0, 5), [-1.0, 1.0], [-1.0, 0.0, 1.0]),
        mw.BoundarySpec.faces(x="bloch", y="pml", z_low="pec", z_high="pmc", num_layers=3, strength=2.0,
                              bloch_wavevector=(0.1, 0.2, 0.0)),
        mw.SubpixelSpec(samples=4, averaging="polarized"),
        mw.grid.MeshOverrideStructure(box, 0.01),
        mw.grid.LayerRefinementSpec(axis="z", bounds=(-0.01, 0.01), bounds_refinement=mw.grid.GridRefinement(dl=0.001),
                                    corner_finder=mw.grid.CornerFinderSpec(),
                                    corner_refinement=mw.grid.GridRefinement(refinement_factor=2)),
        mw.grid.GridRefinement(num_cells=3), mw.grid.CornerFinderSpec(angle_threshold=0.2),
        mw.MaterialRegion("design", box, torch.rand((3, 4, 5), dtype=torch.float64), eps_bounds=(1.0, 4.0),
                          filter_radius=0.01, projection_beta=4.0),
        mw.fdtd.MurABC(order=2), mw.fdtd.ModeABC(mode_index=1, polarization="Ey", frequency_range=(1e9, 2e9)),
        mw.fdtd.InternalAbsorber(position=(0.0, 0.0, 0.5), size=(math.inf, math.inf, 0.0)),
        # media
        mw.Material(eps_r=2.0, mu_r=1.5, sigma_m=0.1, mass_density=1000.0, debye_poles=(pole_d,),
                    drude_poles=(pole_dr,), lorentz_poles=(pole_l,), mu_debye_poles=(mw.DebyePole(0.1, 1e-10),),
                    modulation=mw.media.ModulationSpec(frequency=0.35e9, amplitude=0.2, phase=0.4)),
        mw.Material(eps_r=4.0, breakdown=mw.breakdown.DielectricBreakdown(critical_field=1e6,
                                                                          post_breakdown_conductivity=1.0)),
        mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0), kerr_chi3=1e-20),
        mw.Material(epsilon_tensor=mw.media.Tensor3x3(((2.0, 0.1, 0.0), (0.1, 3.0, 0.0), (0.0, 0.0, 4.0)))),
        mw.media.ComplexTensor3x3(((2.0 + 0.1j, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0 - 0.2j))),
        mw.Material(nonlinearity=mw.media.NonlinearSusceptibility(chi3=1e-20)),
        mw.Material(nonlinearity=mw.media.TensorNonlinearSusceptibility(chi2=torch.zeros((3, 3, 3)))),
        mw.Material(nonlinearity=mw.media.TwoPhotonAbsorption(beta=1e-11)),
        mw.Material.pec(), mw.Material.pmc(),
        mw.media.CopperFoil(thickness=35e-6, frequency_range=(0.8e9, 1.2e9), mode_count=4),
        mw.media.LossyMetalMedium(conductivity=5.8e7, roughness=mw.media.HammerstadRoughness(rms_height=1e-6)),
        mw.media.LossyMetalMedium(conductivity=1e7, roughness=mw.media.GroisseRoughness(rms_height=1e-6)),
        mw.media.LossyMetalMedium(conductivity=1e7, roughness=mw.media.HurayRoughness(
            sphere_radius=0.4e-6, sphere_density=1.5e11, relative_area=1.3)),
        pole_d, pole_dr, pole_l,
        mw.media.CustomDebyePole(delta_eps=torch.ones(2, 2, 2), tau=1e-11),
        mw.media.CustomDrudePole(plasma_frequency=torch.full((2, 2, 2), 1e9), gamma=1e8),
        mw.media.CustomLorentzPole(delta_eps=torch.ones(2, 2, 2), resonance_frequency=2e9, gamma=1e8),
        mw.media.Graphene(chemical_potential=0.2, scattering_time=1.0e-13, temperature=300.0),
        mw.media.Medium2D(sigma_s=2.5e-3, name="sheet"),
        mw.media.GyromagneticFerrite(eps_r=14.5, saturation_magnetization=1.40e5, bias_field=(0.0, 0.0, 1.75e5),
                                     gilbert_damping=2.0e-3),
        mw.media.PerturbationMedium(mw.Material(eps_r=3.0, debye_poles=(pole_d,)), perturbation=perturbation,
                                    eps_sensitivity=0.5),
        impedance, mw.media.SurfaceImpedanceMedium(impedance=impedance, name="coating"),
        mw.breakdown.SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5),
        mw.media.WidebandDebye(eps_inf=3.0, delta_eps=0.1, f1=1e3, f2=1e12),
        mw.media.PolePermittivity(3.0, (pole_d,)),
        mw.mom.LayerStack((mw.mom.Layer("air", 1e-3),), mw.mom.HalfSpace(), mw.mom.HalfSpace()),
        # sources
        mw.TFSF(((-0.01, 0.01),) * 3), mw.TFSF.slab(axis="z", bounds=(-0.01, 0.01)),
        mw.CW(1e9, amplitude=1.7, phase=0.37), mw.GaussianPulse(1e9, 0.2e9, delay=1e-9),
        mw.sources.RickerWavelet(1e9), mw.sources.BroadbandPulse(freq_range=(1e9, 3e9)),
        mw.sources.CustomSourceTime(times=_f64(0.0, 1e-9, 2e-9), amplitudes=_f64(0.0, 1.0, 0.0),
                                    characteristic_frequency=1e9),
        mw.sources.BasebandGaussianPulse(twidth=1e-10), mw.sources.BasebandRectangularPulse(10e-12, 200e-12),
        mw.PointDipole((0.0, 0.0, 0.01), polarization=(0.0, 0.6, 0.8), profile="gaussian", width=0.005),
        mw.PlaneWave(direction=(0.2, 0.0, 1.0), polarization=(1.0, 0.0, -0.2), source_time=mw.CW(1e9)),
        mw.GaussianBeam(beam_waist=0.01, focus=(0.0, 0.0, 0.01)),
        mw.sources.AstigmaticGaussianBeam(beam_waist=(0.01, 0.02)),
        mw.ModeSource(position=(0.0, 0.0, 0.0), size=(0.0, 0.01, 0.01), direction="+"),
        mw.UniformCurrentSource(size=(0.01, 0.0, 0.01), polarization="Ez"),
        mw.sources.CustomCurrentSource(mw.sources.CurrentDataset(coords, currents), source_time=mw.CW(1e9)),
        mw.sources.CustomFieldSource(mw.sources.FieldDataset(([0.0], [-0.01, 0.0, 0.01], [-0.01, 0.0, 0.01]), fields),
                                     source_time=mw.CW(1e9)),
        waveform,
        mw.esd.ESDCurrentSource("gun", port="feed", waveform=waveform),
        mw.esd.ESDVoltageSource("gun2", port="feed", waveform=waveform, discharge_resistance=330.0,
                                storage_capacitance=150e-12),
        mw.esd.MeasuredWaveform(np.linspace(0.0, 1e-9, 5), np.array([0.0, 1.0, 2.0, 1.0, 0.0]), units="A",
                                bandwidth=1e9, provenance={"probe": "unit-test"}),
        # monitors
        mw.PointMonitor("point", (0.0, 0.0, 0.01), fields=("Ez",)),
        mw.PlaneMonitor("plane", axis="x", position=0.01, fields=("Ey", "Hz"), frequencies=(1e9,)),
        mw.FinitePlaneMonitor("finite", position=(0.0, 0.0, 0.01), size=(0.02, 0.02, 0.0)),
        mw.FluxMonitor("flux", axis="z", position=0.015, normal_direction="-"),
        mw.monitors.ClosedSurfaceMonitor.box("closed", size=(0.04,) * 3),
        mw.ModeMonitor("mode", position=(0.0, 0.0, 0.0), size=(0.0, 0.01, 0.01), direction="+"),
        mw.monitors.AngleProjectionMonitor("angle", size=(0.04,) * 3, theta=(0.4,), phi=(0.2,), radius=1.0),
        mw.monitors.CartesianProjectionMonitor("cartesian", size=(0.04,) * 3, x=(0.0,), y=(0.0,), distance=1.0),
        mw.monitors.KSpaceProjectionMonitor("kspace", size=(0.04,) * 3, ux=(0.0,), uy=(0.0,)),
        mw.monitors.PermittivityMonitor("eps", size=(0.02,) * 3),
        mw.monitors.MediumMonitor("medium", size=(0.02,) * 3),
        mw.monitors.DiffractionMonitor("diffraction", position=(0.0, 0.0, 0.01), size=(math.inf, math.inf, 0.0)),
        mw.monitors.DipoleEmissionMonitor("emission", source_name="dipole"),
        mw.monitors.FieldTimeMonitor("time", components=("Ez",), position=(0.0, 0.0, 0.0), interval=2),
        mw.monitors.AuxFieldTimeMonitor("aux", components=("Pz",), position=(0.0, 0.0, 0.0)),
        mw.monitors.FluxTimeMonitor("flux_time", axis="z", position=0.0),
        mw.monitors.SurfaceFieldTimeMonitor("surface", size=(0.02, 0.02, 0.02)),
        mw.monitors.GaussianOverlapMonitor("gauss", position=(0.0, 0.0, 0.01), size=(0.02, 0.02, 0.0), beam_waist=0.005),
        mw.monitors.AstigmaticGaussianOverlapMonitor("astig", position=(0.0, 0.0, 0.01), size=(0.02, 0.02, 0.0),
                                                     beam_waist=(0.005, 0.006)),
        mw.monitors.PowerLossMonitor("loss", position=(0.0, 0.0, 0.0), size=(0.02, 0.02, 0.02)),
        mw.rf.WireMonitor("wire_state", "wire", frequencies=(3e9,)),
        mw.esd.ComponentStressMonitor("stress", port="feed", rating=mw.esd.ComponentRating(voltage=1.0),
                                      voltage_series="voltage", current_series="current"),
        mw.sar.IncidentPowerDensityMonitor("ipd", axis="z", position=0.0),
        mw.esd.BreakdownMonitor("breakdown", region=box, critical_field=1e6),
        # ports and circuits elements
        mw.rf.TerminalPort("terminal", mw.rf.TerminalRef("signal"), mw.rf.TerminalRef("ground"), mw.AxisPath("z"),
                           reference_plane=0.5e-3, reference_impedance=50.0, current_surface=ports_box),
        mw.ModePort("mode_port", position=(0.0, 0.0, 0.0), size=(0.0, 0.01, 0.01), direction="+"),
        mw.LumpedPort("feed", positive=(0.0, 0.0, -0.015), negative=(0.0, 0.0, -0.02), voltage_path=mw.AxisPath("z"),
                      current_surface=mw.Box(position=(0.0, 0.0, -0.0175), size=(0.01, 0.01, 0.0)),
                      reference_impedance=50.0 + 2j, termination=mw.rf.SeriesRLC(r=50.0, l=1e-9, c=1e-12)),
        mw.rf.WavePort("wave", position=(0.0, 0.0, 0.0), size=(0.0, 0.01, 0.01), direction="+", reference_plane=0.0,
                       modes=(mw.rf.WaveModeSpec("te"),)),
        mw.rf.WaveModeSpec("tem", 0, name="even", polarization="Ez", voltage_path=((0.0, 0.05, 0.02), (0.0, 0.05, 0.16)),
                           current_contour=mw.Box(position=(0.005, 0.05, 0.0), size=(0.0, 0.07, 0.07))),
        mw.rf.TerminalWavePort("P1", position=(0, 0, 4e-4), size=(0, 2e-3, 8e-4), direction="+", reference_plane=0.0,
                               terminals=(mw.rf.TerminalRef("a"),), reference=mw.rf.TerminalRef("g")),
        mw.rf.CoaxialLumpedPort("coax", center=(0, 0, 1e-3), inner_diameter=1e-3, outer_diameter=3e-3,
                                normal_axis="z", direction="+", reference_impedance=50.0),
        mw.AxisPath("z"), mw.rf.TerminalRef("signal"), mw.rf.WireNodeRef("wire", 0),
        mw.rf.WirePortBinding.nodes(negative=mw.rf.WireNodeRef("wire", 0), positive=mw.rf.WireNodeRef("wire", 1)),
        mw.rf.PECFrame(3),
        segment, mw.rf.PolylinePath(((0.0, 0.0, 0.0), (0.0, 1e-3, 0.0), (0.0, 1e-3, 1e-3))),
        mw.rf.CircleLoop((0, -3e-4, 2e-4), 1.6e-4, "x", 16),
        mw.rf.PolygonLoop(((0, -5e-4, 0.5e-4), (0, -1e-4, 0.5e-4), (0, -3e-4, 4e-4))), loop,
        mw.rf.CustomVoltageIntegral(segment), mw.rf.CustomCurrentIntegral(loop),
        mw.rf.CompositeVoltageIntegral(((0.5, mw.rf.CustomVoltageIntegral(segment)),
                                        (0.5, mw.rf.CustomVoltageIntegral(segment)))),
        mw.rf.CompositeCurrentIntegral(((1.0, mw.rf.CustomCurrentIntegral(loop)),)),
        mw.rf.Resistor("r", (0.0, 0.0, 0.0), (0.0, 0.0, 0.01), 50.0),
        mw.rf.Capacitor("c", (0.0, 0.0, 0.0), (0.0, 0.0, 0.01), 1e-12),
        mw.rf.Inductor("l", (0.0, 0.0, 0.0), (0.0, 0.0, 0.01), 1e-9),
        mw.rf.SeriesRLC(r=50.0, l=1e-9, c=1e-12), mw.rf.ParallelRLC(r=100.0, l=2e-9, c=2e-12),
        mw.rf.CoaxialLumpedResistor("load", center=(0, 0, 0), inner_diameter=1e-3, outer_diameter=3e-3,
                                    normal_axis="z", resistance=50.0),
        _network_data(),
        mw.rf.NetworkBlock("block", network=_network_data(), connections={1: "p1", 2: "p2"}),
        mw.rf.NetworkBlock("modeled", network=_network_data(), connections={"a": "p1", "b": "p2"},
                           model=_state_space(), fit=False),
        mw.rf.TouchstoneNetwork("touchstone", connections={1: "p1", 2: "p2"}, network=_network_data()),
        _state_space(), model, mw.rf.rational.RationalFitConfig(order=6, band=(1e9, 2e9)),
        _fit_report(mw.rf.rational.FitReport), _fit_report(mw.rf.rational.NetworkFitReport),
        wire, mw.rf.ThinWire("loop", [(0., 0., 0.), (.01, 0., 0.), (.01, .01, 0.), (0., 0., 0.)], 1e-4,
                             mw.rf.WireConductor.pec()),
        mw.rf.WireConductor.pec(), mw.rf.WireEnd.grounded(structure="ground"),
        node_a, mw.circuits.PortBinding("feed", node_a, ground), mw.circuits.MNAConfig(),
        mw.circuits.PulseWaveform(0.0, 1.0, delay=1e-9, rise=1e-10, fall=1e-10, width=1e-9, period=4e-9),
        mw.circuits.SineWaveform(0.0, 1.0, 1e9), mw.circuits.PiecewiseLinearWaveform((0.0, 1e-9), (0.0, 1.0)),
        mw.circuits.VoltageSource("V", node_a, ground, voltage=1.0),
        mw.circuits.CurrentSource("I", node_a, ground, current=1e-3),
        mw.circuits.VoltageControlledVoltageSource("E", node_a, ground, node_a, ground, 2.0),
        mw.circuits.VoltageControlledCurrentSource("G", node_a, ground, node_a, ground, 1e-3),
        mw.circuits.CurrentControlledVoltageSource("H", node_a, ground, "V", 10.0),
        mw.circuits.CurrentControlledCurrentSource("F", node_a, ground, "V", 2.0),
        mw.circuits.MutualInductor("K", "L1", "L2", 0.5),
        mw.circuits.TimedSwitch("S", node_a, ground, transition_times=(7.5e-12, 15.5e-12), on_resistance=25.0,
                                off_resistance=1e6),
        mw.circuits.Diode("D", node_a, ground, saturation_current=1e-14),
        mw.circuits.PiecewiseLinearIV("J", node_a, ground, voltages=(-1.0, 0.0, 1.0), currents=(-1e-3, 0.0, 2e-3)),
        mw.circuits.PolynomialIV("P", node_a, ground, coefficients=(0.0, 1e-3)),
        mw.circuits.VoltageDependentCapacitor("C", node_a, ground, q_coefficients=torch.tensor([0.0, 1e-12])),
        mw.circuits.NonlinearSolveConfig(), _circuit(),
        mw.electrostatic.ElectrostaticTerminal("t", geometry=mw.Box(position=(0., 0., .5), size=(.1, .1, .1)),
                                               potential=1.0),
        mw.electrostatic.ChargeDensity(mw.Box(position=(0., 0., .5), size=(.1, .1, .1)), 1e-6),
        mw.coordinates.BendCoordinates(curvature=2.0),
    ]


SAMPLES = _samples()


def _round_trip(sample):
    arrays = {}
    writer = _Writer(None)
    encoded = writer.encode(sample, "sample")
    text = json.dumps({"value": encoded, "materials": writer.materials}, allow_nan=False)
    document = json.loads(text)
    restored = _Reader(document, arrays).decode(document["value"], "sample")
    again = _Writer(None)
    assert json.dumps({"value": again.encode(restored, "sample"), "materials": again.materials},
                      allow_nan=False) == text, type(sample).__name__
    return restored


def _tags(value):
    if isinstance(value, list):
        return {tag for item in value for tag in _tags(item)}
    if isinstance(value, dict):
        own = {value["type"]} if isinstance(value.get("type"), str) else set()
        return own.union(*(_tags(item) for item in value.values()))
    return set()


def test_every_registered_type_has_a_sample():
    written = set()
    for sample in SAMPLES:
        writer = _Writer(None)
        written |= _tags(writer.encode(sample, "sample")) | _tags(writer.materials)
    registered = {codec.tag for codec in _codecs().values()}
    assert registered <= written, sorted(registered - written)


def _structure(value):
    """canonical() without the tensor-revision stamps (data pointers) of a tensor-authored Mesh."""
    if isinstance(value, dict):
        return {key: ("revision" if key == "_topology_diagnostic_revision" else _structure(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_structure(item) for item in value]
    return value


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: f"{type(sample).__name__}")
def test_registered_type_round_trips_exactly(sample):
    restored = _round_trip(sample)
    assert type(restored) is type(sample)
    assert _structure(canonical(restored)) == _structure(canonical(sample))


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: f"{type(sample).__name__}")
def test_every_constructor_value_is_written(sample):
    codec = _codecs()[type(sample)]
    record = _Writer(None).record(sample, "sample")
    assert set(record) == {"type", *codec.fields}
