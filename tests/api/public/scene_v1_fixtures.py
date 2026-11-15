"""Scenes of the committed ``witwin.scene/1`` fixtures (``tests/api/public/data/scene_v1``).

The fixtures were written by the last ``witwin.scene/1`` writer (master 3c2631231, the parent of 0.7.0
phase 8a) from these builders; ``test_scene_migrate.py`` migrates them and compares with the scenes the
same builders give today. Only API shared by both trees is used here. Regenerate with
``python scripts/dev/write_scene_v1_fixtures.py <tree at 3c2631231>``.
"""

import math

import numpy as np
import torch
import witwin.maxwell as mw

FREQUENCY = 1e9


def dielectrics():
    lam = 299792458.0 / FREQUENCY
    return mw.Scene(
        domain=mw.Domain(((-1.5 * lam, 1.5 * lam),) * 3), grid=mw.GridSpec.uniform(lam / 16),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        structures=[mw.Structure(mw.Sphere(position=(.3 * lam, .1 * lam, .1 * lam), radius=.4 * lam),
                                 mw.Material(eps_r=4)),
                    mw.Structure(mw.Box(position=(-.5 * lam, .5 * lam, .2 * lam), size=(.44 * lam, .6 * lam, .36 * lam),
                                        rotation=(.21, -.32, .43)), mw.Material(eps_r=6, sigma_e=.01))],
        sources=[mw.PointDipole((-.75 * lam, -.625 * lam, -.46875 * lam), polarization="Ez", profile="ideal",
                                source_time=mw.CW(FREQUENCY, amplitude=1.7, phase=.37))],
        monitors=[mw.monitors.PermittivityMonitor("eps", size=(2 * lam,) * 3)],
        subpixel=mw.SubpixelSpec(samples=4, averaging="polarized"),
    )


def pcb():
    nodes = np.array([-.03, -.025, -.02, -.015, -.01, -.005, 0., .005, .01, .015, .02, .026, .03])
    dielectric = mw.Material(eps_r=3.2, sigma_e=.01)
    foil = mw.media.CopperFoil(thickness=35e-6, frequency_range=(.8e9, 1.2e9), mode_count=4)
    polygon = [(-.01, -.01), (.01, -.01), (.01, .01), (-.01, .01)]
    hole = [(-.003, -.003), (.003, -.003), (.003, .003), (-.003, .003)]
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.),
        structures=[
            mw.Structure(mw.PolySlab(polygon, bounds=(-.01, -.005)), dielectric, name="board1"),
            mw.Structure(mw.Box(position=(0., 0., .0025), size=(.02, .02, .005)), dielectric, name="board2"),
            mw.Structure(mw.Box(position=(0., 0., -.005), size=(.01, .01, 0.)), foil, name="trace1"),
            mw.Structure(mw.geometry.PlanarPEC(polygon, holes=(hole,), coordinate=.015), mw.Material.pec(),
                         name="perforated", tags=("sheet",), metadata={"layer": "top", "net": [1, 2]}),
            mw.Structure(mw.geometry.SurfaceMesh([(-.01, -.01, .02), (.01, -.01, .02), (.01, .01, .02),
                                                  (-.01, .01, .02)], [(0, 1, 2), (0, 2, 3)]),
                         mw.Material.pec(), name="surface"),
        ],
        ports=[mw.LumpedPort("feed", positive=(0., 0., -.015), negative=(0., 0., -.02), voltage_path=mw.AxisPath("z"),
                             current_surface=mw.Box(position=(0., 0., -.0175), size=(.01, .01, 0.)),
                             reference_impedance=50. + 2j, termination=mw.rf.SeriesRLC(r=50., l=1e-9, c=1e-12))],
        lumped_elements=[mw.rf.Resistor("resistor", (.01, 0., -.015), (.01, 0., -.02), 75.)],
        sources=[mw.PointDipole((-.015, 0., 0.), profile="ideal", source_time=mw.CW(FREQUENCY))],
        metadata={"case": "pcb", "nested": {"values": [1, True, None]}},
    )


def periodic():
    material = mw.Material(eps_r=2., drude_poles=(mw.DrudePole(plasma_frequency=.2e9, gamma=.1e9),),
                           lorentz_poles=(mw.LorentzPole(delta_eps=.4, resonance_frequency=2e9, gamma=.1e9),))
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml", num_layers=2, strength=1.,
                                       bloch_wavevector="auto"),
        structures=[mw.Structure(mw.Box(size=(.01, .01, .01)), material)],
        sources=[mw.PlaneWave(direction=(.3, .4, math.sqrt(.75)), polarization=(.8, -.6, 0.),
                              source_time=mw.CW(FREQUENCY, phase=.2), injection=mw.TFSF.slab(axis="z", bounds=(-.02, .02)))],
        ports=[mw.LumpedPort("loaded", positive=(0., 0., .015), negative=(0., 0., .01), voltage_path=mw.AxisPath("z"),
                             current_surface=mw.Box(position=(0., 0., .0125), size=(.01, .01, 0.)),
                             termination=mw.rf.ParallelRLC(r=100., l=2e-9, c=2e-12))],
    )


def wire():
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.faces(x="pec", y="pmc", z_low="pec", z_high="pmc"),
        structures=[mw.Structure(mw.geometry.Difference(mw.Box(size=(.01,) * 3), mw.Sphere(radius=.003)),
                                 mw.Material(eps_r=2.5))],
        thin_wires=[mw.rf.ThinWire("dipole", [(-.015, 0., -.01), (-.015, 0., .01)], radius=.0001,
                                   conductor=mw.rf.WireConductor.pec(), snap="nearest"),
                    mw.rf.ThinWire("loop", [(0., 0., 0.), (.01, 0., 0.), (.01, .01, 0.), (0., 0., 0.)], .0001,
                                   mw.rf.WireConductor.finite(5.8e7))],
        sources=[mw.PointDipole((.015, 0., 0.), profile="gaussian", width=.005,
                                source_time=mw.GaussianPulse(FREQUENCY, fwidth=.2e9))],
        monitors=[mw.PointMonitor("point", (0., 0., .01)),
                  mw.PlaneMonitor("plane", axis="x", position=.01, fields=("Ey", "Hz")),
                  mw.FinitePlaneMonitor("finite", position=(0., 0., .01), size=(.02, .02, 0.)),
                  mw.FluxMonitor("flux", axis="z", position=.015),
                  mw.monitors.ClosedSurfaceMonitor.box("closed", size=(.04,) * 3),
                  mw.monitors.AngleProjectionMonitor("angle", size=(.04,) * 3, theta=(.4,), phi=(.2,), radius=1.),
                  mw.monitors.KSpaceProjectionMonitor("kspace", size=(.04,) * 3, ux=(0.,), uy=(0.,))],
    )


def mode_abc():
    lam = 299792458.0 / FREQUENCY
    return mw.Scene(
        domain=mw.Domain(((-lam, lam),) * 3), grid=mw.GridSpec.uniform(lam / 16),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=4, z_high="mode_abc", x_low="pec",
                                       mode_abc=mw.fdtd.ModeABC(order=2, frequency_range=(.8 * FREQUENCY, 1.2 * FREQUENCY))),
        structures=[mw.Structure(mw.Box(size=(.6 * lam, 1.2 * lam, 2 * lam)), mw.Material(eps_r=4))],
        sources=[mw.PointDipole((0., 0., -.5 * lam), polarization="Ey", profile="ideal", source_time=mw.CW(FREQUENCY))],
    )


def large_grid():
    """A custom grid over 4096 nodes and a float32 host mesh: arrays-table references."""
    vertices = np.array([(0., 0., 0.), (.01, 0., 0.), (0., .01, 0.), (0., 0., .01)])
    return mw.Scene(
        domain=mw.Domain(((-1., 1.), (-1., 1.), (-1., 1.))),
        grid=mw.GridSpec.custom(np.linspace(-1., 1., 4097), [-1., 0., 1.], [-1., 0., 1.]),
        structures=[mw.Structure(mw.Mesh(vertices, [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)], recenter=False),
                                 mw.Material(eps_r=2.))],
    )


def euler_rotation():
    """Lossy in form: witwin.scene/1 stored an Euler-tensor rotation as its quaternion."""
    return mw.Scene(
        domain=mw.Domain(((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
        structures=[mw.Structure(mw.Box(size=(.01, .02, .01), rotation=torch.tensor([.1, .2, .3])),
                                 mw.Material(eps_r=3.))],
    )


EXACT = {"dielectrics": dielectrics, "pcb": pcb, "periodic": periodic, "wire": wire, "mode_abc": mode_abc,
         "large_grid": large_grid}
LOSSY = {"euler_rotation": euler_rotation}
