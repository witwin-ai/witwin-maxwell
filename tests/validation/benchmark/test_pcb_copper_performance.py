"""Physical reference planes and independent heat checks for copper timings."""
import math

import numpy as np
import pytest
import torch
import witwin.maxwell as mw

from scripts.dev.fdtd.pcb_copper_performance import (
    EPS0, ETA0, MU0, SIGMA, film_reference, film_scene, normal_grid, skin_depth, pcb_volume_scene,
)


@pytest.mark.parametrize("thickness", (18e-6, 35e-6))
@pytest.mark.parametrize("frequency", (1e9, 2.45e9, 4e9))
def test_film_transfer_absorption_matches_independent_volume_joule_integral(thickness, frequency):
    reflection, transmission, absorption = film_reference(thickness, frequency)
    # An independently integrated interior field verifies the small heat term,
    # which a near-unit reflection comparison alone cannot constrain.
    nodes, weights = np.polynomial.legendre.leggauss(256)
    x = (nodes+1)*thickness/2
    s = -2j*math.pi*frequency
    gamma = np.sqrt(s*MU0*(SIGMA+s*EPS0))
    field = ((1+reflection)*np.sinh(gamma*(thickness-x))
             + transmission*np.sinh(gamma*x))/np.sinh(gamma*thickness)
    heat = ETA0*SIGMA*thickness/2*np.sum(weights*abs(field)**2)
    assert absorption > 0
    assert heat == pytest.approx(absorption, rel=2e-10)


@pytest.mark.parametrize("thickness", (18e-6, 35e-6))
def test_normal_grid_resolves_both_faces_without_a_spurious_tiny_transition_cell(thickness):
    frequency = 2.45e9
    x = normal_grid(thickness, frequency, 4)
    n = math.ceil(thickness*4/skin_depth(frequency))
    np.testing.assert_allclose(x, -x[::-1], atol=1e-18, rtol=0)
    inside = x[(x >= -thickness/2) & (x <= thickness/2)]
    assert len(inside) == n+1
    assert inside[0] == -thickness/2
    assert inside[-1] == thickness/2
    assert np.diff(x).min() == pytest.approx(thickness/n, rel=1e-10)
    assert np.diff(inside).max() <= skin_depth(frequency)/4*(1+1e-10)


def test_surface_and_volume_launch_have_identical_distance_from_physical_faces():
    thickness = 18e-6
    surface = film_scene("surface", thickness, 2.45e9, 4, device="cpu")
    volume = film_scene("volume", thickness, 2.45e9, 4, device="cpu")
    assert surface.ports[0].positive[0] == pytest.approx(volume.ports[0].positive[0]+thickness/2)
    assert volume.structures[0].material.sigma_e == SIGMA
    assert volume.structures[0].geometry.size[0] == pytest.approx(thickness)
    assert surface.structures[0].geometry.size[0] == 0
    assert surface.structures[0].material.thickness == thickness


@pytest.mark.parametrize("family,thickness,bottom", (("flexible",18e-6,50e-6),("rigid",35e-6,400e-6)))
def test_pcb_volume_control_restores_real_thickness_conductivity_and_feed_plane(family, thickness, bottom):
    scene = pcb_volume_scene(family,2.45e9,4,device="cpu")
    copper = [s for s in scene.structures if s.name in ("positive-pad","negative-pad")]
    assert len(copper) == 2
    for structure in copper:
        assert structure.material.sigma_e == SIGMA
        assert float(structure.geometry.size[2]) == pytest.approx(thickness)
        assert float(structure.geometry.position[2]) == pytest.approx(bottom+thickness/2)
    for port in scene.ports:
        assert port.positive[2] == pytest.approx(bottom+thickness)
        assert port.negative[2] == pytest.approx(bottom+thickness)
    z = scene.grid.z_coords
    assert float(np.diff(z).min()) >= skin_depth(2.45e9)/4*.9
    for coordinate in (0.,bottom,bottom+thickness):
        assert np.min(abs(z-coordinate)) < 1e-15


@pytest.mark.parametrize("family",("flexible","rigid"))
@pytest.mark.parametrize("kind",("surface","volume"))
def test_pcb_campaign_declares_the_same_physical_frequency_and_complete_power_outputs(family,kind):
    from scripts.dev.fdtd.pcb_copper_campaign import scene_control
    scene,definition = scene_control(family,kind,2.45e9,4,device="cpu")
    assert {monitor.name for monitor in scene.monitors} == {"radiation","bulk_loss"}
    assert len(scene.ports) == 1
    assert len([s for s in scene.structures if s.name in ("positive-pad","negative-pad")]) == 2
    for conductor in definition["conductors"]:
        low,high = conductor["bounds_m"][4:6]
        assert (high-low == 0) == (kind == "surface")
    assert next(m for m in scene.monitors if m.name == "bulk_loss").frequencies == (2.45e9,)


@pytest.mark.parametrize("family",("flexible","rigid"))
def test_edge_refinement_preserves_physical_ports_and_compiles_strict_native_contours(family):
    from scripts.dev.fdtd.pcb_copper_grid import refine_pcb_edges
    from witwin.maxwell.scene import prepare_scene
    scene = pcb_volume_scene(family,2.45e9,4,device="cpu")
    refined = refine_pcb_edges(scene,2.45e9,4)
    assert refined.ports == scene.ports
    for axis in "xy":
        coordinates = getattr(refined.grid,axis+"_coords")
        assert np.diff(coordinates).min() > skin_depth(2.45e9)/4*.45
        assert len(coordinates) > len(getattr(scene.grid,axis+"_coords"))
    prepared = prepare_scene(refined,lazy_meshgrid=True)
    port, = prepared.compile_ports(device="cpu")
    assert port.voltage_indices.numel() > 0
    assert all(value.numel() > 0 for value in port.current_indices)


def test_edge_axis_metric_matches_independent_closed_form_cell_positions():
    from scripts.dev.fdtd.pcb_copper_grid import graded_edge_axis
    skin,h = 1e-6,.25e-6
    nodes = graded_edge_axis([-10e-6,10e-6],[0.],[],[],skin=skin,cells_per_skin=4)
    np.testing.assert_allclose(nodes,-nodes[::-1],rtol=0,atol=1e-18)
    positive = nodes[nodes >= 0]
    cells = math.ceil(2*skin/h+4*math.log((h+.25*8*skin)/h))
    assert len(positive) == cells+1
    metric = np.where(positive <= 2*skin,positive/h,
        2*skin/h+4*np.log1p(.25*np.maximum(positive-2*skin,0)/h))
    np.testing.assert_allclose(metric,np.linspace(0,metric[-1],cells+1),rtol=1e-13,atol=1e-13)


def test_isolated_material_heat_projections_use_identical_native_fields_and_volumes():
    from scripts.dev.fdtd.pcb_copper_campaign import volume_material_heat
    from witwin.maxwell.scene import prepare_scene
    # This is a postprocessing algebra control, not a CPU field solve.
    scene = mw.Scene(domain=mw.Domain(bounds=((- .015,.015),)*3),grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.none(),device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=4,averaging="arithmetic"),
        structures=(mw.Structure(name="positive-pad",geometry=mw.Box(position=(0.,0.,.05),
            size=(1.,1.,.1)),material=mw.Material(eps_r=1.,sigma_e=SIGMA)),
            mw.Structure(name="board",geometry=mw.Box(position=(0.,0.,-.05),size=(1.,1.,.1)),
                material=mw.Material(eps_r=2.,sigma_e=.1))),
        monitors=(mw.PowerLossMonitor("bulk_loss",position=(0.,0.,0.),size=(.04,.04,.04)),))
    prepared = prepare_scene(scene,lazy_meshgrid=True)
    shape = (prepared.Nx,prepared.Ny,prepared.Nz)
    fields = {name:torch.full(tuple(n-(dim == axis) for axis,n in enumerate(shape)),value,
        dtype=torch.complex128) for dim,(name,value) in enumerate((("EX",2.),("EY",3.),("EZ",5.)))}
    result = mw.Result(method="fdtd",scene=scene,prepared_scene=prepared,frequency=2.45e9,fields=fields)
    copper = volume_material_heat(result,copper=True)
    substrate = volume_material_heat(result,copper=False)
    # Complementary half-spaces make the exact native volume fraction 1/2
    # without a curved-corner subpixel quadrature error. Retain the full native
    # endpoint control volumes selected by this deliberately enclosing monitor.
    weighted_volume = 0.
    for dim,value in enumerate((2.,3.,5.)):
        volume = math.prod(float(np.sum(getattr(prepared,"d"+axis+(
            "_primal64" if index == dim else "_dual64")))) for index,axis in enumerate("xyz"))
        weighted_volume += value**2*volume
    assert float(copper[0]) == pytest.approx(.25*SIGMA*weighted_volume,rel=2e-6)
    assert float(substrate[0]) == pytest.approx(.25*.1*weighted_volume,rel=2e-6)


@pytest.mark.parametrize("kind",("volume","surface"))
def test_throughput_snapshot_keeps_actual_material_grid_and_rf_excitation(kind):
    from scripts.dev.fdtd.pcb_copper_campaign import scene_control
    from scripts.dev.fdtd.pcb_copper_throughput import physical_snapshot
    scene,_ = scene_control("flexible",kind,2.45e9,4,device="cpu")
    snapshot = physical_snapshot(scene,2.45e9)
    np.testing.assert_array_equal(snapshot["grid_nodes_m"]["z"],scene.grid.z_coords)
    copper = [s for s in snapshot["structures"] if s["name"] in ("positive-pad","negative-pad")]
    assert all(s["material"]["sigma_e" if kind == "volume" else "conductivity"] == SIGMA for s in copper)
    assert snapshot["drive"] == {"kind":"GaussianPulse","frequency":2.45e9,"fwidth":2.45e9,"amplitude":1.}
    assert snapshot["ports"][0]["positive_m"] == list(scene.ports[0].positive)
