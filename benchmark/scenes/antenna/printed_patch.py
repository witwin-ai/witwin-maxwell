"""Fixed-geometry printed patch with finite-area galvanic probe excitation.

All dimensions, feed aperture and current reference plane remain fixed under
independent lateral and substrate grid refinement. Air and PML are physical
inputs rather than cell counts that resize the antenna when a grid changes.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import witwin.maxwell as mw

PATCH_LENGTH = .028
PATCH_WIDTH = .034
SUBSTRATE_HEIGHT = .003
SUBSTRATE_EPS = 2.2
GROUND_HALF_X = .020
GROUND_HALF_Y = .023
FEED_X = -.005
FEED_WIDTH = .002
FEED_GAP = .001
CURRENT_RADIUS = .0025
BOUNDS = ((-.035, .035), (-.040, .040), (-.020, .033))


def _physical_box(position, size):
    """Retain declared SI faces before geometric inclusion on the native grid."""
    return mw.Box(position=torch.tensor(position, dtype=torch.float64),
                  size=torch.tensor(size, dtype=torch.float64))


def _axis_nodes(bounds, anchors, refinement, fine_dl, air_dl, half_planes, local_refinements=()):
    """Piecewise uniform SI grid retaining fixed node and half-node references."""
    low, high = bounds
    intervals = [(p-fine_dl/2, p+fine_dl/2) for p in half_planes]
    required = sorted(set((low, high, *anchors, *refinement,
                           *(x for pair in intervals for x in pair),
                           *(x for lo, hi, _ in local_refinements for x in (lo, hi)))))
    nodes = [required[0]]
    for first, second in zip(required[:-1], required[1:]):
        if first < low or second > high:
            raise ValueError("Printed-patch grid anchors must stay inside the domain.")
        if any(first >= a-1e-14 and second <= b+1e-14 for a, b in intervals):
            # The desired current reference is a native half-grid coordinate.
            count = 1
        else:
            midpoint = (first+second)/2
            target = fine_dl if refinement[0] <= midpoint <= refinement[1] else air_dl
            for lo, hi, step in local_refinements:
                if lo <= midpoint <= hi:
                    target = min(target, step)
            count = max(1, math.ceil((second-first)/target-1e-12))
        nodes.extend(np.linspace(first, second, count+1)[1:])
    nodes = np.asarray(nodes, dtype=np.float64)
    for plane in half_planes:
        if not np.any(np.isclose((nodes[:-1]+nodes[1:])/2, plane, rtol=0, atol=1e-14)):
            raise ValueError("A required physical anchor conflicts with a current reference half plane.")
    return nodes


def printed_patch_scene(*, lateral_dl=.001, substrate_cells=3, air_dl=.002,
                        frequencies=(3.0e9, 3.2e9, 3.4e9), pml_layers=8,
                        air_padding=0., device="cuda", radiation_monitor=True, edge_dl=None,
                        metal="pec", dielectric_sigma=0., copper_conductivity=5.8e7,
                        copper_frequency_range=(.1e9, 10e9)):
    """Create one fixed physical antenna with PEC or finite 35 um copper foils.

    The 2 mm square probe joins the sheet patch to the positive source terminal
    at z=1 mm. A power-dual 2 mm square aperture drives the physical 1 mm gap to
    ground. The current contour is fixed at z=0.5 mm with 5 mm square extent.
    """
    if metal not in ("pec", "copper") or dielectric_sigma < 0:
        raise ValueError("metal must be pec or copper, and dielectric_sigma nonnegative.")
    if edge_dl is not None and edge_dl <= 0:
        raise ValueError("Printed-patch edge_dl must be positive.")
    if lateral_dl <= 0 or air_dl <= 0 or substrate_cells < 1:
        raise ValueError("Printed-patch grid steps and substrate cell count must be positive.")
    if air_padding < 0:
        raise ValueError("Printed-patch air_padding must be nonnegative.")
    layer_dl = SUBSTRATE_HEIGHT/int(substrate_cells)
    bounds = tuple((lo-air_padding, hi+air_padding) for lo, hi in BOUNDS)
    x_edges, y_edges = [], []
    if edge_dl is not None:
        x_edges = [(value-.002, value+.002, edge_dl)
                   for value in (-GROUND_HALF_X, GROUND_HALF_X, -PATCH_LENGTH/2, PATCH_LENGTH/2)]
        x_edges.extend((value-.001, value+.001, edge_dl) for value in (FEED_X-FEED_WIDTH/2, FEED_X+FEED_WIDTH/2))
        y_edges = [(value-.002, value+.002, edge_dl)
                   for value in (-GROUND_HALF_Y, GROUND_HALF_Y, -PATCH_WIDTH/2, PATCH_WIDTH/2)]
        y_edges.extend((value-.001, value+.001, edge_dl) for value in (-FEED_WIDTH/2, FEED_WIDTH/2))
    x = _axis_nodes(bounds[0], (-PATCH_LENGTH/2, PATCH_LENGTH/2, FEED_X,
                              FEED_X-FEED_WIDTH/2, FEED_X+FEED_WIDTH/2),
                    (-GROUND_HALF_X, GROUND_HALF_X), lateral_dl, air_dl,
                    (FEED_X-CURRENT_RADIUS, FEED_X+CURRENT_RADIUS), x_edges)
    y = _axis_nodes(bounds[1], (-PATCH_WIDTH/2, PATCH_WIDTH/2, 0., -FEED_WIDTH/2, FEED_WIDTH/2),
                    (-GROUND_HALF_Y, GROUND_HALF_Y), lateral_dl, air_dl,
                    (-CURRENT_RADIUS, CURRENT_RADIUS), y_edges)
    z = _axis_nodes(bounds[2], (0., FEED_GAP, SUBSTRATE_HEIGHT),
                    (-.002, SUBSTRATE_HEIGHT+.002), layer_dl, air_dl, (FEED_GAP/2,))
    def foil(name, half_x, half_y, z):
        if metal == "pec":
            geometry = mw.PlanarPEC.rectangle(((-half_x, half_x), (-half_y, half_y)),
                                              coordinate=z, physical_thickness=35e-6)
            material = mw.Material.pec()
        else:
            geometry = mw.Box(position=torch.tensor((0., 0., z), dtype=torch.float64),
                              size=torch.tensor((2*half_x, 2*half_y, 0.), dtype=torch.float64))
            material = mw.CopperFoil(thickness=35e-6, conductivity=copper_conductivity,
                                    frequency_range=copper_frequency_range)
        return mw.Structure(geometry, material, name=name)

    structures = [
        mw.Structure(_physical_box(position=(0, 0, SUBSTRATE_HEIGHT/2),
                            size=(2*GROUND_HALF_X, 2*GROUND_HALF_Y, SUBSTRATE_HEIGHT)),
                     mw.Material(eps_r=SUBSTRATE_EPS, sigma_e=dielectric_sigma), name="substrate"),
        foil("ground", GROUND_HALF_X, GROUND_HALF_Y, 0.),
        foil("patch", PATCH_LENGTH/2, PATCH_WIDTH/2, SUBSTRATE_HEIGHT),
        mw.Structure(_physical_box(position=(FEED_X, 0., (FEED_GAP+SUBSTRATE_HEIGHT)/2),
                            size=(FEED_WIDTH, FEED_WIDTH, SUBSTRATE_HEIGHT-FEED_GAP)),
                     mw.Material.pec(), name="probe"),
    ]
    port = mw.LumpedPort("feed", positive=(FEED_X, 0., FEED_GAP), negative=(FEED_X, 0., 0.),
        voltage_path=mw.AxisPath("z"), reference_impedance=50.,
        current_surface=_physical_box(position=(FEED_X, 0., FEED_GAP/2), size=(2*CURRENT_RADIUS, 2*CURRENT_RADIUS, 0.)),
        voltage_aperture=_physical_box(position=(FEED_X, 0., FEED_GAP/2), size=(FEED_WIDTH, FEED_WIDTH, FEED_GAP)))
    monitors = []
    if dielectric_sigma:
        monitors.append(mw.PowerLossMonitor("substrate_loss",
            position=(0., 0., SUBSTRATE_HEIGHT/2),
            size=(2*GROUND_HALF_X, 2*GROUND_HALF_Y, SUBSTRATE_HEIGHT),
            frequencies=tuple(frequencies), channels=("conduction",)))
    if radiation_monitor:
        monitors.append(mw.ClosedSurfaceMonitor.box("radiation", position=(0., 0., .0065),
                         size=(.052, .060, .033), frequencies=tuple(frequencies)))
    return mw.Scene(domain=mw.Domain(bounds=bounds), grid=mw.GridSpec.custom(x, y, z),
                    boundary=mw.BoundarySpec.pml(num_layers=pml_layers), device=device,
                    structures=structures, ports=[port], monitors=monitors,
                    metadata={"physical_model": "fixed-geometry-printed-patch-v1",
                              "lateral_dl_m": lateral_dl, "edge_dl_m": edge_dl, "substrate_cells": int(substrate_cells),
                              "air_dl_m": air_dl, "physical_copper_thickness_m": 35e-6,
                              "metal": metal, "dielectric_sigma_S_per_m": dielectric_sigma,
                              "copper_conductivity_S_per_m": copper_conductivity if metal == "copper" else None,
                              "copper_frequency_range_hz": tuple(copper_frequency_range) if metal == "copper" else None})
