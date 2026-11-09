"""Explicit graded Cartesian edge grids for physical volume-copper controls."""
from __future__ import annotations

import math
import numpy as np
import witwin.maxwell as mw

from scripts.dev.fdtd.pcb_copper_performance import skin_depth


def graded_edge_axis(original, edges, mandatory, half_planes, *, skin, cells_per_skin):
    """Integrate an analytic spacing metric and preserve all physical anchors.

    Spacing is h within two skin depths of a metal edge, then grows linearly
    at slope 1/4. Every original node remains an anchor except roundoff duplicates
    or the interior of an explicitly protected half-grid pair.
    """
    original = np.asarray(original,dtype=np.float64)
    edges = np.unique(np.asarray(edges,dtype=np.float64))
    h = skin/cells_per_skin
    base = float(np.diff(original).max())
    anchors = original.copy()
    for value in np.concatenate((edges,np.asarray(mandatory,dtype=np.float64))):
        uncertainty = 16*np.finfo(np.float64).eps*max(abs(value),abs(original[0]),abs(original[-1]))
        anchors = np.append(anchors[abs(anchors-value) > uncertainty],value)
    for plane in half_planes:
        if len(edges) and np.min(abs(edges-plane)) <= 2*h:
            raise ValueError("Physical H-contour plane intersects the protected copper-edge grid.")
        anchors = anchors[abs(anchors-plane) > 1.5*h]
        anchors = np.append(anchors,[plane-h/2,plane+h/2])
    anchors = np.unique(anchors)

    def spacing(x):
        distance = np.min(abs(np.asarray(x)[...,None]-edges),axis=-1)
        return np.minimum(base,h+.25*np.maximum(distance-2*skin,0.))

    # Metric slopes change only at these analytic breakpoints. Integrate each
    # linear-spacing segment exactly, including the constant-spacing plateau.
    breaks = np.concatenate((edges,edges-2*skin,edges+2*skin,
        edges-(2*skin+4*(base-h)),edges+(2*skin+4*(base-h)),.5*(edges[1:]+edges[:-1])))
    points = [anchors[0]]
    for low,high in zip(anchors[:-1],anchors[1:]):
        if high-low <= h*(1+1e-10):
            points.append(high)
            continue
        pieces = np.unique(np.concatenate(([low,high],breaks[(breaks > low) & (breaks < high)])))
        left,right = spacing(pieces[:-1]),spacing(pieces[1:])
        slope = (right-left)/np.diff(pieces)
        constant = abs(slope) < 1e-12
        measure = np.empty_like(slope)
        measure[constant] = np.diff(pieces)[constant]/left[constant]
        measure[~constant] = np.log(right[~constant]/left[~constant])/slope[~constant]
        cumulative = np.concatenate(([0.],np.cumsum(measure)))
        cells = math.ceil(float(cumulative[-1]))
        metric = np.linspace(0.,cumulative[-1],cells+1)[1:-1]
        index = np.minimum(np.searchsorted(cumulative,metric,side="right")-1,len(slope)-1)
        local = metric-cumulative[index]
        offsets = np.empty_like(local)
        flat = constant[index]
        offsets[flat] = left[index[flat]]*local[flat]
        offsets[~flat] = left[index[~flat]]*np.expm1(slope[index[~flat]]*local[~flat])/slope[index[~flat]]
        points.extend(pieces[index]+offsets)
        points.append(high)
    return np.asarray(points,dtype=np.float64)


def refine_pcb_edges(scene,frequency,cells_per_skin):
    axes = [getattr(scene.grid,axis+"_coords") for axis in "xyz"]
    copper = [s for s in scene.structures if s.name in ("positive-pad","negative-pad")]
    for axis in (0,1):
        edges,mandatory,half_planes = [],[],[]
        for structure in copper:
            center,size = float(structure.geometry.position[axis]),float(structure.geometry.size[axis])
            edges.extend((center-size/2,center+size/2))
        for port in scene.ports:
            mandatory.extend((port.positive[axis],port.negative[axis]))
            aperture = port.voltage_aperture
            mandatory.extend((float(aperture.position[axis]-aperture.size[axis]/2),
                              float(aperture.position[axis]+aperture.size[axis]/2)))
            contour = port.current_surface
            if float(contour.size[axis]) == 0:
                half_planes.append(float(contour.position[axis]))
            else:
                half_planes.extend((float(contour.position[axis]-contour.size[axis]/2),
                                    float(contour.position[axis]+contour.size[axis]/2)))
        axes[axis] = graded_edge_axis(axes[axis],edges,mandatory,half_planes,
            skin=skin_depth(frequency),cells_per_skin=cells_per_skin)
    return scene.clone(grid=mw.GridSpec.custom(*axes),metadata={**scene.metadata,
        "edge_cells_per_skin":cells_per_skin,
        "qualification":"actual edge-refined input; response and heat convergence remain unqualified"})
