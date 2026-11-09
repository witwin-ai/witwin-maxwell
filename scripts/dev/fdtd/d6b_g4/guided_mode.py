"""Fourier-mode diagnostic of an infinitely long strip in a PEC enclosure.

Only the existing host compiler and a GPU Hermitian eigensolve are used. A
single x cell is analytically continued by exp(i beta x). This is a compiled
operator audit, not a new periodic runtime or a time-domain simulation. The
on-plane complex is the native PEC limit. Voltage is a strip-to-wall E line
integral; current is a surrounding H contour integral. Vacuum impedance is
normalized to one, so Z is reported in units of eta0.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import torch

import witwin.maxwell as mw
from scripts.dev.fdtd.d6b_g4.operator_probe import diagnostic_compiler
from tests.materials.thin_sheet.harness import rectangle_mesh, pec_material
from witwin.maxwell.scene import prepare_scene


def measure(zoff, mode, beta_h, cells, width_cells, tilt=0., y_offset=.21):
    h = .00375
    nodes = np.arange(-cells//2, cells//2+1)*h
    angle = math.radians(tilt)
    geometry = rectangle_mesh((.5*h, y_offset*h, zoff*h), (0., math.sin(angle), math.cos(angle)),
                               10*h, .5*width_cells*h, u_direction=(1., 0., 0.),
                               sheet_model="conformal_two_sided")
    public = mw.Scene(domain=mw.Domain(bounds=((0., h), (nodes[0], nodes[-1]), (nodes[0], nodes[-1]))),
                      grid=mw.GridSpec.custom(np.array((0., h)), nodes, nodes),
                      boundary=mw.BoundarySpec.faces(default="pec", x="pmc"), device="cuda",
                      structures=[mw.Structure(geometry=geometry, material=pec_material())])
    scene = prepare_scene(public)
    c = diagnostic_compiler(mode)(scene, scene.structures[0], margin=100)
    labels = [(int(a), int(idx[1]), int(idx[2]), int(side))
              for a, idx, side in zip(c.e_component, c.e_index, c.e_side)]
    unique = sorted(set(labels))
    lookup = {key: i for i, key in enumerate(unique)}
    group = np.array([lookup[key] for key in labels])
    volumes = np.bincount(group, weights=c.e_volume, minlength=len(unique))
    fixed = np.zeros(len(unique), dtype=bool)
    np.logical_or.at(fixed, group, c.e_fixed)
    phase = np.exp(1j*beta_h*(c.e_index[:, 0]+.5*(c.e_component==0)))
    T = sp.csr_matrix((phase, (np.arange(len(labels)), group)), shape=(len(labels), len(unique)))
    rows = np.flatnonzero(c.h_index[:, 0]==0)
    K = sp.csr_matrix((c.k_vals, (c.k_rows, c.k_cols)), shape=(len(c.h_area), len(c.e_volume)))
    reduced = K[rows] @ T
    reduced = sp.diags(np.exp(-.5j*beta_h*(c.h_component[rows]!=0))) @ reduced
    dual = c.h_dual[rows].copy()
    dual[c.h_component[rows]==0] = h
    keep = np.flatnonzero(~fixed)
    B = sp.diags(np.sqrt(dual/c.h_area[rows])) @ reduced[:, keep] @ sp.diags(1/np.sqrt(volumes[keep]))
    A = torch.as_tensor((B.conj().T@B).toarray()*h*h, device="cuda", dtype=torch.complex128)
    values, vectors = torch.linalg.eigh(A)
    values = values.cpu().numpy()
    positive = np.flatnonzero(values > 1e-8)
    target = (2*np.sin(beta_h/2))**2
    # Report all low branches so a spurious mode cannot silently replace TEM.
    branch = int(positive[np.argmin(abs(values[positive]-target))])
    e = np.zeros(len(unique), dtype=complex)
    e[keep] = vectors[:, branch].cpu().numpy()/np.sqrt(volumes[keep])
    omega = np.sqrt(values[branch])/h
    H = np.asarray(reduced@e)/(1j*omega*c.h_area[rows])
    h_lookup = {(int(c.h_component[row]), int(c.h_index[row, 1]), int(c.h_index[row, 2])): pos
                for pos, row in enumerate(rows) if c.h_extra[row] < 0}
    # Voltage along z at the closest y node to the strip centre, above sheet.
    j = int(np.argmin(abs(nodes-y_offset*h)))
    V = 0j
    for key, gid in lookup.items():
        a, jj, k, side = key
        if a != 2 or jj != j:
            continue
        ids = np.flatnonzero(group==gid)
        representative = ids[0]
        if c.e_position[representative, 2] > zoff*h-math.tan(angle)*(nodes[j]-y_offset*h):
            V += e[gid]*c.e_length[representative]
    # H contour follows dual edges at least two cells outside the strip.
    jl = int(np.searchsorted(nodes, -.5*width_cells*h))-3
    jr = int(np.searchsorted(nodes, .5*width_cells*h))+2
    kl = int(np.searchsorted(nodes, zoff*h))-3
    kr = int(np.searchsorted(nodes, zoff*h))+2
    I = sum((H[h_lookup[(1, jj, kl)]]-H[h_lookup[(1, jj, kr)]])*h for jj in range(jl+1, jr+1))
    I += sum((H[h_lookup[(2, jr, kk)]]-H[h_lookup[(2, jl, kk)]])*h for kk in range(kl+1, kr+1))
    velocity = np.sqrt(values[branch])/beta_h
    impedance = float(abs(V/I))
    electric_energy = float(np.sum(abs(e)**2*volumes))
    magnetic_energy = float(np.sum(abs(H)**2*c.h_area[rows]*dual))
    longitudinal_rows = int(np.sum((c.h_extra>=0) & (c.h_component!=0) & (c.h_index[:, 0]==0))
                            + np.sum((c.e_component==0) & c.e_split & c.e_fixed)/2)
    outside = c.e_split & (c.e_component!=0) & ~c.surface.inside(c.surface.project(c.e_position), 1e-10*h)
    phantom_rows = len(set((int(c.e_component[i]), int(c.e_index[i, 1]), int(c.e_index[i, 2]))
                           for i in np.flatnonzero(outside)))
    return dict(z=zoff, tilt=tilt, y_offset=y_offset, mode=mode, beta_h=beta_h, cells=cells, width_cells=width_cells,
                matrix_size=len(keep), phase_velocity_over_c=float(velocity),
                velocity_dispersion_corrected=float(np.sqrt(values[branch]/target)),
                impedance_over_eta0=impedance,
                capacitance_normalized=1/(velocity*impedance),
                inductance_normalized=impedance/velocity,
                capacitance_energy_normalized=electric_energy/(h*abs(V)**2),
                inductance_energy_normalized=magnetic_energy/(h*abs(I)**2),
                longitudinal_current_rows=longitudinal_rows, phantom_transverse_edge_rows=phantom_rows,
                sheet_height_span_cells=width_cells*math.sin(angle),
                low_branches_velocity_over_c=(np.sqrt(values[positive[:8]])/beta_h).tolist(),
                longitudinal_electric_energy_fraction=float(
                    np.sum(abs(e[[i for i, key in enumerate(unique) if key[0]==0]])**2 *
                           volumes[[i for i, key in enumerate(unique) if key[0]==0]]) /
                    np.sum(abs(e)**2*volumes)))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out")
    parser.add_argument("--z", default="0,0.05,0.3,0.5")
    parser.add_argument("--modes", default="baseline")
    parser.add_argument("--tilts", default="0")
    parser.add_argument("--cells", type=int, default=20)
    parser.add_argument("--width-cells", type=float, default=4.)
    parser.add_argument("--beta-h", type=float, default=.05)
    parser.add_argument("--y-offset", type=float, default=.21)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"cases": []}
    for zoff in map(float, args.z.split(",")):
        for tilt in map(float, args.tilts.split(",")):
            for mode in args.modes.split(","):
                start = time.time()
                row = measure(zoff, mode, args.beta_h, args.cells, args.width_cells, tilt, args.y_offset)
                row.update(seconds=time.time()-start, memory_gib=torch.cuda.max_memory_allocated()/2**30)
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
                out.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
