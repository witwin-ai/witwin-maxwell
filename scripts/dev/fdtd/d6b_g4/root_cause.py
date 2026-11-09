"""Read-only compiler audit and process-local diagnostic interventions for D6b G4.

No library files are changed. Run through the prescribed GPU-0 environment.
Mass interventions are attribution experiments, not proposed discretizations.
The original mass-interventions JSON files predate the clock discovery: their
f_res values use the empty clock and must not be read as physical frequencies.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.dev.fdtd.d6b_g4 import isotropy as probe
from witwin.maxwell.compiler.thin_sheet import compile_thin_sheet
from witwin.maxwell.scene import prepare_scene


def summarize(c, scene):
    spacing = np.array([np.diff(getattr(scene, a + "_nodes64")).mean() for a in "xyz"])
    h = spacing[2]
    nodes = [getattr(scene, a + "_nodes64") for a in "xyz"]
    parents = np.unique(np.c_[c.e_component, c.e_index], axis=0)
    total = float(c.e_volume.sum() / spacing.prod())
    rows = []
    for comp in range(3):
        for z in np.unique(c.e_index[:, 2]):
            sel = (c.e_component == comp) & (c.e_index[:, 2] == z)
            changed = sel & (c.e_modified | c.e_split)
            if not changed.any():
                continue
            rows.append(dict(component="xyz"[comp], z=float(nodes[2][z] / h),
                             count=int(changed.sum()),
                             volumes=np.unique(np.round(c.e_volume[changed] / spacing.prod(), 9)).tolist()))
    # For horizontal sheets, accepted chords form a grid graph. Count full
    # four-chord cell loops, and the boundary of their union, independently.
    split_h = {(int(a), *map(int, idx)) for a, idx, extra in
               zip(c.h_component, c.h_index, c.h_extra) if extra >= 0}
    cells = set()
    for a, i, j, k in split_h:
        if a == 0 and all(key in split_h for key in
                         ((0, i + 1, j, k), (1, i, j, k), (1, i, j + 1, k))):
            cells.add((i, j))
    perimeter = sum(spacing[1] * ((i+1, j) not in cells) + spacing[1] * ((i-1, j) not in cells)
                    + spacing[0] * ((i, j+1) not in cells) + spacing[0] * ((i, j-1) not in cells)
                    for i, j in cells)
    length, width = .47 * probe.C0 / probe.F0, .015
    return dict(statistics=c.statistics, spacing=spacing.tolist(), n_e=len(c.e_volume), n_h=len(c.h_area),
                volume_cells=total, unique_parent_count=len(parents),
                closed_cell_area=len(cells) * spacing[0]*spacing[1], true_area=length * width,
                closed_cell_perimeter=perimeter, true_perimeter=2*(length+width),
                masses=rows)


def intervention(c, scene, mode):
    """Change a compiled object only, for a causal diagnostic, never a fix."""
    h = float(np.diff(scene.z_nodes64).mean())
    volume = float(np.prod([np.diff(getattr(scene, a+"_nodes64")).mean() for a in "xyz"]))
    zsheet = float(c.surface.vertices[:, 2].mean())
    z = c.e_position[:, 2]
    tangent = c.e_component != 2
    if mode == "baseline":
        return
    if mode == "no_partition":
        c.e_volume[tangent & c.e_modified] = volume
    elif mode in ("dual_all", "dual_interior"):
        # Diagonal dual weights of a 1D nonuniform mesh with a held PEC trace
        # at the actual sheet plane: half the distance to the next E node plus
        # half the distance to that held trace. Its own mass is not donated.
        near = tangent & (np.abs(z-zsheet) < h * (1-1e-8))
        if mode == "dual_interior":
            near &= c.e_modified
        else:
            low, high = c.surface.vertices.min(0), c.surface.vertices.max(0)
            near &= ((c.e_position[:, :2] >= low[:2]-h) &
                     (c.e_position[:, :2] <= high[:2]+h)).all(1)
        c.e_volume[near] = .5 * (1 + np.abs(z[near]-zsheet)/h) * volume
        c.e_modified[near] = True
    elif mode == "normal_equal":
        # Deliberately remove only the split normal mass contrast. This is an
        # attribution test, not a consistent discretization.
        near = (c.e_component == 2) & c.e_split
        c.e_volume[near] = .5*volume
    else:
        raise ValueError(mode)


def pole(t, signal, start, order=8):
    dt = float(t[1]-t[0])
    stride = max(1, int(round(1/(8*1.4*probe.F0*dt))))
    y = signal[t >= start][::stride]
    rows = len(y)//2
    matrix = np.array([y[i:i+rows+1] for i in range(len(y)-rows)])
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    v = vh[:order].conj().T
    roots = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    s = np.log(roots)/(stride*dt)
    residue = np.linalg.lstsq(roots[None, :]**np.arange(len(y))[:, None],
                             y.astype(complex), rcond=None)[0]
    keep = (s.imag/(2*np.pi) > .1*probe.F0) & (s.imag/(2*np.pi) < 1.5*probe.F0)
    ids = np.flatnonzero(keep)
    k = ids[np.argmax(np.abs(residue[keep]))]
    return dict(f_res=float(s[k].imag/(2*np.pi)), decay=float(-s[k].real),
                poles=[dict(f=float(s[i].imag/(2*np.pi)), decay=float(-s[i].real),
                            residue=float(abs(residue[i]))) for i in ids],
                singular_values=singular[:12].tolist())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("audit", "run"))
    parser.add_argument("out")
    parser.add_argument("--z", default="0.05,0.3,0.5,0.75")
    parser.add_argument("--modes", default="baseline")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--dl", type=float, default=.00375)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(dl=args.dl, steps=args.steps, cases=[])
    original = compile_thin_sheet
    import witwin.maxwell.fdtd.thin_sheet as runtime
    for zoff in map(float, args.z.split(",")):
        scene, axis = probe.scene_for(0, probe.pec_material(), report["dl"], with_sheet=True,
                                      offset=(.37, .21), zoff=zoff)
        if args.command == "audit":
            prepared = prepare_scene(scene)
            c = original(prepared, prepared.structures[0])
            report["cases"].append(dict(z=zoff, **summarize(c, prepared)))
            print(json.dumps(report["cases"][-1]), flush=True)
        else:
            empty, _ = probe.scene_for(0, probe.pec_material(), report["dl"], with_sheet=False,
                                       zoff=zoff)
            for mode in args.modes.split(","):
                start = time.time()
                def compiler(scene, structure):
                    c = original(scene, structure)
                    intervention(c, scene, mode)
                    return c
                runtime.compile_thin_sheet = compiler
                dt_empty, dt_loaded = probe.native_dt(empty), probe.native_dt(scene)
                dt = .999*min(dt_empty, dt_loaded)
                steps = math.ceil(args.steps*dt_empty/dt)
                t, incident = probe.projected_series(empty, axis, steps, dt)
                t_loaded, total = probe.projected_series(scene, axis, steps, dt)
                assert np.array_equal(t, t_loaded)
                scattered = total-incident
                fit_start = float(t[np.argmax(abs(incident))]) + 6/probe.F0
                result = dict(z=zoff, mode=mode, **pole(t, scattered, fit_start),
                              dt=dt, dt_empty=dt_empty, dt_loaded=dt_loaded, steps=steps,
                              seconds=time.time()-start,
                              memory_gib=torch.cuda.max_memory_allocated()/2**30)
                report["cases"].append(result)
                np.savez(out.with_name(out.stem+f"-{zoff}-{mode}.npz"),
                         t=t, t_loaded=t_loaded, scattered=scattered, incident=incident)
                print(json.dumps(result), flush=True)
                out.write_text(json.dumps(report, indent=2)+"\n")
            runtime.compile_thin_sheet = original
        out.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
