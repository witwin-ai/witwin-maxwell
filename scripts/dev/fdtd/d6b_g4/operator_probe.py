"""Finite-sheet operator attribution, with process-local diagnostic compilers.

No library files change. All propagation pairs use the fixed probe's shared
explicit timestep, common physical duration and identical-clock assertion.
"""
from __future__ import annotations

import argparse
import functools
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import witwin.maxwell as mw
from scripts.dev.fdtd.d6b_g4 import isotropy as probe
from scripts.dev.fdtd.d6b_g4.root_cause import intervention, pole
import witwin.maxwell.compiler.thin_sheet as compiler
import witwin.maxwell.fdtd.thin_sheet as runtime
from tests.materials.thin_sheet.harness import rectangle_mesh
from witwin.maxwell.scene import prepare_scene


ORIGINAL = compiler.compile_thin_sheet


def diagnostic_compiler(mode):
    if mode in ("endpoints", "endpoints_no_floor"):
        source = inspect.getsource(ORIGINAL)
        old = "if surface.inside(path_midpoint(path)[None], region_tol)[0]:"
        new = "if surface.inside(np.stack((path[0], path[-1])), region_tol).all():"
        assert source.count(old) == 1
        namespace = dict(vars(compiler))
        exec(compile(source.replace(old, new), "<dev-endpoint-diagnostic>", "exec"), namespace)
        fn = namespace["compile_thin_sheet"]
        return functools.partial(fn, stiffness_ratio=1e6, minimum_area=1e-9) if mode.endswith("no_floor") else fn
    def build(scene, structure, **options):
        kw = dict(stiffness_ratio=1e6, minimum_area=1e-9) if mode == "no_floor" else {}
        c = ORIGINAL(scene, structure, **options, **kw)
        if mode not in ("baseline", "no_floor"):
            intervention(c, scene, mode)
        return c
    return build


def scene_for(dl, zoff, tilt, *, present, native=False):
    scene, axis = probe.scene_for(0, probe.pec_material(), dl, with_sheet=present,
                                  offset=(.37, .21), zoff=zoff, native=native)
    if tilt:
        assert not native
        theta = math.radians(tilt)
        normal = np.array((0., math.sin(theta), math.cos(theta)))
        center = np.array((.37*dl, .21*dl, zoff*dl))
        if present:
            geometry = rectangle_mesh(center, normal, .235*probe.C0/probe.F0, .0075,
                                      u_direction=axis, sheet_model="conformal_two_sided")
            scene.structures[0] = mw.Structure(geometry=geometry, material=probe.pec_material())
        scene.sources[0] = mw.PlaneWave(direction=tuple(-normal), polarization=tuple(axis),
                                       source_time=scene.sources[0].source_time)
        scene.monitors[0] = mw.FieldTimeMonitor("obs", components=("Ex", "Ey"),
            position=tuple(center + .4*probe.C0/probe.F0*normal))
    return scene, axis


def audit(c, scene):
    # Copper adds explicit chord paths to precisely the same Cartesian complex.
    h = np.array([np.diff(getattr(scene, a+"_nodes64")).mean() for a in "xyz"])
    paths = c.chord_path
    outside_lengths, partial, outside_ends = 0., 0, 0
    for path in paths:
        hit = c.surface.inside(path, 1e-7*h.min())
        outside_ends += int((~hit).sum())
        partial += int(not hit.all())
        # Exact length clipping to this rectangle in local coordinates.
        facet = c.surface.facets[0]
        bounds = facet.local(c.surface.vertices)
        low, high = bounds.min(0), bounds.max(0)
        for start, end in zip(path[:-1], path[1:]):
            p, q = facet.local(np.stack((start, end)))
            delta = q-p
            t0, t1 = 0., 1.
            for a in range(2):
                if abs(delta[a]) > 1e-14:
                    left, right = sorted(((low[a]-p[a])/delta[a], (high[a]-p[a])/delta[a]))
                    t0, t1 = max(t0, left), min(t1, right)
            outside_lengths += np.linalg.norm(end-start)*(1-max(0., t1-t0))
    phantom = (c.e_component == 2) & c.e_split & ~c.surface.inside(c.surface.project(c.e_position), 1e-7*h.min())
    local = None
    if phantom.any():
        ids = np.flatnonzero(phantom)
        chosen = int(ids[np.argmin(abs(c.e_position[ids, 0]))])
        parent = c.e_index[chosen]
        pair = np.flatnonzero((c.e_component == 2) & (c.e_index == parent).all(1))
        local_rows = np.unique(c.k_rows[np.isin(c.k_cols, pair)])
        row_data = []
        for row in local_rows[c.h_component[local_rows] == 1]:
            entries = c.k_rows == row
            cols = c.k_cols[entries]
            row_data.append(dict(component=int(c.h_component[row]), index=c.h_index[row].tolist(),
                area_cells=float(c.h_area[row]/(h[0]*h[2])), extra=int(c.h_extra[row]),
                entries=[dict(component=int(c.e_component[col]), side=int(c.e_side[col]),
                              index=c.e_index[col].tolist(),
                              coefficient_cells=float(value/h[c.e_component[col]]))
                         for col, value in zip(cols, c.k_vals[entries])]))
        local = dict(parent_index=parent.tolist(),
                     position=c.surface.project(c.e_position[chosen:chosen+1])[0].tolist(),
                     side_lengths_cells=(c.e_length[pair]/h[2]).tolist(),
                     side_masses_cells=(c.e_volume[pair]/h.prod()).tolist(), h_rows=row_data)
    return dict(statistics=c.statistics, spacing=h.tolist(),
                chords=len(paths), partial_chords=partial, outside_endpoints=outside_ends,
                chord_counts_by_face_normal={str(a): int(np.sum(c.chord_face[:, 0]==a)) for a in (-1, 0, 1, 2)},
                outside_chord_length=float(outside_lengths),
                chord_weighted_area=float(np.sum(c.chord_length*c.chord_width)),
                sheet_loop_area=float(c.sheet_face_area.sum()/2),
                true_area=.47*probe.C0/probe.F0*.015,
                electric_volume_by_component=[float(c.e_volume[c.e_component==a].sum()) for a in range(3)],
                min_split_mass_fraction=float(np.min(c.e_volume[c.e_split]/h.prod())),
                magnetic_area_ratio_min=float(np.min(c.h_true_area/c.h_area)),
                phantom_normal_parent_count=len(np.unique(c.e_index[phantom], axis=0)),
                representative_phantom=local)


def native_dump(dl):
    from witwin.maxwell.compiler.copper_foil import compile_copper_faces, compile_copper_response
    public, _ = scene_for(dl, 0, 0, present=True, native=True)
    material = mw.CopperFoil(thickness=35e-6, frequency_range=(.5e9, 1.5e9))
    public.structures[0] = mw.Structure(geometry=public.structures[0].geometry, material=material)
    scene = prepare_scene(public)
    faces = compile_copper_faces(scene, device="cuda")
    h = np.array([np.diff(getattr(scene, a+"_nodes64")).mean() for a in "xyz"])
    trace = []
    for face in faces:
        for index in face.indices.cpu().numpy():
            p = np.array([getattr(scene, a+"_nodes64")[index[i]] for i, a in enumerate("xyz")])
            p[face.component] += .5*h[face.component]
            trace.append(dict(component=int(face.component), index=index.tolist(), position=p.tolist(),
                              length=float(h[face.component]), width=float(h[1-face.component]),
                              side_volumes=[float(h.prod()/2)]*2))
    response = compile_copper_response(material)
    # Record actual native rim conductance scaling as well as bare material D.
    # A fixed safe timestep makes these implicit update coefficients reviewable.
    solver = mw.Simulation.fdtd(public, frequencies=[probe.F0], backend="native",
                               run_time=mw.TimeConfig(time_steps=1, dt=2e-12)).prepare().solver
    updates, scale_by_trace = [], {}
    for entry in solver._copper_foils:
        component = int(entry["face"].component)
        index = entry["indices"].cpu().numpy()
        scale = entry["admittance_scale"].cpu().numpy()
        for ijk, value in zip(index, scale):
            scale_by_trace[(component, *map(int, ijk))] = float(value)
        updates.append(dict(component=component, trace_count=len(index),
            admittance_scale_min=float(scale.min()), admittance_scale_max=float(scale.max()),
            capacity_min=float(entry["capacity"].min()), capacity_max=float(entry["capacity"].max()),
            bulk_conductance_max=float(entry["conductance"].max()),
            effective_two_port=entry["effective"].cpu().tolist()))
        np.savez(f"/tmp/d6b-native-update-{component}.npz", indices=index, admittance_scale=scale,
                 capacity=entry["capacity"].cpu().numpy(), conductance=entry["conductance"].cpu().numpy(),
                 effective=entry["effective"].cpu().numpy(), inverse=entry["inverse"].cpu().numpy())
    for item in trace:
        item["admittance_scale"] = scale_by_trace[(item["component"], *item["index"])]
    np.savez("/tmp/d6b-native-foil.npz", component=[t["component"] for t in trace],
             index=[t["index"] for t in trace], position=[t["position"] for t in trace],
             length=[t["length"] for t in trace], width=[t["width"] for t in trace],
             side_volumes=[t["side_volumes"] for t in trace])
    return dict(trace_count=len(trace), counts=[sum(t["component"]==a for t in trace) for a in (0, 1)],
                weighted_area=sum(t["length"]*t["width"] for t in trace),
                dc_sheet_conductance=material.conductivity*material.thickness,
                two_port_D=response.state_space.D.tolist(), arrays="/tmp/d6b-native-foil.npz",
                update_dt=float(solver.dt), native_updates=updates,
                memory_gib=torch.cuda.max_memory_allocated()/2**30,
                representative_traces=[t for t in trace if abs(t["position"][0]) < h[0]])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("audit", "run", "native"))
    parser.add_argument("out")
    parser.add_argument("--dl", type=float, default=.00375)
    parser.add_argument("--z", default="0.5")
    parser.add_argument("--tilts", default="0")
    parser.add_argument("--modes", default="baseline,endpoints")
    parser.add_argument("--duration", type=float, default=39e-9)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(dl=args.dl, duration=args.duration, cases=[])
    if args.command == "native":
        report.update(native_dump(args.dl))
        out.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({key: value for key, value in report.items() if key != "traces"}))
        return
    for z in map(float, args.z.split(",")):
        for tilt in map(float, args.tilts.split(",")):
            for mode in args.modes.split(","):
                start = time.time()
                build = diagnostic_compiler(mode)
                loaded, axis = scene_for(args.dl, z, tilt, present=True)
                row = dict(z=z, tilt=tilt, mode=mode)
                if args.command == "audit":
                    loaded.structures[0] = mw.Structure(geometry=loaded.structures[0].geometry,
                        material=mw.CopperFoil(thickness=35e-6, frequency_range=(.5e9, 1.5e9)))
                    scene = prepare_scene(loaded)
                    c = build(scene, scene.structures[0])
                    row.update(audit(c, scene))
                    arrays = {key: value for key, value in vars(c).items() if isinstance(value, np.ndarray)}
                    array_path = Path("/tmp")/f"d6b-complex-{args.dl}-{z}-{tilt}-{mode}.npz"
                    np.savez(array_path, **arrays)
                    row["arrays"] = str(array_path)
                else:
                    empty, _ = scene_for(args.dl, z, tilt, present=False)
                    runtime.compile_thin_sheet = build
                    try:
                        dt_empty, dt_loaded = probe.native_dt(empty), probe.native_dt(loaded)
                        dt = .999*min(dt_empty, dt_loaded)
                        steps = math.ceil(args.duration/dt)
                        t, incident = probe.projected_series(empty, axis, steps, dt)
                        t_loaded, total = probe.projected_series(loaded, axis, steps, dt)
                        assert np.array_equal(t, t_loaded)
                    finally:
                        runtime.compile_thin_sheet = ORIGINAL
                    late = float(t[np.argmax(abs(incident))])+6/probe.F0
                    row.update(pole(t, total-incident, late), dt=dt, steps=steps,
                               total_only=pole(t, total, late))
                    np.savez(Path("/tmp")/f"d6b-fields-{args.dl}-{z}-{tilt}-{mode}.npz",
                             t=t, incident=incident, total=total)
                row.update(seconds=time.time()-start, memory_gib=torch.cuda.max_memory_allocated()/2**30)
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
                out.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
