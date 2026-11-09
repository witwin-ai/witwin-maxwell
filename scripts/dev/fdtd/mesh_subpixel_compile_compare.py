"""Fresh-interpreter GPU snapshots for exact material/compiler comparisons.

Export master with git archive, then run this script once per revision, scene
and backend through the GPU wrapper. Timings include FDFD's lazy material init.
Dataset conversion reads the source dataset and writes only to scratch.
"""
from __future__ import annotations

import tempfile
import argparse
import hashlib
import importlib.metadata
import json
import os
from dataclasses import replace
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import witwin.maxwell as mw

SCRATCH = Path(tempfile.gettempdir()) / 'witwin_cov_meshperf'
CASES = ('repro64', 'p16s0', 'p16s1178', 'mesh', 'sphere', 'box', 'samples1',
         'overlap_mesh', 'overlap_primitive', 'primitive_first', 'boundary_touch',
         'boundary_cross', 'transformed', 'open', 'surface', 'mm', 'periodic', 'graded')
sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_scene(case):
    if case.startswith('p16s'):
        position = int(case[4:])
        path = SCRATCH / f'problem_{position}.json'
        sys.path.insert(0, os.environ['WITWIN_INSTAMAX_TOKENS_ROOT'])
        from emfc.shard import index as dataset_index
        from scripts.convert_p16s_to_scene_dict import DATA, convert_scene, read_scene
        with dataset_index(DATA) as index:
            record, metadata = read_scene(index, position)
            problem = convert_scene(record, metadata['source_index'])
        path.write_text(json.dumps(problem))
        return mw.Scene.from_dict(problem['scene'], device='cuda'), problem['analysis']['frequencies'][0]
    from mesh_subpixel_compile_perf import make_scene as repro_scene
    args = SimpleNamespace(mesh=SCRATCH / 'mesh168.npz', cells=64 if case == 'repro64' else 8,
                           pml=8 if case == 'repro64' else 2, samples=4)
    scene = repro_scene(args, 'sphere' if case == 'sphere' else 'mesh')
    mesh = scene.structures[0].geometry
    if case == 'samples1':
        scene.subpixel = mw.SubpixelSpec(samples=1, averaging='arithmetic')
    if case in ('repro64', 'sphere', 'samples1'):
        return scene, 1e9
    data = np.load(args.mesh)
    vertices, faces = data['vertices'], data['faces']
    center = (vertices.min(0) + vertices.max(0)) / 2
    extent = np.ptp(vertices, axis=0).max()
    if case == 'box':
        scene.structures[0] = replace(scene.structures[0],
            geometry=mw.Box(position=center, size=(extent,) * 3))
    elif case == 'overlap_mesh':
        scene.add_structure(mw.Structure(mw.Mesh(vertices, faces, recenter=False,
            position=(extent * .2, 0, 0), fill_mode='solid'), mw.Material(eps_r=3.2), priority=1))
    elif case in ('overlap_primitive', 'primitive_first'):
        primitive = mw.Structure(mw.Box(position=center, size=(extent * .8,) * 3),
                                 mw.Material(eps_r=2.7), priority=1 if case == 'overlap_primitive' else -1)
        scene.add_structure(primitive)
    elif case in ('boundary_touch', 'boundary_cross'):
        lo = vertices.min(0)
        shift = scene.domain.bounds[0][0] - lo[0]
        if case == 'boundary_cross':
            shift -= extent * .3
        mesh.position[0] = float(shift)
    elif case == 'transformed':
        scene.structures[0] = replace(scene.structures[0], geometry=mw.Mesh(
            vertices, faces, position=center, recenter=True, scale=(1.3, .6, .8),
            rotation=(.9238795325, 0, 0, .3826834324), fill_mode='solid'))
    elif case in ('open', 'surface'):
        scene.structures[0] = replace(scene.structures[0], geometry=mw.Mesh(
            vertices, faces[:-5], recenter=False, fill_mode='solid' if case == 'open' else 'surface'))
    elif case == 'mm':
        # Set the largest mesh extent to one millimetre.
        factor = .001 / float(extent)
        scene = mw.Scene(domain=mw.Domain(tuple((a * factor, b * factor) for a, b in scene.domain.bounds)),
            grid=mw.GridSpec.uniform(float(scene.grid.dx) * factor), boundary=mw.BoundarySpec.pml(num_layers=2),
            subpixel_samples=mw.SubpixelSpec(samples=4, averaging='polarized'), device='cuda',
            structures=[mw.Structure(mw.Mesh(vertices * factor, faces, recenter=False, fill_mode='solid'), mw.Material(eps_r=6.83))])
    elif case == 'periodic':
        scene.boundary = mw.BoundarySpec.periodic()
    elif case == 'graded':
        scene.grid = mw.GridSpec.custom(*(np.linspace(a, b, 9) if axis == 0 else
            a + (b - a) * np.linspace(0, 1, 9) ** (1 + .1 * axis)
            for axis, (a, b) in enumerate(scene.domain.bounds)))
    elif case != 'mesh':
        raise ValueError(case)
    return scene, 1e9


def leaves(value, path=''):
    if torch.is_tensor(value):
        yield path, value.detach().cpu()
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f'{path}/{key}')
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from leaves(item, f'{path}/{index}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--backend', choices=('fdfd', 'fdtd'), required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--reference')
    args = parser.parse_args()
    torch.cuda.init()
    from witwin.core.geometry import mesh_sdf
    assert mesh_sdf._MESH_SDF_ACCELERATOR is None
    if args.backend == 'fdtd':
        os.environ.setdefault('MAX_JOBS', '4')
        from witwin.maxwell.fdtd.cuda.backend import get_compiled_extension
        get_compiled_extension()
    scene, frequency = make_scene(args.case)
    print(json.dumps(dict(starting=args.case, backend=args.backend, label=args.label,
        meshes=[dict(faces=g.face_count, watertight=g.is_watertight, mode=g.fill_mode)
                for s in scene.structures if isinstance(g := s.geometry, mw.Mesh)])), flush=True)
    # Warm the exact geometry data and device context outside compile timing.
    q = torch.zeros((1, 1, 1), device='cuda')
    for structure in scene.structures:
        structure.geometry.signed_distance(q, q, q)
    print(json.dumps(dict(case=args.case, backend=args.backend, label=args.label,
                          phase='prepare')), flush=True)
    environment = dict(torch=torch.__version__, core=importlib.metadata.version('witwin'),
        device=torch.cuda.get_device_name(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(mw.__file__).parent / name for name in
                ('compiler/subpixel.py', 'scene.py', 'fdfd/materials.py'))})
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        if args.backend == 'fdfd':
            prepared = mw.Simulation.fdfd(scene, frequency=frequency).prepare()
            prepared.solver._ensure_material_components()
        else:
            prepared = mw.Simulation.fdtd(scene, frequency=frequency,
                run_time=mw.TimeConfig(time_steps=1)).prepare()
    except ValueError as error:
        if (args.case, args.backend) != ('periodic', 'fdfd'):
            raise
        assert str(error) == "FDFD supports per-face 'none', 'pml' and 'pec' boundaries."
        result = dict(case=args.case, backend=args.backend, label=args.label,
            status='unsupported', error=str(error), seconds=None, tensors=0,
            mismatches=[], **environment)
        stem = f'{args.case}_{args.backend}'
        if args.reference:
            reference = json.loads((SCRATCH / f'{args.reference}_{stem}.json').read_text())
            assert reference['status'] == result['status'] and reference['error'] == result['error']
            result['same_rejection'] = True
        (SCRATCH / f'{args.label}_{stem}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    solver = prepared.solver
    # This is a cache hit on the same compiled model the FDFD consumer reads.
    if args.backend == 'fdfd':
        model = solver.scene.compile_materials()
        model = {key: value for key, value in model.items() if key != 'edge_components'}
    else:
        model = solver._compiled_material_model
        assert model['edge_components'] is not None
    tensors = dict(leaves(model, 'model'))
    # Every tensor stored directly or in tensor containers on the prepared solver,
    # including material update coefficients and CPML coefficients/memory.
    tensors.update(dict(leaves(vars(solver), 'solver')))
    # The FDFD master cache contains unused edges; compare those in the FDTD run.
    tensors = {key: value for key, value in tensors.items() if '/edge_components/' not in key or args.backend == 'fdtd'}
    result = dict(case=args.case, backend=args.backend, label=args.label,
                  seconds=seconds, tensors=len(tensors), mismatches=[],
                  **environment)
    stem = f'{args.case}_{args.backend}'
    if args.reference:
        reference = torch.load(SCRATCH / f'{args.reference}_{stem}.pt', weights_only=True)
        result['same_keys'] = reference.keys() == tensors.keys()
        result['mismatches'] = [key for key in reference.keys() & tensors.keys() if not torch.equal(reference[key], tensors[key])]
        assert result['same_keys'] and not result['mismatches'], result
    torch.save(tensors, SCRATCH / f'{args.label}_{stem}.pt')
    (SCRATCH / f'{args.label}_{stem}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
