"""Solve a conforming spatial refinement of the fixed dual-conductive control."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import witwin.maxwell as mw
from scripts.dev.solve_mom_dual_harmonic_power_control import setup, portable


def refined_setup(levels):
    if type(levels) is not int or levels < 1:
        raise ValueError('Spatial refinement levels must be positive integers')
    scene, config = setup()
    volume = config.harmonic_balance.volumes[0].refine(levels)
    boundary = volume.boundary
    refined = mw.Scene(structures=[mw.Structure(
        geometry=mw.Mesh(boundary.vertices, boundary.faces, recenter=False, fill_mode='solid'),
        material=scene.structures[0].material)], sources=scene.sources)
    return refined, replace(config, harmonic_balance=replace(config.harmonic_balance, volumes=(volume,)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--levels', type=int, default=1)
    parser.add_argument('--source-max-order', type=int, default=32)
    args = parser.parse_args()
    root = args.destination
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    paths = list(Path(mw.__file__).parent.rglob('*.py')) + [Path(__file__),
        Path(__file__).with_name('solve_mom_dual_harmonic_power_control.py')]
    sources = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    record = {'version': 'dual-spatial-control-v1', 'sources': sources, 'levels': args.levels,
              'frequency_hz': 3e8, 'source_max_order': args.source_max_order,
              'status': 'running', 'phase': 'setup',
              'physical_qualification': False, 'spatial_convergence_verified': False}
    def write(name, value):
        (root/name).write_text(json.dumps(portable(value), sort_keys=True, indent=2, allow_nan=False))
    write('start.json', record)
    try:
        scene, config = refined_setup(args.levels)
        config = replace(config, harmonic_balance=replace(config.harmonic_balance, source_max_order=args.source_max_order))
        volume = config.harmonic_balance.volumes[0]
        record.update(volume_cells=len(volume.cells), surface_faces=len(volume.boundary.faces), phase='solve')
        write('mesh.json', {'vertices': volume.vertices, 'cells': volume.cells,
                            'boundary_vertices': volume.boundary.vertices, 'boundary_faces': volume.boundary.faces})
        print('solve', record['volume_cells'], record['surface_faces'], flush=True)
        result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
        result.save(root/'result.pt')
        write('solver-stats.json', result.stats())
        record['result_sha256'] = hashlib.sha256((root/'result.pt').read_bytes()).hexdigest()
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in sources.items()):
            raise RuntimeError('Source bytes changed during the spatial control')
        record.update(status='solved_unqualified', phase='complete')
        print(record['status'], flush=True)
    except BaseException as error:
        record.update(status='failed', error={'type': type(error).__name__, 'message': str(error),
                                             'receipt': getattr(error, 'receipt', None)})
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-start
        write('terminal.json', record)


if __name__ == '__main__':
    main()
