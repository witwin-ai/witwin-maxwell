"""V4 stripline Cu (sigma=0) two-length attenuation: run columns and reduce.

Runs the four 64 ns columns (two lengths x two drives) of the V4 material
control ``Cu 35 um, sigma 0`` with the current checkout (same case, sources,
stations and horizon as the V4 controls), and reduces the end-station V/I
transfer with the same independent two-length extraction. Applied to the V4
controls' retained ``execution.json`` files, ``reduce`` reproduces their
published Cu alpha exactly.

    python -m scripts.dev.fdtd.stripline_cu_two_length run --output DIR --length L --drive D
    python -m scripts.dev.fdtd.stripline_cu_two_length reduce DIR
"""
import argparse, json
from pathlib import Path
import numpy as np


def v4_case(length_index):
    from scripts.dev.fdtd.printed_line_loss_execution import PilotCase
    separation = (.40, .48)[length_index]
    end = separation / 2 - .025
    return PilotCase(dx=.0025, longitudinal_dl=None, metal='copper', probe_reference_dx=.0025,
                     record_legacy_native_vi=True, axial_half_length=.30, padding=.08, separation=separation,
                     sigma=0., positions=(-end, -.015, 0., .015, end))


def decode(value):
    if isinstance(value, dict):
        if set(value) == {'real', 'imag'}:
            return complex(value['real'], value['imag'])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def reduce(root):
    from scripts.dev.fdtd.printed_line_two_length import physical_transfer, two_length_modes
    root = Path(root)
    states, spans = [], []
    for length in (0, 1):
        columns = []
        for drive in (0, 1):
            report = decode(json.loads((root / f'length-{length}-drive-{drive}' / 'execution.json').read_text()))
            columns.append(np.asarray(report['physical_state'], dtype=np.complex128))
            positions = report['case']['positions']
        states.append(np.stack(columns, -1))
        spans.append(positions[-1] - positions[0])
    frequencies = report['case']['frequencies']
    rows = []
    for f, frequency in enumerate(frequencies):
        transfers = [physical_transfer(s[[0, -1], 0:1, f, :], s[[0, -1], 1:2, f, :])['matrix'] for s in states]
        modes = two_length_modes(*transfers, lengths=tuple(spans), frequency=frequency)['forward_modes']
        rows.append(dict(frequency_hz=frequency, alpha_np_per_m=modes[0]['alpha_np_per_m'],
                         beta_rad_per_m=modes[0]['beta_rad_per_m']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--length', type=int, choices=(0, 1), required=True)
    run.add_argument('--drive', type=int, choices=(0, 1), required=True)
    red = sub.add_parser('reduce')
    red.add_argument('root', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        from scripts.dev.fdtd.printed_line_loss_execution import DRIVE_BASIS, run_drive
        report = run_drive(v4_case(args.length), DRIVE_BASIS[args.drive],
                           args.output / f'length-{args.length}-drive-{args.drive}', graph=True, resume=False)
        print(json.dumps({'planned_steps': report['planned_steps']}))
    else:
        print(json.dumps(reduce(args.root), indent=1))


if __name__ == '__main__':
    main()
