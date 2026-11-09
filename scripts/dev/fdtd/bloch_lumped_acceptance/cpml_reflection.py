"""Propagating fixed-k reflection of the native FDTD CPML across the fixture band.

Project the source-free CUDA one-step map onto the invariant transverse TE plane
wave. A harmonic current sheet drives its GPU complex128 resolvent; fitting the
interior Ex to the two discrete Yee waves gives the returning/incident amplitude.
This measures the actual float32 field/CPML operator without a TFSF provider or
finite-record transients. Both boundaries retain the fixture's 16-cell layers.
"""
import argparse
import json
import math
from pathlib import Path

from common import C0, F0, FREQUENCIES, SPACING, mw, torch
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block


def operator(angle, alpha):
    ky = 2 * math.pi * F0 / C0 * math.sin(math.radians(angle))
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0., 2 * SPACING), (0., 2 * SPACING), (-.02, .02))),
        grid=mw.GridSpec.uniform(SPACING), device="cuda:0",
        boundary=mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml", num_layers=16,
                                      bloch_wavevector=(0., ky, 0.)),
    )
    config = {"memory_mode": "slab"}
    if alpha is not None:
        config["alpha_max"] = alpha
    solver = mw.Simulation.fdtd(scene, frequencies=FREQUENCIES, cpml_config=config,
        full_field_dft=False, cuda_graph=False, run_time=mw.TimeConfig(time_steps=1)).prepare().solver
    entries = []
    for name, field, scale in (("Ex", "Ex", solver.eps0 ** .5),
                               ("Hy", "Hy", solver.mu0 ** .5),
                               ("Hz", "Hz", solver.mu0 ** .5),
                               ("psi_ex_z", "Ex", solver.mu0 ** .5 * SPACING),
                               ("psi_hy_z", "Hy", solver.eps0 ** .5 * SPACING)):
        y = torch.as_tensor(component_coords(solver.scene, field)[1], device=solver.device)
        phase = torch.exp(1j * ky * y).to(torch.complex64)[None, :, None]
        entries.append((name, getattr(solver, name), getattr(solver, name + "_imag"), phase, scale))
    size = sum(real.shape[-1] for _, real, _, _, _ in entries)
    vector = torch.zeros(size, dtype=torch.complex64, device=solver.device)
    output = torch.empty(size, dtype=torch.complex128, device=solver.device)

    def unpack():
        offset = 0
        for _, real, imag, phase, _ in entries:
            wave = vector[offset:offset + real.shape[-1]][None, None, :] * phase
            real.copy_(wave.real)
            imag.copy_(wave.imag)
            offset += real.shape[-1]

    def pack():
        offset = 0
        for _, real, imag, phase, _ in entries:
            wave = torch.complex(real.double(), imag.double()) * phase.conj().to(torch.complex128)
            output[offset:offset + real.shape[-1]].copy_(wave.mean((0, 1)))
            offset += real.shape[-1]

    unpack()
    _field_update_block(solver, 0.)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unpack()
        _field_update_block(solver, 0.)
        pack()
    matrix = torch.empty((size, size), device=solver.device, dtype=torch.complex128)
    for column in range(size):
        vector.zero_()
        vector[column] = 1.
        graph.replay()
        matrix[:, column].copy_(output)
    scales = torch.cat([torch.full((real.shape[-1],), scale, dtype=torch.float64,
                                 device=solver.device) for _, real, _, _, scale in entries])
    matrix.mul_(scales[:, None]).div_(scales[None, :])
    return solver, ky, matrix, scales


def reflection(angle, alpha):
    solver, ky, matrix, scales = operator(angle, alpha)
    z = torch.as_tensor(solver.scene.z_nodes64, dtype=torch.float64, device=solver.device)
    drive = torch.zeros(matrix.shape[0], dtype=matrix.dtype, device=solver.device)
    source = z.abs().argmin()
    drive[source] = scales[source]
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=solver.device)
    select = (z > .004) & (z < .018)
    rows = []
    for frequency in FREQUENCIES:
        omega = 2 * math.pi * frequency
        normal = (math.sin(.5 * omega * solver.dt) / (solver.c * solver.dt)) ** 2 \
            - (math.sin(.5 * ky * SPACING) / SPACING) ** 2
        assert normal > 0., (angle, frequency)
        kz = 2 / SPACING * math.asin(SPACING * math.sqrt(normal))
        phase = complex(math.cos(omega * solver.dt), -math.sin(omega * solver.dt))
        state = torch.linalg.solve(phase * identity - matrix, drive) / scales
        basis = torch.stack((torch.exp(1j * kz * z[select]), torch.exp(-1j * kz * z[select])), -1)
        electric = state[:z.numel()][select]
        amplitudes = torch.linalg.lstsq(basis, electric).solution
        fit = torch.linalg.vector_norm(basis @ amplitudes - electric) / torch.linalg.vector_norm(electric)
        ratio = (amplitudes[1] / amplitudes[0]).abs()
        assert fit < 1e-4, (angle, frequency, float(fit))
        rows.append(dict(angle=angle, frequency=frequency, alpha=solver.cpml_config["alpha_max"],
                         amplitude=float(ratio), fit_error=float(fit)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angle", type=float, nargs="+", default=(10., 30., 60.))
    parser.add_argument("--margin", type=float, default=1e-5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = []
    print("angle frequency_GHz non_CFS_amplitude CFS_amplitude", flush=True)
    for angle in args.angle:
        baseline, shifted = reflection(angle, 0.), reflection(angle, None)
        for unshifted, cfs in zip(baseline, shifted):
            row = dict(cfs, non_cfs_amplitude=unshifted["amplitude"])
            row["pass"] = row["amplitude"] <= max(1e-4, row["non_cfs_amplitude"] + args.margin)
            rows.append(row)
            print(f"{angle:g} {row['frequency']/1e9:g} {row['non_cfs_amplitude']:.9g} "
                  f"{row['amplitude']:.9g} {'PASS' if row['pass'] else 'FAIL'}", flush=True)
        if args.output:
            args.output.write_text(json.dumps(rows, indent=2) + "\n")
    assert all(row["pass"] for row in rows), "CFS CPML increases propagating reflection"
    print("PASS native CPML reflection", flush=True)


if __name__ == "__main__":
    main()
