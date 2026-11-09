"""Native source-free one-step spectrum of a small loaded Bloch cell.

The complex CUDA electric kernels support float32 only. Assemble their map in
float32, diagonalize it in float64, and measure the leading mode's long-run
amplitude growth with the actual kernels. The open cell is the control.
Apply energy conditioning only to the assembled float64 matrix so coordinate
conversion does not add float32 rounding to the native unit-vector responses.
Optional CPML and a tiny split patch isolate the absorbing-boundary coupling.
There is no material ADE, source, or spectral accumulation in this cell.
Only last_voltage_after and active L/C coordinates feed the next lumped step;
the other last_* values are outputs, and scratch is overwritten before use.
The reported split uses physical field/L/C energy coordinates and scaled CPML
and voltage-history norms, which are not independent physical stored energy.
Float32 assembly has a spectral-radius floor of order 1e-7. Finite-time norm
growth can also include neutral/Jordan modes and roundoff-driven transients.
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "4"

from common import C0, F0, FIELDS, mw, torch
from witwin.maxwell.fdtd.checkpoint import _CPML_STATE_NAMES
from witwin.maxwell.fdtd.ports import apply_port_runtimes
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block


def cell(angle, model, *, cpml=True, patch=False, alpha=None, conductivity=False):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.002, 0.002),) * 3),
        grid=mw.GridSpec.uniform(.001),
        boundary=mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml" if cpml else "bloch",
            num_layers=2, bloch_wavevector=(0., 2 * math.pi * F0 / C0
                                             * math.sin(math.radians(angle)), 0.)),
        device="cuda:0",
    )
    if patch:
        for name, bounds in (("left", (-.002, -.001)), ("right", (.001, .002))):
            scene.add_structure(mw.Structure(
                geometry=mw.PlanarPEC.rectangle((bounds, (-.001, .001)),
                                                normal_axis="z", coordinate=0.),
                material=mw.Material.pec(), name=name))
    scene.add_port(mw.LumpedPort(
        "load", negative=(-.001, 0., 0.), positive=(.001, 0., 0.),
        voltage_path=mw.AxisPath("x"),
        current_surface=mw.Box(position=(-.0005, 0., 0.), size=(0., .003, .003)),
        termination=model,
    ))
    cpml_config = {"memory_mode": "slab"}
    if alpha is not None:
        cpml_config["alpha_max"] = alpha
    solver = mw.Simulation.fdtd(scene, frequency=F0, cuda_graph=False,
        cpml_config=cpml_config,
        run_time=mw.TimeConfig(time_steps=2)).prepare().solver
    if conductivity:
        # One interior Ex edge with a 50-ohm shunt, expressed entirely through
        # the ordinary static-conductivity coefficients. No lumped state exists.
        edge = (2, 2, solver.Ex.shape[2] // 2)
        half = .5 * solver.dt / (solver.eps0 * .001 * 50.)
        solver.cex_decay[edge] = (1. - half) / (1. + half)
        solver.cex_curl[edge] /= 1. + half
    return solver


def coordinates(solver):
    """Energy-scaled coordinates, including each bank's carried circuit state."""
    entries = []
    banks = ("", "_imag") if solver.complex_fields_enabled else ("",)
    for suffix in banks:
        for name in FIELDS:
            field = getattr(solver, name + suffix)
            mass = getattr(solver, "eps_" + name) if name.startswith("E") else solver.mu0
            volume = torch.ones_like(field)
            for axis, coordinate in enumerate("xyz"):
                primal = (coordinate == name[1].lower()) == name.startswith("E")
                widths = torch.as_tensor(getattr(solver.scene,
                    f"d{coordinate}_{'primal' if primal else 'dual'}64"),
                    device=solver.device, dtype=field.dtype)
                shape = [1, 1, 1]
                shape[axis] = widths.numel()
                volume.mul_(widths.view(shape))
            scale = volume * mass
            entries.append(("fields", suffix, field, scale.sqrt()))
        if solver.uses_cpml:
            for name in _CPML_STATE_NAMES:
                tensor = getattr(solver, name + suffix)
                if tensor.numel():
                    mass = solver.mu0 if name.startswith("psi_e") else solver.eps0
                    entries.append(("cpml", suffix, tensor, (mass * .001 ** 3) ** .5 * .001))
    load = solver._port_runtimes[0].lumped
    if load is not None:
        for suffix in banks:
            bank = load if suffix == "" else load.imaginary
            # Voltage history has no independent physical storage. Its scale is
            # a local electric-energy scale for conditioning the similarity map.
            entries.append(("history", suffix, bank.last_voltage_after,
                            (solver.eps0 * .001 ** 3 / .002 ** 2) ** .5))
            if bank.inductance_active:
                entries.append(("lumped", suffix, bank.inductor_current, bank.inductance.sqrt()))
            if bank.capacitance_active:
                entries.append(("lumped", suffix, bank.capacitor_voltage, bank.capacitance.sqrt()))
    return entries


def spectrum(angle, label, model, steps, *, cpml=True, patch=False, alpha=None):
    solver = cell(angle, model, cpml=cpml, patch=patch, alpha=alpha,
                  conductivity=label == "conductivity")
    alpha = solver.cpml_config["alpha_max"]
    entries = coordinates(solver)
    size = sum(t.numel() for _, _, t, _ in entries)
    assert size < 6000, size
    vector = torch.zeros(size, device=solver.device)
    output = torch.empty_like(vector)

    def unpack():
        start = 0
        for _, _, tensor, _ in entries:
            tensor.copy_(vector[start:start + tensor.numel()].view_as(tensor))
            start += tensor.numel()

    def pack():
        start = 0
        for _, _, tensor, _ in entries:
            output[start:start + tensor.numel()].copy_(tensor.flatten())
            start += tensor.numel()

    def step():
        _field_update_block(solver, 0.)
        apply_port_runtimes(solver)

    unpack()
    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unpack()
        step()
        pack()
    print(f"assembling angle={angle:g} load={label} cpml={cpml} patch={patch} "
          f"alpha={alpha:g} states={size}", flush=True)
    matrix = torch.empty((size, size), dtype=torch.float64, device=solver.device)
    for column in range(size):
        vector.zero_()
        vector[column] = 1.
        graph.replay()
        matrix[:, column].copy_(output)
    scales = torch.cat([
        torch.broadcast_to(torch.as_tensor(scale, device=solver.device, dtype=torch.float64),
                           tensor.shape).flatten()
        for _, _, tensor, scale in entries
    ])
    matrix.mul_(scales[:, None]).div_(scales[None, :])
    # This is diagnostic dense algebra, not a CPU field-solver fallback.
    import scipy.linalg
    eigenvalues, eigenvectors = scipy.linalg.eig(matrix.cpu().numpy())
    index = abs(eigenvalues).argmax()
    rho = abs(eigenvalues[index])
    mode = torch.as_tensor(eigenvectors[:, index], device=solver.device)
    energies = {"fields": 0., "lumped": 0., "history": 0., "cpml": 0., "real": 0., "imag": 0.}
    start = 0
    for kind, suffix, tensor, _ in entries:
        weight = mode[start:start + tensor.numel()].abs().square().sum().item()
        energies[kind] += weight
        energies["imag" if suffix else "real"] += weight
        start += tensor.numel()
    print(f"angle={angle:g} load={label} states={size} dtype=float32 "
          f"rho={rho:.12g} lambda={eigenvalues[index]} split={energies}", flush=True)
    # A complex eigenvector of a real map is represented by two real trials.
    # Their combined norm avoids a spurious oscillatory endpoint growth rate.
    # Replay contiguous native steps without repacking every field each step.
    block_steps = 64
    advance = torch.cuda.CUDAGraph()
    with torch.cuda.graph(advance):
        for _ in range(block_steps):
            step()
    norms = []
    for part in (mode.real, mode.imag):
        vector.copy_(part / scales)
        unpack()
        initial = (vector.double() * scales).square().sum()
        for _ in range(steps // block_steps):
            advance.replay()
        for _ in range(steps % block_steps):
            step()
        pack()
        norms.append((initial, (output.double() * scales).square().sum()))
    rate = .5 * torch.log(sum(n[1] for n in norms) / sum(n[0] for n in norms)) / steps
    print(f"angle={angle:g} load={label} native_steps={steps} "
          f"log_amplitude_growth_per_step={rate.item():.12g} "
          f"finite_time_amplitude_factor_per_step={math.exp(rate.item()):.12g}", flush=True)
    return dict(angle=angle, load=label, cpml=cpml, patch=patch, alpha=alpha,
                states=size, dtype="float32", rho=float(rho), split=energies,
                steps=steps, log_amplitude_growth_per_step=rate.item(),
                finite_time_amplitude_factor_per_step=math.exp(rate.item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--cpml", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--alpha", type=float, help="Override the runtime's default CFS profile")
    parser.add_argument("--angle", type=float, nargs="+", default=(0., 30.))
    parser.add_argument("--models", nargs="+", choices=("open", "R", "LC", "RLC", "conductivity"),
                        default=("open", "R", "LC", "RLC", "conductivity"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = []
    models = {"open": None, "R": mw.SeriesRLC(r=50.), "LC": mw.SeriesRLC(l=5e-9, c=.5e-12),
              "RLC": mw.SeriesRLC(r=50., l=5e-9, c=.5e-12), "conductivity": None}
    for angle in args.angle:
        for label in args.models:
            results.append(spectrum(angle, label, models[label], args.steps,
                                    cpml=args.cpml, patch=args.patch, alpha=args.alpha))
            if args.output is not None:
                args.output.write_text(json.dumps(results, indent=2) + "\n")
    # Two float32 ULPs cover the native open-cell neutral-mode assembly floor.
    floor = 2 * torch.finfo(torch.float32).eps
    failures = [(row["angle"], row["load"], row["rho"]) for row in results
                if not row["rho"] <= 1. + floor]
    assert not failures, failures
    print(f"PASS one-step radii within 1+{floor:.9g} float32 floor", flush=True)


if __name__ == "__main__":
    main()
