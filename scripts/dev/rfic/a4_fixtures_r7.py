"""A4 fixture producers (R7-2): MoM values of F-D (MIM), F-E (microstrip, CPW) and F-F (octagonal differential inductor).

python a4_fixtures_r7.py <fd|fe-microstrip|fe-cpw|ff> --edge <m> [--max <m>] --out FILE.json [--cap-gib 7.5]

F-D: C = 1/(omega Im Z) of the one-port per plate; C' = dC/dL from two lengths at equal width (de-embedded).
F-E: de-embedded line quantities: two-line gamma (deembed.line_parameters, error-box free) -> eps_eff(f); open-end
     length differencing of C_in = 1/(omega Im Z11_open) at 10 MHz -> C' (ports and ends cancel); Z0 = sqrt(eps)/(c C').
F-F: differential L = -Im(Z)/omega and R of the single floating port.
Every record holds the exact repeat check (torch.equal of S on a second run of the first case).
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered import extraction as ex
from witwin.maxwell.mom.layered.deembed import line_parameters
from tests.support import rfic_a4_fixtures as fx

C0 = 299792458.0


def checks(network):
    y = network.to_y()
    asymmetry = (torch.linalg.matrix_norm(y - y.transpose(-1, -2)) / torch.linalg.matrix_norm(y)).max().item()
    return dict(reciprocity=asymmetry, max_singular_S=torch.linalg.matrix_norm(network.s, ord=2).max().item())


def open_capacitance(network, frequency_index):
    z = ex.terminal_impedance(network, network.port_names[0], others="open")[frequency_index]
    omega = 2 * math.pi * network.frequencies[frequency_index].item()
    return 1 / (omega * z.imag.item())


def fd(edge, maximum):
    frequencies = (1e8, 1e9)
    options = fx.mim_options(edge) if maximum is None else dict(fx.mim_options(edge), max_edge_length_m=maximum)
    cases, repeat = [], None
    for width, length in fx.MIM_CASES:
        result = fx.simulate(fx.mim(width, length), frequencies, options)
        network = result.network
        if repeat is None:
            repeat = torch.equal(fx.simulate(fx.mim(width, length), frequencies, options).network.s, network.s)
        z = ex.terminal_impedance(network, "p", others="open")
        omega = 2 * math.pi * network.frequencies
        cases.append(dict(width_m=width, length_m=length, unknowns=network.metadata["unknowns"],
                          C_F=(1 / (omega * z.imag)).tolist(), R_ohm=z.real.tolist(), **checks(network)))
        del result
        torch.cuda.empty_cache()
    # Length differencing at equal width: C' = dC/dL (ends, corners and the port cancel).
    per_length = {}
    for width in sorted({c["width_m"] for c in cases}):
        pair = sorted((c for c in cases if c["width_m"] == width and c["length_m"] != width),
                      key=lambda c: c["length_m"])
        per_length[f"{width * 1e6:g}um"] = [(b - a) / (pair[1]["length_m"] - pair[0]["length_m"])
                                            for a, b in zip(pair[0]["C_F"], pair[1]["C_F"])]
    return dict(fixture="F-D", edge_m=edge, max_edge_m=options["max_edge_length_m"], frequency_Hz=list(frequencies),
                cases=cases, C_per_length_F_per_m=per_length, exact_repeat=repeat)


def fe(kind, edge, maximum):
    if kind in ("microstrip", "microstrip-air"):
        build, lengths, options = fx.microstrip, fx.MS_LENGTHS, fx.microstrip_options(edge)
        if kind == "microstrip-air":
            def build(length):
                return fx.microstrip(length, eps=1.)
        frequencies = (1e7, 3e8, 1e9, 3e9)
    else:
        build, lengths, options = fx.cpw, fx.CPW_LENGTHS, fx.cpw_options(edge)
        frequencies = (1e7, 2e9, 5e9, 1e10, 2e10)
    if maximum is not None:
        options = dict(options, max_edge_length_m=maximum)
    networks, unknowns = [], []
    for length in lengths:
        result = fx.simulate(build(length), frequencies, options)
        networks.append(result.network)
        unknowns.append(result.network.metadata["unknowns"])
        del result
        torch.cuda.empty_cache()
    repeat = torch.equal(fx.simulate(build(lengths[0]), frequencies, options).network.s, networks[0].s)
    delta = lengths[1] - lengths[0]
    line = line_parameters(*networks, delta)
    capacitance = [open_capacitance(n, 0) for n in networks]
    c_per_length = (capacitance[1] - capacitance[0]) / delta
    eps = line["eps_eff"][1:].tolist()
    return dict(fixture="F-E", kind=kind, edge_m=edge, max_edge_m=options["max_edge_length_m"],
                lengths_m=list(lengths), frequency_Hz=list(frequencies), unknowns=unknowns,
                eps_eff=eps, attenuation_np_per_m=line["attenuation_np_per_m"][1:].tolist(),
                impedance_two_line_ohm=line["impedance_two_line"].real[1:].tolist(),
                open_capacitance_10MHz_F=capacitance, C_per_length_F_per_m=c_per_length,
                Z0_from_C_ohm=[math.sqrt(e) / (C0 * c_per_length) for e in eps],
                checks=[checks(n) for n in networks], exact_repeat=repeat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", choices=("fd", "fe-microstrip", "fe-microstrip-air", "fe-cpw", "ff"))
    parser.add_argument("--edge", type=float, required=True)
    parser.add_argument("--max", type=float)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap-gib", type=float, default=7.5)
    args = parser.parse_args()
    print("witwin.maxwell", mw.__file__, flush=True)
    assert mw.__file__.startswith(str(Path(__file__).resolve().parents[3])), mw.__file__
    torch.cuda.set_per_process_memory_fraction(args.cap_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    if args.fixture == "fd":
        record = fd(args.edge, args.max)
    elif args.fixture == "ff":
        from scripts.dev.rfic.octagon_reference_r7 import mom_record
        record = mom_record(args.edge, args.max)
    else:
        record = fe(args.fixture[3:], args.edge, args.max)
    record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("cases", "checks")}), flush=True)
