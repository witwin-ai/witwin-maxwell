"""Acquire surface_admittance (A1) evidence outside pytest: strip, thin strip, F-A, F-B.

Usage: python admittance_fixture_a1.py OUTPUT.json --case strip|fa|fb [--model ...] [--frequencies ...]
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import witwin.maxwell as mw
from tests.support.rfic_fixtures import fixture_fa
from tests.support.rfic_closed_references import strip_fixture, network_metrics

UNIT = round(1e-6 * 2**60) / 2**60
FB_PRIMARY_XY = None


def fixture_fb(*, model, maximum, edge, thin=False, substrate=True):
    """F-B interleaved transformer (two turns per winding, 20/5 um, ring 430/490 um).

    Copied from the F-B acquisition worktree; p1/p2 primary, p3/p4 secondary.
    """
    from tests.support.rfic_fixtures import FB_PRIMARY_XY as primary, FB_SECONDARY_XY as secondary
    eps_si, eps_ox = (11.9, 4.1) if substrate else (1., 1.)
    sigma = 10. if substrate else 0.
    stack = mw.mom.LayerStack((mw.mom.Layer("silicon", 300*UNIT, eps_si, sigma), mw.mom.Layer("oxide", 6*UNIT, eps_ox)),
                          mw.mom.HalfSpace(eps_si, sigma), mw.mom.HalfSpace())
    scene = mw.Scene(background=stack)
    material = mw.media.LossyMetalMedium(conductivity=5.8e7)

    def strip(a, b, z, thickness):
        nonlocal scene
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        vertices = torch.tensor(((x0-10, y0-10), (x1+10, y0-10), (x1+10, y1+10), (x0-10, y1+10)),
                                dtype=torch.float64)*UNIT
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(mw.geometry.PlanarPEC(vertices, normal_axis="z", coordinate=z*UNIT,
                                      physical_thickness=thickness*UNIT), material),
        ))
    top, lower = (.1, .1) if thin else (2., 1.)
    for path in (primary, secondary):
        strip((195, path[0][1]), path[0], 305, top)
        for a, b in zip(path[:-1], path[1:]):
            strip(a, b, 305, top)
        x, y = path[-1]
        strip((x, y), (195, y), 302, lower)
        strip((195, y), (195, y), 305, top)
        scene = scene.updated(structures=(
            *scene.structures,
            *(mw.Structure(mw.Box(position=torch.tensor((vx, y, 303.5), dtype=torch.float64)*UNIT,
                                  size=torch.tensor((4, 4, 3), dtype=torch.float64)*UNIT), material) for vx in (x, 195)),
        ))
    outer = torch.tensor(((-245, -245), (245, -245), (245, 245), (-245, 245)), dtype=torch.float64)*UNIT
    inner = torch.tensor(((-215, -215), (215, -215), (215, 215), (-215, 215)), dtype=torch.float64)*UNIT
    scene = scene.updated(structures=(
        *scene.structures,
        mw.Structure(mw.geometry.PlanarPEC(outer, holes=[inner], normal_axis="z", coordinate=305*UNIT,
                                  physical_thickness=top*UNIT), material),
    ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(mw.LumpedPort(name, positive=(205*UNIT, y*UNIT, 305*UNIT), negative=(215*UNIT, y*UNIT, 305*UNIT),
                        voltage_path=mw.AxisPath("x"), reference_impedance=50.) for name, y in (("p1", 170), ("p2", 70), ("p3", 145), ("p4", 45))),
    ))
    config = mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=20*UNIT, max_edge_length_m=maximum*UNIT,
                                                          edge_cell_m=edge*UNIT, growth_ratio=1.5,
                                                          conductor_model=model))
    return scene, config


def transformer_metrics(result):
    """D=((1,0),(-1,0),(0,1),(0,-1)); Zw = D^T Z D; L, R, M, k per frequency."""
    z = torch.linalg.inv(result.network.to_y())
    d = torch.tensor(((1., 0.), (-1., 0.), (0., 1.), (0., -1.)), dtype=z.dtype, device=z.device)
    zw = d.mT @ z @ d
    f = result.network.frequencies.to(z.device)
    omega = 2*math.pi*f
    lmat = -zw.imag/omega[:, None, None]
    l1, l2, m = lmat[:, 0, 0], lmat[:, 1, 1], lmat[:, 0, 1]
    return dict(frequency_Hz=f.tolist(), L1_H=l1.tolist(), L2_H=l2.tolist(), M_H=m.tolist(),
                k=(m/torch.sqrt(l1*l2)).tolist(), R1_ohm=zw.real[:, 0, 0].tolist(), R2_ohm=zw.real[:, 1, 1].tolist())


def simulation(args):
    if args.case == "strip":
        scene, config = strip_fixture(model=args.model, maximum_um=args.maximum, edge_um=args.edge,
                                      thickness_um=args.thickness)
    elif args.case == "fa":
        fixture = fixture_fa(substrate=args.substrate, mesh="test", top_thickness_um=args.thickness)
        scene = fixture.scene
        config = replace(fixture.config, layered=replace(fixture.config.layered, conductor_model=args.model,
                         max_edge_length_m=args.maximum*1e-6, edge_cell_m=args.edge*1e-6))
    else:
        scene, config = fixture_fb(model=args.model, maximum=args.maximum, edge=args.edge, thin=args.thin)
    return mw.Simulation.mom(scene, frequencies=tuple(args.frequencies), config=config)


def main(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prepared = simulation(args).prepare()
    native = prepared.solver.prepared
    print("prepared", native.metadata["unknowns"], native.metadata.get("admittance_bodies"), flush=True)
    result = prepared.run()
    seconds = time.perf_counter() - start
    data = transformer_metrics(result) if args.case == "fb" else network_metrics(result)
    if args.repeat:
        again = simulation(args).run()
        data["repeat_bitwise"] = bool(torch.equal(again.network.s, result.network.s) and
                                      torch.equal(mw.mom.surface_currents(again).currents, mw.mom.surface_currents(result).currents))
    y = result.network.to_y()
    data.update(case=args.case, model=args.model, maximum_um=args.maximum, edge_um=args.edge,
                thickness_um=args.thickness, seconds=seconds, peak_bytes=torch.cuda.max_memory_allocated(),
                reciprocity=((y - y.mT).abs().max()/y.abs().max()).item(),
                max_singular_S=torch.linalg.matrix_norm(result.network.s, ord=2).max().item(),
                metadata={k: v for k, v in result.network.metadata.items() if isinstance(v, (int, float, str, tuple, list))},
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path("witwin/maxwell/mom/layered").rglob("*"))
                               if p.suffix in (".py", ".cu", ".cuh", ".cpp")})
    args.output.write_text(json.dumps(data, indent=2, default=str))
    print(json.dumps({k: v for k, v in data.items() if k not in ("metadata", "source_sha256")}, default=str), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--case", choices=("strip", "fa", "fb"), default="strip")
    p.add_argument("--model", choices=("sheet", "closed", "surface_admittance"), default="surface_admittance")
    p.add_argument("--maximum", type=float, default=20.)
    p.add_argument("--edge", type=float, default=2.)
    p.add_argument("--thickness", type=float, default=2.)
    p.add_argument("--thin", action="store_true")
    p.add_argument("--substrate", action="store_true")
    p.add_argument("--repeat", action="store_true")
    p.add_argument("--memory-fraction", type=float, default=None)
    p.add_argument("--frequencies", type=float, nargs="+", default=(1e8, 1e9))
    a = p.parse_args()
    if a.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    main(a)
