"""R6-4 probes: vertical closed-body ports on a two-strip line in homogeneous oxide (writes JSON under $RFIC_OUT).

Signal 10 um wide on z = 25 um over a 40 um ground strip on z = 15 um, eps_r = 4 everywhere, full-width
vertical inter-layer edge ports at both line ends, two-line eps_eff (= eps_r exactly for a TEM line)
and impedance from 300/600 um lines at 10/20 GHz. Cases: sheet; closed PEC t = 2, 1, 0.5 um (thin limit
towards the sheet line); surface_admittance PEC (bitwise to closed) and copper t = 2 um; closed copper.
Run from the worktree root: python scripts/dev/rfic/r6_4_probes.py
"""
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import witwin.maxwell as mw  # noqa: E402
from witwin.maxwell.mom.layered.deembed import line_parameters  # noqa: E402
from witwin.maxwell.mom.layered.port_system import edge_port  # noqa: E402

print(mw.__file__, flush=True)
assert mw.__file__.startswith(str(ROOT))
UM, EPS_R, FREQUENCIES, LENGTHS = 1e-6, 4.0, (10e9, 20e9), (300., 600.)
OUT = pathlib.Path(os.environ["RFIC_OUT"])
OUT.mkdir(parents=True, exist_ok=True)


def scene(length, model, thickness, copper):
    stack = mw.mom.LayerStack((mw.mom.Layer("oxide", 40 * UM, EPS_R),), mw.mom.HalfSpace(EPS_R), mw.mom.HalfSpace(EPS_R))
    s = mw.Scene(background=stack)
    material = mw.media.LossyMetalMedium(conductivity=5.8e7) if copper else mw.Material.pec()
    for name, z, width in (("gnd", 15 * UM, 40 * UM), ("sig", 25 * UM, 10 * UM)):
        vertices = torch.tensor(((0., -width / 2), (length * UM, -width / 2), (length * UM, width / 2), (0., width / 2)),
                                dtype=torch.float64)
        extra = {} if model == "sheet" and not copper else dict(physical_thickness=thickness * UM)
        s = s.updated(structures=(
            *s.structures,
            mw.Structure(mw.geometry.PlanarPEC(vertices, normal_axis="z", coordinate=z, **extra),
                         material, name=name),
        ))
    s = s.updated(ports=(
        *s.ports,
        *(edge_port(s, name, signal="sig", side=side, reference="gnd") for name, side in (("p1", "-x"), ("p2", "+x"))),
    ))
    return s


def run(tag, model, thickness=2., copper=False, edge=1.):
    config = mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(
        bridge_width_m=10 * UM, max_edge_length_m=10 * UM, edge_cell_m=edge * UM, conductor_model=model))
    out, networks = dict(model=model, thickness_um=thickness, copper=copper, edge_cell_um=edge), []
    for length in LENGTHS:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = mw.Simulation.mom(scene(length, model, thickness, copper), frequencies=FREQUENCIES, config=config).run()
        network = result.network
        y = network.to_y()
        out[f"L{length:g}"] = dict(
            seconds=time.perf_counter() - start, peak_bytes=torch.cuda.max_memory_allocated(),
            unknowns=network.metadata["unknowns"], port_blocks=network.metadata.get("port_blocks"),
            port_attachment=network.metadata.get("port_attachment"), ports=network.metadata["ports"],
            reciprocity=((y - y.transpose(-1, -2)).abs().max() / y.abs().max()).item(),
            max_singular=torch.linalg.matrix_norm(network.s, ord=2).max().item(),
            min_eig_re_y=torch.linalg.eigvalsh(((y + y.conj().transpose(-1, -2)) / 2)).min().item(),
            s_re=network.s.real.tolist(), s_im=network.s.imag.tolist())
        if length == LENGTHS[0] and model == "closed" and not copper and thickness == 2.:
            again = mw.Simulation.mom(scene(length, model, thickness, copper), frequencies=FREQUENCIES, config=config).run()
            out["repeat_equal"] = bool(torch.equal(again.network.s, network.s))
        networks.append(network)
        print(tag, length, out[f"L{length:g}"]["seconds"], out[f"L{length:g}"]["unknowns"], flush=True)
    line = line_parameters(*networks, (LENGTHS[1] - LENGTHS[0]) * UM)
    out.update(eps_eff=line["eps_eff"].tolist(), eps_relative=[e / EPS_R - 1 for e in line["eps_eff"].tolist()],
               z0_two_line=[[z.real, z.imag] for z in line["impedance_two_line"].tolist()],
               attenuation_np_per_m=line["attenuation_np_per_m"].tolist())
    print(tag, json.dumps({k: out[k] for k in ("eps_eff", "eps_relative", "z0_two_line")}), flush=True)
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=1))
    return networks


cases = [("sheet", "sheet", 2., False), ("closed-t2", "closed", 2., False), ("closed-t1", "closed", 1., False),
         ("closed-t0.5", "closed", .5, False), ("sa-pec-t2", "surface_admittance", 2., False),
         ("sa-cu-t2", "surface_admittance", 2., True), ("closed-cu-t2", "closed", 2., True)]
results = {tag: run(tag, model, thickness, copper) for tag, model, thickness, copper in cases}
# Edge-cell refinement of the closed t = 2 line.
results["closed-t2-e0.5"] = run("closed-t2-e0.5", "closed", 2., False, edge=.5)
summary = dict(sa_pec_equals_closed=bool(torch.equal(results["sa-pec-t2"][0].s, results["closed-t2"][0].s)))
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print("summary", summary, flush=True)
