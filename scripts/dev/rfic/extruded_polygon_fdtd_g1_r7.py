"""G1-style FDTD gate for ExtrudedPolygon through master's core-path capture helpers (benchmark/core_path/gates.py).

Arm A: the existing Prism (pentagon). Arm B: ExtrudedPolygon with Prism's vertices. Same scene otherwise
(_grid_scene 48^3 dipole, default FDTD execution, 256 steps). Pinned items: setup tensors, fields, phasors,
monitors, DFT accumulators; torch.equal required (T0). Arm C: a non-convex L ExtrudedPolygon vs the same L as
the maxwell PolySlab: Rule-2 setup-level tiers on the setup tensors and the runs' fields. Plus a repeat of arm B
(determinism, torch.equal).
"""
import json, math, sys
WT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, WT)
import torch
import witwin.maxwell as mw, witwin.core
assert str(__import__("pathlib").Path(mw.__file__).resolve()).startswith(WT) and hasattr(witwin.core, "ExtrudedPolygon"), (mw.__file__, witwin.core.__file__)
print(mw.__file__, witwin.core.__file__, flush=True)
if torch.cuda.get_device_properties(0).total_memory < 60e9:
    torch.cuda.set_per_process_memory_fraction(7.0 / 47.99)
from benchmark.core_path import gates
from benchmark.core_path.scenes import CoreCase, _grid_scene

def pentagon():
    angles = torch.linspace(0, 2 * math.pi, 6)[:-1]
    return torch.stack([0.2 * torch.cos(angles), 0.2 * torch.sin(angles)], 1)

L = [(-0.2, -0.2), (0.2, -0.2), (0.2, 0.0), (0.0, 0.0), (0.0, 0.2), (-0.2, 0.2)]
GEOMETRIES = {
    "prism": lambda: mw.geometry.Prism(position=(0.05, 0.0, 0.0), radius=0.2, height=0.3, num_sides=5, axis="y", device="cpu"),
    "extruded": lambda: mw.geometry.ExtrudedPolygon(pentagon(), height=0.3, position=(0.05, 0.0, 0.0), axis="y", device="cpu"),
    "l_extruded": lambda: mw.geometry.ExtrudedPolygon(L, height=0.3, position=(0.0, 0.0, 0.05), axis="z", device="cpu"),
    "l_polyslab": lambda: mw.PolySlab(L, bounds=(-0.15 + 0.05, 0.15 + 0.05), axis="z", device="cpu"),
}

def capture(name):
    scene = _grid_scene((48, 48, 48))
    scene = scene.updated(structures=(
        *scene.structures,
        mw.Structure(geometry=GEOMETRIES[name](), material=mw.Material(eps_r=4.0), name="body"),
    ))
    case = CoreCase(name=name, scene=scene)
    prepared = gates.make_simulation(case, steps=256, full_field_dft=True).prepare()
    solver = prepared.solver
    setup = gates.setup_tensors(solver)
    result = prepared.run()
    solver = prepared.solver
    payload = gates.tensors({n: getattr(solver, n) for n in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "fields")
    payload.update(gates.result_tensors(result))
    payload.update(gates.tensors(solver._dft_batched_fields, "dft_accumulators"))
    return setup, payload, bool(getattr(solver, "_core_step_graph_active", False))

out = {}
arms = {name: capture(name) for name in ("prism", "extruded", "l_extruded", "l_polyslab")}
repeat = capture("extruded")
def equal(a, b):
    assert a.keys() == b.keys()
    return {k: bool(torch.equal(a[k], b[k])) for k in a}
for label, (x, y) in {"prism_vs_extruded": (arms["prism"], arms["extruded"]),
                      "extruded_repeat": (arms["extruded"], repeat)}.items():
    eq_setup, eq_run = equal(x[0], y[0]), equal(x[1], y[1])
    out[label] = dict(setup_items=len(eq_setup), run_items=len(eq_run), all_equal=all(eq_setup.values()) and all(eq_run.values()),
                      unequal=[k for k, v in {**eq_setup, **eq_run}.items() if not v], core_step_graph=(x[2], y[2]))
def setup_and_fields(reference, candidate):
    rows = gates.rule2_rows(reference[0], candidate[0], level="setup")
    fields = {k: v for k, v in reference[1].items() if k.startswith(("fields/", "phasors/"))}
    diff = {k: float((candidate[1][k] - v).abs().max() / v.abs().max().clamp_min(1e-30)) for k, v in fields.items()}
    return dict(tiers=gates.tier_summary(rows), worst=sorted(((r["ratio_to_bound"], k) for k, r in rows.items()
                                                               if r["tier"] != "T0"), reverse=True)[:5],
                changed_samples={k: int((reference[0][k] != candidate[0][k]).sum()) for k in reference[0]
                                 if k.startswith("setup/eps_") and not torch.equal(reference[0][k], candidate[0][k])},
                fields_max_rel=max(diff.values()))
out["prism_vs_extruded_rule2"] = setup_and_fields(arms["prism"], arms["extruded"])
rows = gates.rule2_rows(arms["l_polyslab"][0], arms["l_extruded"][0], level="setup")
out["l_extruded_vs_polyslab_setup"] = dict(tiers=gates.tier_summary(rows),
    worst=sorted(((r["ratio_to_bound"], k) for k, r in rows.items() if r["tier"] != "T0"), reverse=True)[:5])
fields = {k: v for k, v in arms["l_polyslab"][1].items() if k.startswith(("fields/", "phasors/"))}
diff = {k: float((arms["l_extruded"][1][k] - v).abs().max() / v.abs().max().clamp_min(1e-30)) for k, v in fields.items()}
out["l_extruded_vs_polyslab_fields_max_rel"] = max(diff.values())
print(json.dumps(out, indent=1, default=str), flush=True)
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
