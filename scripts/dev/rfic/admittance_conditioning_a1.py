"""Conditioning of surface_admittance systems on the straight strip, 1 kHz .. 100 GHz.

Reports the 2-norm condition of the equilibrated loop/tree exterior system (with the
per-body DSA blocks folded) and of the diagonally equilibrated interior T_c, T_b.
"""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import witwin.maxwell as mw
from tests.support.rfic_closed_references import strip_fixture
from witwin.maxwell.mom.layered.lowfreq import scaled_system
from witwin.maxwell.mom.layered.admittance import EPS0, MU0, interior_operators

output = Path(sys.argv[1])
torch.cuda.set_per_process_memory_fraction(float(sys.argv[2]))
frequencies = [10.0 ** e for e in range(3, 12)]
scene, config = strip_fixture(model="surface_admittance", maximum_um=20, edge_um=2)
prepared = mw.Simulation.mom(scene, frequencies=frequencies, config=config).prepare().solver.prepared
body = prepared.admittance.bodies[0]
q, loops = body.transform.to_dense(), body.loops
t = body.vertices[body.local_faces]
area = torch.linalg.vector_norm(torch.linalg.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), dim=-1) / 2
density = body.charges.to_dense() / area[:, None]
rows = []
for k, f in enumerate(frequencies):
    system = scaled_system(prepared, frequency_start=k, frequency_stop=k + 1)
    exterior = torch.linalg.cond(system.matrix[0]).item()
    del system
    omega = 2 * math.pi * f
    eps_c = EPS0 + 1j * body.sigma / omega
    kc, kb = omega * (MU0 * eps_c) ** .5, omega * (MU0 * EPS0) ** .5
    ops = interior_operators(body, kc, kb)
    interior = {}
    for name, v, p, eps in (("T_c", ops["vc"], ops["pc"], eps_c), ("T_b", ops["vb"], ops["pb"], EPS0)):
        a = 1j * omega * MU0 * torch.complex(q.mT @ v.real @ q, q.mT @ v.imag @ q)
        a[loops:, loops:] -= (1j / (omega * eps)) * torch.complex(density.mT @ p.real @ density,
                                                                  density.mT @ p.imag @ density)
        s = a.diagonal().abs().rsqrt()
        interior[name] = torch.linalg.cond(s[:, None] * a * s[None]).item()
    del ops
    rows.append(dict(frequency_Hz=f, exterior_equilibrated=exterior, **interior))
    print(rows[-1], flush=True)
output.write_text(json.dumps(dict(case="strip 20/2 um surface_admittance", unknowns=prepared.metadata["unknowns"],
                                  body_rwg=len(body.basis[0]), rows=rows), indent=2))
