"""Copper (finite-conductivity) vs PEC ThinWire dipole stability at long horizons.

Run: python -m benchmark.datagen.repro.copper_dipole_stability <segments> <steady_cycles> <n> [cu]
Example: 19 40 64 cu diverges (|S11| > 1, power 1e30 or NaN); 19 40 64 (PEC) is stable.
"""
"""f4 dipole at n=64 with a chosen segments-per-arm count; long horizon; report power growth."""
import sys, math, numpy as np, torch, witwin.maxwell as mw
import benchmark.datagen.scenes as S
segs = int(sys.argv[1]); cycles = int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
orig_ceil = math.ceil
def patched_wire(name, *, n, kind, count, finite, device):
    frequency = 2.4e9; wavelength = S.C0 / frequency; span = 1.6 * wavelength; dx = span / n
    scene = S._cube(n, span, device=device)
    conductor = mw.WireConductor.finite(conductivity=5.8e7) if finite else mw.WireConductor.pec()
    for side, sign in (("negative", -1), ("positive", 1)):
        pts = [(0., 0., sign * v) for v in np.linspace(dx / 2, .24 * wavelength, segs + 1)]
        scene.add_thin_wire(mw.ThinWire(side, pts, .12 * dx, conductor, snap="continuous"))
    neg, pos = mw.WireNodeRef("negative", 0), mw.WireNodeRef("positive", 0)
    scene.add_port(mw.LumpedPort("feed", wire_binding=mw.WirePortBinding.nodes(negative=neg, positive=pos), reference_impedance=50.))
    freqs = tuple(np.linspace(.8 * frequency, 1.2 * frequency, count))
    S._surface(scene, freqs, (1.2 * wavelength,) * 3)
    return scene, freqs
scene, freqs = patched_wire("x", n=n, kind="dipole", count=11, finite=(len(sys.argv) > 4 and sys.argv[4] == "cu"), device=torch.device("cuda:0"))
kw = S._kwargs(freqs, antenna=True); kw["run_time"] = mw.TimeConfig.auto(steady_cycles=cycles, transient_cycles=8)
r = mw.Simulation.fdtd(scene, **kw).prepare().run()
p = r.port("feed") if hasattr(r, "port") else None
print(f"wire={sys.argv[4] if len(sys.argv)>4 else 'pec'} n={n} segs={segs} cycles={cycles} steps={r.solver_stats['steps_run']} max_incident_power={float(torch.as_tensor(p.incident_power).abs().max()):.3e} max|S11|={float(torch.as_tensor(p.reflection_coefficient).abs().max()):.3f}")
