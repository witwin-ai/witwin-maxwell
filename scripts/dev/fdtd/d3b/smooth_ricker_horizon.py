import os, sys, torch
sys.path.insert(0, 'scripts/dev/fdtd'); sys.path.insert(0, '.')
import witwin.maxwell as mw
from thin_sheet_overhead import SCENARIOS, _scene
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
def sim(n):
    scene = _scene(SCENARIOS["copper_patch"], True)
    scene.sources = [mw.PointDipole(position=s.position, polarization=s.polarization, profile="gaussian", width=1.5 * SCENARIOS["copper_patch"]["cell"],
                                    source_time=mw.RickerWavelet(frequency=2.4e9)) for s in scene.sources]
    return mw.Simulation.fdtd(scene, frequencies=[2.4e9], run_time=mw.TimeConfig(time_steps=n), cuda_graph=True)
def run(kernel, n):
    os.environ["WITWIN_MAXWELL_THIN_SHEET_KERNEL"] = kernel
    sol = sim(n).run().solver
    return {f: getattr(sol, f).double().clone() for f in FIELDS} | {"loss": sol._thin_sheets[0].loss.double().clone()}
peak = {f: 0.0 for f in FIELDS}; rows = []
for n in (300, 600, 1000, 2000):
    a, b = run("fused", n), run("torch", n)
    for f in FIELDS: peak[f] = max(peak[f], float(torch.linalg.vector_norm(b[f])))
    rows.append((n, {f: float(torch.linalg.vector_norm(a[f]-b[f])) for f in FIELDS}, {f: float(torch.linalg.vector_norm(b[f])) for f in FIELDS},
                 abs(float(a["loss"]-b["loss"]))/max(abs(float(b["loss"])),1e-300)))
for n, err, nrm, loss in rows:
    print(n, "E/H %.1e" % ((sum(nrm[f]**2 for f in FIELDS[:3])**.5)/max((sum(nrm[f]**2 for f in FIELDS[3:])**.5),1e-300)),
          "final-rel", {f: f"{err[f]/max(nrm[f],1e-300):.1e}" for f in FIELDS},
          "peak-rel", {f: f"{err[f]/peak[f]:.1e}" for f in FIELDS}, "loss %.1e" % loss, flush=True)
