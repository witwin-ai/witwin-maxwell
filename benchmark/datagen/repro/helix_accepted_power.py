"""PEC thin-wire helix: per-frequency accepted power and |S11| under the data-plan decay rule.

Builder: benchmark.datagen.scenes.wire(kind="helix"); catalog scene f4_helix_48 (n=48, 21 frequencies over
0.8-1.2 x 2.4 GHz, PEC WireConductor, LumpedPort "feed" 50 ohm, Gaussian pulse fwidth 0.8 f_c,
shutoff 1e-5 with a TimeConfig.auto(steady_cycles=40, transient_cycles=8) cap).
Run: python -m benchmark.datagen.repro.helix_accepted_power [scene_name] [steady_cycles_cap]
"""
import sys
import torch
import witwin.maxwell as mw
from benchmark.datagen.scenes import scene_builders

name = sys.argv[1] if len(sys.argv) > 1 else "f4_helix_48"
sample = {b.name: b for b in scene_builders()}[name].build(device=torch.device("cuda:0"))
kwargs = dict(sample.fdtd_kwargs)
if len(sys.argv) > 2:  # optional longer horizon cap, in steady cycles
    kwargs["run_time"] = mw.TimeConfig.auto(steady_cycles=int(sys.argv[2]), transient_cycles=8)
result = mw.Simulation.fdtd(sample.scene, **kwargs).prepare().run()
stats = result.solver_stats
print(name, "steps", stats["steps_run"], "of", stats["time_steps"], "dt", stats["dt"])
port = result.port("feed")
for i, (f, s11, pa, pi) in enumerate(zip(port.frequencies, port.reflection_coefficient,
                                         port.accepted_power, port.incident_power)):
    flag = "  <-- P_acc<=0" if float(pa) <= 0 else ""
    print(f"{i:2d} f={float(f) / 1e9:.3f} GHz |S11|={abs(complex(s11)):.4f} P_acc={float(pa):+.3e} P_inc={float(pi):.3e}{flag}")
