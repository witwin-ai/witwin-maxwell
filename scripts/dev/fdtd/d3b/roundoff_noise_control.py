import os, sys, torch, types
sys.path.insert(0, 'scripts/dev/fdtd'); sys.path.insert(0, '.')
os.environ["WITWIN_MAXWELL_THIN_SHEET_KERNEL"] = "torch"
import importlib.util
spec = importlib.util.spec_from_file_location("s2", "scripts/dev/fdtd/d3b/step2_scenes.py"); s2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2)
sim = s2.simulation("copper_patch", 2000)
sim.config.cuda_graph = False
prepared = sim.prepare()
sheet = prepared.solver._thin_sheets[0]
gen = torch.Generator(device="cuda").manual_seed(7)
original = sheet.electric
def noisy(self, solver):
    original(solver)
    for t in (self.chord_e, self.copper["state"]):
        sign = torch.randint(0, 2, t.shape, generator=gen, device=t.device, dtype=t.dtype) * 2 - 1
        t.mul_(1 + sign * 2.0**-52)
sheet.electric = types.MethodType(noisy, sheet)
solver = prepared.run().solver
state = {n: getattr(solver, n).cpu() for n in s2.FIELDS}; state["loss"] = solver._thin_sheets[0].loss.cpu()
torch.save(dict(scenario="copper_patch", kernel="torch", steps=2000, state=state, peak_allocated_gib=0., peak_reserved_gib=0.), sys.argv[1])
