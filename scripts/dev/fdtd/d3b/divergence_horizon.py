import os, sys, torch
sys.path.insert(0, 'scripts/dev/fdtd'); sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location("s2", "scripts/dev/fdtd/d3b/step2_scenes.py"); s2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2)
def run(kernel, n):
    os.environ["WITWIN_MAXWELL_THIN_SHEET_KERNEL"] = kernel
    sol = s2.simulation("copper_patch", n).run().solver
    return {f: getattr(sol, f).double().clone() for f in s2.FIELDS} | {"loss": sol._thin_sheets[0].loss.double().clone()}
for n in (10, 50, 200, 800):
    a, b = run("fused", n), run("torch", n)
    row = {k: float(torch.linalg.vector_norm(a[k]-b[k]) / max(float(torch.linalg.vector_norm(b[k])), 1e-300)) for k in a}
    ulps = {k: int(((a[k].float().view(torch.int32).long()) - (b[k].float().view(torch.int32).long())).abs().max()) for k in s2.FIELDS}
    print(n, {k: f"{v:.1e}" for k, v in row.items()}, "max_ulp", max(ulps.values()), flush=True)
