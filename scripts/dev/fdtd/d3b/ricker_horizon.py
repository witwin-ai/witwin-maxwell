import os, sys, torch
sys.path.insert(0, 'scripts/dev/fdtd'); sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location("s2", "scripts/dev/fdtd/d3b/step2_scenes.py"); s2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2)
def run(kernel, n):
    os.environ["WITWIN_MAXWELL_THIN_SHEET_KERNEL"] = kernel
    sol = s2.simulation("copper_patch_ricker", n).run().solver
    return {f: getattr(sol, f).double().clone() for f in s2.FIELDS}
for n in (300, 600, 1000, 2000):
    a, b = run("fused", n), run("torch", n)
    e = float(sum(torch.linalg.vector_norm(b[f])**2 for f in ("Ex","Ey","Ez"))**.5)
    h = float(sum(torch.linalg.vector_norm(b[f])**2 for f in ("Hx","Hy","Hz"))**.5)
    rel = {f: float(torch.linalg.vector_norm(a[f]-b[f]) / torch.linalg.vector_norm(b[f])) for f in s2.FIELDS}
    print(n, "|E| %.2e |H| %.2e" % (e, h), {k: f"{v:.1e}" for k, v in rel.items()}, flush=True)
