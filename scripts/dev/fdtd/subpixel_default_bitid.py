"""Two-tree snapshots for default-policy movers and explicit-policy bit identity.

Run the dump in a fresh interpreter per tree, then compare::

    git archive master | tar -x -C /tmp/master_tree
    PYTHONPATH=/tmp/master_tree python scripts/dev/fdtd/subpixel_default_bitid.py dump master.pt
    PYTHONPATH=. python scripts/dev/fdtd/subpixel_default_bitid.py dump branch.pt
    python scripts/dev/fdtd/subpixel_default_bitid.py compare master.pt branch.pt

Every tensor attribute of the prepared FDTD solver (compiled eps/mu/sigma
edge coefficients, update coefficients, fields after 80 steps) and a probe trace
are compared with torch.equal for representative default scenes:
dielectric, explicit arithmetic/staircase spec, dispersive, diagonal
anisotropic, PEC (staircase and samples=1 conformal), CopperFoil + PlanarPEC on
a substrate, and a MaterialRegion.
"""
import sys

import torch


def dump(out_path):
    import witwin.maxwell as mw

    F0 = 1.5e9
    D = 0.004


    def base(spec=None, boundary=None):
        kwargs = {} if spec is None else {"subpixel_samples": spec}
        return mw.Scene(domain=mw.Domain(bounds=((-0.06, 0.06),) * 3), grid=mw.GridSpec.uniform(D),
                        boundary=boundary or mw.BoundarySpec.pml(num_layers=6), device="cuda", **kwargs)


    def scenes():
        s = base()
        s.add_structure(mw.Structure(geometry=mw.Sphere(position=(0.0031, -0.0017, 0.0023), radius=0.021), material=mw.Material(eps_r=4.0)))
        s.add_structure(mw.Structure(geometry=mw.Box(position=(0, 0, -0.03), size=(0.1, 0.1, 0.0213)), material=mw.Material(eps_r=2.25, sigma_e=0.01)))
        yield "dielectric", s
        s = base(mw.SubpixelSpec(samples=1, averaging="arithmetic", pec="staircase"))
        s.add_structure(mw.Structure(geometry=mw.Sphere(position=(0.0031, -0.0017, 0.0023), radius=0.021), material=mw.Material(eps_r=4.0)))
        yield "dielectric_explicit_spec", s
        s = base()
        s.add_structure(mw.Structure(geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.019),
                                     material=mw.Material.lorentz(eps_inf=2.0, delta_eps=2.0, resonance_frequency=2.0e9, gamma=2.0e8)))
        s.add_structure(mw.Structure(geometry=mw.Box(position=(0.03, 0, 0), size=(0.02, 0.03, 0.03)),
                                     material=mw.Material.drude(eps_inf=1.0, plasma_frequency=2.0e9, gamma=1.0e8)))
        yield "dispersive", s
        s = base()
        s.add_structure(mw.Structure(geometry=mw.Box(position=(0.001, 0.002, -0.0013), size=(0.03, 0.025, 0.0217)),
                                     material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.5))))
        yield "anisotropic", s
        for pec in ("staircase", "conformal"):
            s = base(mw.SubpixelSpec(samples=1, averaging="arithmetic", pec=pec))
            s.add_structure(mw.Structure(geometry=mw.Sphere(position=(0.0013, 0.0021, -0.0007), radius=0.017), material=mw.Material.pec()))
            yield f"pec_{pec}", s
        s = base()
        s.add_structure(mw.Structure(mw.Box(position=(0, 0, -0.0125), size=(0.2, 0.2, 0.025)), mw.Material(eps_r=2.25, sigma_e=0.01), name="substrate"))
        s.add_structure(mw.Structure(mw.PlanarPEC.rectangle(((-0.1, 0.1), (-0.012, -0.004)), coordinate=0.0, physical_thickness=35e-6),
                                     mw.Material.pec(), name="pec_strip"))
        s.add_structure(mw.Structure(mw.Box(position=(0.0, 0.008, 0.0), size=(0.2, 0.008, 0.0)),
                                     mw.CopperFoil(thickness=35e-6, frequency_range=(0.8e9, 2.0e9)), name="foil"))
        yield "copperfoil_planarpec", s
        s = base()
        g = torch.Generator().manual_seed(7)
        density = torch.rand((8, 8, 8), generator=g)
        s.add_material_region(mw.MaterialRegion(name="design", geometry=mw.Box(position=(0.001, 0, 0), size=(0.032, 0.032, 0.032)),
                                                density=density, eps_bounds=(1.0, 4.0)))
        yield "material_region", s


    results = {}
    for name, scene in scenes():
        scene.add_source(mw.PointDipole(position=(-0.031, 0.0007, 0.0213), polarization="Ez",
                                        source_time=mw.GaussianPulse(frequency=F0, fwidth=0.7e9), name="src"))
        scene.add_monitor(mw.FieldTimeMonitor("probe", components=("Ez", "Hx"), position=(0.021, -0.0011, 0.0031)))
        sim = mw.Simulation.fdtd(scene, frequency=F0, run_time=mw.TimeConfig(time_steps=80), full_field_dft=False)
        result = sim.run()
        solver = result.solver
        tensors = {}
        for key, value in vars(solver).items():
            if torch.is_tensor(value):
                tensors[key] = value.detach().cpu().clone()
        payload = result.monitor("probe")
        for comp, trace in payload["components"].items():
            tensors["probe_" + comp] = torch.as_tensor(trace).detach().cpu().clone()
        results[name] = tensors
        print(name, len(tensors), "tensors", flush=True)
    torch.save(results, out_path)


def compare(path_a, path_b):
    a = torch.load(path_a); b = torch.load(path_b)
    ok = True
    for scene in a:
        ta, tb = a[scene], b[scene]
        keys = sorted(set(ta) | set(tb))
        missing = [k for k in keys if k not in ta or k not in tb]
        bad = [k for k in keys if k in ta and k in tb and not (ta[k].shape == tb[k].shape and ta[k].dtype == tb[k].dtype and torch.equal(ta[k], tb[k]))]
        fields = [k for k in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") if k in ta]
        print(f"{scene}: {len(keys)} tensors compared, mismatched={bad}, missing={missing}, field tensors equal={all(torch.equal(ta[k], tb[k]) for k in fields)} ({fields}), "
              f"max|Ez|={float(ta['Ez'].abs().max()):.3e}")
        ok &= not bad and not missing
    print("ALL BIT-IDENTICAL" if ok else "DIFFERENCES FOUND")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif sys.argv[1] == "compare":
        raise SystemExit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    else:
        raise SystemExit("usage: dump OUT.pt | compare A.pt B.pt")
