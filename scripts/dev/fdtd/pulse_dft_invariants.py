"""Fresh-interpreter pulse/CW ratio comparison against archived source trees.

Run with the project's GPU wrapper. Both refs are extracted with git archive;
``--branch-ref`` defaults to HEAD. Outputs and native builds live in --output.
Complex comparisons retain phase and use max-norm relative tolerance 1e-5.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def scene_case(kind):
    import witwin.maxwell as mw

    spec = importlib.util.spec_from_file_location("horizon", Path(__file__).with_name("dft_horizon_normalisation_repro.py"))
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    scene, frequencies, excitations = builder.build_scene("dipole" if kind in ("wire", "cw") else "vacuum", 24, 3)
    if kind == "cw":
        excitations = (mw.PortExcitation("feed", source_time=mw.CW(frequency=builder.FC)),)
    if kind == "tfsf":
        scene.sources.clear()
        scene.add_source(mw.PlaneWave(
            direction=(1., 0., 0.), polarization=(0., 0., 1.),
            source_time=mw.GaussianPulse(frequency=builder.FC, fwidth=0.8 * builder.FC),
            injection=mw.TFSF(bounds=((-0.065, 0.065),) * 3),
        ))
        scene.add_structure(mw.Structure(
            mw.Sphere(position=(0., 0., 0.), radius=0.02), mw.Material(eps_r=3.),
        ))
    return scene, frequencies, excitations


def probe(output):
    import witwin.maxwell as mw
    from witwin.maxwell.postprocess import FieldProjector, compute_directivity

    payload = {"module": mw.__file__}
    for kind in ("dipole", "wire", "tfsf", "cw"):
        for normalized in (False, True):
            scene, frequencies, excitations = scene_case(kind)
            prepared = mw.Simulation.fdtd(
                scene, frequencies=frequencies, excitations=excitations,
                run_time=mw.TimeConfig(time_steps=1200), shutoff=0., full_field_dft=True,
                spectral_sampler=mw.SpectralSampler(normalize_source=normalized, window="none"),
            ).prepare()
            result = prepared.run()
            row = {}

            def save(name, value):
                row[name] = torch.as_tensor(value).detach().cpu().clone()

            point = torch.as_tensor(result.monitor("point")["Ez"])
            plane = torch.as_tensor(result.monitor("plane")["Ez"]).reshape(3, -1)
            flux = torch.as_tensor(result.monitor("flux")["flux"])
            closed = torch.as_tensor(result.monitor("radiation")["flux"])
            save("point", point)
            save("plane", plane)
            save("flux", flux)
            save("closed", closed)
            for component, field in result.fields.items():
                if isinstance(field, torch.Tensor):
                    save("full/" + component, field)
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                save("final/" + component, getattr(prepared.solver, component))
            save("ratio/point_plane", point / torch.linalg.vector_norm(plane, dim=1))
            save("ratio/flux_closed", flux / closed)
            projection = FieldProjector.from_monitor(result, "radiation", device=scene.device).angles(
                np.linspace(0., math.pi, 9), np.linspace(0., 2 * math.pi, 17),
            )
            save("far_E", projection.E)
            directivities = [compute_directivity(dict(
                theta=projection.theta, phi=projection.phi, radius=projection.radius,
                power_density=projection.power_density[index], frequency=frequency,
            ))["D_max"] for index, frequency in enumerate(frequencies)]
            save("ratio/directivity", directivities)
            if kind in ("wire", "cw"):
                port = result.port("feed")
                for name in ("voltage", "current", "a", "b", "incident_power", "reflected_power",
                             "accepted_power", "delivered_power", "available_power"):
                    save("port/" + name, getattr(port, name))
                save("ratio/S11", port.reflection_coefficient)
                save("ratio/accepted_incident", port.accepted_power / port.incident_power)
                save("ratio/impedance", port.voltage / port.current)
                save("ratio/incident_available", port.incident_power / port.available_power)
            payload[f"{kind}/{int(normalized)}"] = row
            print(f"probe {kind} normalize_source={normalized}", flush=True)
    torch.save(payload, output)


def compare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    payloads = []
    refs = []
    for label, ref in (("master", args.base_ref), ("branch", args.branch_ref)):
        revision = subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
        refs.append(revision)
        tree = args.output / label
        tree.mkdir()
        archive = args.output / f"{label}.tar"
        subprocess.run(["git", "archive", "-o", str(archive), revision, "witwin"], cwd=ROOT, check=True)
        with tarfile.open(archive) as handle:
            handle.extractall(tree, filter="data")
        env = dict(os.environ, PYTHONPATH=str(tree))
        # The two archives have identical native sources; share the wrapper's
        # private build while importing each Python tree in a fresh process.
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--probe", str(args.output / f"{label}.pt")],
                       cwd=args.output, env=env, check=True)
        payload = torch.load(args.output / f"{label}.pt", weights_only=False)
        assert Path(payload.pop("module")).is_relative_to(tree)
        payloads.append(payload)
    rows = []
    for case, base in payloads[0].items():
        current = payloads[1][case]
        names = [name for name in base if name.startswith("ratio/")]
        if case.endswith("/1"):
            names += [name for name in base if name in ("point", "plane", "flux", "closed", "far_E") or name.startswith("full/")]
        errors = {}
        for name in names:
            left, right = base[name], current[name]
            error = float((right - left).abs().max() / left.abs().max().clamp_min(1e-30))
            assert error < 1e-5, (case, name, error)
            errors[name] = error
        identical = None
        if case == "cw/0":
            identical = all(torch.equal(base[name], current[name]) for name in base)
            assert identical
        rows.append(dict(case=case, max_relative_error=max(errors.values()), errors=errors,
                         bit_identical=identical, compared_outputs=len(base)))
    report = dict(refs=refs, relative_tolerance=1e-5, rows=rows)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--base-ref", default="master")
    parser.add_argument("--branch-ref", default="HEAD")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.probe:
        probe(args.probe)
    else:
        compare(args)
