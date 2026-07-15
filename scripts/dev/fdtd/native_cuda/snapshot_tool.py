"""Correctness snapshot for native CUDA FDTD performance work.

Usage (from maxwell/, witwin2 env, with the extension built):

    WITWIN_MAXWELL_FDTD_BACKEND=cuda WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION=1 \
    python scripts/dev/fdtd/native_cuda/snapshot_tool.py save
    ... make a perf change, rebuild ...
    python scripts/dev/fdtd/native_cuda/snapshot_tool.py check

`save` records forward solve outputs (final fields + DFT monitor data) for
shortened benchmark scenarios plus adjoint gradients for a differentiable
scene. `check` re-runs the same workloads and asserts the outputs match the
stored snapshot within tight float32 tolerances. This is the backend-independent
guardrail for every optimization commit.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

import witwin.maxwell as mw
from benchmark.runner import _clone_scene
from benchmark.scenes import SCENARIOS
from witwin.maxwell.simulation import Simulation, TimeConfig

SNAPSHOT_DIR = Path(os.environ.get("WITWIN_MAXWELL_SNAPSHOT_DIR", ".bench_tmp/snapshots"))
FORWARD_STEPS = 1200
ADJOINT_STEPS = 300

FIELD_RTOL = 1e-5
FIELD_ATOL = 1e-6
DFT_RTOL = 1e-4
DFT_ATOL = 1e-6
GRAD_RTOL = 1e-4
GRAD_ATOL = 1e-7


def forward_outputs(name: str) -> dict[str, torch.Tensor]:
    scenario = SCENARIOS[name]
    scene = _clone_scene(scenario.builder(), device="cuda")
    source = scene.sources[0] if scene.sources else None
    normalize = isinstance(source, mw.PlaneWave) and getattr(source, "injection", "soft") == "soft"
    sim = Simulation.fdtd(
        scene,
        frequencies=scenario.frequencies,
        run_time=TimeConfig(time_steps=FORWARD_STEPS),
        spectral_sampler=mw.SpectralSampler(normalize_source=normalize),
    )
    prepared = sim.prepare()
    result = prepared.run()
    solver = prepared.solver
    out = {}
    for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        out[f"field_{field}"] = getattr(solver, field).detach().cpu()
    batched = getattr(solver, "_dft_batched_fields", None) or {}
    for component, parts in batched.items():
        for part_name in ("real", "imag"):
            tensor = parts.get(part_name)
            if tensor is not None:
                out[f"dft_{component}_{part_name}"] = tensor.detach().cpu()
    payload = result.monitor(scenario.display_monitor)
    components = payload.get("components", {})
    for component, comp_payload in components.items():
        data = comp_payload.get("data") if isinstance(comp_payload, dict) else comp_payload
        if torch.is_tensor(data):
            out[f"monitor_{component}_real"] = data.real.detach().cpu().float()
            out[f"monitor_{component}_imag"] = data.imag.detach().cpu().float()
    del result, prepared
    return out


class _AdjointScene(mw.SceneModule):
    def __init__(self, shape=(8, 8, 8), init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full(shape, float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.03),
            boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.12), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
        return scene


def adjoint_outputs(adjoint_backend: str) -> dict[str, torch.Tensor]:
    os.environ["WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND"] = adjoint_backend
    try:
        torch.manual_seed(5)
        model = _AdjointScene()
        sim = Simulation.fdtd(
            model,
            frequencies=[1e9],
            run_time=TimeConfig(time_steps=ADJOINT_STEPS),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        )
        result = sim.run()
        value = result.monitor("probe")["components"]["Ez"]
        loss = (value.real**2 + value.imag**2).sum()
        loss.backward()
        return {
            f"adjoint_{adjoint_backend}_grad": model.logits.grad.detach().cpu(),
            f"adjoint_{adjoint_backend}_loss": loss.detach().cpu().reshape(1),
        }
    finally:
        os.environ.pop("WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND", None)


def collect() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in ("dipole_vacuum", "planewave_vacuum"):
        for key, value in forward_outputs(name).items():
            out[f"{name}/{key}"] = value
    out.update(adjoint_outputs("auto"))
    return out


def tolerances_for(key: str) -> tuple[float, float]:
    if "dft" in key:
        return DFT_RTOL, DFT_ATOL
    if "grad" in key or "loss" in key:
        return GRAD_RTOL, GRAD_ATOL
    return FIELD_RTOL, FIELD_ATOL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["save", "check"])
    args = parser.parse_args()

    snapshot_path = SNAPSHOT_DIR / "cuda_correctness_snapshot.pt"
    if args.mode == "save":
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        data = collect()
        torch.save(data, snapshot_path)
        print(f"saved {len(data)} tensors to {snapshot_path}")
        return

    reference = torch.load(snapshot_path, weights_only=True)
    current = collect()
    missing = sorted(set(reference) - set(current))
    extra = sorted(set(current) - set(reference))
    if missing or extra:
        print(f"SNAPSHOT MISMATCH: missing={missing} extra={extra}")
        sys.exit(1)
    failures = []
    for key in sorted(reference):
        rtol, atol = tolerances_for(key)
        ref = reference[key]
        got = current[key]
        if not torch.allclose(got, ref, rtol=rtol, atol=atol):
            max_abs = (got - ref).abs().max().item()
            scale = ref.abs().max().item()
            failures.append(f"  {key}: max|diff|={max_abs:.3e} ref_max={scale:.3e} (rtol={rtol}, atol={atol})")
    if failures:
        print("SNAPSHOT CHECK FAILED:")
        print("\n".join(failures))
        sys.exit(1)
    print(f"snapshot check OK ({len(reference)} tensors)")


if __name__ == "__main__":
    main()
