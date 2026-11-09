"""Measure the exported fixed-k incident spectrum's one-step label rotation.

CUDA runs the full native scene twice. CPU prepares the same scene with native
admission/loading mocked, then advances only its PyTorch incident line: the CPU
reference backend does not admit Bloch boundaries. Neither mode changes library
code, source sampling, normalization, or the checkpointed accumulator.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0
from witwin.maxwell.fdtd.excitation import fixed_k


FREQUENCIES = (0.8e9, 0.9e9, 1.0e9, 1.1e9, 1.2e9)
EXPORT = fixed_k.FixedKIncidentLine.incident_spectrum


def _previous_labels(line, solver):
    # Revert only physical_phase = exp(+i*omega*dt) to exp(0). Limit the
    # module-local substitution to export, after all field/DFT updates finish.
    def label_phase(argument):
        assert torch.equal(argument, 1j * line.omega * line.dt)
        return torch.exp(argument * 0)

    with patch.object(fixed_k, "torch", SimpleNamespace(exp=label_phase)):
        return EXPORT(line, solver)


def _simulation(device, steps):
    sine = 0.25
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.06, 0.06), (-0.06, 0.06), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.015), device=device,
        boundary=mw.BoundarySpec.faces(
            x="bloch", y="bloch", z="pml", num_layers=4,
            bloch_wavevector=(sine * 2 * math.pi * 1e9 / C_0, 0., 0.)),
    )
    scene.add_source(mw.PlaneWave(
        direction=(sine, 0., math.sqrt(1 - sine**2)), polarization="Ey",
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.2e9),
        injection=mw.TFSF.slab(axis="z", bounds=(-0.15, 0.15)),
    ))
    return mw.Simulation.fdtd(
        scene, frequencies=FREQUENCIES, field_dtype=torch.float32,
        run_time=mw.TimeConfig(time_steps=steps), cuda_graph=False,
        full_field_dft=False, shutoff=0.,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
    )


@torch.no_grad()
def _run(device, steps, *, previous_labels):
    with ExitStack() as stack:
        if previous_labels:
            stack.enter_context(patch.object(fixed_k.FixedKIncidentLine,
                                            "incident_spectrum", _previous_labels))
        if torch.device(device).type == "cpu":
            # Preparation-only mocks, matching the host fixed-k provider tests.
            for name in ("witwin.maxwell.simulation._require_cuda_scene",
                         "witwin.maxwell.fdtd.solver._require_cuda_scene",
                         "witwin.maxwell.fdtd.runtime.initialization.require_cuda_scene"):
                stack.enter_context(patch(name, return_value=None))
            for name in ("witwin.maxwell.fdtd.solver._get_fdtd_module",
                         "witwin.maxwell.fdtd.runtime.initialization.get_fdtd_module"):
                stack.enter_context(patch(name, return_value=object()))
        prepared = _simulation(device, steps).prepare()
        solver = prepared.solver
        assert solver._tfsf_state["provider"] == "plane_wave_fixed_k"
        line = solver._tfsf_state["auxiliary_grid"]
        if torch.device(device).type == "cpu":
            for _ in range(steps):
                line.advance_magnetic()
                line.advance_electric()
            # This is the exact exporter used for Result metadata on CUDA.
            exported = line.incident_spectrum(solver)
        else:
            exported = prepared.run().metadata["tfsf_incident_reference"]
        state = {name: value.clone() for name, value in line.state_tensors().items()}
        if torch.device(device).type == "cuda":
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                state["grid/" + name] = getattr(solver, name).clone()
                state["grid/" + name + "_imag"] = getattr(solver, name + "_imag").clone()
        normalization = (solver._normalization_time_offset,
                         solver._normalization_label_offset, solver._spectral_reference_frequency)
        return exported, state, normalization


def _complex(value):
    return dict(real=float(value.real), imag=float(value.imag))


def _json_safe(value):
    # Keep failed zero/nonfinite-spectrum measurements visible in valid JSON.
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    current, state_a, normalization_a = _run(args.device, args.steps, previous_labels=False)
    previous, state_b, normalization_b = _run(args.device, args.steps, previous_labels=True)
    equal_state = all(torch.equal(value, state_b[name]) for name, value in state_a.items())
    equal_metadata = all(current[name] == previous[name] for name in current if name != "electric")
    equal_normalization = normalization_a == normalization_b
    # Ey is the driven, nonzero TE component; symmetry-zero Ex/Ez have no ratio.
    a, b = current["electric"][:, 1], previous["electric"][:, 1]
    ratio = a / b
    omega_dt = torch.tensor(FREQUENCIES, device=a.device, dtype=torch.float64) * (2 * math.pi * current["dt"])
    magnitude_error = (ratio.abs() - 1).abs()
    phase_residual = torch.angle(ratio * torch.exp(-1j * omega_dt)).abs()
    bound = 64 * torch.finfo(torch.float64).eps
    passed = (bool(torch.isfinite(ratio).all()) and bool((b.abs() > 0).all())
              and float(magnitude_error.max()) <= bound and float(phase_residual.max()) <= bound
              and equal_state and equal_metadata and equal_normalization)
    report = dict(
        device=args.device, scope="incident-line-only" if torch.device(args.device).type == "cpu" else "full-native-scene",
        steps=args.steps, component="Ey", dt=current["dt"],
        label_before="post-update E at step n labelled n*dt",
        label_after="post-update E at step n labelled (n+1)*dt",
        label_offset_before=0, label_offset_after=1,
        delta_label=1, fourier_sign=1, expected_factor="exp(+i*omega*dt)",
        state_equal=equal_state, metadata_equal=equal_metadata, normalization_equal=equal_normalization,
        max_magnitude_error=float(magnitude_error.max()), max_phase_residual_rad=float(phase_residual.max()),
        roundoff_bound=bound, passed=passed,
        classification="(A) follow-up label-only mover" if passed else "T2: investigate beyond-label change",
        frequencies=[dict(frequency_hz=f, spectrum_a=_complex(a[i]), spectrum_b=_complex(b[i]),
                          c=_complex(ratio[i]), magnitude_error=float(magnitude_error[i]),
                          expected_phase_rad=float(omega_dt[i]), phase_residual_rad=float(phase_residual[i]))
                     for i, f in enumerate(FREQUENCIES)],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_json_safe(report), indent=2, allow_nan=False)
    args.output.write_text(serialized + "\n")
    print(serialized)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
