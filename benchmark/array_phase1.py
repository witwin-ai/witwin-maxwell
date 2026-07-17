from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch

import witwin.maxwell as mw
from benchmark.scenes.array.four_element_linear import (
    ARRAY_BENCHMARK_FREQUENCY,
    benchmark_weight_tensor,
    build_four_element_linear_scene,
)
from witwin.maxwell.array import ARRAY_ACCEPTANCE_BUDGET, _solid_angle_weights
from witwin.maxwell.postprocess.antenna import _far_fields_from_result


def _source_time():
    return mw.CW(frequency=ARRAY_BENCHMARK_FREQUENCY)


def _simulation(scene, excitations):
    return mw.Simulation.fdtd(
        scene,
        frequency=ARRAY_BENCHMARK_FREQUENCY,
        excitations=excitations,
        run_time=mw.TimeConfig(time_steps=ARRAY_ACCEPTANCE_BUDGET.phase1_steps),
        absorber="stablepml",
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    )


def _build_basis(scene):
    result = _simulation(
        scene,
        mw.PortSweep(source_time=_source_time()),
    ).run()
    basis = result.array_basis(
        monitor="array_nf2ff",
        theta_points=ARRAY_ACCEPTANCE_BUDGET.phase1_angular_shape[0],
        phi_points=ARRAY_ACCEPTANCE_BUDGET.phase1_angular_shape[1],
        radius=1.0,
        batch_size=1024,
    )
    return result, basis


def _direct_beam(scene, basis, weights):
    calibration = basis.metadata["measured_incident_waves"][0]
    amplitudes = weights / calibration
    excitations = tuple(
        mw.PortExcitation(
            name,
            amplitude=amplitude.detach().cpu(),
            source_impedance="matched",
            source_time=_source_time(),
        )
        for name, amplitude in zip(basis.port_names, amplitudes)
    )
    result = _simulation(scene, excitations).run()
    transformed = _far_fields_from_result(
        result,
        surface="array_nf2ff",
        frequencies=basis.frequencies,
        theta=basis.eep.theta,
        phi=basis.eep.phi,
        radius=1.0,
        phase_center=basis.eep.phase_center,
        frame=basis.eep.frame,
        batch_size=1024,
    )
    direct_a = torch.stack([result.port(name).a for name in basis.port_names], dim=-1)
    direct_b = torch.stack([result.port(name).b for name in basis.port_names], dim=-1)
    torch.cuda.synchronize()
    return transformed, direct_a, direct_b


def _weighted_complex_l2(actual, expected, theta):
    weights = _solid_angle_weights(expected.far_field.theta, expected.far_field.phi)
    numerator = torch.sum(
        weights
        * (
            torch.abs(actual["e_theta"] - expected.far_field.e_theta).square()
            + torch.abs(actual["e_phi"] - expected.far_field.e_phi).square()
        )
    )
    denominator = torch.sum(
        weights
        * (
            torch.abs(expected.far_field.e_theta).square()
            + torch.abs(expected.far_field.e_phi).square()
        )
    )
    return float(torch.sqrt(numerator / denominator))


def _phase_rms(actual, expected):
    expected_magnitude = torch.sqrt(
        torch.abs(expected.far_field.e_theta).square()
        + torch.abs(expected.far_field.e_phi).square()
    )
    support = expected_magnitude >= (
        ARRAY_ACCEPTANCE_BUDGET.fdtd_phase_support_fraction
        * torch.amax(expected_magnitude)
    )
    coherent = (
        actual["e_theta"] * torch.conj(expected.far_field.e_theta)
        + actual["e_phi"] * torch.conj(expected.far_field.e_phi)
    )
    return float(
        torch.sqrt(torch.mean(torch.angle(coherent[support]).square()))
        * 180.0
        / math.pi
    )


def _relative_power_error(actual, expected):
    floor = torch.finfo(actual.dtype).tiny
    return float(torch.amax(torch.abs(actual - expected) / torch.abs(expected).clamp_min(floor)))


def _cuda_elapsed(function):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    value = function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1.0e-3, value


def _qualification(scene):
    result, basis = _build_basis(scene)
    weights = benchmark_weight_tensor(device=basis.device, dtype=basis.dtype)
    beams = basis.combine(weights)
    comparisons = {}
    for beam_index, label in ((0, "broadside"), (8, "endfire")):
        direct, direct_a, direct_b = _direct_beam(
            scene,
            basis,
            weights[beam_index, 0],
        )
        expected = basis.combine(weights[beam_index])
        comparisons[label] = {
            "weighted_complex_l2": _weighted_complex_l2(direct, expected, basis.eep.theta),
            "phase_rms_deg": _phase_rms(direct, expected),
            "incident_power_wave_relative_error": _relative_power_error(
                torch.abs(direct_a).square(),
                expected.network.incident_power_per_port,
            ),
            "reflected_power_wave_relative_error": _relative_power_error(
                torch.abs(direct_b).square(),
                expected.network.reflected_power_per_port,
            ),
            "accepted_power_relative_error": _relative_power_error(
                torch.sum(torch.abs(direct_a).square(), dim=-1)
                - torch.sum(torch.abs(direct_b).square(), dim=-1),
                expected.network.accepted_power,
            ),
        }
    closure = torch.abs(beams.network.accepted_power - beams.antenna.p_rad) / (
        beams.network.incident_power
    )
    metrics = {
        "comparisons": comparisons,
        "max_physical_power_residual": float(torch.amax(closure)),
        "grid_cells": tuple(int(value - 1) for value in (len(result.prepared_scene.x), len(result.prepared_scene.y), len(result.prepared_scene.z))),
        "pml_cells_per_face": result.scene.boundary.num_layers,
        "steps": int(
            result._array_run_data.column_results[0][0].stats()["time_steps"]
        ),
        "angular_shape": tuple(int(value) for value in basis.eep.theta.shape),
        "beam_count": int(weights.shape[0]),
    }
    return basis, weights, metrics


def _timings(scene, basis, weights, *, warmups: int, samples: int, rounds: int):
    def basis_workflow():
        result, measured_basis = _build_basis(scene)
        measured_basis.combine(weights)
        torch.cuda.synchronize()
        del measured_basis, result

    def direct_workflow():
        for beam_index in range(weights.shape[0]):
            _direct_beam(scene, basis, weights[beam_index, 0])

    def combine_workflow():
        basis.combine(weights)

    for _ in range(warmups):
        _cuda_elapsed(basis_workflow)
        _cuda_elapsed(direct_workflow)
        _cuda_elapsed(combine_workflow)

    basis_times = []
    direct_times = []
    combine_times = []
    for round_index in range(rounds):
        order = (
            (("basis", basis_workflow), ("direct", direct_workflow))
            if round_index % 2 == 0
            else (("direct", direct_workflow), ("basis", basis_workflow))
        )
        for _ in range(samples):
            for name, operation in order:
                elapsed, _ = _cuda_elapsed(operation)
                (basis_times if name == "basis" else direct_times).append(elapsed)
            elapsed, _ = _cuda_elapsed(combine_workflow)
            combine_times.append(elapsed)
    basis_median = statistics.median(basis_times)
    direct_median = statistics.median(direct_times)
    combine_median = statistics.median(combine_times)
    return {
        "warmups": warmups,
        "samples_per_round": samples,
        "alternating_rounds": rounds,
        "basis_plus_16_combine_seconds_median": basis_median,
        "sixteen_direct_seconds_median": direct_median,
        "sixteen_combine_seconds_median": combine_median,
        "basis_to_direct_ratio": basis_median / direct_median,
        "combine_to_one_solve_ratio": combine_median / (direct_median / 16.0),
        "basis_samples_seconds": basis_times,
        "direct_samples_seconds": direct_times,
        "combine_samples_seconds": combine_times,
    }


def _assert_gates(metrics, timings, *, qualifying):
    for comparison in metrics["comparisons"].values():
        assert comparison["weighted_complex_l2"] <= ARRAY_ACCEPTANCE_BUDGET.fdtd_complex_l2
        assert comparison["phase_rms_deg"] <= ARRAY_ACCEPTANCE_BUDGET.fdtd_phase_rms_deg
        assert comparison["incident_power_wave_relative_error"] <= ARRAY_ACCEPTANCE_BUDGET.port_power_relative_error
        assert comparison["reflected_power_wave_relative_error"] <= ARRAY_ACCEPTANCE_BUDGET.port_power_relative_error
        assert comparison["accepted_power_relative_error"] <= ARRAY_ACCEPTANCE_BUDGET.port_power_relative_error
    assert metrics["max_physical_power_residual"] <= ARRAY_ACCEPTANCE_BUDGET.physical_power_residual
    assert tuple(metrics["grid_cells"]) == ARRAY_ACCEPTANCE_BUDGET.phase1_grid_shape
    assert metrics["pml_cells_per_face"] == 8
    assert metrics["steps"] == ARRAY_ACCEPTANCE_BUDGET.phase1_steps
    assert tuple(metrics["angular_shape"]) == ARRAY_ACCEPTANCE_BUDGET.phase1_angular_shape
    assert metrics["beam_count"] == 16
    assert timings["basis_to_direct_ratio"] <= ARRAY_ACCEPTANCE_BUDGET.phase1_basis_direct_time_ratio
    assert timings["combine_to_one_solve_ratio"] <= ARRAY_ACCEPTANCE_BUDGET.phase1_combine_solve_time_ratio
    if qualifying:
        assert timings["warmups"] == ARRAY_ACCEPTANCE_BUDGET.timing_warmups
        assert timings["samples_per_round"] == ARRAY_ACCEPTANCE_BUDGET.timing_samples
        assert timings["alternating_rounds"] == ARRAY_ACCEPTANCE_BUDGET.timing_order_rounds


def _hardware_metadata():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    driver = "unavailable"
    pci_bus = "unavailable"
    try:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()[torch.cuda.current_device()]
        driver, pci_bus = (part.strip() for part in query.split(",", maxsplit=1))
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        pass
    return {
        "device": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "driver": driver,
        "pci_bus": pci_bus,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=ARRAY_ACCEPTANCE_BUDGET.timing_warmups)
    parser.add_argument("--samples", type=int, default=ARRAY_ACCEPTANCE_BUDGET.timing_samples)
    parser.add_argument("--rounds", type=int, default=ARRAY_ACCEPTANCE_BUDGET.timing_order_rounds)
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="Allow non-frozen timing counts and mark the output non-qualifying.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmups, args.samples, args.rounds) < 0 or args.samples == 0 or args.rounds == 0:
        parser.error("warmups must be non-negative and samples/rounds must be positive")
    exact_timing_protocol = (
        args.warmups == ARRAY_ACCEPTANCE_BUDGET.timing_warmups
        and args.samples == ARRAY_ACCEPTANCE_BUDGET.timing_samples
        and args.rounds == ARRAY_ACCEPTANCE_BUDGET.timing_order_rounds
    )
    if not exact_timing_protocol and not args.exploratory:
        parser.error(
            "non-frozen timing counts require --exploratory; qualifying runs use 3/5/4"
        )

    scene = build_four_element_linear_scene()
    basis, weights, metrics = _qualification(scene)
    timings = _timings(
        scene,
        basis,
        weights,
        warmups=args.warmups,
        samples=args.samples,
        rounds=args.rounds,
    )
    payload = {
        "qualifying": exact_timing_protocol and not args.exploratory,
        "hardware": _hardware_metadata(),
        "metrics": metrics,
        "timings": timings,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    _assert_gates(metrics, timings, qualifying=payload["qualifying"])


if __name__ == "__main__":
    main()
