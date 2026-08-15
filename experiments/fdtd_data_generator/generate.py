#!/usr/bin/env python3
"""Generate converged randomized 3-D WiTwin Maxwell CW-FDTD samples."""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Prefer the repository wheel unpacked beside this experiment. It contains the
# native CUDA library that an editable source checkout intentionally omits.
WHEEL_RUNTIME = Path(__file__).with_name("_wheel_runtime")
if WHEEL_RUNTIME.is_dir():
    sys.path.insert(0, str(WHEEL_RUNTIME))

C0 = 299_792_458.0
PULSE_FWIDTH = 0.25e9
CHANNELS = ("Ex_real", "Ex_imag", "Ey_real", "Ey_imag", "Ez_real", "Ez_imag")
FIELD_COMPONENTS = ("Ex", "Ey", "Ez")
GENERATOR_CONTRACT = "pulse-dft-stable6000-v1"
LAYOUTS = ("centered", "staggered")
# Fields added to Config after datasets were already written, with the value that reproduces
# the behaviour those datasets were generated under.  Used only to compare a stored config.
CONFIG_DEFAULTS = {"layout": "centered"}


@dataclass(frozen=True)
class Config:
    output_dir: str
    num_samples: int = 100
    grid_size: int = 256
    frequency: float = 2.0e9
    ppw: int = 16
    pml_layers: int = 10
    time_steps: int | None = None
    steady_cycles: int = 80
    max_transient_cycles: int = 640
    transient_cycles: int = 15
    steady_check_cycles: int = 4
    steady_tolerance: float = 0.01
    require_steady: bool = True
    seed: int = 20260715
    min_objects: int = 1
    max_objects: int = 4
    compression: str = "none"
    layout: str = "centered"
    cuda_graph: bool = False

    @property
    def grid_spacing(self) -> float:
        return C0 / self.frequency / self.ppw

    @property
    def domain_size(self) -> float:
        return self.grid_size * self.grid_spacing


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sample_paths(output_dir: Path, sample_id: int) -> tuple[Path, Path]:
    stem = output_dir / f"sample_{sample_id:05d}"
    return stem.with_suffix(".npz"), stem.with_name(stem.name + "_meta.json")


def centered_component(component: Any, axis: int, size: int) -> np.ndarray:
    array = component.detach().to(device="cpu").numpy()
    pml_cells = (array.shape[axis] - size) // 2
    expected = tuple(
        size + 2 * pml_cells + (dimension != axis) for dimension in range(3)
    )
    if array.shape != expected:
        raise ValueError(f"Unexpected Yee shape {array.shape}; expected {expected}")

    slices = []
    for dimension in range(3):
        width = size if dimension == axis else size + 1
        slices.append(slice(pml_cells, pml_cells + width))
    centered = array[tuple(slices)]

    for dimension in range(3):
        if dimension == axis:
            continue
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[dimension] = slice(0, -1)
        upper[dimension] = slice(1, None)
        centered = 0.5 * (centered[tuple(lower)] + centered[tuple(upper)])
    return np.asarray(centered, dtype=np.complex64)


def save_npz(
    output_path: Path,
    field: np.ndarray,
    permittivity: np.ndarray,
    compression: str,
) -> None:
    temporary = output_path.with_suffix(".npz.tmp")
    saver = np.savez_compressed if compression == "compressed" else np.savez
    with temporary.open("wb") as handle:
        saver(handle, field=field, permittivity=permittivity)
    temporary.replace(output_path)


def save_npz_arrays(output_path: Path, arrays: dict[str, Any], compression: str) -> None:
    """Atomic ``.npz`` write of a named array set, for layouts other than the centered one."""

    temporary = output_path.with_suffix(".npz.tmp")
    saver = np.savez_compressed if compression == "compressed" else np.savez
    with temporary.open("wb") as handle:
        saver(handle, **arrays)
    temporary.replace(output_path)


def pml_thickness(scene: Any) -> tuple[tuple[int, int], ...]:
    """The absorbing layer on each side of each axis.

    Read per face rather than inferred as ``(padded - physical) // 2``.  A face carrying a symmetry
    plane, a PEC boundary or a periodic axis gets no layer at all, so the two sides are not always
    equal and the halved difference is simply wrong -- measured, a PEC low face makes it 5 where the
    truth is ``(0, 10)``.  This reads the same authority that padded the grid in the first place, so
    the crop and the padding cannot disagree.
    """

    return tuple(
        (
            int(scene.pml_thickness_for_face(axis, "low")),
            int(scene.pml_thickness_for_face(axis, "high")),
        )
        for axis in ("x", "y", "z")
    )


def staggered_component(
    component: Any, axis: int, size: int, pml: tuple[tuple[int, int], ...]
) -> np.ndarray:
    """One Yee E component cropped to the physical domain, with no interpolation at all.

    A component is a cell-centre sample along its own axis and a node sample on the other two, so
    the physical crop takes ``size`` samples along its own axis and ``size + 1`` on the others,
    each starting at that axis's low-side layer count.  The padded shape is asserted rather than
    assumed: an off-by-one here would still look like a plausible field.
    """

    array = component.detach().to(device="cpu").numpy()
    expected = tuple(
        size + pml[dimension][0] + pml[dimension][1] + (dimension != axis)
        for dimension in range(3)
    )
    if array.shape != expected:
        raise ValueError(f"Unexpected Yee shape {array.shape}; expected {expected}")
    slices = tuple(
        slice(pml[dimension][0], pml[dimension][0] + size + (dimension != axis))
        for dimension in range(3)
    )
    return np.ascontiguousarray(array[slices], dtype=np.complex64)


def node_material(tensor: Any, size: int, pml: tuple[tuple[int, int], ...]) -> np.ndarray:
    """A node-sampled material array cropped to the physical domain's ``size + 1`` nodes.

    Kept on the nodes deliberately.  The runtime that produced these samples forms the value each
    update actually uses with ``average_node_to_component`` -- a two-point mean of this array along
    the component's own axis -- so storing the nodes preserves exactly what the kernels saw, where
    the cell-centred 8-point average the centered layout writes does not.
    """

    array = np.asarray(tensor.detach().to(device="cpu"), dtype=np.float32)
    expected = tuple(
        size + pml[dimension][0] + pml[dimension][1] + 1 for dimension in range(3)
    )
    if array.shape != expected:
        raise ValueError(f"Unexpected material shape {array.shape}; expected {expected}")
    slices = tuple(
        slice(pml[dimension][0], pml[dimension][0] + size + 1) for dimension in range(3)
    )
    return np.ascontiguousarray(array[slices])


def node_axis(tensor: Any, size: int, layers: tuple[int, int]) -> np.ndarray:
    """The physical node coordinates of one axis, in float64.

    Nodes, not cell centres: centres fix the nodes only up to one telescoping constant, so a
    consumer that has to weight a difference by the true sample separation cannot recover the cell
    widths from centres alone.
    """

    array = _as_numpy(tensor)
    if np.asarray(array).dtype != np.float64:
        raise ValueError(
            f"node coordinates must already be float64, got {np.asarray(array).dtype}; widening a "
            "float32 coordinate array back to float64 would look precise and not be"
        )
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    expected = size + layers[0] + layers[1] + 1
    if array.shape != (expected,):
        raise ValueError(f"Unexpected node coordinate shape {array.shape}; expected {expected}")
    return np.ascontiguousarray(array[layers[0] : layers[0] + size + 1])


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fit_cw_phasor(
    trace: np.ndarray,
    times: np.ndarray,
    frequency: float,
) -> tuple[complex, float]:
    """Least-squares CW phasor and relative non-sinusoidal residual."""

    omega_t = 2.0 * np.pi * frequency * times
    design = np.column_stack((np.cos(omega_t), np.sin(omega_t), np.ones_like(times)))
    coefficients, _, _, _ = np.linalg.lstsq(
        design, trace.astype(np.float64), rcond=None
    )
    fitted = design @ coefficients
    denominator = max(float(np.linalg.norm(trace)), 1e-30)
    residual = float(np.linalg.norm(trace - fitted) / denominator)
    return complex(float(coefficients[0]), -float(coefficients[1])), residual


def steady_state_certificate(
    result: Any,
    probe_names: list[str],
    config: Config,
) -> dict[str, Any]:
    """Compare two adjacent tail windows at every point probe."""

    window_s = config.steady_check_cycles / config.frequency
    records: list[dict[str, Any]] = []
    for name in probe_names:
        payload = result.monitor(name)
        times = np.asarray(_as_numpy(payload["t"]), dtype=np.float64)
        if times.size < 4:
            raise RuntimeError(f"steady-state probe {name!r} has too few samples")
        end = float(times[-1])
        previous_mask = (times >= end - 2.0 * window_s) & (times < end - window_s)
        latest_mask = times >= end - window_s
        if int(previous_mask.sum()) < 8 or int(latest_mask.sum()) < 8:
            raise RuntimeError(
                f"steady-state probe {name!r} does not cover two "
                f"{config.steady_check_cycles}-cycle windows"
            )

        previous: list[complex] = []
        latest: list[complex] = []
        latest_residuals: list[float] = []
        latest_norms: list[float] = []
        for component in FIELD_COMPONENTS:
            trace = np.asarray(
                _as_numpy(payload["components"][component]), dtype=np.float64
            )
            old_phasor, _ = _fit_cw_phasor(
                trace[previous_mask], times[previous_mask], config.frequency
            )
            new_phasor, residual = _fit_cw_phasor(
                trace[latest_mask], times[latest_mask], config.frequency
            )
            previous.append(old_phasor)
            latest.append(new_phasor)
            latest_residuals.append(residual)
            latest_norms.append(float(np.linalg.norm(trace[latest_mask])))

        previous_vector = np.asarray(previous, dtype=np.complex128)
        latest_vector = np.asarray(latest, dtype=np.complex128)
        amplitude = float(np.linalg.norm(latest_vector))
        residual_numerator = math.sqrt(
            sum(
                (residual * norm) ** 2
                for residual, norm in zip(latest_residuals, latest_norms)
            )
        )
        fit_residual = residual_numerator / max(
            math.sqrt(sum(norm**2 for norm in latest_norms)), 1e-30
        )
        relative_change = float(
            np.linalg.norm(latest_vector - previous_vector) / max(amplitude, 1e-30)
        )
        records.append(
            {
                "name": name,
                "amplitude": amplitude,
                "relative_phasor_change": relative_change,
                "latest_window_relative_fit_residual": fit_residual,
                "previous_samples": int(previous_mask.sum()),
                "latest_samples": int(latest_mask.sum()),
            }
        )

    amplitude_floor = (
        max((record["amplitude"] for record in records), default=0.0) * 1e-6
    )
    checked = [record for record in records if record["amplitude"] >= amplitude_floor]
    max_change = max(
        (record["relative_phasor_change"] for record in checked), default=0.0
    )
    max_residual = max(
        (record["latest_window_relative_fit_residual"] for record in checked),
        default=0.0,
    )
    passed = bool(
        checked
        and max_change <= config.steady_tolerance
        and max_residual <= config.steady_tolerance
    )
    return {
        "method": "adjacent_tail_cw_phasor_and_sinusoid_fit",
        "check_cycles_per_window": config.steady_check_cycles,
        "relative_tolerance": config.steady_tolerance,
        "amplitude_floor_relative_to_max_probe": 1e-6,
        "checked_probe_count": len(checked),
        "max_relative_phasor_change": max_change,
        "max_latest_window_relative_fit_residual": max_residual,
        "passed": passed,
        "probes": records,
    }


def spectral_metadata(result: Any) -> dict[str, Any]:
    solver = result.solver
    starts = getattr(solver, "_dft_start_steps", ())
    ends = getattr(solver, "_dft_end_steps", ())
    normalizations = getattr(solver, "_dft_window_normalization_values", ())
    return {
        "start_steps": [int(value) for value in starts],
        "end_steps": [int(value) for value in ends],
        "window_normalizations": [float(value) for value in normalizations],
    }


def sample_is_complete(config: Config, sample_id: int) -> bool:
    data_path, metadata_path = sample_paths(Path(config.output_dir), sample_id)
    if not data_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    convergence = metadata.get("convergence", {})
    convergence_complete = (
        convergence.get("passed") is True
        if config.require_steady
        else convergence.get("skipped") is True
    )
    return bool(
        metadata.get("generator_contract") == GENERATOR_CONTRACT
        and metadata.get("sample_id") == sample_id
        and metadata.get("seed") == config.seed + sample_id
        and metadata.get("grid_shape") == [config.grid_size] * 3
        and metadata.get("frequency_hz") == config.frequency
        and metadata.get("requested_time_steps") == config.time_steps
        and metadata.get("steady_cycles") == config.steady_cycles
        and metadata.get("max_transient_cycles") == config.max_transient_cycles
        and metadata.get("transient_cycles") == config.transient_cycles
        and metadata.get("layout", "centered") == config.layout
        and convergence_complete
        and metadata.get("storage_bytes") == data_path.stat().st_size
    )


def build_scene(config: Config, sample_seed: int, device: str):
    import witwin.maxwell as mw

    rng = np.random.default_rng(sample_seed)
    dl = config.grid_spacing
    half = config.domain_size / 2.0
    interior_half = half - (config.pml_layers + 5) * dl
    if interior_half <= 8 * dl:
        raise ValueError("Grid is too small for PML plus random-scene margins")

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(dl),
        boundary=mw.BoundarySpec.pml(num_layers=config.pml_layers, strength=1.0),
        device=device,
        subpixel_samples=1,
    )

    source_position = rng.uniform(-0.72 * interior_half, 0.72 * interior_half, size=3)
    source_width = float(rng.uniform(1.5, 3.0) * dl)
    source_polarization = random.Random(sample_seed).choice(("Ex", "Ey", "Ez"))
    occupied: list[tuple[np.ndarray, float]] = [(source_position, 5.0 * source_width)]
    geometries: list[dict[str, Any]] = []
    object_count = int(rng.integers(config.min_objects, config.max_objects + 1))

    for object_id in range(object_count):
        shape = str(rng.choice(("box", "sphere", "cylinder")))
        for attempt in range(100):
            center = rng.uniform(-0.72 * interior_half, 0.72 * interior_half, size=3)
            scale = float(rng.uniform(0.045, 0.13) * config.domain_size)
            if all(
                np.linalg.norm(center - other_center) > scale + other_radius
                for other_center, other_radius in occupied
            ):
                break
        else:
            continue

        eps_r = float(rng.uniform(2.0, 12.0))
        if shape == "box":
            size = rng.uniform(0.65, 1.25, size=3) * scale
            geometry = mw.Box(position=tuple(center), size=tuple(size))
            parameters: dict[str, Any] = {"size": size.tolist()}
        elif shape == "sphere":
            radius = float(0.5 * scale)
            geometry = mw.Sphere(position=tuple(center), radius=radius)
            parameters = {"radius": radius}
        else:
            radius = float(rng.uniform(0.28, 0.48) * scale)
            height = float(rng.uniform(0.7, 1.25) * scale)
            axis = str(rng.choice(("x", "y", "z")))
            geometry = mw.Cylinder(
                position=tuple(center), radius=radius, height=height, axis=axis
            )
            parameters = {"radius": radius, "height": height, "axis": axis}

        scene.add_structure(
            mw.Structure(
                name=f"object_{object_id:02d}",
                geometry=geometry,
                material=mw.Material(eps_r=eps_r),
            )
        )
        occupied.append((center, scale))
        geometries.append(
            {
                "type": shape,
                "center": center.tolist(),
                "eps_r": eps_r,
                **parameters,
            }
        )

    scene.add_source(
        mw.PointDipole(
            name="source",
            position=tuple(source_position),
            polarization=source_polarization,
            width=source_width,
            source_time=mw.GaussianPulse(
                frequency=config.frequency,
                fwidth=PULSE_FWIDTH,
                amplitude=1.0,
            ),
        )
    )

    direction = -source_position
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.asarray((1.0, 0.0, 0.0))
    else:
        direction /= norm
    near_source = source_position + direction * (0.5 * C0 / config.frequency)
    probes = [
        {
            "name": "steady_probe_near_source",
            "position": near_source.tolist(),
            "kind": "near_source",
        }
    ]
    probes.extend(
        {
            "name": f"steady_probe_object_{index:02d}",
            "position": geometry["center"],
            "kind": "object_center",
        }
        for index, geometry in enumerate(geometries)
    )
    for probe in probes:
        scene.add_monitor(
            mw.FieldTimeMonitor(
                probe["name"],
                components=("Ex", "Ey", "Ez"),
                position=tuple(probe["position"]),
                interval=1,
            )
        )

    return scene, {
        "source": {
            "position": source_position.tolist(),
            "width": source_width,
            "polarization": source_polarization,
            "time_kind": "gaussian_pulse",
            "fwidth_hz": PULSE_FWIDTH,
            "frequency_hz": config.frequency,
        },
        "geometries": geometries,
        "steady_probes": probes,
    }


def generate_one(config: Config, sample_id: int, device: str) -> dict[str, Any]:
    import torch
    import witwin.maxwell as mw

    output_dir = Path(config.output_dir)
    data_path, metadata_path = sample_paths(output_dir, sample_id)
    sample_seed = config.seed + sample_id
    started = time.perf_counter()
    scene, scene_description = build_scene(config, sample_seed, device)
    scene_seconds = time.perf_counter() - started

    torch.cuda.reset_peak_memory_stats()
    probe_names = [probe["name"] for probe in scene_description["steady_probes"]]
    propagation_transient_cycles = math.ceil(
        10.0 * config.domain_size * config.frequency / C0
    )
    effective_transient_cycles = max(
        config.transient_cycles, propagation_transient_cycles
    )
    convergence_attempts: list[dict[str, Any]] = []
    solve_seconds = 0.0
    while True:
        solve_started = time.perf_counter()
        run_time = (
            mw.TimeConfig(time_steps=config.time_steps)
            if config.time_steps is not None
            else mw.TimeConfig.auto(
                steady_cycles=config.steady_cycles,
                transient_cycles=effective_transient_cycles,
            )
        )
        result = mw.Simulation.fdtd(
            scene,
            frequencies=(config.frequency,),
            run_time=run_time,
            spectral_sampler=mw.SpectralSampler(window="none", normalize_source=True),
            full_field_dft=True,
            cuda_graph=config.cuda_graph,
        ).run()
        torch.cuda.synchronize()
        attempt_seconds = time.perf_counter() - solve_started
        solve_seconds += attempt_seconds
        solver_stats = {
            key: value
            for key, value in result.stats().items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        actual_steps = int(solver_stats["time_steps"])

        if config.require_steady:
            convergence = steady_state_certificate(result, probe_names, config)
        else:
            convergence = {
                "method": "disabled_by_configuration",
                "passed": None,
                "skipped": True,
            }
        convergence_attempts.append(
            {
                "steady_cycles": (
                    config.steady_cycles if config.time_steps is None else None
                ),
                "transient_cycles": (
                    effective_transient_cycles if config.time_steps is None else None
                ),
                "time_steps": actual_steps,
                "solve_seconds": attempt_seconds,
                "passed": convergence.get("passed"),
                "max_relative_phasor_change": convergence.get(
                    "max_relative_phasor_change"
                ),
                "max_latest_window_relative_fit_residual": convergence.get(
                    "max_latest_window_relative_fit_residual"
                ),
            }
        )
        if not config.require_steady or convergence["passed"]:
            break
        if (
            config.time_steps is not None
            or effective_transient_cycles >= config.max_transient_cycles
        ):
            worst_change = max(
                convergence["probes"],
                key=lambda probe: probe["relative_phasor_change"],
            )
            worst_fit = max(
                convergence["probes"],
                key=lambda probe: probe["latest_window_relative_fit_residual"],
            )
            failed_cycles = effective_transient_cycles
            del result, scene
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                "steady-state convergence failed after maximum runtime: "
                f"transient_cycles={failed_cycles}, "
                f"phasor_change={convergence['max_relative_phasor_change']:.6g}, "
                f"worst_change_probe={worst_change['name']}, "
                f"fit_residual={convergence['max_latest_window_relative_fit_residual']:.6g}, "
                f"worst_fit_probe={worst_fit['name']}, "
                f"tolerance={config.steady_tolerance:.6g}"
            )
        del result
        gc.collect()
        torch.cuda.empty_cache()
        effective_transient_cycles = min(
            2 * effective_transient_cycles,
            config.max_transient_cycles,
        )

    peak_gpu_bytes = int(torch.cuda.max_memory_allocated())

    save_started = time.perf_counter()
    size = config.grid_size
    if config.layout not in LAYOUTS:
        raise ValueError(f"unknown layout {config.layout!r}; choose one of {sorted(LAYOUTS)}")
    if config.layout == "staggered":
        layout_metadata = save_staggered_sample(
            data_path, result, scene, config, size
        )
        save_seconds = time.perf_counter() - save_started
        data_bytes = data_path.stat().st_size
        field = permittivity = None
    else:
        layout_metadata = {
            "data_format": "physical_domain_cell_centered",
            "channels": list(CHANNELS),
        }
        save_centered_sample(data_path, result, config, size)
        save_seconds = time.perf_counter() - save_started
        data_bytes = data_path.stat().st_size
        field = permittivity = None
    total_seconds = time.perf_counter() - started

    metadata = _sample_metadata(
        config=config,
        sample_id=sample_id,
        sample_seed=sample_seed,
        size=size,
        layout_metadata=layout_metadata,
        actual_steps=actual_steps,
        propagation_transient_cycles=propagation_transient_cycles,
        effective_transient_cycles=effective_transient_cycles,
        convergence_attempts=convergence_attempts,
        scene_description=scene_description,
        result=result,
        convergence=convergence,
        scene_seconds=scene_seconds,
        solve_seconds=solve_seconds,
        save_seconds=save_seconds,
        total_seconds=total_seconds,
        data_bytes=data_bytes,
        peak_gpu_bytes=peak_gpu_bytes,
        solver_stats=solver_stats,
    )
    atomic_json(metadata_path, metadata)

    del field, permittivity, result, scene
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def save_centered_sample(data_path: Path, result: Any, config: "Config", size: int) -> None:
    """The original layout: components interpolated to cell centres, permittivity to cell centres.

    Left byte-for-byte as it was.  The existing corpora have to stay reproducible, so the staggered
    layout is a second path rather than a change to this one.
    """

    field = np.empty((size, size, size, 6), dtype=np.float32)
    for component_id, (component, axis) in enumerate(
        ((result.E.x, 0), (result.E.y, 1), (result.E.z, 2))
    ):
        centered = centered_component(component, axis, size)
        field[..., 2 * component_id] = centered.real
        field[..., 2 * component_id + 1] = centered.imag
        del centered

    eps_nodes = np.asarray(
        result.materials.eps.scalar.detach().to(device="cpu"), dtype=np.float32
    )
    expected_nodes = size + 2 * config.pml_layers + 1
    if eps_nodes.shape != (expected_nodes,) * 3:
        raise ValueError(f"Unexpected permittivity shape {eps_nodes.shape}")
    pml = config.pml_layers
    eps_nodes = eps_nodes[
        pml : pml + size + 1,
        pml : pml + size + 1,
        pml : pml + size + 1,
    ]
    permittivity = 0.125 * (
        eps_nodes[:-1, :-1, :-1]
        + eps_nodes[1:, :-1, :-1]
        + eps_nodes[:-1, 1:, :-1]
        + eps_nodes[:-1, :-1, 1:]
        + eps_nodes[1:, 1:, :-1]
        + eps_nodes[1:, :-1, 1:]
        + eps_nodes[:-1, 1:, 1:]
        + eps_nodes[1:, 1:, 1:]
    )
    save_npz(data_path, field, permittivity, config.compression)


def save_staggered_sample(
    data_path: Path, result: Any, scene: Any, config: "Config", size: int
) -> dict[str, Any]:
    """The Yee layout, written as the solver computed it -- no interpolation anywhere.

    Emits each E component on its own sample set, the physical **node** coordinates of each axis,
    and the node permittivity (and permeability, only when it is not identically 1).  Together
    those are enough to reconstruct exactly what every update kernel saw: the value a component's
    update used is a two-point mean of the node array along that component's own axis.

    The cell-centred layout cannot support that.  Averaging a component onto the centres is not
    invertible, and the permittivity it stores has been averaged twice over -- which is the measured
    reason material recovery on that layout stops at O(1) relative error at interfaces.
    """

    require_storable_materials(result)
    pml = pml_thickness(scene)
    prepared = result.prepared_scene
    arrays: dict[str, Any] = {}
    for name, component, axis in (
        ("ex", result.E.x, 0),
        ("ey", result.E.y, 1),
        ("ez", result.E.z, 2),
    ):
        arrays[name] = staggered_component(component, axis, size, pml)

    for index, axis in enumerate(("x", "y", "z")):
        nodes = getattr(prepared, f"{axis}_nodes64", None)
        if nodes is None:
            raise RuntimeError(
                f"this runtime exposes no {axis}_nodes64; the float32 {axis} coordinates are not "
                "an acceptable substitute for a layout whose whole point is exact sample positions"
            )
        arrays[f"{axis}_nodes"] = node_axis(nodes, size, pml[index])

    material_metadata: dict[str, Any] = {}
    for family, accessor in (("eps", result.materials.eps), ("mu", result.materials.mu)):
        components = [
            node_material(getattr(accessor, axis), size, pml) for axis in ("x", "y", "z")
        ]
        isotropic = all(np.array_equal(components[0], other) for other in components[1:])
        if family == "mu" and isotropic and np.array_equal(
            components[0], np.ones_like(components[0])
        ):
            material_metadata[family] = "unit"
            continue
        if isotropic:
            arrays[f"{family}_nodes"] = components[0]
            material_metadata[family] = "isotropic"
        else:
            for axis, values in zip(("x", "y", "z"), components):
                arrays[f"{family}_{axis}_nodes"] = values
            material_metadata[family] = "anisotropic"

    save_npz_arrays(data_path, arrays, config.compression)
    return {
        "data_format": "physical_domain_yee_staggered",
        "components": {
            "ex": list(arrays["ex"].shape),
            "ey": list(arrays["ey"].shape),
            "ez": list(arrays["ez"].shape),
        },
        "coordinates": "primal_nodes",
        "materials": material_metadata,
        "material_sampling": "primal_nodes",
        "component_material_rule": (
            "0.5 * (node[:-1] + node[1:]) along the component's own axis; the solver averages "
            "eps_r * eps_0 rather than eps_r, so the two differ by at most one float32 ulp"
        ),
        "pml_layers_per_face": {
            axis: list(layers) for axis, layers in zip(("x", "y", "z"), pml)
        },
        "runtime": _runtime_provenance(),
    }


def require_storable_materials(result: Any) -> None:
    """Refuse a scene whose updates depend on something this layout does not store.

    The layout stores the node permittivity and permeability, and the claim that follows is that a
    component's update coefficient is recoverable from them.  That claim is false the moment the
    update also depends on a conductivity, a dispersive pole, a nonlinearity or a PEC fill -- a PEC
    cell in particular forces ``E = 0`` while its stored permittivity still reads as background, so
    the sample would record vacuum where the solver saw metal.  None of these occur in the scenes
    this generator builds, which is exactly why the failure would be silent if it ever did.
    """

    solver = result.solver
    unsupported = [
        name
        for name, flag in (
            ("electric conductivity", getattr(solver, "conductive_enabled", False)),
            ("magnetic conductivity", getattr(solver, "magnetically_conductive_enabled", False)),
            ("dispersion", getattr(solver, "dispersive_enabled", False)),
            ("nonlinearity", getattr(solver, "nonlinear_enabled", False)),
            ("modulation", getattr(solver, "modulation_enabled", False)),
            ("PEC fill", getattr(solver, "pec_enabled", False)),
        )
        if flag
    ]
    if unsupported:
        raise NotImplementedError(
            "the staggered layout stores only the node permittivity and permeability, but this "
            f"scene's updates also depend on: {', '.join(unsupported)}. Storing it would silently "
            "record a material the solver did not use."
        )


def _runtime_provenance() -> dict[str, Any]:
    """Which copy of ``witwin.maxwell`` ran, recorded in every staggered sample.

    Load-bearing rather than decorative.  The pinned wheel runtime forms a component's material by
    a node->edge two-point mean, which is why storing the node array is sufficient; the source tree
    has since moved to an edge-native subpixel blend with no such average, under which it would not
    be.  A consumer must be able to tell which contract its samples were written under.
    """

    import importlib.metadata

    import witwin.maxwell as mw

    try:
        version = importlib.metadata.version("witwin-maxwell")
    except importlib.metadata.PackageNotFoundError:
        version = None
    module = getattr(mw, "__file__", None)
    return {
        "version": version,
        "wheel_runtime": bool(module and str(WHEEL_RUNTIME) in str(module)),
    }


def _sample_metadata(
    *,
    config: "Config",
    sample_id: int,
    sample_seed: int,
    size: int,
    layout_metadata: dict[str, Any],
    actual_steps: int,
    propagation_transient_cycles: int,
    effective_transient_cycles: int,
    convergence_attempts: list[dict[str, Any]],
    scene_description: dict[str, Any],
    result: Any,
    convergence: dict[str, Any],
    scene_seconds: float,
    solve_seconds: float,
    save_seconds: float,
    total_seconds: float,
    data_bytes: int,
    peak_gpu_bytes: int,
    solver_stats: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "generator_contract": GENERATOR_CONTRACT,
        "sample_id": sample_id,
        "seed": sample_seed,
        "grid_shape": [size, size, size],
        "grid_spacing_m": config.grid_spacing,
        "domain_size_m": config.domain_size,
        "frequency_hz": config.frequency,
        "pml_layers": config.pml_layers,
        "time_steps": actual_steps,
        "requested_time_steps": config.time_steps,
        "runtime_mode": "fixed" if config.time_steps is not None else "auto",
        "steady_cycles": config.steady_cycles,
        "max_transient_cycles": config.max_transient_cycles,
        "propagation_transient_cycles": propagation_transient_cycles,
        "effective_transient_cycles": effective_transient_cycles,
        "convergence_attempts": convergence_attempts,
        "transient_cycles": config.transient_cycles,
        "layout": config.layout,
        "dtype": "float32",
        "compression": config.compression,
        "scene": scene_description,
        "spectral_sampling": {
            "window": "none",
            "normalize_source": True,
            **spectral_metadata(result),
        },
        "convergence": convergence,
        "timing_seconds": {
            "scene": scene_seconds,
            "solve": solve_seconds,
            "save": save_seconds,
            "total": total_seconds,
        },
        "storage_bytes": data_bytes,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "solver_stats": solver_stats,
    }
    metadata.update(layout_metadata)
    return metadata


def worker(
    config_dict: dict[str, Any],
    gpu_id: int,
    sample_ids: list[int],
    queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = Config(**config_dict)
    try:
        for sample_id in sample_ids:
            data_path, metadata_path = sample_paths(Path(config.output_dir), sample_id)
            if data_path.exists() or metadata_path.exists():
                if sample_is_complete(config, sample_id):
                    queue.put(
                        {"kind": "skipped", "sample_id": sample_id, "gpu": gpu_id}
                    )
                else:
                    queue.put(
                        {
                            "kind": "failure",
                            "sample_id": sample_id,
                            "gpu": gpu_id,
                            "traceback": (
                                "existing sample is partial, unconverged, or belongs to a "
                                "different generator configuration; choose a fresh output "
                                "directory instead of overwriting it"
                            ),
                        }
                    )
                continue
            try:
                metadata = generate_one(config, sample_id, "cuda:0")
                queue.put(
                    {
                        "kind": "success",
                        "sample_id": sample_id,
                        "gpu": gpu_id,
                        "metadata": metadata,
                    }
                )
            except Exception:
                queue.put(
                    {
                        "kind": "failure",
                        "sample_id": sample_id,
                        "gpu": gpu_id,
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        queue.put({"kind": "worker_done", "gpu": gpu_id})


def summarize(config: Config, wall_seconds: float) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    items = []
    for sample_id in range(config.num_samples):
        _, metadata_path = sample_paths(output_dir, sample_id)
        if sample_is_complete(config, sample_id):
            items.append(json.loads(metadata_path.read_text(encoding="utf-8")))

    data_files = [sample_paths(output_dir, item["sample_id"])[0] for item in items]
    metadata_files = [sample_paths(output_dir, item["sample_id"])[1] for item in items]
    sample_bytes = sum(path.stat().st_size for path in data_files)
    metadata_bytes = sum(path.stat().st_size for path in metadata_files)
    solves = [item["timing_seconds"]["solve"] for item in items]
    totals = [item["timing_seconds"]["total"] for item in items]
    steps = [item["time_steps"] for item in items]
    effective_cycles = [item["effective_transient_cycles"] for item in items]
    convergence = [
        item["convergence"]
        for item in items
        if item["convergence"].get("passed") is True
    ]
    return {
        "generator_contract": GENERATOR_CONTRACT,
        "config": asdict(config),
        "completed_samples": len(items),
        "converged_samples": len(convergence),
        "requested_samples": config.num_samples,
        "wall_seconds_this_run": wall_seconds,
        "sum_sample_total_seconds": float(sum(totals)),
        "mean_sample_total_seconds": float(np.mean(totals)) if totals else None,
        "mean_solve_seconds": float(np.mean(solves)) if solves else None,
        "min_solve_seconds": float(np.min(solves)) if solves else None,
        "max_solve_seconds": float(np.max(solves)) if solves else None,
        "min_time_steps": min(steps) if steps else None,
        "max_time_steps": max(steps) if steps else None,
        "min_effective_transient_cycles": min(effective_cycles) if items else None,
        "max_effective_transient_cycles": max(effective_cycles) if items else None,
        "max_relative_phasor_change": (
            max(item["max_relative_phasor_change"] for item in convergence)
            if convergence
            else None
        ),
        "max_latest_window_relative_fit_residual": (
            max(item["max_latest_window_relative_fit_residual"] for item in convergence)
            if convergence
            else None
        ),
        "sample_data_bytes": sample_bytes,
        "metadata_bytes": metadata_bytes,
        "total_dataset_bytes": sample_bytes + metadata_bytes,
        "mean_data_bytes_per_sample": sample_bytes / len(items) if items else None,
        "filesystem_free_bytes_after_run": shutil.disk_usage(output_dir).free,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("dataset_256_steady")),
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--frequency", type=float, default=2.0e9)
    parser.add_argument("--ppw", type=int, default=16)
    parser.add_argument("--pml-layers", type=int, default=10)
    parser.add_argument(
        "--time-steps",
        type=int,
        help="fixed diagnostic step count; omit for TimeConfig.auto steady-state runtime",
    )
    parser.add_argument("--steady-cycles", type=int, default=80)
    parser.add_argument("--max-transient-cycles", type=int, default=640)
    parser.add_argument("--transient-cycles", type=int, default=15)
    parser.add_argument("--steady-check-cycles", type=int, default=4)
    parser.add_argument("--steady-tolerance", type=float, default=0.01)
    parser.add_argument("--skip-steady-check", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--compression", choices=("none", "compressed"), default="none")
    parser.add_argument(
        "--layout",
        choices=("centered", "staggered"),
        default="centered",
        help=(
            "centered interpolates every component onto cell centres (the original, "
            "unchanged); staggered writes the Yee components as the solver computed them, "
            "plus the physical node coordinates and the node materials"
        ),
    )
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--cuda-graph", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_samples <= 0 or args.grid_size <= 0:
        raise SystemExit("num-samples and grid-size must be positive")
    if args.frequency <= 0.0 or args.ppw <= 0 or args.pml_layers <= 0:
        raise SystemExit("frequency, ppw, and pml-layers must be positive")
    if args.time_steps is not None and args.time_steps <= 0:
        raise SystemExit("time-steps must be positive when explicitly provided")
    if args.steady_cycles <= 0 or args.transient_cycles <= 0:
        raise SystemExit("steady-cycles and transient-cycles must be positive")
    if args.max_transient_cycles < args.transient_cycles:
        raise SystemExit("max-transient-cycles must be >= transient-cycles")
    if args.steady_check_cycles <= 0:
        raise SystemExit("steady-check-cycles must be positive")
    propagation_cycles = math.ceil(10.0 * args.grid_size / args.ppw)
    if args.time_steps is None and args.max_transient_cycles < propagation_cycles:
        raise SystemExit(
            "max-transient-cycles must cover the domain propagation-time floor"
        )
    if args.time_steps is None and 2 * args.steady_check_cycles > args.steady_cycles:
        raise SystemExit("steady-cycles must cover two steady-check windows")
    if not 0.0 < args.steady_tolerance < 1.0:
        raise SystemExit("steady-tolerance must be in (0, 1)")
    if args.min_objects < 0 or args.max_objects < args.min_objects:
        raise SystemExit("Require 0 <= min-objects <= max-objects")
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise SystemExit("At least one GPU ID is required")

    config = Config(
        output_dir=str(Path(args.output_dir).resolve()),
        num_samples=args.num_samples,
        grid_size=args.grid_size,
        frequency=args.frequency,
        ppw=args.ppw,
        pml_layers=args.pml_layers,
        time_steps=args.time_steps,
        steady_cycles=args.steady_cycles,
        transient_cycles=args.transient_cycles,
        steady_check_cycles=args.steady_check_cycles,
        max_transient_cycles=args.max_transient_cycles,
        steady_tolerance=args.steady_tolerance,
        require_steady=not args.skip_steady_check,
        seed=args.seed,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        compression=args.compression,
        layout=args.layout,
        cuda_graph=args.cuda_graph,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_payload = asdict(config)
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        # A directory written before a field existed is not a different configuration -- it is the
        # same one with that field at its default.  Comparing raw dicts would make every new field
        # break resume against every existing dataset, which is what happened when `layout` landed.
        existing_config = {**CONFIG_DEFAULTS, **existing_config}
        if existing_config != config_payload:
            raise SystemExit(
                f"{config_path} belongs to a different generator configuration; "
                "choose a fresh output directory"
            )
    else:
        atomic_json(config_path, config_payload)

    assignments = {gpu_id: [] for gpu_id in gpu_ids}
    for sample_id in range(config.num_samples):
        assignments[gpu_ids[sample_id % len(gpu_ids)]].append(sample_id)

    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(asdict(config), gpu_id, assignments[gpu_id], queue),
        )
        for gpu_id in gpu_ids
    ]
    started = time.perf_counter()
    for process in processes:
        process.start()

    done_workers = 0
    failures = 0
    while done_workers < len(processes):
        event = queue.get()
        kind = event["kind"]
        if kind == "worker_done":
            done_workers += 1
        elif kind == "success":
            metadata = event["metadata"]
            timing = metadata["timing_seconds"]
            size_gib = metadata["storage_bytes"] / 1024**3
            convergence = metadata["convergence"]
            steady = (
                f"steady={convergence['max_relative_phasor_change']:.3g}/"
                f"{convergence['max_latest_window_relative_fit_residual']:.3g}"
                if convergence.get("passed") is True
                else "steady=unchecked"
            )
            print(
                f"[{event['sample_id'] + 1:03d}/{config.num_samples:03d}] "
                f"GPU {event['gpu']} steps={metadata['time_steps']} "
                f"transient={metadata['effective_transient_cycles']} "
                f"attempts={len(metadata['convergence_attempts'])} "
                f"solve={timing['solve']:.2f}s total={timing['total']:.2f}s "
                f"{steady} size={size_gib:.3f} GiB",
                flush=True,
            )
        elif kind == "skipped":
            print(
                f"[skip] sample {event['sample_id']:05d} already complete", flush=True
            )
        else:
            failures += 1
            print(
                f"[failed] sample {event['sample_id']:05d} on GPU {event['gpu']}\n"
                f"{event['traceback']}",
                file=sys.stderr,
                flush=True,
            )

    for process in processes:
        process.join()
    wall_seconds = time.perf_counter() - started
    summary = summarize(config, wall_seconds)
    summary["failures_this_run"] = failures
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if failures or summary["completed_samples"] != config.num_samples else 0


if __name__ == "__main__":
    raise SystemExit(main())
