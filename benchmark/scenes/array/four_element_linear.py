from __future__ import annotations

import math

import torch

import witwin.maxwell as mw


ARRAY_BENCHMARK_FREQUENCY = 1.0e9
_C0 = 299792458.0
_ELEMENT_COUNT = 4
_GRID_STEP = 0.0075
ARRAY_BENCHMARK_WEIGHTS = tuple(
    tuple(
        0.5 * complex(math.cos(index * 2.0 * math.pi * beam / 16.0), math.sin(index * 2.0 * math.pi * beam / 16.0))
        for index in range(_ELEMENT_COUNT)
    )
    for beam in range(16)
)


def _port(name: str, x: float, *, x_index: int, x_nodes: torch.Tensor) -> mw.LumpedPort:
    lower = 0.5 * float(x_nodes[x_index - 1] + x_nodes[x_index])
    upper = 0.5 * float(x_nodes[x_index] + x_nodes[x_index + 1])
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, _GRID_STEP),
        negative=(x, 0.0, -_GRID_STEP),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.5 * (lower + upper), 0.0, -0.5 * _GRID_STEP),
            size=(upper - lower, _GRID_STEP, 0.0),
        ),
        reference_impedance=50.0,
    )


def _dipole_arm(name: str, x: float, sign: float) -> mw.Structure:
    inner = _GRID_STEP
    outer = 0.06
    center = sign * 0.5 * (inner + outer)
    return mw.Structure(
        name=name,
        geometry=mw.Box(
            position=(x, 0.0, center),
            size=(2.0 * _GRID_STEP, 2.0 * _GRID_STEP, outer - inner),
        ),
        material=mw.Material.pec(),
    )


def build_four_element_linear_scene(*, device="cuda") -> mw.Scene:
    """Return the frozen 96^3-cell, half-wavelength four-element benchmark."""

    spacing = _C0 / (2.0 * ARRAY_BENCHMARK_FREQUENCY)
    positions = tuple((index - 1.5) * spacing for index in range(_ELEMENT_COUNT))
    x_nodes = torch.linspace(-0.3, 0.3, 81, dtype=torch.float64)
    x_indices = (10, 30, 50, 70)
    x_nodes[torch.tensor(x_indices)] = torch.tensor(
        positions,
        dtype=torch.float64,
    )
    transverse_nodes = torch.linspace(-0.3, 0.3, 81, dtype=torch.float64)
    ports = tuple(
        _port(
            f"element_{index + 1}",
            x,
            x_index=x_index,
            x_nodes=x_nodes,
        )
        for index, (x, x_index) in enumerate(zip(positions, x_indices))
    )
    structures = tuple(
        _dipole_arm(f"element_{index + 1}_{side}", x, sign)
        for index, x in enumerate(positions)
        for side, sign in (("lower", -1.0), ("upper", 1.0))
    )
    surface = mw.ClosedSurfaceMonitor.box(
        "array_nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.54, 0.18, 0.18),
        frequencies=(ARRAY_BENCHMARK_FREQUENCY,),
    )
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3),) * 3),
        grid=mw.GridSpec.custom(x_nodes, transverse_nodes, transverse_nodes),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        structures=structures,
        ports=ports,
        monitors=(surface,),
        device=device,
    )


def benchmark_weight_tensor(*, device, dtype) -> torch.Tensor:
    return torch.tensor(ARRAY_BENCHMARK_WEIGHTS, device=device, dtype=dtype).unsqueeze(1)


__all__ = [
    "ARRAY_BENCHMARK_FREQUENCY",
    "ARRAY_BENCHMARK_WEIGHTS",
    "benchmark_weight_tensor",
    "build_four_element_linear_scene",
]
