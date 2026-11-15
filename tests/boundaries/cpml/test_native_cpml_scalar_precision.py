"""Native dense CPML (the fused core Yee launch) must preserve its declared scalar arithmetic precision."""
from fractions import Fraction

import numpy as np
import pytest
import torch


_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _local_fixture(component, active):
    """Static nonuniform inverse-spacing inputs and nonzero CPML history."""
    electric = component.startswith("E")
    own = "xyz".index(component[1])
    nodes = (7, 6, 5)
    shapes = {
        kind + "xyz"[axis]: tuple(
            n - int((dimension == axis) == (kind == "E"))
            for dimension, n in enumerate(nodes)
        )
        for kind in ("E", "H")
        for axis in range(3)
    }
    first, second = (
        (("Hy", "Hz"), ("Hx", "Hz"), ("Hx", "Hy"))
        if electric
        else (("Ey", "Ez"), ("Ex", "Ez"), ("Ex", "Ey"))
    )[own]
    shape = shapes[component]
    a, b = (1 if own == 0 else 0), (1 if own == 2 else 2)
    rng = np.random.default_rng(50721 + own + 10 * electric)
    fields = [rng.normal(size=shapes[name]).astype(np.float64) for name in (first, second)]
    vector = lambda axis, value: np.asarray(
        [value * (1 + .03125 * index) for index in range(shape[axis])], dtype=np.float64
    )
    if active:
        profiles = [
            vector(a, .65432198765), vector(a, .76543219876), vector(a, -.04321098765),
            vector(b, .6987654321), vector(b, .7432198765), vector(b, -.03219876543),
        ]
        memories = [rng.normal(size=shape).astype(np.float64) for _ in range(2)]
    else:
        profiles = [
            np.ones(shape[a]), np.zeros(shape[a]), np.zeros(shape[a]),
            np.ones(shape[b]), np.zeros(shape[b]), np.zeros(shape[b]),
        ]
        memories = [np.zeros(shape), np.zeros(shape)]
    arrays = [
        *fields, np.full(shape, .875), np.full(shape, .125), *profiles,
        vector(a, 198.765432109), vector(b, 201.234567891),
        *memories, rng.normal(size=shape).astype(np.float64),
    ]
    return {"component": component, "electric": electric, "own": own,
            "first": first, "second": second, "shape": shape,
            "axes": (a, b), "point": (2, 2, 2), "arrays": arrays}


def _physical_curl_reference(fixture):
    """Exact-input Maxwell curl and two independent CPML memory recurrences."""
    arrays, point = fixture["arrays"], fixture["point"]
    scalar = lambda value: Fraction.from_float(float(value))
    memories, memory_scales, curl_terms = [], [], []
    for slot, direction, source_index, source_name, profile_index in (
        (0, fixture["axes"][0], 1, fixture["second"], 4),
        (1, fixture["axes"][1], 0, fixture["first"], 7),
    ):
        lower, upper = list(point), list(point)
        if fixture["electric"]:
            lower[direction] -= 1
        else:
            upper[direction] += 1
        field = arrays[source_index]
        derivative = (
            scalar(field[tuple(upper)]) - scalar(field[tuple(lower)])
        ) * scalar(arrays[10 + slot][point[direction]])
        kappa_inverse, decay, drive = [
            scalar(arrays[profile_index + j][point[direction]]) for j in range(3)
        ]
        old_memory = scalar(arrays[12 + slot][point])
        memory = decay * old_memory + drive * derivative
        memories.append(memory)
        memory_scales.append(abs(decay * old_memory) + abs(drive * derivative))
        # epsilon_ijk gives the physical orientation, independent of CUDA branches.
        own, source_axis = fixture["own"], "xyz".index(source_name[1])
        epsilon = (own - direction) * (direction - source_axis) * (source_axis - own) // 2
        assert epsilon in (-1, 1)
        curl_terms.append(epsilon * (kappa_inverse * derivative + memory))
    field_decay = scalar(arrays[2][point])
    field_curl = scalar(arrays[3][point])
    old_field = scalar(arrays[14][point])
    sign = 1 if fixture["electric"] else -1
    value = field_decay * old_field + sign * field_curl * sum(curl_terms)
    scale = abs(field_decay * old_field) + abs(field_curl) * sum(abs(x) for x in curl_terms)
    return [value, *memories], [scale, *memory_scales]


def _normalized_errors(actual, expected, scales):
    errors = []
    for value, reference, scale in zip(actual, expected, scales):
        difference = abs(Fraction.from_float(float(value)) - reference)
        if scale == 0:
            assert difference == 0, "Nonzero error with exact-zero reference scale"
            errors.append(0.)
        else:
            errors.append(float(difference / scale))
    return errors


def _core_fixture(component, active):
    """The fixture of ``_local_fixture`` written into a prepared float64 dense-CPML core solver.

    The per-component CPML kernels were deleted in W7 row 16e; the fused core Yee launch
    now carries the dense CPML update. Fields, inverse spacings, CPML profiles and memories
    are bound by storage, so they are filled in place; the decay/curl coefficients are
    compressed when the core binds, so they are set first and the core is rebound.
    """
    from benchmark.core_path.gates import make_simulation
    from benchmark.core_path.scenes import build_core_case
    from tests.support.core_solver import bind_core

    solver = make_simulation(build_core_case("fp64_vacuum_port"), steps=1, full_field_dft=False).prepare().solver
    assert solver.Ex.dtype == torch.float64 and solver.uses_cpml and solver._cpml_memory_mode == "dense"
    electric = component.startswith("E")
    family = "e" if electric else "h"
    own = "xyz".index(component[1])
    first, second = (
        (("Hy", "Hz"), ("Hx", "Hz"), ("Hx", "Hy"))
        if electric
        else (("Ey", "Ez"), ("Ex", "Ez"), ("Ex", "Ey"))
    )[own]
    a, b = (1 if own == 0 else 0), (1 if own == 2 else 2)
    rng = np.random.default_rng(50721 + own + 10 * electric)
    fill = lambda tensor, values: tensor.copy_(torch.as_tensor(values, dtype=tensor.dtype, device=tensor.device))
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        fill(getattr(solver, name), rng.normal(size=tuple(getattr(solver, name).shape)))
    for axis in "xyz":
        for kind, value in (("decay", .875), ("curl", .125)):
            name = f"c{family}{axis}_{kind}"
            getattr(solver, name).fill_(value)
            solver._coefficient_uniformity[name] = None  # indexed coefficient tables, as dense per-cell arrays
    vector = lambda axis, value: np.asarray(
        [value * (1 + .03125 * index) for index in range(getattr(solver, f"cpml_b_{family}_{'xyz'[axis]}").numel())],
        dtype=np.float64,
    )
    spacing = {a: 198.765432109, b: 201.234567891, own: 200.0}
    for axis, value in spacing.items():
        inv = getattr(solver, f"inv_d{'xyz'[axis]}_{family}")
        fill(inv, [value * (1 + .03125 * index) for index in range(inv.numel())])
    for axis in range(3):
        name = "xyz"[axis]
        if active and axis == a:
            values = (vector(a, .65432198765), vector(a, .76543219876), vector(a, -.04321098765))
        elif active and axis == b:
            values = (vector(b, .6987654321), vector(b, .7432198765), vector(b, -.03219876543))
        else:
            count = getattr(solver, f"cpml_b_{family}_{name}").numel()
            values = (np.ones(count), np.zeros(count), np.zeros(count))
        for profile, value in zip(("inv_kappa", "b", "c"), values):
            fill(getattr(solver, f"cpml_{profile}_{family}_{name}"), value)
    for target, axes in (("x", "yz"), ("y", "xz"), ("z", "xy")):
        for axis in axes:
            psi = getattr(solver, f"psi_{family}{target}_{axis}")
            assert tuple(psi.shape) == tuple(getattr(solver, family.upper() + target).shape)  # dense memory
            fill(psi, rng.normal(size=tuple(psi.shape)) if active else np.zeros(tuple(psi.shape)))
    bind_core(solver)
    own_name, axis_a, axis_b = component[1].lower(), "xyz"[a], "xyz"[b]
    tensors = [
        getattr(solver, first), getattr(solver, second),
        getattr(solver, f"c{family}{own_name}_decay"), getattr(solver, f"c{family}{own_name}_curl"),
        *(getattr(solver, f"cpml_{profile}_{family}_{axis_a}") for profile in ("inv_kappa", "b", "c")),
        *(getattr(solver, f"cpml_{profile}_{family}_{axis_b}") for profile in ("inv_kappa", "b", "c")),
        getattr(solver, f"inv_d{axis_a}_{family}"), getattr(solver, f"inv_d{axis_b}_{family}"),
        getattr(solver, f"psi_{family}{own_name}_{axis_a}"), getattr(solver, f"psi_{family}{own_name}_{axis_b}"),
        getattr(solver, component),
    ]
    shape = tuple(tensors[14].shape)
    point = tuple(size // 2 for size in shape)
    fixture = {"component": component, "electric": electric, "own": own,
               "first": first, "second": second, "shape": shape, "axes": (a, b), "point": point,
               "arrays": [tensor.detach().cpu().numpy().copy() for tensor in tensors]}
    return solver, fixture, tensors


@pytest.mark.parametrize("component", _COMPONENTS)
def test_fraction_reference_matches_analytic_linear_curl(component):
    fixture = _local_fixture(component, False)
    arrays, point = fixture["arrays"], fixture["point"]
    analytic_terms = []
    for source_index, direction, source_name, slot in (
        (0, fixture["axes"][1], fixture["first"], 1),
        (1, fixture["axes"][0], fixture["second"], 0),
    ):
        arrays[source_index] = np.indices(arrays[source_index].shape)[direction].astype(np.float64)
        own, source_axis = fixture["own"], "xyz".index(source_name[1])
        epsilon = (own - direction) * (direction - source_axis) * (source_axis - own) // 2
        analytic_terms.append(epsilon * Fraction.from_float(float(arrays[10 + slot][point[direction]])))
    expected, _ = _physical_curl_reference(fixture)
    f = lambda value: Fraction.from_float(float(value))
    analytic = f(arrays[2][point]) * f(arrays[14][point])
    analytic += (1 if fixture["electric"] else -1) * f(arrays[3][point]) * sum(analytic_terms)
    assert expected == [analytic, Fraction(0), Fraction(0)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA execution.")
@pytest.mark.parametrize("component", _COMPONENTS)
@pytest.mark.parametrize("active", (False, True), ids=("inactive", "active"))
def test_native_dense_cpml_keeps_scalar_precision(component, active, record_property):
    solver, fixture, tensors = _core_fixture(component, active)
    expected, scales = _physical_curl_reference(fixture)
    if fixture["electric"]:
        solver._core_runtime.electric()
    else:
        solver._core_runtime.magnetic()
    actual = [tensors[index][fixture["point"]].detach().cpu().item() for index in (14, 12, 13)]
    errors = _normalized_errors(actual, expected, scales)
    record_property("maximum_normalized_update_error", max(errors))
    record_property("field_and_two_memories_compared", 3)
    record_property("native_field_dtype", str(tensors[14].dtype))
    assert max(errors) <= 2e-10
