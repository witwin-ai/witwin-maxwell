"""Native dense CPML must preserve its declared scalar arithmetic precision."""
from fractions import Fraction

import numpy as np
import pytest
import torch

from witwin.maxwell.fdtd.cuda import backend

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


def _kernel_arguments(fixture, tensors):
    component = fixture["component"]
    a, b = ["xyz"[axis] for axis in fixture["axes"]]
    prefix_b, prefix_c = ("B", "C") if fixture["electric"] else ("By", "Cy")
    kwargs = {
        component: tensors[14], fixture["first"]: tensors[0], fixture["second"]: tensors[1],
        component + "Decay": tensors[2], component + "Curl": tensors[3],
        "InvKappa" + component + a.upper(): tensors[4],
        prefix_b + component + a.upper(): tensors[5], prefix_c + component + a.upper(): tensors[6],
        "InvKappa" + component + b.upper(): tensors[7],
        prefix_b + component + b.upper(): tensors[8], prefix_c + component + b.upper(): tensors[9],
        "invD" + a: tensors[10], "invD" + b: tensors[11],
        "Psi" + component + a.upper(): tensors[12], "Psi" + component + b.upper(): tensors[13],
        "uniformDecay": None, "uniformCurl": None,
    }
    if fixture["electric"]:
        kwargs.update({a + "LowBoundaryMode": 2, a + "HighBoundaryMode": 2,
                       b + "LowBoundaryMode": 2, b + "HighBoundaryMode": 2})
    name = (
        "updateElectricField" + component + "Cpml3D"
        if fixture["electric"] else "updateMagneticField" + component + "3D"
    )
    return name, kwargs


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
    fixture = _local_fixture(component, active)
    expected, scales = _physical_curl_reference(fixture)
    tensors = [torch.as_tensor(x, device="cuda", dtype=torch.float64).contiguous() for x in fixture["arrays"]]
    module = backend.get_native_fdtd_module()
    name, kwargs = _kernel_arguments(fixture, tensors)
    getattr(module, name)(**kwargs).launchRaw()
    actual = [tensors[index][fixture["point"]].detach().cpu().item() for index in (14, 12, 13)]
    errors = _normalized_errors(actual, expected, scales)
    record_property("maximum_normalized_update_error", max(errors))
    record_property("field_and_two_memories_compared", 3)
    record_property("native_field_dtype", str(tensors[14].dtype))
    assert max(errors) <= 2e-10
