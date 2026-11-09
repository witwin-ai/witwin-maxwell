"""CPU sampling/operator contracts; synthetic arrays are not PDE evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_fixture(name, relative):
    path = Path(__file__).parent / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rotated = _load_fixture("rotated_slab_contract", "combinations/test_rotated_birefringent_dispersive_slab.py")
_magnetic = _load_fixture("magnetic_slab_contract", "dispersive/test_fdtd_dispersive.py")


@pytest.mark.parametrize("frequency", [0.8e9, 1.2e9])
@pytest.mark.parametrize("epsilon", [1.4, 1.8])
@pytest.mark.parametrize("backward", [0.0, 0.37 - 0.29j])
def test_bulk_index_ignores_internal_counterwave(frequency, epsilon, backward):
    h, dt = 0.006, 1e-11
    n = np.sqrt(epsilon + _rotated._POLE.susceptibility_at_freq(frequency))
    omega_discrete = 2 * np.sin(np.pi * frequency * dt) / dt
    q = 2 * np.arcsin(omega_discrete * h * n / (2 * _rotated._C)) / h
    xs = np.linspace(-0.75, 0.75, 251)
    field = np.exp(1j * q * xs) + backward * np.exp(-1j * q * xs)
    actual = _rotated._bulk_index_from_line(xs, field, frequency, dt, 1)
    assert abs(actual - n) / abs(n) < 2e-13
    conjugate = _rotated._bulk_index_from_line(xs, field.conjugate(), frequency, dt, -1)
    assert abs(conjugate - n) / abs(n) < 2e-13


@pytest.mark.parametrize("value", [0.0, np.nan, np.inf])
def test_bulk_index_requires_finite_nonzero_signal(value):
    xs = np.linspace(-0.75, 0.75, 251)
    with pytest.raises(ValueError, match="finite nonzero"):
        _rotated._bulk_index_from_line(xs, np.full(len(xs), value), 1.2e9, 1e-11, 1)


@pytest.mark.parametrize("frequency,dt,sign", [(0.0, 1e-11, 1), (1.2e9, 0.0, 1), (1.2e9, 1e-11, 0)])
def test_bulk_index_requires_actual_clock_and_direction(frequency, dt, sign):
    xs = np.linspace(-0.75, 0.75, 251)
    with pytest.raises(ValueError, match="positive carrier/dt"):
        _rotated._bulk_index_from_line(xs, np.ones(len(xs)), frequency, dt, sign)


@pytest.mark.parametrize("padding", [0, 3, 6])
def test_physical_aperture_excludes_added_pml_samples(padding):
    h = 0.04
    y = np.linspace(-0.4 - padding * h, 0.4 + padding * h, 21 + 2 * padding)
    z = (y[:-1] + y[1:]) / 2
    mask = (np.abs(y[:, None]) <= 0.4) & (np.abs(z[None, :]) <= 0.4)
    field = np.where(mask, 0.8 + 0.0j, 0.0j)
    assert _magnetic._physical_plane_field_mean(field, (y, z), ((-0.4, 0.4), (-0.4, 0.4))) == pytest.approx(0.8)
    if padding == 6:
        # This is a sampling-functional counterexample, not a simulated field.
        assert _magnetic._plane_field_mean(field) < 0.5


@pytest.mark.parametrize("case", ["shape", "axes", "bounds", "empty", "nonfinite"])
def test_physical_aperture_rejects_invalid_contract(case):
    axes = (np.linspace(-0.4, 0.4, 21), np.linspace(-0.38, 0.38, 20))
    field = np.ones((21, 20), dtype=complex)
    bounds = ((-0.4, 0.4), (-0.4, 0.4))
    if case == "shape":
        field = field[:-1]
    elif case == "axes":
        axes = (axes[0][:, None], axes[1])
    elif case == "bounds":
        bounds = ((0.4, -0.4), (-0.4, 0.4))
    elif case == "empty":
        bounds = ((1.0, 2.0), (1.0, 2.0))
    else:
        field[5, 5] = np.nan
    with pytest.raises(ValueError):
        _magnetic._physical_plane_field_mean(field, axes, bounds)
