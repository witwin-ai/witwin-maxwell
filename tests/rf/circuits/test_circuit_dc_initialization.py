"""Coupled DC admission uses operating-point precision without an ampere floor."""

from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.mna import (
    CompiledStampPlan,
    LinearMNASystem,
    _terminal_voltage,
    compile_coupled_mna_system,
)
from witwin.maxwell.fdtd import circuits as runtime


def test_coupled_mna_rejects_circuit_without_em_port_binding():
    """The DC port-current bound in circuits.py relies on an EM binding and a non-ground node."""
    circuit = mw.Circuit("unbound")

    with pytest.raises(ValueError, match="requires at least one EM port binding"):
        compile_coupled_mna_system(circuit, dt=1e-12, device="cpu", dtype=torch.float64)


@pytest.fixture(params=(
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable."
    )),
))
def device(request, monkeypatch):
    if request.param == "cpu":
        # Exercise the real assembly/LU/initialization on host tensors without
        # changing the production compiler's CUDA-only admission contract.
        def compile_cpu(circuit, *, dt, device, dtype):
            return LinearMNASystem(circuit, CompiledStampPlan(
                graph=circuit.compile(),
                device=torch.device(device),
                dtype=dtype,
                dt=dt,
                inductor_names=(),
                inductance_matrix=torch.empty((0, 0), device=device, dtype=dtype),
            ))

        monkeypatch.setattr(runtime, "compile_coupled_mna_system", compile_cpu)
    return torch.device(request.param)


def _prepare(circuit, impedance_scale, device, dtype):
    solver = SimpleNamespace(
        scene=SimpleNamespace(circuits=(circuit,)),
        device=device,
        dt=torch.tensor(1e-12, device=device, dtype=dtype),
        Ex=torch.zeros(1, device=device, dtype=dtype),
    )
    port = SimpleNamespace(
        port=SimpleNamespace(name="feed"), geometry=None, field_name="Ez",
    )
    field = SimpleNamespace(
        discrete_port_impedance=torch.tensor(7.1 * impedance_scale, device=device, dtype=dtype),
        coupling_impedance=torch.tensor(14.2 * impedance_scale, device=device, dtype=dtype),
    )
    return runtime.prepare_circuit_runtimes(
        solver, (port,), field_couplings={"feed": field},
    )[0]


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("integration", ("trapezoidal", "backward_euler"))
@pytest.mark.parametrize("impedance_scale", (1.0, 1e-6))
@pytest.mark.parametrize("voltage_scale", (1.0, 1e-6))
def test_biased_series_dc_block_admits_roundoff_port_current(
    device, dtype, integration, impedance_scale, voltage_scale,
):
    circuit = mw.Circuit("dc_blocked", config=mw.MNAConfig(integration=integration))
    source = circuit.node("source")
    bias = circuit.node("bias")
    feed = circuit.node("feed")
    voltage = torch.tensor(3.7 * voltage_scale, device=device, dtype=dtype)
    r1 = torch.tensor(2.3 * impedance_scale, device=device, dtype=dtype)
    r2 = torch.tensor(4.9 * impedance_scale, device=device, dtype=dtype)
    circuit.add(mw.VoltageSource("V1", source, circuit.ground, voltage))
    circuit.add(mw.Resistor("R1", source, bias, r1))
    circuit.add(mw.Resistor("R2", bias, circuit.ground, r2))
    circuit.add(mw.Capacitor("C1", source, feed, 1e-12 / impedance_scale))
    circuit.bind_port("feed", positive=feed, negative=bias)

    prepared = _prepare(circuit, impedance_scale, device, dtype)

    # At DC the capacitor is open, so feed and bias have the same voltage.
    # Their independently solved values can differ by LU roundoff.
    port_voltage = _terminal_voltage(prepared.initial_solution, prepared.system.graph, feed, bias)
    current = -prepared.ports[0].conductance(integration) * port_voltage
    assert current.abs() <= 64 * torch.finfo(dtype).eps * voltage / impedance_scale
    torch.testing.assert_close(
        prepared.initial_solution[-1], -voltage / (r1 + r2),
        rtol=8 * torch.finfo(dtype).eps, atol=0,
    )
    torch.testing.assert_close(
        prepared.state.capacitor_voltage["C1"], voltage * r1 / (r1 + r2),
        rtol=8 * torch.finfo(dtype).eps, atol=0,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("integration", ("trapezoidal", "backward_euler"))
@pytest.mark.parametrize("impedance_scale", (1.0, 1e-6))
@pytest.mark.parametrize("voltage_scale", (1.0, 1e-6, 1e-18))
def test_dc_driven_port_rejects_even_tiny_physical_current(
    device, dtype, integration, impedance_scale, voltage_scale,
):
    circuit = mw.Circuit("dc_driven", config=mw.MNAConfig(integration=integration))
    feed = circuit.node("feed")
    circuit.add(mw.VoltageSource(
        "V1", feed, circuit.ground,
        torch.tensor(3.7 * voltage_scale, device=device, dtype=dtype),
    ))
    circuit.add(mw.Resistor("R1", feed, circuit.ground, 2.3 * impedance_scale))
    circuit.bind_port("feed", positive=feed, negative=circuit.ground)

    with pytest.raises(ValueError, match="nonzero DC port current.*zero initial Yee field"):
        _prepare(circuit, impedance_scale, device, dtype)
