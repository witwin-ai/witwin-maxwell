"""Contract test: ``finalize_embedded_networks`` must detach ``state_norm``.

``state_norm`` is a reported diagnostic, never an optimization observable
(docs/reference/network-embedding.md). The finalize path builds it with
``torch.linalg.vector_norm(runtime.state.detach())`` so the contract holds by
construction even when the runtime state becomes graph-connected. This test
feeds finalize a deliberately graph-connected ``runtime.state`` and asserts the
resulting ``state_norm`` carries no autograd graph. Without the ``.detach()``
the whole state recurrence would leak into the Result, so this test goes red if
the detach is ever removed.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from witwin.maxwell.fdtd.networks import finalize_embedded_networks


def _solver_with_graph_connected_state() -> tuple[SimpleNamespace, dict]:
    frequencies = torch.tensor([1.0e9, 2.0e9], dtype=torch.float64)
    port_data = SimpleNamespace(
        frequencies=frequencies,
        voltage=torch.tensor([1.0 + 0.1j, 0.8 - 0.2j], dtype=torch.complex128),
        current=torch.tensor([0.02 + 0.001j, 0.016 - 0.002j], dtype=torch.complex128),
    )
    ports = {"feed": port_data}
    compiled = SimpleNamespace(
        connection_names=("feed",),
        port_order=("feed",),
        model_id="net-model",
        fit_report=None,
        frequency_band=(1.0e9, 2.0e9),
        delay=None,
    )
    # A graph-connected runtime state: requires_grad and carries a grad_fn.
    base = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
    state = base * 2.0
    runtime = SimpleNamespace(
        compiled=compiled,
        state=state,
        name="net",
        runtime_warnings=(),
        port_energy=torch.tensor([0.1], dtype=torch.float64),
        absorbed_energy=torch.tensor(0.2, dtype=torch.float64),
        generated_energy=torch.tensor(0.0, dtype=torch.float64),
        loop_condition=1.0,
    )
    solver = SimpleNamespace(_network_runtimes=(runtime,))
    return solver, ports


def test_finalize_detaches_state_norm_from_a_graph_connected_state():
    solver, ports = _solver_with_graph_connected_state()

    # Guard the guard: the runtime state must actually be graph-connected, or
    # this test would pass trivially and could never detect a regression.
    graph_state = solver._network_runtimes[0].state
    assert graph_state.requires_grad is True
    assert graph_state.grad_fn is not None

    output = finalize_embedded_networks(solver, ports)
    state_norm = output["net"].state_norm

    assert state_norm.requires_grad is False
    assert state_norm.grad_fn is None
