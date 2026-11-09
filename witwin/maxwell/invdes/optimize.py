"""Explicit optimization-loop helpers: bound clipping, history and continuation.

These helpers never call ``loss.backward()`` and never choose the optimizer. A
loop stays plain PyTorch::

    loop = DesignLoop(torch.optim.Adam(model.parameters(), lr=0.05),
                      bounds=(0.0, 1.0), beta_schedule=BetaSchedule())
    for iteration in range(100):
        model.design.beta = loop.beta
        loop.zero_grad()
        loss = objective(model)
        loss.backward()
        loop.step(loss, gray=gray_level_penalty(model.design()))
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from .projection import BetaSchedule


@torch.no_grad()
def clip_to_bounds_(parameters: Iterable[torch.Tensor], lower=0.0, upper=1.0) -> None:
    """Clamp parameters in place to ``[lower, upper]`` (scalars or broadcastable tensors)."""
    for parameter in parameters:
        lo = torch.as_tensor(lower, device=parameter.device, dtype=parameter.dtype)
        hi = torch.as_tensor(upper, device=parameter.device, dtype=parameter.dtype)
        parameter.copy_(torch.maximum(torch.minimum(parameter, hi), lo))


def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("history metrics must be scalars.")
        value = value.detach()
        if value.is_complex():
            raise ValueError("history metrics must be real.")
        return float(value.item())
    return float(value)


class OptimizationHistory:
    """Per-iteration record of the loss, beta, gradient norm and named metrics."""

    def __init__(self):
        self.records: list[dict[str, float]] = []

    def append(self, iteration: int, loss, **metrics) -> dict[str, float]:
        record = {"iteration": int(iteration), "loss": _scalar(loss)}
        record.update({name: _scalar(value) for name, value in metrics.items() if value is not None})
        self.records.append(record)
        return record

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def column(self, name: str) -> list[float]:
        """Values of one metric over iterations (``nan`` where it was not recorded)."""
        return [record.get(name, math.nan) for record in self.records]

    @property
    def losses(self) -> list[float]:
        return self.column("loss")

    def best(self) -> dict[str, float]:
        """Record with the smallest loss."""
        if not self.records:
            raise ValueError("history is empty.")
        return min(self.records, key=lambda record: record["loss"])

    def to_dict(self) -> dict[str, list[float]]:
        """Column-oriented copy suitable for ``json.dump``."""
        names: list[str] = []
        for record in self.records:
            names.extend(name for name in record if name not in names)
        return {name: self.column(name) for name in names}


class DesignLoop:
    """Bookkeeping around a user-owned PyTorch optimizer.

    ``step(loss, **metrics)`` must be called after the caller's own
    ``loss.backward()``. It records the loss, the current beta, the global
    gradient norm and any extra metrics, applies ``optimizer.step()``, clips the
    parameters to ``bounds`` and advances the iteration counter.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        bounds: tuple | None = (0.0, 1.0),
        beta_schedule: BetaSchedule | None = None,
        history: OptimizationHistory | None = None,
    ):
        self.optimizer = optimizer
        self.bounds = bounds
        self.beta_schedule = beta_schedule
        self.history = OptimizationHistory() if history is None else history
        self.iteration = 0

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [p for group in self.optimizer.param_groups for p in group["params"]]

    @property
    def beta(self) -> float | None:
        """Continuation beta for the current iteration (``None`` without a schedule)."""
        return None if self.beta_schedule is None else self.beta_schedule(self.iteration)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def gradient_norm(self) -> float:
        grads = [p.grad for p in self.parameters if p.grad is not None]
        if not grads:
            raise RuntimeError("no parameter has a gradient; call loss.backward() before DesignLoop.step().")
        return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])))

    def step(self, loss, **metrics) -> dict[str, float]:
        grad_norm = self.gradient_norm()
        record = self.history.append(self.iteration, loss, beta=self.beta, grad_norm=grad_norm, **metrics)
        self.optimizer.step()
        if self.bounds is not None:
            clip_to_bounds_(self.parameters, *self.bounds)
        self.iteration += 1
        return record
