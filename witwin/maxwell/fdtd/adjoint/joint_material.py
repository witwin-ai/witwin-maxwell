"""GPU PyTorch VJP for the linear conductive electric material recurrence."""
from copy import copy

import torch

from .profiler import _ReverseStepResult


def qualifies_joint_material(solver, forward_state):
    return forward_state["Ex"].is_cuda


def reverse_joint_material(solver, forward_state, adjoint_state, *, time_value,
                           eps_ex, eps_ey, eps_ez, resolved_source_terms=None, profiler=None):
    """Differentiate the same GPU field recurrence with checkpoint-sized storage.

    Circuit operators have already been transposed by the shared RF pullback.
    This specialization differentiates the free Maxwell/material update only.
    It never copies state to a CPU and does not fall back to a different solver.
    """
    from .core import _step_state

    proxy = copy(solver)
    proxy._port_runtimes = ()
    proxy._lumped_element_runtimes = ()
    proxy._circuit_runtimes = ()
    proxy._network_runtimes = ()
    proxy._mur_state = ()  # The outer bridge already transposes the boundary tail.
    proxy.conductive_enabled = True
    names = tuple(name for name, value in forward_state.items() if value.is_floating_point())
    state = {name: value.detach().requires_grad_(name in names) for name, value in forward_state.items()}
    sigmas = tuple(getattr(solver, f"sigma_e_{name}").detach().requires_grad_(True) for name in ("Ex", "Ey", "Ez"))
    proxy._adjoint_sigma_reference = {name: getattr(solver, f"sigma_e_{name}").detach() for name in ("Ex", "Ey", "Ez")}
    for name, sigma in zip(("Ex", "Ey", "Ez"), sigmas):
        setattr(proxy, f"sigma_e_{name}", sigma)
    with torch.enable_grad():
        pole_leaves = []
        pole_targets = []
        for attribute in ("_dispersive_templates", "_magnetic_dispersive_templates"):
            templates = getattr(solver, attribute, {})
            copied = {component: dict(values) for component, values in templates.items()}
            for component, values in templates.items():
                for kind in ("debye", "drude", "lorentz"):
                    copied[component][kind] = [dict(entry) for entry in values[kind]]
                    for index, entry in enumerate(copied[component][kind]):
                        targets = getattr(solver, "_pole_coefficient_gradients", None)
                        if targets is None:
                            continue
                        for key, target in targets[(component, kind, index)].items():
                            entry[key] = torch.as_tensor(entry[key], device=target.device, dtype=target.dtype).detach().requires_grad_(True)
                            pole_leaves.append(entry[key])
                            pole_targets.append(target)
            setattr(proxy, attribute, copied)
        from .copper_foil import differentiable_copper_entries
        proxy._copper_foils, copper_leaves = differentiable_copper_entries(
            solver, dict(zip(("Ex", "Ey", "Ez"), (eps_ex, eps_ey, eps_ez))),
            dict(zip(("Ex", "Ey", "Ez"), sigmas)))
        coefficient_inputs = tuple(value for group in copper_leaves for value in group)
        output = _step_state(proxy, state, time_value=time_value,
                             eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez,
                             resolved_source_terms=None)
        output_names = tuple(name for name, value in output.items() if value.requires_grad)
        inputs = tuple(state[name] for name in names) + (eps_ex, eps_ey, eps_ez) + sigmas + coefficient_inputs + tuple(pole_leaves)
        gradients = torch.autograd.grad(tuple(output[name] for name in output_names), inputs,
                                        tuple(adjoint_state[name] for name in output_names), allow_unused=True)
    state_grads = dict(zip(names, gradients[:len(names)]))
    pre = {name: (state_grads[name] if state_grads.get(name) is not None else (
        torch.zeros_like(value) if name in output else adjoint_state[name])) for name, value in forward_state.items()}
    material_grads = tuple(gradient if gradient is not None else torch.zeros_like(value)
                          for gradient, value in zip(gradients[len(names):], (eps_ex, eps_ey, eps_ez, *sigmas)))
    from .copper_foil import COPPER_COEFFICIENTS
    copper_gradients = []
    cursor = len(names) + 6
    for entry, leaves in zip(getattr(solver, "_copper_foils", ()), copper_leaves):
        record = {key: gradient if gradient is not None else torch.zeros_like(leaf)
            for key, gradient, leaf in zip(COPPER_COEFFICIENTS, gradients[cursor:cursor + len(leaves)], leaves)}
        record["structure_index"] = next(index for index, structure in enumerate(solver.scene.structures)
                                         if structure.material is entry["face"].material)
        record["response_alpha"] = entry.get("response_alpha", 1.)
        record["edge_geometry"] = entry.get("edge_geometry")
        copper_gradients.append(record)
        cursor += len(leaves)
    for target, gradient in zip(pole_targets, gradients[cursor:]):
        if gradient is not None:
            target.add_(gradient)
    return _ReverseStepResult(pre_step_adjoint=pre, grad_eps_ex=material_grads[0],
        grad_eps_ey=material_grads[1], grad_eps_ez=material_grads[2],
        grad_sigma_ex=material_grads[3], grad_sigma_ey=material_grads[4], grad_sigma_ez=material_grads[5],
        grad_copper_coefficients=tuple(copper_gradients), backend="torch_cuda_joint_material")
