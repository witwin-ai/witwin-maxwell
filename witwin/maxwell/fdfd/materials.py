"""Material components for the FDFD exp(-i omega t) convention."""
from __future__ import annotations
import math
import torch
from ..compiler.materials import evaluate_material_components
from ..constants import EPSILON_0, MU_0
from ..compiler.materials import compile_materials


def compile_fdfd_materials(scene, frequency, *, edges=False):
    """Preserve native material graphs while folding both conductivities once.

    Shared pole susceptibilities already use exp(-i omega t). The shared static
    electric-conductivity sampler uses the opposite loss sign, and does not
    fold magnetic conductivity into mu. Correct those terms at this boundary;
    conjugating entire components would incorrectly reverse the pole loss.
    """
    model = compile_materials(scene)
    return _evaluate_edge_components(model, frequency) if edges else _evaluate_components(model, frequency)


def _evaluate_components(model, frequency):
    epsilon, permeability = evaluate_material_components(model, frequency)
    omega = 2*math.pi*frequency
    # The shared sampler subtracts 1j*sigma/(omega eps0) only for nonzero
    # conductivity, so the +i correction is 2j over it and 1j without it. A
    # trainable conductivity is folded even at zero: skipping it there would
    # drop its (nonzero) derivative from the material graph.
    shared_loss = any(bool((value != 0).any()) for value in model['sigma_e_components'].values())
    if shared_loss or any(value.requires_grad for value in model['sigma_e_components'].values()):
        fold = 2j if shared_loss else 1j
        epsilon = {axis:value.to(torch.complex64)+fold*model['sigma_e_components'][axis]/(omega*EPSILON_0)
                   for axis,value in epsilon.items()}
    if any(value.requires_grad or bool((value != 0).any()) for value in model['sigma_m_components'].values()):
        permeability = {axis:value.to(torch.complex64)+1j*model['sigma_m_components'][axis]/(omega*MU_0)
                        for axis,value in permeability.items()}
    return epsilon, permeability


def _evaluate_edge_components(model, frequency):
    """Evaluate the shared Yee samples with the FDFD conductivity convention.

    Pole weights follow the same compiled scalar/graded policy as FDTD ADE.
    """
    from ..fdtd.runtime.materials import (
        average_node_to_component, average_node_to_magnetic_component,
        pole_component_weight,
    )

    edge = model['edge_components']
    averages = (average_node_to_component, average_node_to_magnetic_component)
    if edge is None and not model.get('pole_edge_background'):
        nodes = _evaluate_components(model, frequency)
        return tuple({axis: average(None, values[axis], prefix + axis)
                      for axis in 'xyz'}
                     for values, average, prefix in zip(nodes, averages, ('E', 'H')))

    sampled = dict(model)
    for key, channel, prefix in (('eps_components', 'eps', 'E'),
                                 ('mu_components', 'mu', 'H'),
                                 ('sigma_e_components', 'sigma_e', 'E'),
                                 ('sigma_m_components', 'sigma_m', 'H')):
        average = averages[0 if prefix == 'E' else 1]
        sampled[key] = {axis: (edge[channel][prefix + axis] if edge is not None else
                               average(None, model[key][axis], prefix + axis)) for axis in 'xyz'}
        for axis in 'xyz':
            name = prefix + axis
            if channel in ('eps', 'mu') and name in model.get('pole_edge_background', {}):
                sampled[key][axis] = model['pole_edge_background'][name]
    for average, prefix, pole_prefix in zip(averages, ('E', 'H'), ('', 'mu_')):
        for kind in ('debye', 'drude', 'lorentz'):
            key = pole_prefix + kind + '_poles'
            sampled[key] = [dict(entry, axes=(axis,),
                weight=pole_component_weight(entry, prefix + axis, average))
                for entry in model[key] for axis in entry.get('axes') or 'xyz']
    return _evaluate_components(sampled, frequency)
