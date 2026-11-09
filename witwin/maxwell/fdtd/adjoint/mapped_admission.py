"""Admission of fixed-chart electrical material dependencies only.

This checks the declarative Scene received by the system. User code that calls
item()/detach() before constructing that Scene has the usual autograd boundary.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import torch
import numpy as np


def tensor_leaves(tensor):
    """Find leaf ancestry structurally, including dependencies with zero VJP."""
    if not tensor.requires_grad:
        return {}
    if tensor.is_leaf:
        return {id(tensor): tensor}
    pending = [tensor.grad_fn]
    visited = set()
    found = {}
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        leaf = getattr(node, 'variable', None)
        if isinstance(leaf, torch.Tensor) and leaf.requires_grad:
            found[id(leaf)] = leaf
        pending.extend(other for other, _ in node.next_functions if other is not None)
    if not found:
        raise NotImplementedError('Differentiable material value has unresolved leaf ancestry.')
    return found


def _tensors(value, path, ancestors=()):
    if isinstance(value, torch.Tensor):
        yield path, value, None, None
        return
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
        return
    if id(value) in ancestors:
        return
    ancestors = (*ancestors, id(value))
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, (tuple, list)):
        items = enumerate(value)
    elif hasattr(value, '__dict__'):
        items = vars(value).items()
    else:
        return
    for name, item in items:
        if isinstance(item, torch.Tensor):
            yield f'{path}.{name}', item, value, name
        else:
            yield from _tensors(item, f'{path}.{name}', ancestors)


@dataclass(frozen=True)
class MappedMaterialAdmission:
    material_slots: tuple[str, ...]
    leaf_ids: frozenset[int]


def assess_mapped_materials(scene, *, module, config, excitations, port_sweep, frequencies):
    """Return None for detached scenes; reject every unregistered active use."""
    from witwin.core import Box
    from ...coordinates import BendCoordinates
    from ...media import Material
    from ...copper_foil import CopperFoil
    from ...ports import LumpedPort
    from ...monitors import PointMonitor, ClosedSurfaceMonitor, PowerLossMonitor

    # Scene metadata and prepared caches are not physical inputs. Enumerate the
    # actual declarative families rather than following arbitrary Python objects.
    names = ('domain', 'grid', 'boundary', 'coordinates', 'structures', 'sources',
             'ports', 'monitors', 'material_regions', 'thin_wires', 'circuits',
             'networks', 'lumped_elements', 'subpixel', 'background')
    semantic = {name: getattr(scene, name, None) for name in names}
    semantic.update(excitations=excitations, port_sweep=port_sweep, config=config)
    active = [entry for entry in _tensors(semantic, 'scene') if entry[1].requires_grad]
    parameters = tuple(module.named_parameters()) if module is not None else ()
    buffers = tuple(module.named_buffers()) if module is not None else ()
    if not active and not any(value.requires_grad for _, value in (*parameters, *buffers)):
        return None
    if not isinstance(scene.coordinates, BendCoordinates):
        raise NotImplementedError('Electrical mapped adjoints require fixed BendCoordinates.')
    if config is None or config.run_time is None or config.run_time.dt is None or config.run_time.time_steps is None:
        raise NotImplementedError('Electrical mapped adjoints require explicit fixed dt and time_steps.')
    if frequencies is None or len(frequencies) != 1:
        raise NotImplementedError('Electrical mapped adjoints currently require one fixed frequency.')
    if config.field_dtype != torch.float32:
        raise NotImplementedError('Electrical mapped adjoints currently require FP32 primary fields.')
    if config.parallel is not None or config.shutoff or config.initial_condition is not None:
        raise NotImplementedError('Electrical mapped adjoints require fixed-duration single-device execution without initial conditions.')
    if any(str(kind) != 'pml' for pair in scene.boundary.face_kinds for kind in pair):
        raise NotImplementedError('Electrical mapped adjoints currently require all six PML faces.')
    if scene.grid.is_auto:
        raise NotImplementedError('Electrical mapped adjoints require a fixed grid; auto-grid derivatives are unqualified.')
    if scene.sources or scene.material_regions or scene.thin_wires or scene.circuits or scene.networks or scene.lumped_elements or port_sweep is not None:
        raise NotImplementedError('Electrical mapped adjoints currently require direct fixed Norton feeds without other source/device runtimes.')
    if any(type(port) is not LumpedPort or port.termination is not None for port in scene.ports):
        raise NotImplementedError('Electrical mapped adjoints currently require fixed LumpedPort Norton geometry.')
    if any(type(monitor) not in (PointMonitor, ClosedSurfaceMonitor, PowerLossMonitor) for monitor in scene.monitors):
        raise NotImplementedError('Electrical mapped adjoints currently support fixed point, closed-surface and bulk-loss observations.')
    from ...sources import CW, GaussianPulse, RickerWavelet, CustomSourceTime
    for excitation in excitations:
        waveform = excitation.source_time
        if waveform is not None and (type(waveform) not in (CW, GaussianPulse, RickerWavelet, CustomSourceTime)
                                     or getattr(waveform, '_fn', None) is not None):
            raise NotImplementedError('Electrical mapped adjoints require declarative fixed temporal waveforms; custom callable dependencies are unqualified.')
    allowed = {}
    for index, structure in enumerate(scene.structures):
        material = structure.material
        if structure.phase_screen is not None or structure.uv is not None:
            raise NotImplementedError('Electrical mapped adjoints do not qualify optical phase-screen or UV assignments.')
        if type(structure.geometry) is not Box:
            raise NotImplementedError('Electrical mapped adjoints currently require fixed Box geometry.')
        if getattr(material, 'is_pec', False):
            raise NotImplementedError('Electrical mapped adjoints do not yet qualify PEC material combinations.')
        if getattr(material, 'is_pmc', False):
            raise NotImplementedError('Electrical mapped adjoints do not yet qualify PMC material combinations.')
        if type(material) not in (Material, CopperFoil):
            raise NotImplementedError('Electrical mapped adjoints require scalar Material or CopperFoil declarations; ADE and other constitutive classes are unqualified.')
        if any(getattr(material, name, False) for name in ('is_anisotropic', 'is_dispersive', 'is_nonlinear', 'is_gyromagnetic', 'is_medium2d', 'is_lossy_metal', 'is_surface_impedance')):
            raise NotImplementedError('Electrical mapped adjoints require scalar nondispersive constitutive laws.')
        if getattr(material, 'modulation', None) is not None or getattr(material, 'breakdown', None) is not None:
            raise NotImplementedError('Electrical mapped adjoints require static constitutive declarations.')
        if isinstance(material, CopperFoil) and material.mode_count is None:
            raise NotImplementedError('Electrical mapped copper adjoints require fixed mode_count.')
        if not structure.enabled:
            continue
        fields = ('thickness', 'conductivity') if isinstance(material, CopperFoil) else ('eps_r', 'sigma_e')
        for name in fields:
            allowed[id(material), name] = f'structures[{index}].material.{name}'
    covered = {}
    slots = []
    for path, tensor, owner, name in active:
        slot = allowed.get((id(owner), name))
        if slot is None:
            raise NotImplementedError('Unregistered differentiable mapped input: ' + path)
        if tensor.ndim != 0 or not tensor.is_floating_point():
            raise NotImplementedError('Registered electrical material values must be real floating scalars.')
        covered.update(tensor_leaves(tensor))
        slots.append(slot)
    if module is not None:
        frontier = {id(value) for _, value in parameters if value.requires_grad}
        if not covered.keys() <= frontier:
            raise NotImplementedError('Electrical mapped module dependencies must originate from registered SceneModule parameters; independent buffer or plain-Tensor leaves are unqualified.')
    storage = {}
    for identifier, leaf in covered.items():
        if not leaf.is_floating_point():
            raise NotImplementedError('Complex material parameter leaves are unqualified.')
        key = str(leaf.device), leaf.untyped_storage().data_ptr()
        if key in storage and storage[key] != identifier:
            raise NotImplementedError('Distinct trainable material leaves share storage; input deduplication is unsafe.')
        storage[key] = identifier
    for name, parameter in parameters:
        if parameter.requires_grad and id(parameter) not in covered:
            raise NotImplementedError('Unaccounted trainable module parameter: ' + name)
    for name, buffer in buffers:
        if buffer.requires_grad and not set(tensor_leaves(buffer)) <= covered.keys():
            raise NotImplementedError('Unaccounted trainable module buffer: ' + name)
    return MappedMaterialAdmission(tuple(slots), frozenset(covered))


def _physical_scene_inputs(scene):
    """Exclude allocator identities, provenance and caches, not physical values."""
    names = ('domain', 'grid', 'boundary', 'coordinates', 'sources', 'ports', 'monitors',
             'material_regions', 'thin_wires', 'circuits', 'networks', 'lumped_elements',
             'subpixel', 'background', 'symmetry', 'electrostatic_terminals', 'charge_densities', 'device')
    inputs = {name: getattr(scene, name, None) for name in names}
    inputs['structures'] = tuple({
        name: getattr(structure, name, None)
        for name in ('geometry', 'material', 'name', 'priority', 'enabled', 'phase_screen', 'uv', 'face_uv')
    } for structure in scene.structures)
    return inputs


def _freeze_physical_inputs(value, *, clone_tensors, path, grad_flags):
    """Capture values without keeping a rematerialized autograd graph alive."""
    if isinstance(value, torch.Tensor):
        grad_flags[path] = value.requires_grad
        data = value.detach()
        return ('tensor', data.clone() if clone_tensors else data)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            entries = tuple(_freeze_physical_inputs(item, clone_tensors=clone_tensors,
                path=path + '[' + str(index) + ']', grad_flags=grad_flags)
                for index, item in enumerate(value.flat))
            return ('object_array', value.shape, entries)
        return ('array', value.copy() if clone_tensors else value)
    if isinstance(value, Mapping):
        return ('mapping', {name: _freeze_physical_inputs(item, clone_tensors=clone_tensors,
            path=path + '.' + str(name), grad_flags=grad_flags) for name, item in value.items()})
    if isinstance(value, (tuple, list)):
        return ('sequence', type(value), tuple(_freeze_physical_inputs(item, clone_tensors=clone_tensors,
            path=path + '[' + str(index) + ']', grad_flags=grad_flags) for index, item in enumerate(value)))
    if callable(value):
        return ('callable', value)
    if hasattr(value, '__dict__'):
        # PhysicalMaterial ids are allocated anew on each declarative build.
        # They identify owners and are not constitutive values.
        members = {name: item for name, item in vars(value).items() if name != 'material_id'}
        return ('object', type(value), _freeze_physical_inputs(members,
            clone_tensors=clone_tensors, path=path, grad_flags=grad_flags))
    return ('scalar', type(value), value)


def capture_grad_mode_scene(scene):
    """Own the numerical declarations before a second callback can mutate them."""
    flags = {}
    values = _freeze_physical_inputs(_physical_scene_inputs(scene), clone_tensors=True,
        path='scene', grad_flags=flags)
    return values, flags


def validate_grad_mode_scene(numerical_scene, dependency_scene, *, snapshot):
    """Require the exact same physical declarations across internal grad mode.

    Value snapshots also detect callbacks that mutate a cached Scene in place.
    CUDA predicates reduce once per device in preparation, never per field step.
    """
    pending = {}
    def mismatch(path):
        raise NotImplementedError('Mapped SceneModule changes physical declarations with grad mode: ' + path)
    def compare(left, right, path):
        if left[0] != right[0]:
            mismatch(path)
        kind = left[0]
        if kind == 'tensor':
            a, b = left[1], right[1]
            if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device or a.layout != b.layout:
                mismatch(path)
            pending.setdefault(a.device, []).append(torch.eq(a, b).all())
        elif kind == 'array':
            a, b = left[1], right[1]
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
                mismatch(path)
        elif kind == 'mapping':
            a, b = left[1], right[1]
            if a.keys() != b.keys():
                mismatch(path)
            for name in a:
                compare(a[name], b[name], path + '.' + str(name))
        elif kind in ('sequence', 'object_array'):
            if left[1] != right[1] or len(left[2]) != len(right[2]):
                mismatch(path)
            for index, (a, b) in enumerate(zip(left[2], right[2])):
                compare(a, b, path + '[' + str(index) + ']')
        elif kind == 'object':
            if left[1] is not right[1]:
                mismatch(path)
            compare(left[2], right[2], path)
        elif kind == 'callable':
            if left[1] is not right[1]:
                mismatch(path)
        elif left[1] is not right[1] or left[2] != right[2]:
            mismatch(path)
    before, before_flags = snapshot
    after_flags, dependency_flags = {}, {}
    numerical_values = _freeze_physical_inputs(_physical_scene_inputs(numerical_scene),
        clone_tensors=False, path='scene', grad_flags=after_flags)
    dependency_values = _freeze_physical_inputs(_physical_scene_inputs(dependency_scene),
        clone_tensors=False, path='scene', grad_flags=dependency_flags)
    if before_flags != after_flags:
        mismatch('callback mutated numerical Tensor dependency flags')
    compare(before, numerical_values, 'numerical scene after dependency callback')
    compare(before, dependency_values, 'dependency scene')
    for flags in pending.values():
        if not bool(torch.stack(flags).all()):
            mismatch('tensor values')
