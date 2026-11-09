"""Private genuine-native normal-plane split/CN integration preparation.

Host code compiles real Scene grid/source metadata only. Actual field execution
requires a real CUDA PreparedSimulation and uses its native background kernels.
The adapter exposes exterior observations and separate side-aware cut records;
it never publishes discarded native interior samples as a physical field.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import compile_fdtd_sources
from witwin.maxwell.scene import PreparedScene
from witwin.maxwell.copper_foil import EPSILON_0, MU_0
from scripts.dev.fdtd.split_copper_plane import (
    compile_split_plane, compile_local_cn_support, compiled_local_cn_algebra,
)


@dataclass(frozen=True)
class NativePlaneLayout:
    geometry: object
    support: dict
    electric_name: str
    magnetic_name: str
    magnetic_sign: float
    native_node_bounds: tuple[int, int]
    native_electric: torch.Tensor
    local_electric: torch.Tensor
    native_magnetic: torch.Tensor
    local_magnetic: torch.Tensor
    discarded_native_magnetic: torch.Tensor
    outside_magnetic: tuple[int, int]
    line_shape: tuple[int, int]
    physical_line_area: torch.Tensor
    dependency_audit: dict
    source_metadata: tuple

    def assert_exterior_query(self, x):
        lo, hi = self.support['bounds']
        h = float((self.geometry.original_nodes[1:]-self.geometry.original_nodes[:-1]).max())
        if lo-h <= float(x) <= hi+h:
            raise NotImplementedError('Ordinary native queries may not consume the replaced patch or discarded samples; use side-aware cut records.')


def native_source_identity():
    from scripts.dev.fdtd.split_copper_plane_validation import source_identity
    result = source_identity()
    root = Path(__file__).resolve().parents[3]
    for relative in ('scripts/dev/fdtd/native_split_copper_plane.py',
                     'scripts/dev/fdtd/native_split_copper_validation.py',
                     'tests/materials/surface_impedance/test_native_split_copper_plane.py',
                     'native-split-host-control.py'):
        path = root/relative
        if path.exists():
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def compile_outside_dependency_audit(nodes, bounds, discarded):
    """Exact native 1D derivative dependencies for a full transverse slab.

    This proves stencil ownership for the restricted constant-transverse mode,
    not for arbitrary transverse derivatives or curved surfaces.
    """
    nodes = torch.as_tensor(nodes, dtype=torch.float64)
    indices = [torch.nonzero(nodes == float(value)).flatten() for value in bounds]
    if any(len(index) != 1 for index in indices):
        raise ValueError('Patch boundaries must be exact native original nodes.')
    low, high = (int(index[0]) for index in indices)
    C = torch.zeros((len(nodes)-1, len(nodes)), dtype=torch.float64)
    rows = torch.arange(len(nodes)-1)
    C[rows, rows], C[rows, rows+1] = -1, 1
    outside_e = torch.cat((torch.arange(low), torch.arange(high+1, len(nodes))))
    outside_h = torch.cat((torch.arange(low), torch.arange(high, len(nodes)-1)))
    interior_e = torch.arange(low+1, high)
    inside_h = torch.arange(low, high)
    discarded = torch.as_tensor(discarded, dtype=torch.int64)
    leaked = C[discarded][:, outside_e] if len(discarded) else C.new_zeros((0, len(outside_e)))
    if bool(leaked.any()) or bool(C[inside_h][:, outside_e].any()) or bool(C[outside_h][:, interior_e].any()):
        raise ValueError('The proposed discarded field has a real exterior stencil dependency.')
    return dict(native_incidence=C.tolist(), outside_electric_nodes=outside_e.tolist(),
        outside_magnetic_intervals=outside_h.tolist(), replaced_electric_nodes=list(range(low, high+1)),
        replaced_magnetic_intervals=inside_h.tolist(), discarded_native_magnetic=discarded.tolist(),
        discarded_to_exterior_electric_nonzeros=int(torch.count_nonzero(leaked)),
        transverse_assumption='constant fields on both periodic transverse axes; separately validated at runtime',
        provisional_samples_are_physical=False)


def compile_native_plane_layout(scene, *, cut, polarization, implicit_bounds):
    """Compile actual Scene/PreparedScene metadata without a CPU field solver."""
    if not isinstance(scene, mw.Scene):
        raise TypeError('Native plane compilation requires a real Scene.')
    if polarization not in ('Ey', 'Ez'):
        raise ValueError('Normal incidence requires Ey or Ez tangential polarization.')
    if scene.coordinates is not None or scene.background is not None:
        raise NotImplementedError('The first native plane diagnostic requires Cartesian vacuum.')
    for name in ('structures', 'material_regions', 'ports', 'thin_wires', 'networks', 'lumped_elements', 'circuits'):
        if getattr(scene, name, ()):
            raise NotImplementedError(f'The first uniform native plane diagnostic rejects {name}; no geometry or feed is removed.')
    if scene.boundary.axis_kind('x') != 'pml' or any(scene.boundary.axis_kind(axis) not in ('periodic', 'bloch') for axis in 'yz'):
        raise NotImplementedError('The first native plane requires x PML and two periodic or zero-Bloch transverse axes.')
    if scene.boundary.bloch_wavevector != (0., 0., 0.):
        raise NotImplementedError('A transverse Bloch phase would invalidate independent constant-transverse lines.')
    if len(scene.sources) != 1 or not isinstance(scene.sources[0], mw.PlaneWave):
        raise NotImplementedError('The first native diagnostic requires one actual normal PlaneWave source.')
    source = scene.sources[0]
    if not isinstance(source.source_time, mw.CW):
        raise NotImplementedError('The first native plane diagnostic requires an explicit CW carrier.')
    if float(source.source_time.amplitude) == 0:
        raise ValueError('A native scattering qualification requires a nonzero actual incident carrier.')
    expected = (0., 1., 0.) if polarization == 'Ey' else (0., 0., 1.)
    if tuple(source.direction) != (1., 0., 0.) or tuple(source.polarization) != expected:
        raise NotImplementedError('Oblique, reverse, or mixed-polarization sources need another native operator qualification.')
    injection = source.injection
    if not isinstance(injection, mw.TFSF) or injection.mode != 'slab' or injection.axis != 'x':
        raise NotImplementedError('The actual source must use an x-normal slab TFSF.')
    if torch.device(scene.device).type == 'cpu':
        prepared_scene = scene if isinstance(scene, PreparedScene) else PreparedScene(scene, lazy_meshgrid=True)
    else:
        if not isinstance(scene, PreparedScene):
            raise TypeError('CUDA layout compilation must consume the actually prepared native scene.')
        prepared_scene = scene
    geometry = compile_split_plane(prepared_scene.x_nodes64, cut)
    support = compile_local_cn_support(geometry, implicit_bounds)
    low, high = support['bounds']
    h = float((geometry.original_nodes[1:]-geometry.original_nodes[:-1]).max())
    if not scene.domain.bounds[0][0]+h < low < high < scene.domain.bounds[0][1]-h:
        raise NotImplementedError('The actual patch must remain outside PML and its interface stencil.')
    if not injection.axis_bounds[0]+h < low < high < injection.axis_bounds[1]-h:
        raise NotImplementedError('The actual patch overlaps a TFSF correction face or its stencil.')
    nodes = geometry.original_nodes
    ni, nj = (int(torch.nonzero(nodes == value).flatten()[0]) for value in (low, high))
    local_positions = geometry.electric_positions[support['electric']]
    native_e = torch.arange(ni, nj+1)
    local_e = torch.tensor([int(torch.nonzero(local_positions == nodes[index]).flatten()[0]) for index in native_e])
    native_h, local_h, discarded = [], [], []
    for index in range(ni, nj):
        matches = torch.nonzero((geometry.electric_positions[geometry.edge_low[support['magnetic']]] == nodes[index])
            & (geometry.electric_positions[geometry.edge_high[support['magnetic']]] == nodes[index+1])).flatten()
        if len(matches) == 1:
            native_h.append(index); local_h.append(int(matches[0]))
        else:
            discarded.append(index)
    if polarization == 'Ey':
        area = torch.as_tensor(prepared_scene.dy_primal64)[:, None]*torch.as_tensor(prepared_scene.dz_dual64)[None, :]
        area[:, (0, -1)] *= .5
        magnetic_name, sign = 'Hz', 1.
    else:
        area = torch.as_tensor(prepared_scene.dy_dual64)[:, None]*torch.as_tensor(prepared_scene.dz_primal64)[None, :]
        area[(0, -1), :] *= .5
        magnetic_name, sign = 'Hy', -1.
    audit = compile_outside_dependency_audit(nodes, (low, high), discarded)
    layout = NativePlaneLayout(geometry, support, polarization, magnetic_name, sign, (ni, nj), native_e, local_e,
        torch.tensor(native_h), torch.tensor(local_h), torch.tensor(discarded, dtype=torch.int64),
        (ni-1, nj), tuple(area.shape), area, audit, tuple(compile_fdtd_sources(prepared_scene, default_frequency=source.source_time.frequency)))
    for monitor in scene.monitors:
        if not isinstance(monitor, mw.PointMonitor):
            raise NotImplementedError('Only ordinary exterior PointMonitor records are qualified in the first native integration.')
        layout.assert_exterior_query(monitor.position[0])
    return layout


def require_native_plane_execution(simulation, material):
    """Host-only validation of a real declarative execution request."""
    if not isinstance(simulation, mw.Simulation):
        raise TypeError('A native plane execution request requires a real Simulation.')
    if simulation.has_trainable_parameters or simulation.config.parallel is not None or simulation.config.cuda_graph:
        raise NotImplementedError('Private split-plane integration requires detached single-device eager execution.')
    if simulation.config.full_field_dft or len(simulation.frequencies) != 1:
        raise NotImplementedError('Full-field/native-interior output and multifrequency execution are not qualified.')
    if simulation.config.shutoff:
        raise NotImplementedError('Cut-state-aware automatic shutoff is not implemented.')
    if simulation.config.spectral_sampler.normalize_source:
        raise NotImplementedError('Native source-normalized observations require joint cut-spectrum normalization.')
    if any(torch.is_tensor(value) and value.requires_grad for value in (material.thickness, material.conductivity)):
        raise NotImplementedError('The private native cut stage does not yet support material gradients.')


def require_native_solve_options(options):
    """Reject unsupported lifecycle paths before any real native kernel runs."""
    if options.get('resume_from') is not None or options.get('stop_step') is not None:
        raise NotImplementedError('Native cut states are not registered for checkpoint/resume.')
    if options.get('use_cuda_graph') or options.get('full_field_dft') or options.get('normalize_source'):
        raise NotImplementedError('The native cut diagnostic requires eager exterior-only unnormalized observations.')


class NativeSplitPlane:
    """Private eager coupling to a real prepared CUDA FDTD background.

    This class is not a solver replacement and does not return a public Result
    carrying invalid native interior fields. Graph, AD, resume, and distributed
    execution are deliberately rejected until their actual state is integrated.
    """
    def __init__(self, prepared, *, cut, polarization, material, implicit_bounds):
        from witwin.maxwell.simulation import PreparedSimulation
        from witwin.maxwell.fdtd.solver import FDTD
        if not isinstance(prepared, PreparedSimulation) or not isinstance(prepared.solver, FDTD):
            raise TypeError('NativeSplitPlane requires an actual Scene -> Simulation.prepare() FDTD object.')
        solver, simulation = prepared.solver, prepared.simulation
        if torch.device(solver.device).type != 'cuda':
            raise ValueError('Actual native background field advancement requires CUDA.')
        require_native_plane_execution(simulation, material)
        self.prepared, self.solver, self.material = prepared, solver, material
        self.layout = compile_native_plane_layout(solver.scene, cut=cut, polarization=polarization, implicit_bounds=implicit_bounds)
        self.dt = float(solver.dt)
        self.frequency = float(simulation.frequency)
        self.device = torch.device(solver.device)
        self.complex_fields = bool(solver.complex_fields_enabled)
        self.weight_audit, eps, mu = self._audit_actual_weights()
        masses_e = self.layout.geometry.electric_lengths*eps
        masses_h = self.layout.geometry.magnetic_lengths*mu
        compiled = compiled_local_cn_algebra(self.layout.geometry, material, self.dt, implicit_bounds,
            electric_mass=masses_e, magnetic_mass=masses_h, impedance=math.sqrt(mu/eps))
        self.p = {key: value.to(self.device) if torch.is_tensor(value) else value
                  for key, value in compiled.items() if key != 'material'}
        self.a = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in compiled['material'].items()}
        self.maps = {name: getattr(self.layout, name).to(self.device) for name in
            ('native_electric', 'local_electric', 'native_magnetic', 'local_magnetic', 'discarded_native_magnetic')}
        self.maps['outside_magnetic'] = torch.tensor(self.layout.outside_magnetic, device=self.device)
        self.area = self.layout.physical_line_area.flatten().to(self.device)
        reference_h = abs(float(solver.scene.sources[0].source_time.amplitude))/compiled['eta']
        self.uniform_reference_h = torch.tensor(reference_h, device=self.device, dtype=torch.float64)
        lines = self.area.numel()
        self.electric = torch.zeros((len(compiled['electric']), lines), device=self.device, dtype=torch.complex128)
        self.magnetic = torch.zeros((len(compiled['magnetic']), lines), device=self.device, dtype=torch.complex128)
        self.state = torch.zeros((len(self.a['rates']), lines), device=self.device, dtype=torch.complex128)
        self.cumulative_heat = torch.zeros((), device=self.device, dtype=torch.float64)
        self.uniformity_error = torch.zeros_like(self.cumulative_heat)
        self.native_block_calls = 0
        self.last_force = None
        self.total_steps = simulation._resolve_fdtd_time_steps(solver)
        plan = solver._time_plan
        start = int(plan['spectral_start_step'])
        weights = [solver._compute_window_weight(n, start_step=start, end_step=self.total_steps,
                    window_type=str(simulation.config.spectral_sampler.window)) for n in range(self.total_steps)]
        time = torch.arange(self.total_steps, device=self.device, dtype=torch.float64)*self.dt
        w = torch.tensor(weights, device=self.device, dtype=torch.float64)
        self.spectral_weight = w*torch.exp(2j*math.pi*self.frequency*time)
        self.spectral_normalization = w.sum()/float(plan['spectral_phasor_factor'])
        self.electric_spectrum = torch.zeros_like(self.electric)
        self.magnetic_spectrum = torch.zeros_like(self.magnetic)
        self.state_spectrum = torch.zeros_like(self.state)
        self._used = False
        if any(bool(getattr(solver, name).any()) for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')):
            raise NotImplementedError('Initial-field import requires a separate cut-state initialization contract.')

    def _audit_actual_weights(self):
        l, s = self.layout, self.solver
        ni, nj = l.native_node_bounds
        audit = {}
        effective = []
        for field, node_slice, reference, axis_width in (
            (l.electric_name, slice(ni, nj+1), EPSILON_0, s.scene.dx_dual64[ni:nj+1]),
            (l.magnetic_name, slice(ni, nj), MU_0, s.scene.dx_primal64[ni:nj])):
            lower = field.lower()
            gain = getattr(s, 'c'+lower+'_curl')[node_slice].double()
            decay = getattr(s, 'c'+lower+'_decay')[node_slice]
            stored = getattr(s, ('eps_' if field.startswith('E') else 'mu_')+field)[node_slice].double()
            sigma = getattr(s, ('sigma_e_' if field.startswith('E') else 'sigma_m_')+field)[node_slice]
            if not bool((gain > 0).all()) or not bool((decay == 1).all()) or bool(sigma.any()):
                raise NotImplementedError('Actual patch coefficients overlap a suppressed, conductive, or nonvacuum update.')
            if not bool(torch.allclose(stored, torch.full_like(stored, reference), rtol=2e-7, atol=0)):
                raise NotImplementedError('The actual patch material is not the required vacuum.')
            if not bool((gain == gain.flatten()[0]).all()):
                raise NotImplementedError('The native coefficient is not transversely and normally uniform in this patch.')
            permittivity = self.dt/float(gain.flatten()[0].cpu())
            effective.append(permittivity)
            inv = getattr(s, 'inv_dx_e' if field.startswith('E') else 'inv_dx_h')[node_slice].double()
            expected = 1/torch.as_tensor(np.array(axis_width), device=self.device)
            if not bool(torch.allclose(inv, expected, rtol=2e-7, atol=0)):
                raise ValueError('Actual native x curl widths differ from the original physical grid.')
            audit[field] = dict(curl_gain=gain.cpu().tolist(), decay=decay.cpu().tolist(),
                stored_material=stored.cpu().tolist(), inverse_x_spacing=inv.cpu().tolist(),
                effective_mass_coefficient_from_actual_gain=permittivity,
                physical_transverse_area=l.physical_line_area.tolist())
        return audit, effective[0], effective[1]

    def _read(self, name, indices=None):
        real = getattr(self.solver, name)
        imaginary = getattr(self.solver, name+'_imag', None)
        if indices is not None:
            real = real[indices]
            imaginary = imaginary[indices] if imaginary is not None else None
        return torch.complex(real.double(), imaginary.double() if imaginary is not None else torch.zeros_like(real, dtype=torch.float64))

    def _scatter(self, name, indices, value):
        target = getattr(self.solver, name)
        indices = indices.to(self.device)
        target[indices] = value.real.reshape((len(indices),)+self.layout.line_shape).to(target.dtype)
        if self.complex_fields:
            imag = getattr(self.solver, name+'_imag')
            imag[indices] = value.imag.reshape((len(indices),)+self.layout.line_shape).to(imag.dtype)

    def advance_after_native_block(self):
        """Use actual native exterior H; never advance the background twice."""
        l, p, a = self.layout, self.p, self.a
        outside = (self._read(l.magnetic_name, self.maps['outside_magnetic'])*l.magnetic_sign).reshape(2, -1)
        # Normalize against the declared incident carrier as well as the
        # instantaneous field, so a real-CW zero crossing is not a false error.
        scale = torch.maximum(outside.abs().max(), self.uniform_reference_h)
        self.uniformity_error.copy_(torch.maximum(self.uniformity_error, (outside-outside[:, :1]).abs().max()/scale))
        force = torch.zeros_like(self.electric)
        force[0], force[-1] = outside[0], -outside[1]
        self.last_force = force.clone()
        history = a['B'].T.to(self.state.dtype)@(a['inverse_denominator'][:, None]*self.state)
        rhs_e = p['electric_diagonal'][:, None]*self.electric+p['eta']*force
        rhs_e -= p['eta']*p['trace_selector'].T.to(self.state.dtype)@history
        rhs_h = p['magnetic_diagonal'][:, None]*p['eta']*self.magnetic
        midpoint = torch.linalg.lu_solve(p['factor'], p['pivots'], torch.cat((rhs_e, rhs_h)))
        ne = len(self.electric)
        trace = p['trace_selector'].to(self.state.dtype)@midpoint[:ne]
        new_state = a['decay'][:, None]*self.state+a['drive'].to(self.state.dtype)@trace
        zmid = (self.state+new_state)/2
        heat = (a['rates'][:, None]*zmid.abs().square()).sum(0)
        heat += (trace.conj()*(a['D'].to(trace.dtype)@trace)).sum(0).real
        self.cumulative_heat.add_(self.dt*(heat*self.area).sum())
        self.electric.copy_(2*midpoint[:ne]-self.electric)
        self.magnetic.copy_(2*midpoint[ne:]/p['eta']-self.magnetic)
        self.state.copy_(new_state)
        self._scatter(l.electric_name, self.maps['native_electric'], self.electric[self.maps['local_electric']])
        self._scatter(l.magnetic_name, self.maps['native_magnetic'], l.magnetic_sign*self.magnetic[self.maps['local_magnetic']])
        if len(l.discarded_native_magnetic):
            # Bounded nonphysical workspace, excluded by dependency proof and
            # every exposed observer. It is neither one of the physical sides.
            zeros = self.electric.new_zeros((len(l.discarded_native_magnetic), self.area.numel()))
            self._scatter(l.magnetic_name, self.maps['discarded_native_magnetic'], zeros)
        n = self.native_block_calls
        self.electric_spectrum.add_(self.electric*self.spectral_weight[n])
        self.magnetic_spectrum.add_(self.magnetic*self.spectral_weight[n])
        self.state_spectrum.add_(self.state*self.spectral_weight[n])
        self.native_block_calls += 1

    @contextmanager
    def native_context(self):
        from witwin.maxwell.fdtd.runtime import stepping
        original = stepping._field_update_block
        original_solve = self.solver.solve
        solve_signature = inspect.signature(original_solve)
        def block(solver, time_value):
            if solver is not self.solver:
                return original(solver, time_value)
            original(solver, time_value)
            self.advance_after_native_block()
        def solve(*args, **kwargs):
            arguments = solve_signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            require_native_solve_options(arguments.arguments)
            return original_solve(*args, **kwargs)
        with patch.object(stepping, '_field_update_block', block), patch.object(self.solver, 'solve', solve):
            yield

    def run(self):
        if self._used:
            raise RuntimeError('A native cut-plane diagnostic runs once; create a fresh prepared scene.')
        self._used = True
        with self.native_context():
            result = self.prepared.run()
        if self.native_block_calls != self.total_steps or float(self.uniformity_error.cpu()) > 2e-6:
            raise RuntimeError('Actual native execution violated the declared step count or transverse-uniform scope.')
        from witwin.maxwell.fdtd.copper_foil import copper_spectral_power
        electric = self.electric_spectrum/self.spectral_normalization
        magnetic = self.magnetic_spectrum/self.spectral_normalization
        states = self.state_spectrum/self.spectral_normalization
        trace = self.p['trace_selector'].to(electric.dtype)@electric
        current = self.a['D'].to(trace.dtype)@trace+self.a['B'].T.to(trace.dtype)@states
        heat = copper_spectral_power(trace[None], torch.tensor([self.frequency], device=self.device, dtype=torch.float64),
            self.dt, self.a['B'], self.a['rates'], self.a['D'], self.area)
        # Do not return the internal Result: its native last-step fields contain
        # deliberate invalid interior workspace. Only exterior monitors escape.
        monitors = {monitor.name: result.monitor(monitor.name) for monitor in self.prepared.simulation.scene.monitors}
        return dict(exterior_monitors=monitors, local_electric_phasors=electric.detach().cpu(),
            local_magnetic_phasors=magnetic.detach().cpu(), local_skin_state_phasors=states.detach().cpu(),
            local_surface_electric_phasors=trace.detach().cpu(), local_inward_current_phasors=current.detach().cpu(),
            actual_shared_film_spectral_heat_W=float(heat[0].cpu()), cumulative_irreversible_heat=float(self.cumulative_heat.cpu()),
            actual_native_dt_s=self.dt, actual_native_steps=self.native_block_calls,
            transverse_uniformity_relative_error=float(self.uniformity_error.cpu()),
            actual_native_weight_audit=self.weight_audit, dependency_audit=self.layout.dependency_audit,
            local_geometry=dict(electric_x_m=self.layout.geometry.electric_positions[self.layout.support['electric']].tolist(),
                magnetic_x_m=self.layout.geometry.magnetic_positions[self.layout.support['magnetic']].tolist(),
                magnetic_intervals_m=self.layout.geometry.magnetic_lengths[self.layout.support['magnetic']].tolist(),
                native_original_magnetic_samples_invalid=self.layout.discarded_native_magnetic.tolist()),
            observation_contract='Exterior ordinary native PointMonitor data plus explicit two-sided cut records only; no ordinary interior fields.',
            local_magnetic_time_offset_relative_export_E_s=0., exterior_native_H_time_offset_s=-self.dt/2,
            production_3d_curved_copper_qualified=False)
