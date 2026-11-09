"""Complex Yee incident line with the transverse Bloch derivatives held fixed."""
from __future__ import annotations

import math

import torch

from ...sources import evaluate_source_time


class FixedKIncidentLine:
    drive_sample_offset = 0.0

    @property
    def band_edge_transverse_wavenumber(self):
        return math.sqrt(sum(k * k for k in self.k))

    def __init__(self, solver, source, deltas, lower):
        self.axis = "xyz".index(source["injection"]["axis"])
        self.ds = float(deltas["xyz"[self.axis]])
        self.dt = float(solver.dt)
        self.source_time = source["source_time"]
        self.absorber_cells = 40
        self.padding = self.absorber_cells + 24
        self.source_index = self.absorber_cells + 6
        count = (solver.Nx, solver.Ny, solver.Nz)[self.axis] + 2 * self.padding
        self.reference_index = self.padding + lower[self.axis]
        self.direction_sign = 1 if source["direction"][self.axis] > 0 else -1
        if self.direction_sign < 0:
            self.source_index = count - 1 - self.source_index
        dtype = torch.complex128 if solver.Ex.dtype == torch.float64 else torch.complex64
        options = dict(device=solver.device, dtype=dtype)
        self.electric = torch.zeros((3, count), **options)
        self.magnetic = torch.zeros_like(self.electric)
        self.psi_e = torch.zeros_like(self.electric)
        self.psi_h = torch.zeros_like(self.electric)
        self.clock = torch.zeros((), dtype=torch.int64)
        # The fixed-k line keeps its own reference: its incident spectrum is sampled at the
        # TFSF entry plane and exported through incident_spectrum(). It does not use the
        # real-line physical-entry-plane convention (seeding, index translation and its
        # resume signature), so that hook must see no entry-plane reference here.
        self.incident_reference = None
        self.courant = solver.c * self.dt / self.ds
        self.frequencies = tuple(solver._requested_port_frequencies)
        self.omega = torch.as_tensor(self.frequencies, device=solver.device, dtype=torch.float64) * (2 * math.pi)
        self.spectrum = torch.zeros((len(self.frequencies), 3), device=solver.device, dtype=torch.complex128)
        self.k = tuple(0.0 if i == self.axis else 2 * math.sin(k * deltas[a] / 2) / deltas[a]
                       for i, (a, k) in enumerate(zip("xyz", solver.resolved_bloch_wavevector)))
        self.ce = self.dt / solver.eps0
        self.ch = self.dt / solver.mu0
        polarization = list(source["polarization"])
        polarization[self.axis] = 0.0
        norm = math.sqrt(sum(p * p for p in polarization))
        self.drive = self.electric.new_tensor([p / norm for p in polarization])
        # Stretch of the normal derivative only, with the 3D grid's resolved CPML
        # model (including the complex fixed-k CFS shift): one absorber model for
        # the grid and its incident line. An unshifted closure is not passive for
        # the below-light-line content this line carries. The 40-cell ends target
        # a 1e-10 return, because line reflections enter the incident reference.
        from ..boundary.cpml import cpml_line_profiles
        (self.inv_kappa_e, self.b_e, self.c_e), (self.inv_kappa_h, self.b_h, self.c_h) = cpml_line_profiles(
            solver, count, self.ds, self.absorber_cells, reflection=1.e-10)

    def state_tensors(self):
        return {name: getattr(self, name) for name in
                ("electric", "magnetic", "psi_e", "psi_h", "clock", "spectrum")}

    def _derivative(self, fields, component, axis, *, magnetic):
        if axis != self.axis:
            return (1j * self.k[axis]) * fields[component]
        derivative = torch.zeros_like(fields[component])
        if magnetic:
            derivative[:-1] = (fields[component, 1:] - fields[component, :-1]) / self.ds
            memory, b, c, inv_kappa = self.psi_h[component], self.b_h, self.c_h, self.inv_kappa_h
        else:
            derivative[1:-1] = (fields[component, 1:-1] - fields[component, :-2]) / self.ds
            memory, b, c, inv_kappa = self.psi_e[component], self.b_e, self.c_e, self.inv_kappa_e
        memory.mul_(b).add_(c * derivative)
        return derivative * inv_kappa + memory

    def _curl(self, fields, *, magnetic):
        return torch.stack([
            self._derivative(fields, (i + 2) % 3, (i + 1) % 3, magnetic=magnetic) -
            self._derivative(fields, (i + 1) % 3, (i + 2) % 3, magnetic=magnetic)
            for i in range(3)])

    def advance_magnetic(self):
        self.magnetic.add_(self._curl(self.electric, magnetic=True), alpha=-self.ch)

    def advance_electric(self):
        self.electric.add_(self._curl(self.magnetic, magnetic=False), alpha=self.ce)
        # The host clock is checkpointed; waveform evaluation stays outside
        # the captured field-update block, in the existing E-side TFSF hook.
        step = int(self.clock.item())
        value = evaluate_source_time(self.source_time, step * self.dt)
        self.electric[:, self.source_index].add_(self.drive, alpha=float(value) * 2 * self.courant)
        self.electric[:, 0].zero_()
        self.electric[:, -1].zero_()
        self.magnetic[:, -1].zero_()
        phase = torch.exp(1j * self.omega * (step * self.dt))
        self.spectrum.add_(phase[:, None] * self.electric[:, self.reference_index])
        self.clock.add_(1)

    def assert_lossless_samples(self, electric_terms, magnetic_terms):
        """Validate every face gather against the line's CPML ends while constructing it."""
        for term in (*electric_terms, *magnetic_terms):
            index = term["sample_index"]
            for inv_kappa, c in ((self.inv_kappa_e, self.c_e), (self.inv_kappa_h, self.c_h)):
                assert bool(inv_kappa[index] == 1) and bool(c[index] == 0), \
                    "TFSF sample overlaps the fixed-k line absorber"

    def prepare_incident_boundary(self, values=None):
        # The stepping loop stages entry-plane ghost samples on every auxiliary line.
        # The fixed-k line has no entry-plane reference (incident_reference is None),
        # so, like the real line in that case, there is nothing to stage.
        return None

    def incident_spectrum(self, solver):
        # Match the public pulse spectral scale: Fourier integral times frequency.
        reference_frequency = solver._spectral_reference_frequency
        scale = self.dt * (self.omega / (2 * math.pi) if reference_frequency is None else reference_frequency)
        # The retained accumulator labels post-update E with n*dt. Export it at
        # its physical (n+1)*dt, matching observers without changing the drive
        # or checkpointed line state. Raw-exact soft-current driving is separate.
        physical_phase = torch.exp(1j * self.omega * self.dt)
        return dict(frequencies=self.frequencies, electric=self.spectrum * (scale * physical_phase)[:, None],
                    components=("Ex", "Ey", "Ez"), axis="xyz"[self.axis],
                    position=float((solver.scene.x, solver.scene.y, solver.scene.z)[self.axis][
                        self.reference_index - self.padding]),
                    bloch_wavevector=solver.resolved_bloch_wavevector,
                    discrete_wavevector=self.k, ds=self.ds, dt=self.dt,
                    direction_sign=self.direction_sign)


def initialize_fixed_k_slab(solver, source, lower, upper, bounds, deltas):
    from .tfsf_common import slice_coeff_patch
    from .tfsf_specs import E_CURL_ATTR, H_CURL_ATTR, build_slab_tfsf_specs
    from .tfsf_state import _set_tfsf_state, _spec_positions

    line = FixedKIncidentLine(solver, source, deltas, lower)
    electric_specs, magnetic_specs = build_slab_tfsf_specs(lower, upper, axis=source["injection"]["axis"])

    def terms(specs, coefficients):
        result = []
        for spec in specs:
            positions = _spec_positions(solver, spec)
            phase = sum(k * positions[..., i] for i, k in enumerate(solver.resolved_bloch_wavevector)
                        if i != line.axis)
            # Shared electric-face specs sample the negative magnetic carrier.
            # This line stores physical H, so reverse that orientation here.
            orientation = -1 if spec["sample_kind"] == "magnetic" else 1
            profile = orientation * spec["sign"] * slice_coeff_patch(solver, spec, coefficients) * torch.exp(1j * phase)
            result.append(dict(field_name=spec["field_name"], offsets=spec["offsets"],
                               coeff_patch=profile, component="xyz".index(spec["incident_name"][1]),
                               sample_index=line.padding + spec["sample_offsets"][line.axis]))
        return result

    _set_tfsf_state(solver, provider="plane_wave_fixed_k", source=source, lower=lower, upper=upper,
                    bounds=bounds, auxiliary_grid=line,
                    electric_terms=terms(electric_specs, E_CURL_ATTR),
                    magnetic_terms=terms(magnetic_specs, H_CURL_ATTR))
