"""Private 1D split-cell stage for a true off-grid finite-copper interface.

This diagnostic is not a SurfaceMesh compiler or public solver. CPU work is
limited to geometry, local algebra, and an independent frequency-domain oracle.
The executable field advance below requires PyTorch CUDA tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from witwin.maxwell.compiler.copper_foil import compile_copper_response
from witwin.maxwell.copper_foil import EPSILON_0, MU_0


@dataclass(frozen=True)
class SplitPlaneGeometry:
    electric_positions: torch.Tensor
    magnetic_positions: torch.Tensor
    magnetic_lengths: torch.Tensor
    electric_lengths: torch.Tensor
    edge_low: torch.Tensor
    edge_high: torch.Tensor
    traces: torch.Tensor
    cut: float
    aligned: bool
    original_nodes: torch.Tensor

    @property
    def electric_mass(self):
        return EPSILON_0 * self.electric_lengths

    @property
    def magnetic_mass(self):
        return MU_0 * self.magnetic_lengths

    def incidence(self):
        """Compile the same incidence used by both native field reactions."""
        count = len(self.magnetic_lengths)
        result = torch.zeros((count, len(self.electric_positions)),
                             dtype=torch.float64, device=self.edge_low.device)
        rows = torch.arange(count, device=self.edge_low.device)
        result[rows, self.edge_low] = -1
        result[rows, self.edge_high] = 1
        return result

    def stable_dt(self, courant=.4):
        if not math.isfinite(courant) or not 0 < courant < 1:
            raise ValueError("The split-cell Courant fraction must lie in (0, 1).")
        return courant * float(self.magnetic_lengths.min()) * math.sqrt(MU_0 * EPSILON_0)

    def bulk_stable_dt(self, courant=.9):
        """Original mesh CFL, used only with a compatible implicit cut patch."""
        if not math.isfinite(courant) or not 0 < courant < 1:
            raise ValueError("The bulk Courant fraction must lie in (0, 1).")
        width = float((self.original_nodes[1:] - self.original_nodes[:-1]).min())
        return courant * width * math.sqrt(MU_0 * EPSILON_0)


def compile_split_plane(nodes, cut):
    """Split the actual bulk interval and duplicate the physical interface E.

    A cut exactly equal to a node uses that node twice. No zero interval, snap,
    minimum cut size, clipped mass, or epsilon regularization is introduced.
    Interior offsets preserve their actual positions even arbitrarily near a
    node. A genuinely small interval consequently imposes a small explicit dt.
    """
    nodes = torch.as_tensor(nodes, dtype=torch.float64, device="cpu")
    cut = float(cut)
    if nodes.ndim != 1 or len(nodes) < 3 or not bool(torch.isfinite(nodes).all()):
        raise ValueError("Split-plane nodes require a finite 1D grid of at least three nodes.")
    if not bool(((nodes[1:] - nodes[:-1]) > 0).all()):
        raise ValueError("Split-plane nodes must increase strictly.")
    if not math.isfinite(cut) or not float(nodes[0]) < cut < float(nodes[-1]):
        raise ValueError("A finite split plane must lie strictly inside the grid.")
    aligned = bool((nodes == cut).any())
    left = torch.cat((nodes[nodes < cut], nodes.new_tensor([cut])))
    right = torch.cat((nodes.new_tensor([cut]), nodes[nodes > cut]))
    position = torch.cat((left, right))
    low_left = torch.arange(len(left) - 1)
    low_right = torch.arange(len(right) - 1) + len(left)
    low = torch.cat((low_left, low_right))
    high = low + 1
    length = position[high] - position[low]
    if not bool((length > 0).all()):
        raise ValueError("The exact cut produced an unrepresentable positive interval.")
    dual = torch.zeros_like(position)
    dual.index_add_(0, low, length / 2)
    dual.index_add_(0, high, length / 2)
    if not bool((dual > 0).all()):
        raise ValueError("All split electric control volumes must be strictly positive.")
    return SplitPlaneGeometry(position, (position[low] + position[high]) / 2,
        length, dual, low, high, torch.tensor([len(left) - 1, len(left)]),
        cut, aligned, nodes.clone())


def compiled_midpoint_algebra(geometry, material, dt):
    """Local two-face Schur coefficients; compilation is valid on the host."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("The split-plane timestep must be finite and positive.")
    response = compile_copper_response(material)
    rates, B, D = response.rates, response.state_space.B, response.state_space.D
    denominator = 1 + dt * rates / 2
    inverse = denominator.reciprocal()
    drive = dt * inverse[:, None] * B
    effective = D + B.T @ drive / 2
    capacity = geometry.electric_mass[geometry.traces]
    lhs = torch.diag(capacity / dt) + effective / 2
    rhs = torch.diag(capacity / dt) - effective / 2
    return dict(rates=rates, B=B, D=D, inverse_denominator=inverse,
        decay=(1 - dt * rates / 2) * inverse, drive=drive,
        effective=effective, lhs=lhs, rhs=rhs,
        inverse=torch.linalg.inv(lhs), response_error=response.relative_error_bound)


def compile_local_cn_support(geometry, implicit_bounds=None):
    """Actual implicit intervals in a fixed original-mesh patch.

    The default includes the original cut cell and one original neighbor on
    each side. Bounds are exact existing nodes; they cannot clip a real small
    interval or change the interface position. Keeping supplied bounds fixed
    also gives the same implicit physical region in aligned-limit controls.
    """
    nodes = geometry.original_nodes
    if implicit_bounds is None:
        cell = int(torch.searchsorted(nodes, nodes.new_tensor(geometry.cut), right=True))-1
        if not 1 <= cell < len(nodes)-3:
            raise ValueError("The local CN diagnostic requires exterior original cells beyond both patch sides.")
        implicit_bounds = (float(nodes[cell-1]), float(nodes[cell+2]))
    low, high = map(float, implicit_bounds)
    if not (low < geometry.cut < high and bool((nodes == low).any()) and bool((nodes == high).any())):
        raise ValueError("Implicit patch bounds must be exact original nodes surrounding the real cut.")
    if not float(nodes[0]) < low < high < float(nodes[-1]):
        raise ValueError("The implicit patch must leave explicit exterior cells on both sides.")
    edge_low = geometry.electric_positions[geometry.edge_low]
    edge_high = geometry.electric_positions[geometry.edge_high]
    patch = torch.nonzero((edge_low >= low) & (edge_high <= high)).flatten()
    explicit = torch.nonzero(~((edge_low >= low) & (edge_high <= high))).flatten()
    electric = torch.unique(torch.cat((geometry.edge_low[patch], geometry.edge_high[patch])), sorted=True)
    C = geometry.incidence()
    local_curl = C[patch][:, electric]
    selector = torch.zeros((2, len(electric)), dtype=torch.float64)
    selector[torch.arange(2), torch.searchsorted(electric, geometry.traces)] = 1
    if not bool((selector.sum(1) == 1).all()):
        raise ValueError("Both physical interface traces must belong to the actual implicit patch.")
    return dict(bounds=(low, high), electric=electric, magnetic=patch, explicit=explicit,
        local_curl=local_curl, trace_selector=selector)


def compiled_local_cn_algebra(geometry, material, dt, implicit_bounds=None, *,
                             electric_mass=None, magnetic_mass=None, impedance=None):
    """Passive local CN field Schur without an inverse small cut mass.

    Eliminate only the passive material states, retaining every true H interval
    and adjacent E. The unknowns are [E, eta0 H]; exact physical row scaling
    keeps coefficients finite as a cut length approaches zero. It adds no
    regularizer, material fit, dropped interval, or modified field mass.
    """
    if not math.isfinite(dt) or not 0 < dt <= geometry.bulk_stable_dt(.99):
        raise ValueError("Local CN requires the original mesh CFL bound.")
    support = compile_local_cn_support(geometry, implicit_bounds)
    a = compiled_midpoint_algebra(geometry, material, dt)
    ep, hp = support['electric'], support['magnetic']
    supplied = (electric_mass is not None, magnetic_mass is not None, impedance is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("Native field masses and impedance must be supplied together.")
    if all(supplied):
        electric_mass = torch.as_tensor(electric_mass, dtype=torch.float64, device='cpu')
        magnetic_mass = torch.as_tensor(magnetic_mass, dtype=torch.float64, device='cpu')
        if (electric_mass.shape != geometry.electric_mass.shape or magnetic_mass.shape != geometry.magnetic_mass.shape
                or not bool(torch.isfinite(electric_mass).all()) or not bool(torch.isfinite(magnetic_mass).all())
                or not bool((electric_mass > 0).all()) or not bool((magnetic_mass > 0).all())
                or not math.isfinite(float(impedance)) or float(impedance) <= 0):
            raise ValueError("Native field masses require positive finite geometry-compatible vectors and impedance.")
        me, mh, eta = electric_mass[ep], magnetic_mass[hp], float(impedance)
    else:
        me, mh = geometry.electric_mass[ep], geometry.magnetic_mass[hp]
        eta = math.sqrt(MU_0/EPSILON_0)
    C, T = support['local_curl'], support['trace_selector']
    electric_diagonal, magnetic_diagonal = 2*eta*me/dt, 2*mh/(eta*dt)
    matrix = torch.cat((torch.cat((torch.diag(electric_diagonal)+eta*T.T@a['effective']@T, -C.T), 1),
                        torch.cat((C, torch.diag(magnetic_diagonal)), 1)), 0)
    factor, pivots = torch.linalg.lu_factor(matrix.to(torch.complex128))
    return dict(**support, electric_diagonal=electric_diagonal,
        magnetic_diagonal=magnetic_diagonal, matrix=matrix,
        factor=factor, pivots=pivots, eta=eta, material=a)


def local_cn_frequency_oracle(geometry, material, dt, frequency, implicit_bounds=None):
    """Independent full harmonic E/H matrix for the compatible IMEX method.

    The host solves algebra only. Retaining physical H (in eta0 units) avoids
    forming the ill-conditioned 1/alpha second-order electric normal equation.
    No equation contains a reciprocal cut length or an adjusted cut location.
    """
    support = compile_local_cn_support(geometry, implicit_bounds)
    C = geometry.incidence().to(torch.complex128)
    Cexp = C.clone()
    Cexp[support['magnetic']] = 0
    response = compile_copper_response(material)
    s = -2j/dt*math.tan(math.pi*frequency*dt)
    B = response.state_space.B.to(torch.complex128)
    Y = response.state_space.D.to(torch.complex128)+B.T@(B/(s+response.rates[:, None]))
    eta = math.sqrt(MU_0/EPSILON_0)
    ne, nh = C.shape[1], C.shape[0]
    matrix = torch.zeros((ne+nh, ne+nh), dtype=torch.complex128)
    matrix[:ne, :ne] = torch.diag(s*eta*geometry.electric_mass)
    matrix[:ne, ne:] = -(C.T+s*dt*Cexp.T/2)
    matrix[ne:, :ne] = C-s*dt*Cexp/2
    matrix[ne:, ne:] = torch.diag(s/eta*geometry.magnetic_mass)
    trace = geometry.traces
    matrix[trace[:, None], trace[None, :]] += eta*Y
    k, boundary_y = discrete_carrier(geometry, dt, frequency)
    matrix[0, 0] += eta*boundary_y
    matrix[ne-1, ne-1] += eta*boundary_y
    import cmath
    xlow, xhigh = float(geometry.electric_positions[0])-geometry.cut, float(geometry.electric_positions[-1])-geometry.cut
    incident = cmath.exp(1j*k*xlow)
    source = matrix.new_zeros(ne+nh)
    source[0] = 2*eta*boundary_y*incident
    solution = torch.linalg.solve(matrix, source)
    electric = solution[:ne]
    r = (complex(electric[0])-incident)*cmath.exp(1j*k*xlow)
    t = complex(electric[-1])*cmath.exp(-1j*k*xhigh)
    v = electric[trace]
    q = Y@v
    absorption = 1-abs(r)**2-abs(t)**2
    heat_absorption = float(torch.vdot(v, q).real)/boundary_y
    return dict(reflection=r, transmission=t, absorption=absorption,
        heat_absorption=heat_absorption, electric=electric, magnetic_at_E_integer_clock=solution[ne:]/eta,
        carrier_wavenumber=k, boundary_admittance=boundary_y,
        harmonic_residual=float(torch.linalg.vector_norm(matrix@solution-source)/torch.linalg.vector_norm(source)),
        harmonic_matrix_condition=float(torch.linalg.cond(matrix)), implicit_bounds=support['bounds'])


def independent_film_reference(thickness, conductivity, frequency):
    """Full physical Maxwell layer, with conductor displacement current.

    This transfer solve does not call the surface material, its rational
    approximation, the split compiler, or the time integrator.
    """
    import cmath
    s = -2j * math.pi * frequency
    gamma = cmath.sqrt(s * MU_0 * (conductivity + s * EPSILON_0))
    impedance = s * MU_0 / gamma
    eta = math.sqrt(MU_0 / EPSILON_0)
    a = cmath.cosh(gamma * thickness)
    sinh = cmath.sinh(gamma * thickness)
    b, c = impedance * sinh, sinh / impedance
    denominator = 2 * a + b / eta + eta * c
    r, t = (b / eta - eta * c) / denominator, 2 / denominator
    return r, t, 1 - abs(r)**2 - abs(t)**2


def discrete_carrier(geometry, dt, frequency):
    """Native uniform exterior dispersion and its exact carrier termination."""
    width = float(geometry.original_nodes[1] - geometry.original_nodes[0])
    if not bool(torch.allclose(geometry.original_nodes[1:] - geometry.original_nodes[:-1],
                              geometry.original_nodes.new_full((len(geometry.original_nodes)-1,), width),
                              rtol=1e-12, atol=0)):
        raise ValueError("The scattering oracle currently requires uniform exterior intervals.")
    omega = 2 * math.pi * frequency
    speed = 1 / math.sqrt(MU_0 * EPSILON_0)
    argument = width * math.sin(omega * dt / 2) / (speed * dt)
    if not abs(argument) < 1:
        raise ValueError("The requested carrier does not propagate on the exterior Yee grid.")
    k = 2 * math.asin(argument) / width
    admittance = math.sqrt(EPSILON_0 / MU_0) * math.cos(k * width / 2) / math.cos(omega * dt / 2)
    return k, admittance


def frequency_oracle(geometry, material, dt, frequency):
    """Independent finite-dimensional harmonic solve, not CPU time stepping.

    Eliminate bulk H from the modified Yee CN equations. At the two physical
    Et traces evaluate the shared passive material state realization directly.
    This is an algebra/coefficient oracle, not a replacement FDTD execution.
    """
    response = compile_copper_response(material)
    s = -2j / dt * math.tan(math.pi * frequency * dt)
    B = response.state_space.B.to(torch.complex128)
    Y = response.state_space.D.to(torch.complex128) + B.T @ (B / (s + response.rates[:, None]))
    C = geometry.incidence().to(torch.complex128)
    me, mh = geometry.electric_mass, geometry.magnetic_mass
    beta = (1 - (s * dt / 2)**2) / s
    matrix = torch.diag(s * me) + beta * (C.T @ (C / mh[:, None]))
    trace = geometry.traces
    matrix[trace[:, None], trace[None, :]] += Y
    k, boundary_y = discrete_carrier(geometry, dt, frequency)
    matrix[0, 0] += boundary_y
    matrix[-1, -1] += boundary_y
    source = torch.zeros(len(me), dtype=torch.complex128)
    import cmath
    incident = cmath.exp(1j * k * (float(geometry.electric_positions[0]) - geometry.cut))
    source[0] = 2 * boundary_y * incident
    electric = torch.linalg.solve(matrix, source)
    xlow, xhigh = float(geometry.electric_positions[0]) - geometry.cut, float(geometry.electric_positions[-1]) - geometry.cut
    reflection = (complex(electric[0]) - incident) * cmath.exp(1j * k * xlow)
    transmission = complex(electric[-1]) * cmath.exp(-1j * k * xhigh)
    trace_e = electric[trace]
    trace_q = Y @ trace_e
    incoming = boundary_y / 2
    heat = float(torch.vdot(trace_e, trace_q).real) / 2
    absorption = 1 - abs(reflection)**2 - abs(transmission)**2
    return dict(reflection=reflection, transmission=transmission,
        absorption=absorption, heat_absorption=heat/incoming,
        electric=electric, carrier_wavenumber=k, boundary_admittance=boundary_y,
        harmonic_residual=float(torch.linalg.vector_norm(matrix @ electric - source) / torch.linalg.vector_norm(source)))


class SplitPlaneCUDA:
    """Validation-only real split-cell field and passive two-face ADE advance.

    Native tensors and field updates stay entirely on CUDA. This deliberately
    narrow 1D stage carries no claim of arbitrary SurfaceMesh, existing 3D
    solver integration, differentiability, Graph, or distributed qualification.
    """
    def __init__(self, geometry, material, dt, frequency, *, device="cuda", scheme="explicit", implicit_bounds=None):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Actual split-cell field advancement requires CUDA.")
        if scheme not in ('explicit', 'local_cn'):
            raise ValueError("The research split runtime supports explicit or local_cn stepping.")
        self.scheme = scheme
        if scheme == 'explicit' and not 0 < float(dt) <= geometry.stable_dt(courant=.99):
            raise ValueError("The split-cell runtime requires the unregularized minimum-interval CFL bound.")
        if scheme == 'local_cn' and not 0 < float(dt) <= geometry.bulk_stable_dt(.99):
            raise ValueError("Local CN requires the original mesh CFL bound.")
        self.geometry, self.dt = geometry, float(dt)
        self.low, self.high, self.trace = (value.to(device) for value in
            (geometry.edge_low, geometry.edge_high, geometry.traces))
        self.me, self.mh = geometry.electric_mass.to(device), geometry.magnetic_mass.to(device)
        self.algebra = {key: value.to(device) if torch.is_tensor(value) else value
                       for key, value in compiled_midpoint_algebra(geometry, material, dt).items()}
        self.patch = None
        if scheme == 'local_cn':
            prepared = compiled_local_cn_algebra(geometry, material, dt, implicit_bounds)
            self.patch = {key: value.to(device) if torch.is_tensor(value) else value
                          for key, value in prepared.items() if key != 'material'}
            self.explicit = self.patch['explicit']
        else:
            self.explicit = torch.arange(len(self.mh), device=device)
        self.k, self.boundary_y = discrete_carrier(geometry, dt, frequency)
        self.electric = torch.zeros(len(self.me), device=device, dtype=torch.complex128)
        self.magnetic = torch.zeros(len(self.mh), device=device, dtype=torch.complex128)
        self.state = torch.zeros(len(self.algebra["rates"]), device=device, dtype=torch.complex128)
        self.force = torch.zeros_like(self.electric)
        self.cumulative_heat = self.me.new_zeros(())
        self.cumulative_boundary_loss = self.me.new_zeros(())
        self.cumulative_input_work = self.me.new_zeros(())

    def energy(self):
        difference = self.electric[self.high] - self.electric[self.low]
        return ((self.me * self.electric.abs().square()).sum()
            + (self.mh * self.magnetic.abs().square()).sum()
            + self.state.abs().square().sum()
            - self.dt * torch.vdot(self.magnetic[self.explicit], difference[self.explicit]).real) / 2

    def _advance_local_cn_fields(self, old):
        """Solve actual patch fields and copper jointly after exterior H."""
        p, a, dt = self.patch, self.algebra, self.dt
        explicit, ep, hp = self.explicit, p['electric'], p['magnetic']
        self.magnetic[explicit] -= dt*(old[self.high[explicit]]-old[self.low[explicit]])/self.mh[explicit]
        self.force.zero_()
        self.force.index_add_(0, self.low[explicit], -self.magnetic[explicit])
        self.force.index_add_(0, self.high[explicit], self.magnetic[explicit])
        self.electric.copy_(old+dt*self.force/self.me)
        history = a['B'].T.to(old.dtype)@(a['inverse_denominator']*self.state)
        electric_rhs = (p['electric_diagonal']*old[ep]+p['eta']*self.force[ep]
            -p['eta']*p['trace_selector'].T.to(old.dtype)@history)
        magnetic_rhs = p['magnetic_diagonal']*p['eta']*self.magnetic[hp]
        midpoint = torch.linalg.lu_solve(p['factor'], p['pivots'],
            torch.cat((electric_rhs, magnetic_rhs))[:, None])[:, 0]
        ne = len(ep)
        # The exact CN trace midpoint is already an independent solved value.
        # Recovering it from Eold+Enew would cancel O(alpha^-1/2) fast fields
        # and corrupt the skin-state drive on a genuine very small cut.
        self._trace_midpoint = p['trace_selector'].to(old.dtype)@midpoint[:ne]
        self.electric[ep] = 2*midpoint[:ne]-old[ep]
        self.magnetic[hp] = 2*midpoint[ne:]/p['eta']-self.magnetic[hp]

    def step(self, incident_midpoint):
        """Advance actual fields; incident_midpoint is the boundary E average."""
        if not torch.is_tensor(incident_midpoint) or incident_midpoint.device != self.electric.device:
            raise ValueError("The incident sample must already be a tensor on the field CUDA device.")
        dt, a = self.dt, self.algebra
        old = self.electric.clone()
        if self.scheme == 'local_cn':
            self._advance_local_cn_fields(old)
        else:
            self.magnetic.sub_(dt * (old[self.high] - old[self.low]) / self.mh)
            self.force.zero_()
            self.force.index_add_(0, self.low, -self.magnetic)
            self.force.index_add_(0, self.high, self.magnetic)
            self.electric.copy_(old + dt * self.force / self.me)
            history = a["B"].T.to(self.state.dtype) @ (a["inverse_denominator"] * self.state)
            rhs = a["rhs"].to(old.dtype) @ old[self.trace] + self.force[self.trace] - history
            updated = a["inverse"].to(old.dtype) @ rhs
            self.electric[self.trace] = updated
        for index, drive in ((0, 2 * self.boundary_y * incident_midpoint), (-1, 0)):
            self.electric[index] = ((self.me[index]/dt - self.boundary_y/2) * old[index]
                + self.force[index] + drive) / (self.me[index]/dt + self.boundary_y/2)
        average = (old + self.electric) / 2
        trace_midpoint = self._trace_midpoint if self.scheme == 'local_cn' else average[self.trace]
        new_state = a["decay"] * self.state + a["drive"].to(old.dtype) @ trace_midpoint
        state_midpoint = (self.state + new_state) / 2
        heat = dt * ((a["rates"] * state_midpoint.abs().square()).sum()
            + torch.vdot(trace_midpoint, a["D"].to(old.dtype) @ trace_midpoint).real)
        self.cumulative_heat.add_(heat)
        self.cumulative_boundary_loss.add_(dt * self.boundary_y * (average[0].abs().square() + average[-1].abs().square()))
        self.cumulative_input_work.add_(dt * (average[0].conj() * (2 * self.boundary_y * incident_midpoint)).real)
        self.state.copy_(new_state)
        return self.electric, self.magnetic
