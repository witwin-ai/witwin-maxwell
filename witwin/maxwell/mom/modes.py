"""Regular spherical-wave source space through angular spectrum integration.

Modes are defined by their tangential vector-spherical-harmonic plane-wave
density. This fixes phase without mixing conventions from radial formulas.
All integrals and linear algebra execute on CUDA in native PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .quadrature import gauss_rule, triangle_rule
from .operator import local_geometry
from .mesh import SurfaceSystem


def angular_grid(order, *, device, dtype=torch.float64):
    mu, weight = gauss_rule(order, device=device, dtype=dtype)
    mu, weight = 2*mu-1, 2*weight
    phi = torch.arange(2*order, device=device, dtype=dtype)*math.pi/order
    z, p = torch.meshgrid(mu, phi, indexing="ij")
    sine = torch.sqrt(1-z*z)
    direction = torch.stack((sine*torch.cos(p), sine*torch.sin(p), z), -1).reshape(-1, 3)
    tangent = torch.stack((z*torch.cos(p), z*torch.sin(p), -sine), -1).reshape(-1, 3)
    azimuth = torch.stack((-torch.sin(p), torch.cos(p), torch.zeros_like(p)), -1).reshape(-1, 3)
    weights = (weight[:, None].expand(-1, 2*order)*math.pi/order).flatten()
    return direction, weights, tangent, azimuth, z.flatten(), p.flatten()


def _harmonic_derivatives(l, m, mu, phi):
    """Orthonormal Condon--Shortley Y_lm and theta/phi derivatives."""
    am = abs(m)
    sine = torch.sqrt(1-mu*mu)
    pmm = torch.ones_like(mu)
    for j in range(1, am+1):
        pmm = -(2*j-1)*sine*pmm
    pprev = torch.zeros_like(mu)
    p = pmm
    for degree in range(am+1, l+1):
        pprev, p = p, ((2*degree-1)*mu*p-(degree+am-1)*pprev)/(degree-am)
    normalization = math.sqrt((2*l+1)/(4*math.pi)*math.exp(math.lgamma(l-am+1)-math.lgamma(l+am+1)))
    phase = torch.exp(1j*am*phi)
    y = normalization*p*phase
    dtheta = normalization*(l*mu*p-(l+am)*pprev)/sine*phase
    dphi_over_sin = 1j*am*y/sine
    if m < 0:
        y, dtheta, dphi_over_sin = tuple((-1)**am*v.conj() for v in (y, dtheta, dphi_over_sin))
    return y, dtheta, dphi_over_sin


@dataclass
class RegularModes:
    """TE/TM modes ordered by l, then m=-l..l, then TE/TM.

    TE density is d cross grad_S Y_lm / sqrt(l(l+1)); TM density is
    d cross TE. Both carry phase 1/(4*pi*i**l). The origin is a physical
    phase centre in metres. These are entire regular fields with no sources
    inside or outside the enclosing sphere.
    """
    cutoff: int
    medium: object
    angular_order: int = 16
    origin: tuple = (0., 0., 0.)
    device: object = "cuda"
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        if isinstance(self.cutoff, bool) or not isinstance(self.cutoff, int) or self.cutoff < 1:
            raise ValueError("Modal cutoff must be a positive integer")
        if isinstance(self.angular_order, bool) or not isinstance(self.angular_order, int) or self.angular_order < self.cutoff+2:
            raise ValueError("Angular quadrature order must exceed modal cutoff by at least two")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Modal real precision must be float32 or float64")
        if (self.medium.epsilon_r.imag != 0 or self.medium.mu_r.imag != 0
                or self.medium.k.imag != 0 or self.medium.eta.real <= 0):
            raise ValueError("Regular incident modes require a lossless background")
        d, w, theta, phi, mu, azimuth = angular_grid(self.angular_order, device=self.device, dtype=self.dtype)
        densities, ordering = [], []
        for l in range(1, self.cutoff+1):
            for m in range(-l, l+1):
                _, dt, dp = _harmonic_derivatives(l, m, mu, azimuth)
                grad = (dt[:, None]*theta+dp[:, None]*phi)/math.sqrt(l*(l+1))
                te = torch.linalg.cross(d.to(grad.dtype), grad)
                tm = torch.linalg.cross(d.to(grad.dtype), te)
                for kind, density in (("TE", te), ("TM", tm)):
                    ordering.append((l,m,kind))
                    densities.append(density/(4*math.pi*1j**l))
        self.directions, self.weights = d, w
        self.electric_density = torch.stack(densities, 1)
        self.ordering = tuple(ordering)
        self.magnetic_density = torch.linalg.cross(d[:, None].to(self.electric_density.dtype).expand_as(self.electric_density), self.electric_density)/self.medium.eta
        origin = torch.as_tensor(self.origin, device=d.device, dtype=d.dtype)
        if origin.shape != (3,) or not bool(torch.isfinite(origin).all()):
            raise ValueError("Modal origin must be three finite SI coordinates")
        self._origin = origin
        self.metadata = {"source_space": "regular_VSWF_angular_spectrum_v1", "ordering": self.ordering,
                         "phase": "TE=d_cross_gradY; TM=d_cross_TE; both divided by 4*pi*i**l",
                         "radial_equivalence": "TE=M=j_l(kr)*rhat_cross_grad_S_Y/sqrt(l(l+1)); TM=curl(M)/(i*k)",
                         "harmonics": "orthonormal_Condon_Shortley", "origin_m": tuple(self.origin),
                         "angular_order": self.angular_order, "normalization": "raw",
                         "regularity": "entire_source_free_fields", "input_convention": "exp(-i omega t)"}

    @property
    def count(self):
        return len(self.ordering)

    def evaluate(self, points, coefficients=None):
        """Return [query,mode,xyz], or [query,xyz] for a coefficient vector."""
        if points.device != self.directions.device or points.dtype != self.directions.dtype or points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("Mode queries must be CUDA real [N,3] on the modal grid device/dtype")
        if not bool(torch.isfinite(points).all()):
            raise ValueError("Modal query coordinates must be finite")
        electric, magnetic = [], []
        ed, hd = self.electric_density, self.magnetic_density
        if coefficients is not None:
            if coefficients.shape != (self.count,) or coefficients.device != ed.device or coefficients.dtype != ed.dtype:
                raise ValueError("Modal coefficients must match the modal count, dtype and device")
            if not bool(torch.isfinite(coefficients).all()):
                raise ValueError("Modal coefficients must be finite")
            ed = torch.einsum("dmc,m->dc", ed, coefficients)[:, None]
            hd = torch.einsum("dmc,m->dc", hd, coefficients)[:, None]
        for query in points.split(256):
            phase = torch.exp(1j*self.medium.k*((query-self._origin)@self.directions.T))*self.weights
            electric.append(torch.einsum("qd,dmc->qmc", phase, ed))
            magnetic.append(torch.einsum("qd,dmc->qmc", phase, hd))
        if not len(points):
            shape = (0, 3) if coefficients is not None else (0, self.count, 3)
            return ed.new_empty(shape), hd.new_empty(shape)
        e, h = torch.cat(electric), torch.cat(magnetic)
        return (e[:,0],h[:,0]) if coefficients is not None else (e,h)

    def normalize_control_surface(self, radius, *, order=None):
        """Normalize positive E/H control-surface energy (net flux is zero).

        Energy convention: integral (|E|²+eta²|H|²)/(4 eta) dS in watts.
        This is an explicit source-space norm, not outward radiated power.
        """
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Control radius must be finite and positive")
        d,w,*_ = angular_grid(order or self.angular_order, device=self.directions.device, dtype=self.dtype)
        e,h = self.evaluate(self._origin+radius*d)
        eta = self.medium.eta.real
        gram = radius**2/(4*eta)*(torch.einsum("qmc,qnc,q->mn", e.conj(),e,w.to(e.dtype))+eta**2*torch.einsum("qmc,qnc,q->mn",h.conj(),h,w.to(h.dtype)))
        diagonal = gram.diagonal().real
        if bool((diagonal <= 0).any()):
            raise ValueError("Control-surface modal energy is not positive")
        scale = diagonal.rsqrt()
        normalized = scale[:,None]*gram*scale[None,:]
        offdiag = torch.linalg.vector_norm(normalized-torch.eye(self.count, device=gram.device, dtype=gram.dtype))/math.sqrt(self.count)
        if float(offdiag) > max(1e-6, 64*torch.finfo(self.dtype).eps):
            raise ValueError("Angular quadrature has not converged for control-surface normalization")
        self.electric_density = self.electric_density*scale[None,:,None]
        self.magnetic_density = self.magnetic_density*scale[None,:,None]
        self.metadata.update({"normalization": "unit_positive_control_surface_energy_W", "control_radius_m": radius,
                              "gram_relative_error": float(offdiag), "control_angular_order": order or self.angular_order,
                              "coefficient_units": "sqrt(W) under the positive E/H control norm",
                              "net_outward_power_normalized": False})
        return gram

    def validate_enclosing_sphere(self, basis, radius, *, source_positions=None):
        """Reject nonenclosing controls and source singularities inside the ball.

        Positions describe singularities of an external incident field being
        expanded, when supplied. They are not sources of the entire modes.
        Singular sources on or inside the ball need direct analytic traces.
        """
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Enclosing radius must be finite and positive")
        if isinstance(basis, SurfaceSystem):
            basis.assert_current()
            bases = tuple(body.basis for body in basis.bodies)
        else:
            bases = (basis,)
        maximum_radius = 0.
        for body_basis in bases:
            if body_basis.mesh.device != self.directions.device or body_basis.mesh.dtype != self.dtype:
                raise ValueError("Modal basis and geometry must share CUDA device and precision")
            radii = torch.linalg.vector_norm(body_basis.mesh.vertices-self._origin, dim=-1)
            if bool((radii >= radius).any()):
                raise ValueError("Control sphere must strictly enclose every mesh vertex of every body")
            maximum_radius = max(maximum_radius, float(radii.max()))
        if source_positions is not None:
            if not isinstance(source_positions, torch.Tensor) or source_positions.device != self.directions.device:
                raise ValueError("Source singularity positions must be CUDA tensors on the modal device")
            if source_positions.ndim != 2 or source_positions.shape[1] != 3 or not bool(torch.isfinite(source_positions).all()):
                raise ValueError("Source singularity positions must be finite [N,3] coordinates")
            if bool((torch.linalg.vector_norm(source_positions-self._origin, dim=-1) <= radius).any()):
                raise ValueError("Regular expansion excludes source singularities on or inside the enclosing sphere; use analytic incident traces")
        return {"enclosing_radius_m": radius, "geometry_radius_m": maximum_radius,
                "singularity_positions_checked": source_positions is not None}


def build_modal_rhs(basis, modes, *, order=6):
    """Build B_q with physical weak RHS columns for all regular modes."""
    if not isinstance(modes, RegularModes):
        raise TypeError("Modal RHS requires a RegularModes source space")
    if isinstance(basis, SurfaceSystem):
        basis.assert_current()
        result = torch.empty(2*basis.total_edges, modes.count, device=basis.device, dtype=modes.electric_density.dtype)
        for i, body in enumerate(basis.bodies):
            local = build_modal_rhs(body.basis, modes, order=order)
            n = body.basis.n_edges
            result[basis.J_slices[i]], result[basis.M_slices[i]] = local[:n], local[n:]
        return result
    if modes.directions.device != basis.mesh.device or modes.dtype != basis.mesh.dtype:
        raise ValueError("Modal source and RWG basis must share CUDA device and precision")
    edges, free, slopes = local_geometry(basis)
    points, weights = triangle_rule(basis.mesh.triangles, order)
    e,h = modes.evaluate(points.reshape(-1,3))
    e,h = e.reshape(*points.shape[:2],modes.count,3),h.reshape(*points.shape[:2],modes.count,3)
    values = (slopes[:,None,:,None]*(points[:,:,None]-free[:,None])).to(e.dtype)
    result = torch.zeros(2*basis.n_edges,modes.count, device=e.device,dtype=e.dtype)
    for offset,field,sign in ((0,h,-1),(basis.n_edges,e,1)):
        local = sign*torch.einsum("tqlc,tqmc,tq->tlm",values,field,weights.to(e.dtype))
        result.index_add_(0,edges.flatten()+offset,local.reshape(-1,modes.count))
    return result


def modal_response(handle, modes, *, order=6, stopping_rule=None, source_positions=None):
    """Return complete A^-1 B columns and numerical solve costs."""
    from .linalg import solve_reference
    if (modes.medium.frequency != handle.exterior.frequency or modes.medium.epsilon_r != handle.exterior.epsilon_r
            or modes.medium.mu_r != handle.exterior.mu_r):
        raise ValueError("Modal source medium must match the operator exterior")
    radius = modes.metadata.get("control_radius_m")
    if source_positions is not None and radius is None:
        raise ValueError("Checking incident-source applicability requires a declared normalized control sphere")
    geometry = handle.system if hasattr(handle, "system") else handle.basis
    applicability = (modes.validate_enclosing_sphere(geometry, radius, source_positions=source_positions)
                     if radius is not None else {"normalization": "raw; enclosing control not requested"})
    rhs = build_modal_rhs(geometry,modes,order=order)
    solution,log = solve_reference(handle,rhs.T.contiguous(),stopping_rule)
    return solution.T.contiguous(), {"modes": dict(modes.metadata), "solver": log,
                                    "complete_column_count": modes.count, "applicability": applicability,
                                    "response_kind": "unlifted_total_rwg_modal_probe",
                                    "physical_lifted_solution": False,
                                    "spatial_reference_validated": bool(handle.metadata.get("reference_validated", False))}


def calibrate_modal_response(
    handle, *, cutoffs=(1, 3, 5), angular_order=12, control_radius,
    origin=(0., 0., 0.), rhs_order=6, cutoff_tolerance=5e-4,
    angular_tolerance=1e-6, source_positions=None,
):
    """Calibrate complete normalized responses with +2 modal and +4 angular steps.

    Added-column significance is the spectral norm of the newly admitted
    response divided by the full response norm. Currents use an explicitly
    named impedance-balanced surface L2 metric and input coefficients the
    positive control-surface norm. This tests worst-case added unit-energy
    excitations; it is neither a tail bound beyond the largest cutoff nor a
    spatial-discretization certificate. All mode columns share cached GPU LU.
    """
    cutoffs = tuple(cutoffs)
    if len(cutoffs) < 2 or any(isinstance(l, bool) or not isinstance(l, int) or l < 1 for l in cutoffs):
        raise ValueError("Calibration requires at least two positive integer cutoffs")
    if any(b-a != 2 for a, b in zip(cutoffs, cutoffs[1:])):
        raise ValueError("Modal cutoff calibration must advance by exactly two orders")
    if isinstance(angular_order, bool) or not isinstance(angular_order, int) or angular_order < cutoffs[-1]+2:
        raise ValueError("Angular order must exceed the largest cutoff by at least two")
    if any(not math.isfinite(value) or value <= 0 for value in (cutoff_tolerance, angular_tolerance)):
        raise ValueError("Modal calibration tolerances must be finite and positive")
    from .linalg import StoppingRule
    geometry = handle.system if hasattr(handle, 'system') else handle.basis
    if isinstance(geometry, SurfaceSystem):
        dtype = geometry.dtype
        bases = tuple(body.basis for body in geometry.bodies)
        j_slices, m_slices = geometry.J_slices, geometry.M_slices
    else:
        dtype = geometry.mesh.dtype
        bases = (geometry,)
        j_slices, m_slices = (slice(0, geometry.n_edges),), (slice(geometry.n_edges, 2*geometry.n_edges),)
    factors = tuple(torch.linalg.cholesky(basis.mass_matrix().to(handle.dtype)).mH for basis in bases)

    def weighted(response):
        # Apply body factors independently; never allocate a zero-filled global Gram.
        return torch.cat([abs(handle.exterior.eta)*factor@response[sl]
                          for factor, sl in zip(factors, j_slices)]
                         +[factor@response[sl] for factor, sl in zip(factors, m_slices)])
    rows, previous, previous_count = [], None, 0
    rule = StoppingRule(method="direct")
    tiny = torch.finfo(dtype).tiny
    for cutoff in cutoffs:
        responses, modes_by_order, logs = [], [], []
        for order in (angular_order, angular_order+4):
            modes = RegularModes(cutoff, handle.exterior, angular_order=order, origin=origin,
                                 device=handle.device, dtype=dtype)
            modes.validate_enclosing_sphere(geometry, control_radius, source_positions=source_positions)
            modes.normalize_control_surface(control_radius, order=angular_order+4)
            response, log = modal_response(handle, modes, order=rhs_order, stopping_rule=rule,
                                           source_positions=source_positions)
            responses.append(response)
            modes_by_order.append(modes)
            logs.append(log)
        response, modes = responses[-1], modes_by_order[-1]
        balanced = weighted(response)
        total_norm = torch.linalg.matrix_norm(balanced, ord=2)
        angular_error = torch.linalg.matrix_norm(weighted(responses[0]-response), ord=2)/total_norm.clamp_min(tiny)
        if previous is None:
            prefix_error, tail_ratio, added_norm = None, None, None
        else:
            prefix_error = float(torch.linalg.matrix_norm(weighted(response[:, :previous_count]-previous), ord=2)/total_norm.clamp_min(tiny))
            added_norm_tensor = torch.linalg.matrix_norm(balanced[:, previous_count:], ord=2)
            added_norm, tail_ratio = float(added_norm_tensor), float(added_norm_tensor/total_norm.clamp_min(tiny))
        rows.append({"cutoff": cutoff, "mode_count": modes.count, "added_column_count": modes.count-previous_count,
                     "angular_orders": (angular_order, angular_order+4), "angular_relative_error": float(angular_error),
                     "prefix_relative_error": prefix_error, "added_response_operator_norm": added_norm,
                     "added_response_relative_operator_norm": tail_ratio, "full_response_operator_norm": float(total_norm),
                     "solver_logs": [log["solver"] for log in logs]})
        previous, previous_count = response, modes.count
    algebra_ok = all(log["converged"] for row in rows for log in row["solver_logs"])
    angular_ok = all(row["angular_relative_error"] <= angular_tolerance for row in rows)
    cutoff_ok = rows[-1]["added_response_relative_operator_norm"] <= cutoff_tolerance
    report = {"rows": rows, "algebra_converged": algebra_ok, "angular_converged": angular_ok,
              "cutoff_converged": cutoff_ok, "converged": algebra_ok and angular_ok and cutoff_ok,
              "cutoff_tolerance": cutoff_tolerance, "angular_tolerance": angular_tolerance,
              "source_metric": "unit_positive_control_surface_energy_W",
              "response_metric": "impedance_balanced_surface_L2_proxy",
              "uncomputed_tail_bound_certified": False,
              "spatial_reference_validated": bool(handle.metadata.get("reference_validated", False)),
              "largest_mode_metadata": dict(modes.metadata)}
    return response, modes, report
