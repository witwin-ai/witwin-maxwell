"""Unqualified passive Yee/copper response primitive for separate mortars.

Physical plane/shell tests reject this unsplit-field spatial interface. The
public SurfaceMesh-Cu preparation guard remains in place.

The modified Yee energy supplies the positive field mass. Both interface
currents are eliminated together, including the magnetic-to-electric curl
response. No diagonal surface-mass approximation or post-step filter is used.
"""
from __future__ import annotations

import torch

from ..compiler.curved_copper import compile_copper_channels


class JointCopperMortar:
    """Prepared GPU response for a fixed physical surface and static bulk media.

    Inputs use physical native control-volume masses and their compatible
    volume-weighted curl transpose. Columns of electric_trace and curl_transpose
    share the expanded electric support. Magnetic columns share the independent
    magnetic trace support. All numerical work remains on the input device.
    """

    def __init__(self, electric_mass, magnetic_mass, electric_conduction,
                 magnetic_conduction, curl_transpose, electric_trace,
                 magnetic_trace, surface_mass, material, dt):
        if electric_mass.device.type != "cuda":
            raise ValueError("Joint native copper mortars require a CUDA device.")
        self.dt = float(dt)
        self.ge, self.gh, self.mass = electric_trace, magnetic_trace, surface_mass
        self.ct = curl_transpose
        self.channel = compile_copper_channels(material)
        # Scalar material parameters may begin on CPU. This placement preserves
        # their autograd connection, unlike reconstructing detached float values.
        self.coefficients = {key: value.to(electric_mass.device) for key, value in
                             self.channel.midpoint_coefficients(dt).items()}
        self.rates = self.channel.rates.to(electric_mass.device)
        self.a = self.channel.electric_residue.to(electric_mass.device)
        self.b = self.channel.magnetic_residue.to(electric_mass.device)
        self.direct = self.channel.electric_direct.to(electric_mass.device)
        self.tail = self.channel.magnetic_inductance.to(electric_mass.device)
        self.be = (electric_mass+dt*electric_conduction/2).reciprocal()
        self.bh = (magnetic_mass+dt*magnetic_conduction/2).reciprocal()
        self.surface_cholesky = torch.linalg.cholesky(surface_mass)
        tail_factor = surface_mass/self.tail+(magnetic_trace*self.bh[None])@magnetic_trace.T
        self.tail_cholesky = torch.linalg.cholesky(tail_factor)
        self.h_response = self._inverse_h(magnetic_trace.T)
        self.e_response = self.be[:, None]*electric_trace.T
        self.em_response = dt*self.be[:, None]*(curl_transpose@self.h_response)
        self.ge_gain, self.gm_gain = self.coefficients["electric_gain"], self.coefficients["magnetic_gain"]
        self.resistance = torch.block_diag(surface_mass/self.ge_gain, surface_mass/self.gm_gain)
        ree = dt/2*electric_trace@self.e_response
        rem = dt/2*electric_trace@self.em_response
        rhh = dt/2*magnetic_trace@self.h_response
        zeros = torch.zeros_like(rem.T)
        self.response = torch.cat((torch.cat((ree, rem), 1), torch.cat((zeros, rhh), 1)), 0)
        self.factor, self.pivots = torch.linalg.lu_factor(self.resistance+self.response)
        n = len(surface_mass)
        self.ze = electric_mass.new_zeros((len(self.rates), n))
        self.zm = torch.zeros_like(self.ze)
        self.cumulative_loss = electric_mass.new_zeros(())
        self.electric_old = torch.empty_like(electric_mass)
        self.magnetic_old = torch.empty_like(magnetic_mass)
        self.last_electric_effort = electric_mass.new_zeros(n)
        self.last_magnetic_effort = electric_mass.new_zeros(n)
        self.last_electric_current = electric_mass.new_zeros(n)
        self.last_magnetic_current = electric_mass.new_zeros(n)

    def _inverse_h(self, value):
        matrix = value[:, None] if value.ndim == 1 else value
        diagonal = self.bh[:, None]*matrix
        correction = torch.cholesky_solve(self.gh@diagonal, self.tail_cholesky)
        result = diagonal-self.bh[:, None]*(self.gh.T@correction)
        return result[:, 0] if value.ndim == 1 else result

    def begin(self, electric, magnetic):
        self.electric_old.copy_(electric)
        self.magnetic_old.copy_(magnetic)

    def correct_magnetic_predictor(self, magnetic):
        """Add the proper magnetic tail mass before the native electric update."""
        correction = torch.cholesky_solve((self.gh@(magnetic-self.magnetic_old))[:, None],
                                           self.tail_cholesky)[:, 0]
        magnetic.sub_(self.bh*(self.gh.T@correction))
        return magnetic

    def advance(self, electric, magnetic, *, activity=1):
        """Eliminate both currents, then advance both passive skin-state banks."""
        coefficients = self.coefficients
        free = torch.cat((self.ge@(self.electric_old+electric)/2,
                          self.gh@(self.magnetic_old+magnetic)/2))
        history = torch.cat(((coefficients["electric_history"][:, None]*self.ze).sum(0),
                             (coefficients["magnetic_history"][:, None]*self.zm).sum(0)))
        rhs = free+self.resistance@history
        current = torch.linalg.lu_solve(self.factor, self.pivots, rhs[:, None])[:, 0]
        n = len(self.mass)
        j, m = current[:n], current[n:]
        de = -self.dt*(self.e_response@j+self.em_response@m)
        dh = -self.dt*(self.h_response@m)
        electric.add_(de*activity)
        magnetic.add_(dh*activity)
        effort = free-self.response@current
        effort_coefficients = torch.cholesky_solve(torch.stack((effort[:n], effort[n:]), 1),
                                                   self.surface_cholesky)
        ve, vh = effort_coefficients[:, 0], effort_coefficients[:, 1]
        ze = coefficients["decay"][:, None]*self.ze+coefficients["electric_drive"][:, None]*ve
        zm = coefficients["decay"][:, None]*self.zm+coefficients["magnetic_drive"][:, None]*vh
        midpoint_e, midpoint_m = (self.ze+ze)/2, (self.zm+zm)/2
        magnetic_heat = self.b.sqrt()[:, None]*vh-self.rates.sqrt()[:, None]*midpoint_m
        heat = self.dt*(self.direct*(ve@self.mass@ve)+
            (self.rates*torch.einsum("ni,ij,nj->n", midpoint_e, self.mass, midpoint_e)).sum()+
            torch.einsum("ni,ij,nj->", magnetic_heat, self.mass, magnetic_heat))
        self.cumulative_loss.add_(heat*activity)
        self.ze.copy_(self.ze+(ze-self.ze)*activity)
        self.zm.copy_(self.zm+(zm-self.zm)*activity)
        self.last_electric_effort.copy_(ve)
        self.last_magnetic_effort.copy_(vh)
        self.last_electric_current.copy_(j)
        self.last_magnetic_current.copy_(m)
        return electric, magnetic

    def modal_storage(self):
        return (torch.einsum("ni,ij,nj->", self.ze, self.mass, self.ze)+
                torch.einsum("ni,ij,nj->", self.zm, self.mass, self.zm))/2
