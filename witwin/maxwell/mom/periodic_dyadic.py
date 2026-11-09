"""Native tensor Maxwell Bloch spectra and outgoing periodic plane sums."""
from dataclasses import dataclass,replace,field,asdict
import hashlib
import json
import math
import torch
from ..constants import EPSILON_0,MU_0
from .periodic import LatticeSpec
from .tensor_medium import TensorMedium
from .dyadic_green import maxwell_symbol,current_green_spectrum,compile_plane_modes,SpectralModeCache


class PeriodicDyadicConvergenceError(RuntimeError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('Periodic dyadic cutoff estimate failed')


def _validate_cutoff_tolerances(relative,absolute):
    if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (relative,absolute))
            or relative+absolute==0):
        raise ValueError('Dyadic cutoff tolerances must be finite, nonnegative and nonzero')


def _cutoff_check(baseline,refined,relative,absolute,*,increment=None):
    eta0=math.sqrt(MU_0/EPSILON_0)
    scale=baseline.real.new_tensor([1.,1.,1.,eta0,eta0,eta0])
    def norm(value):
        return torch.linalg.matrix_norm(scale[:,None]*value*scale[None,:],ord='fro')
    difference=norm(refined-baseline if increment is None else increment)
    reference=norm(refined)
    threshold=absolute+relative*reference
    return {'passed':bool((torch.isfinite(difference)&torch.isfinite(reference)
                          &torch.isfinite(threshold)&(difference<=threshold)).all()),
        'absolute_difference':difference.detach().cpu().tolist(),
        'threshold':threshold.detach().cpu().tolist(),
        'relative_tolerance':relative,'absolute_tolerance_ohm_per_m2':absolute,
        'norm':'Frobenius_diag_I_eta0_G_diag_I_eta0',
        'certified_error_bound':False,'physical_qualification':False}


@dataclass(frozen=True)
class QuasiperiodicDyadicGreenKernel:
    """Experimental 1D/2D/3D passive-tensor periodic building block.

    Exact requested 3D Fourier coefficients, finite heat-filtered sums, 2D
    outgoing plane sums and 1D outgoing transverse integrals away from the
    source plane are available. Primary
    singular separation and admission to periodic surface assembly remain open. In particular, no finite sampled spectrum
    proves that omitted modes of an indefinite tensor are free of poles.
    """
    lattice: LatticeSpec
    medium: TensorMedium
    pole_relative_tolerance: float = 1e-10
    mode_batch: int = 32
    query_batch: int = 32
    cache_max_bytes: int = 256*1024**2
    mode_cache: SpectralModeCache = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.lattice,LatticeSpec):
            raise ValueError('Dyadic periodic kernels require a LatticeSpec')
        if not isinstance(self.medium,TensorMedium):
            raise ValueError('Periodic dyadic spectra require a TensorMedium')
        if (self.medium.epsilon_r.device!=self.lattice.vectors.device
                or self.medium.epsilon_r.real.dtype!=self.lattice.vectors.dtype):
            raise ValueError('Periodic medium and lattice must share device and precision')
        if type(self.pole_relative_tolerance) not in (int,float) or not math.isfinite(self.pole_relative_tolerance) or self.pole_relative_tolerance<=0:
            raise ValueError('Pole tolerance must be finite and positive')
        if any(type(v) is not int or v<1 for v in (self.mode_batch,self.query_batch)):
            raise ValueError('Dyadic query and mode batches must be positive integers')
        self.medium.assert_current()
        self.lattice.assert_current()
        object.__setattr__(self, 'mode_cache', SpectralModeCache(self.medium, self.cache_max_bytes))

    def _plane_modes(self, key, transverse, medium, weights):
        # One owned lattice/material and explicit grid/frame/pole keys define an entry.
        self.lattice.assert_current()
        self.mode_cache.validate(self.medium)
        def build():
            return (compile_plane_modes(transverse, medium,
                pole_relative_tolerance=self.pole_relative_tolerance), None, weights)
        retain = not any(t.requires_grad for t in (self.medium.epsilon_r, self.medium.mu_r,
            transverse, medium.epsilon_r, medium.mu_r, weights))
        modes, _, retained_weights = self.mode_cache.get(key, build, retain=retain)
        return modes, retained_weights

    def spectral_modes(self,orders):
        """Map unnormalized [Jhat,Mhat] source moments to physical [Ehat,Hhat]."""
        if self.lattice.dimension!=3:
            raise ValueError('Discrete full Bloch spectra require a 3D lattice')
        self.medium.assert_current()
        self.lattice.assert_current()
        sample=self.lattice.vectors
        if (not isinstance(orders,torch.Tensor) or orders.ndim!=2 or orders.shape[1]!=3
                or orders.dtype!=torch.int64 or orders.device!=sample.device
                or len(torch.unique(orders,dim=0))!=len(orders)):
            raise ValueError('Dyadic orders require unique matching CUDA int64 [N,3] indices')
        q=orders.to(sample.dtype)@self.lattice.reciprocal+self.lattice.bloch
        blocks=[]
        for wave in q.split(self.mode_batch):
            wave=wave.to(self.medium.epsilon_r.dtype)
            symbol=maxwell_symbol(wave,self.medium.epsilon_r,self.medium.mu_r,self.medium.frequency)
            singular=torch.linalg.svdvals(symbol)
            threshold=max(self.pole_relative_tolerance,128*torch.finfo(sample.dtype).eps)
            if bool((singular[:,-1]<=threshold*singular[:,0]).any()):
                raise ValueError('Unresolved tensor Maxwell spectral pole; no implicit damping is permitted')
            blocks.append(current_green_spectrum(wave,self.medium)/self.lattice.cell_measure)
        value=torch.cat(blocks) if blocks else self.medium.epsilon_r.new_empty(0,6,6)
        return {'orders':orders.clone(),'wavevectors':q,'current_to_field':value,
            'normalization':'unnormalized_current_moments_to_cell_fourier_fields',
            'global_resolvent_verified':False,'physical_qualification':False}

    def projected_modes(self, orders, *, rank_relative_tolerance):
        """Explicit truncated-SVD modal quantity, never an unprojected resolvent.

        Orthogonality uses the dimensionless Maxwell symbol coordinates
        [E, eta0 H] and its scaled forcing. Removed forcing is described by the
        left-null projector; homogeneous freedom by the right-null projector.
        Rank is a caller-selected numerical policy, not a proof of an exact
        eigenfrequency. Constitutive gradients across rank changes are rejected.
        """
        if self.lattice.dimension != 3:
            raise ValueError('Projected Bloch modes require a 3D lattice')
        self.medium.assert_current()
        self.lattice.assert_current()
        sample = self.lattice.vectors
        if (type(rank_relative_tolerance) not in (int, float)
                or not math.isfinite(rank_relative_tolerance)
                or not 128*torch.finfo(sample.dtype).eps <= rank_relative_tolerance < 1):
            raise ValueError('Rank tolerance must be finite, at least 128 eps and below one')
        if self.medium.epsilon_r.requires_grad or self.medium.mu_r.requires_grad:
            raise NotImplementedError('Projected rank-changing material derivatives are not defined')
        if (not isinstance(orders, torch.Tensor) or orders.ndim != 2 or orders.shape[1] != 3
                or orders.dtype != torch.int64 or orders.device != sample.device
                or len(torch.unique(orders, dim=0)) != len(orders)):
            raise ValueError('Dyadic orders require unique matching CUDA int64 [N,3] indices')
        q = orders.to(sample.dtype)@self.lattice.reciprocal+self.lattice.bloch
        eta0 = math.sqrt(MU_0/EPSILON_0)
        k0 = self.medium.omega*math.sqrt(EPSILON_0*MU_0)
        source_scale = self.medium.epsilon_r.new_tensor([-1j*eta0/k0]*3+[1j/k0]*3)
        field_scale = self.medium.epsilon_r.new_tensor([1.]*3+[1/eta0]*3)
        matrices, left_null, right_null, singular_values, ranks = [], [], [], [], []
        for wave in q.split(self.mode_batch):
            symbol = maxwell_symbol(wave.to(self.medium.epsilon_r.dtype),
                self.medium.epsilon_r, self.medium.mu_r, self.medium.frequency)
            u, singular, vh = torch.linalg.svd(symbol, full_matrices=False)
            retained = singular > rank_relative_tolerance*singular[:, :1]
            # Safe denominators avoid evaluating reciprocal(0) in discarded modes.
            inverse = torch.where(retained, singular, torch.ones_like(singular)).reciprocal()
            inverse = torch.where(retained, inverse, torch.zeros_like(inverse))
            pseudoinverse = (vh.mH*inverse[:, None, :])@u.mH
            removed = (~retained).to(symbol.dtype)
            left = (u*removed[:, None, :])@u.mH
            right = (vh.mH*removed[:, None, :])@vh
            physical = field_scale[:, None]*pseudoinverse*source_scale[None, :]/self.lattice.cell_measure
            if not bool(torch.isfinite(physical).all() & torch.isfinite(left).all() & torch.isfinite(right).all()):
                raise ValueError('Nonfinite projected Maxwell modal quantity')
            matrices.append(physical)
            left_null.append(left)
            right_null.append(right)
            singular_values.append(singular)
            ranks.append(retained.sum(-1))
        def concatenate(parts, shape, *, real=False):
            template = sample if real else self.medium.epsilon_r
            return torch.cat(parts) if parts else template.new_empty(shape)
        return {'orders': orders.clone(), 'wavevectors': q,
            'projected_current_to_field': concatenate(matrices, (0, 6, 6)),
            'removed_forcing_projector': concatenate(left_null, (0, 6, 6)),
            'homogeneous_field_projector': concatenate(right_null, (0, 6, 6)),
            'singular_values': concatenate(singular_values, (0, 6), real=True),
            'retained_rank': torch.cat(ranks) if ranks else orders.new_empty(0),
            'version': 'explicit_truncated_svd_maxwell_modes_v1',
            'quantity': 'projected_scaled_forcing_minimum_norm_modal_response',
            'projector_coordinates': 'dimensionless_symbol_forcing_and_E_eta0H',
            'rank_relative_tolerance': rank_relative_tolerance,
            'normalization': 'unnormalized_current_moments_to_cell_fourier_fields',
            'unprojected_forced_solution': False, 'limiting_absorption_verified': False,
            'exact_nullspace_certified': False, 'global_resolvent_verified': False,
            'physical_qualification': False}

    def apply_modes(self,orders,values,*,adjoint=False):
        """Apply the physical modal map or its complex Euclidean adjoint.

        Adjoint inputs are field covectors, not physical source currents.
        Both paths retain native current and constitutive-tensor derivatives.
        """
        if type(adjoint) is not bool:
            raise TypeError('adjoint must be a boolean')
        modes=self.spectral_modes(orders)
        if (not isinstance(values,torch.Tensor) or values.shape!=(len(orders),6)
                or values.device!=self.medium.epsilon_r.device or values.dtype!=self.medium.epsilon_r.dtype
                or not bool(torch.isfinite(values).all())):
            raise ValueError('Modal values require finite matching CUDA [N,6] tensors')
        matrix=modes['current_to_field']
        return torch.einsum('nij,nj->ni',matrix.mH if adjoint else matrix,values)

    def evaluate_heat(self,displacement,*,shells,heat_time):
        """Finite sum exp(-heat_time*q²) Ghat(q) exp(i q.r).

        Heat filtering is explicit and retains contact distributions. The result
        is neither the unfiltered kernel nor its primary-image regular part.
        No convergence or global pole-free claim is inferred from a finite sum.
        """
        if self.lattice.dimension!=3:
            raise ValueError('Discrete heat-filtered sums require a 3D lattice')
        if type(shells) is not int or shells<0:
            raise ValueError('Dyadic shells must be a nonnegative integer')
        if type(heat_time) not in (int,float) or not math.isfinite(heat_time) or heat_time<=0:
            raise ValueError('Heat time must be finite and strictly positive in m^2')
        sample=self.lattice.vectors
        if (not isinstance(displacement,torch.Tensor) or displacement.ndim!=2 or displacement.shape[1]!=3
                or displacement.device!=sample.device or displacement.dtype!=sample.dtype
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Dyadic queries require finite matching real CUDA [N,3] coordinates')
        self.medium.assert_current()
        self.lattice.assert_current()
        output=self.medium.epsilon_r.new_zeros(len(displacement),6,6)
        width=2*shells+1
        for start in range(0,width**3,self.mode_batch):
            indices=torch.arange(start,min(start+self.mode_batch,width**3),device=sample.device)
            orders=torch.stack((indices//width**2,indices//width%width,indices%width),-1)-shells
            modes=self.spectral_modes(orders)
            q=modes['wavevectors']
            filtered=modes['current_to_field']*torch.exp(-heat_time*q.square().sum(-1))[:,None,None]
            for begin in range(0,len(displacement),self.query_batch):
                end=min(begin+self.query_batch,len(displacement))
                phase=torch.exp(1j*(displacement[begin:end]@q.T))
                output[begin:end]=output[begin:end]+torch.einsum('nm,mij->nij',phase,filtered)
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Nonfinite heat-filtered periodic dyadic kernel')
        return output,{'kernel_id':'quasiperiodic_dyadic_maxwell','version':'bloch_spectral_v1',
            'quantity':'finite_heat_filtered_full_distribution','heat_time_m2':heat_time,
            'shells':shells,'query_count':len(displacement),'primary_singularity_separated':False,
            'global_resolvent_verified':False,'cutoff_converged':False,'physical_qualification':False}

    def evaluate_stacked(self, displacement, *, shells):
        """3D Bloch Green values between lattice planes, without heat filtering.

        Two transverse reciprocal indices remain finite. The third periodic
        direction is summed analytically in each outgoing modal channel. Real
        propagating geometric factors use their Abel continuation; a vanishing
        denominator rejects rather than adding loss. Source-plane/contact
        separation and global spectral convergence are separate obligations.
        """
        if self.lattice.dimension != 3:
            raise ValueError('Stacked dyadic sums require a 3D lattice')
        if type(shells) is not int or shells < 0:
            raise ValueError('Dyadic shells must be a nonnegative integer')
        self.medium.assert_current()
        self.lattice.assert_current()
        sample = self.lattice.vectors
        if (not isinstance(displacement, torch.Tensor) or displacement.ndim != 2 or displacement.shape[1] != 3
                or displacement.device != sample.device or displacement.dtype != sample.dtype
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Dyadic queries require finite matching real CUDA [N,3] coordinates')
        normal = torch.linalg.cross(sample[0], sample[1])
        area = normal.norm()
        normal = normal/area
        period = sample[2]@normal
        if bool(period < 0):
            normal = -normal
            period = -period
        tangent = sample[0]/sample[0].norm()
        cell_index = torch.floor((displacement@normal)/period)
        wrapped = displacement-cell_index[:, None]*sample[2]
        height = wrapped@normal
        scale = torch.maximum(displacement.norm(dim=-1), sample.norm(dim=-1).max())
        plane_tolerance = 128*torch.finfo(sample.dtype).eps*scale
        if bool(((height <= plane_tolerance) | (period-height <= plane_tolerance)).any()):
            raise ValueError('Stacked dyadic source-plane queries require explicit primary/contact separation')
        beta = self.lattice.bloch
        # 2D reciprocal vectors must be tangent to the first two lattice rows.
        plane_q, plane_r = torch.linalg.qr(sample[:2].T, mode="reduced")
        reciprocal = 2*math.pi*torch.linalg.solve_triangular(plane_r, plane_q.T, upper=True)
        beta_plane = beta-(beta@normal)*normal
        k0 = self.medium.omega*math.sqrt(EPSILON_0*MU_0)
        width = 2*shells+1
        output = self.medium.epsilon_r.new_zeros(len(displacement), 6, 6)
        minimum_denominator = float('inf')
        for sign in (1, -1):
            n = sign*normal
            rotation = torch.stack((tangent, torch.linalg.cross(n, tangent), n), -1)
            r = rotation.to(self.medium.epsilon_r.dtype)
            medium = TensorMedium(self.medium.frequency, r.T@self.medium.epsilon_r@r,
                r.T@self.medium.mu_r@r, reciprocal=self.medium.reciprocal)
            transform = torch.block_diag(r, r)
            local_points = (wrapped if sign == 1 else wrapped-sample[2])@rotation
            local_translation = sample[2]@rotation
            local_output = output.new_zeros(len(displacement), 6, 6)
            for start in range(0, width**2, self.mode_batch):
                indices = torch.arange(start, min(start+self.mode_batch, width**2), device=sample.device)
                orders = torch.stack((indices//width, indices%width), -1)-shells
                q = orders.to(sample.dtype)@reciprocal+beta_plane
                weights = sample.new_ones(len(q))/area
                key = ('stacked', sign, shells, start, len(q), self.pole_relative_tolerance)
                modes, weights = self._plane_modes(key, (q@rotation)[:, :2]/k0, medium, weights)
                angle = sign*((modes.transverse@local_translation[:2])[:, None]
                    +modes.longitudinal*local_translation[2]-beta@sample[2])
                denominator = -torch.expm1(1j*angle)
                threshold = max(self.pole_relative_tolerance, 128*torch.finfo(sample.dtype).eps)
                if (not bool(torch.isfinite(denominator).all())
                        or bool((denominator.abs() <= threshold).any())
                        or bool((angle.imag < -threshold).any())):
                    raise ValueError('Unresolved stacked Maxwell spectral pole or growing radiation sheet')
                minimum_denominator = min(minimum_denominator, float(denominator.abs().min().detach()))
                geometric = replace(modes, residues=modes.residues/denominator[..., None, None])
                local_output = local_output+geometric.integrate(local_points, weights, query_batch=self.query_batch)
            phase = 1. if sign == 1 else torch.exp(1j*(beta@sample[2]))
            output = output+phase*(transform@local_output@transform.T)
        output = output*torch.exp(1j*cell_index*(beta@sample[2]))[:, None, None]
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Nonfinite stacked periodic dyadic kernel')
        return output, {'kernel_id': 'quasiperiodic_dyadic_maxwell',
            'version': 'outgoing_stacked_plane_residues_v1',
            'quantity': 'finite_transverse_reciprocal_sum_exact_third_axis_periodization',
            'shells': shells, 'minimum_geometric_denominator': minimum_denominator,
            'source_plane_separated': False, 'contact_distribution_included': False,
            'heat_filter_applied': False, 'cutoff_converged': False,
            'global_resolvent_verified': False, 'physical_qualification': False,
            'spectral_cache': self.mode_cache.stats()}

    def evaluate_open(self,displacement,*,shells):
        """2D tensor Rayleigh sum for strictly nonzero open-axis separation.

        Rotate the constitutive tensors into an oriented lattice frame, integrate
        the open spectral axis by outgoing Maxwell residues, and sum transverse
        reciprocal modes. Primary singular/contact separation at the source
        plane is still unavailable; no coincident or same-plane query is admitted.
        """
        if self.lattice.dimension!=2:
            raise ValueError('Open-axis dyadic sums require a 2D lattice')
        if type(shells) is not int or shells<0:
            raise ValueError('Dyadic shells must be a nonnegative integer')
        self.medium.assert_current()
        self.lattice.assert_current()
        sample=self.lattice.vectors
        if (not isinstance(displacement,torch.Tensor) or displacement.ndim!=2 or displacement.shape[1]!=3
                or displacement.device!=sample.device or displacement.dtype!=sample.dtype
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Dyadic queries require finite matching real CUDA [N,3] coordinates')
        normal=torch.linalg.cross(sample[0],sample[1])/self.lattice.cell_measure
        height=displacement@normal
        if bool((height.abs()<=128*torch.finfo(sample.dtype).eps*displacement.norm(dim=-1)).any()):
            raise ValueError('Dyadic source-plane queries require explicit primary singular/contact separation')
        output=self.medium.epsilon_r.new_zeros(len(displacement),6,6)
        tangent=sample[0]/sample[0].norm()
        k0=self.medium.omega*math.sqrt(EPSILON_0*MU_0)
        width=2*shells+1
        for sign in (1,-1):
            selected=sign*height>0
            if not bool(selected.any()):
                continue
            n=sign*normal
            rotation=torch.stack((tangent,torch.linalg.cross(n,tangent),n),-1)
            r=rotation.to(self.medium.epsilon_r.dtype)
            medium=TensorMedium(self.medium.frequency,r.T@self.medium.epsilon_r@r,r.T@self.medium.mu_r@r,
                                reciprocal=self.medium.reciprocal)
            transform=torch.block_diag(r,r)
            local_points=displacement[selected]@rotation
            local_output=self.medium.epsilon_r.new_zeros(len(local_points),6,6)
            for start in range(0,width**2,self.mode_batch):
                indices=torch.arange(start,min(start+self.mode_batch,width**2),device=sample.device)
                orders=torch.stack((indices//width,indices%width),-1)-shells
                q=orders.to(sample.dtype)@self.lattice.reciprocal+self.lattice.bloch
                weights=sample.new_ones(len(q))/self.lattice.cell_measure
                key=('plane',sign,shells,start,len(q),self.pole_relative_tolerance)
                modes,weights=self._plane_modes(key,(q@rotation)[:,:2]/k0,medium,weights)
                local_output=local_output+modes.integrate(local_points,weights,query_batch=self.query_batch)
            output[selected]=transform@local_output@transform.T
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Nonfinite open-axis periodic dyadic kernel')
        return output,{'kernel_id':'quasiperiodic_dyadic_maxwell','version':'outgoing_plane_residues_v1','spectral_cache':self.mode_cache.stats(),
            'quantity':'finite_2d_reciprocal_sum_at_nonzero_open_separation','shells':shells,
            'pole_guard':'opposite_radiation_sheet_gap_normalized_by_max_k0_abs_qz',
            'pole_relative_tolerance':self.pole_relative_tolerance,
            'primary_singularity_separated':False,'contact_distribution_included':False,
            'global_resolvent_verified':False,'cutoff_converged':False,'physical_qualification':False}

    def evaluate_line(self,displacement,*,normal,shells,transverse_cutoff,quadrature_order,
                      transverse_map='linear'):
        """Finite 1D Bloch sum with a transverse Fourier integral and residues.

        ``normal`` is a fixed unit open direction perpendicular to the lattice.
        Queries on its source plane reject: no contact or primary singular
        separation is implicit. The remaining open wavenumber is integrated
        over [-transverse_cutoff, transverse_cutoff] in rad/m. Each plane mode
        uses the same outgoing Maxwell pole guard as the 2D kernel. Quadrature,
        transverse range and reciprocal range are independent finite budgets.
        """
        from .quadrature import gauss_rule
        if self.lattice.dimension!=1:
            raise ValueError('Line dyadic sums require a 1D lattice')
        self.medium.assert_current();self.lattice.assert_current()
        sample=self.lattice.vectors
        if type(shells) is not int or shells<0:
            raise ValueError('Dyadic shells must be a nonnegative integer')
        if (type(transverse_cutoff) not in (int,float) or not math.isfinite(transverse_cutoff)
                or transverse_cutoff<=0):
            raise ValueError('Transverse cutoff must be positive and finite in rad/m')
        if type(quadrature_order) is not int or quadrature_order<1:
            raise ValueError('Line quadrature order must be a positive integer')
        if transverse_map not in ('linear','sinh'):
            raise ValueError('Transverse map must be linear or sinh')
        if (not isinstance(displacement,torch.Tensor) or displacement.ndim!=2 or displacement.shape[1]!=3
                or displacement.device!=sample.device or displacement.dtype!=sample.dtype
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Dyadic queries require finite matching real CUDA [N,3] coordinates')
        tangent=sample[0]/self.lattice.cell_measure
        tolerance=512*torch.finfo(sample.dtype).eps
        if (not isinstance(normal,torch.Tensor) or normal.shape!=(3,) or normal.device!=sample.device
                or normal.dtype!=sample.dtype or not bool(torch.isfinite(normal).all())
                or bool((normal.norm()-1).abs()>tolerance) or bool((normal@tangent).abs()>tolerance)):
            raise ValueError('Open normal must be a matching unit vector perpendicular to the lattice')
        height=displacement@normal
        if bool((height.abs()<=128*torch.finfo(sample.dtype).eps*displacement.norm(dim=-1)).any()):
            raise ValueError('Dyadic source-plane queries require explicit primary singular/contact separation')
        nodes,weights=gauss_rule(quadrature_order,device=sample.device,dtype=sample.dtype)
        k0=self.medium.omega*math.sqrt(EPSILON_0*MU_0)
        if transverse_map=='linear':
            transverse=(2*nodes-1)*transverse_cutoff
            weights=weights*(2*transverse_cutoff)/(2*math.pi*self.lattice.cell_measure)
        else:
            # A fixed vacuum scale concentrates nodes without depending on material parameters.
            extent=math.asinh(transverse_cutoff/k0)
            if not math.isfinite(extent):
                raise ValueError('Transverse sinh extent is not finite')
            parameter=(2*nodes-1)*extent
            transverse=k0*torch.sinh(parameter)
            weights=weights*(2*extent*k0)*torch.cosh(parameter)/(2*math.pi*self.lattice.cell_measure)
        if not bool(torch.isfinite(transverse).all()&torch.isfinite(weights).all()):
            raise ValueError('Nonfinite transverse quadrature nodes or weights')
        output=self.medium.epsilon_r.new_zeros(len(displacement),6,6)
        count=(2*shells+1)*quadrature_order
        normal_key=tuple(normal.detach().cpu().tolist())
        for sign in (1,-1):
            selected=sign*height>0
            if not bool(selected.any()):
                continue
            n=sign*normal
            rotation=torch.stack((tangent,torch.linalg.cross(n,tangent),n),-1)
            r=rotation.to(self.medium.epsilon_r.dtype)
            medium=TensorMedium(self.medium.frequency,r.T@self.medium.epsilon_r@r,r.T@self.medium.mu_r@r,
                                reciprocal=self.medium.reciprocal)
            transform=torch.block_diag(r,r)
            local_points=displacement[selected]@rotation
            local_output=self.medium.epsilon_r.new_zeros(len(local_points),6,6)
            for start in range(0,count,self.mode_batch):
                flat=torch.arange(start,min(start+self.mode_batch,count),device=sample.device)
                order=flat//quadrature_order-shells
                node=flat%quadrature_order
                parallel=(2*math.pi*order.to(sample.dtype)/self.lattice.cell_measure
                          +self.lattice.bloch@tangent)
                q=torch.stack((parallel,transverse[node]),-1)
                key=('line',normal_key,sign,shells,transverse_cutoff,quadrature_order,
                     transverse_map,start,len(q),self.pole_relative_tolerance)
                modes,node_weights=self._plane_modes(key,q/k0,medium,weights[node])
                local_output=local_output+modes.integrate(local_points,node_weights,query_batch=self.query_batch)
            output[selected]=transform@local_output@transform.T
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Nonfinite open-line periodic dyadic kernel')
        return output,{'kernel_id':'quasiperiodic_dyadic_maxwell',
            'version':'outgoing_line_residues_v1' if transverse_map=='linear' else 'outgoing_line_sinh_residues_v1',
            'quantity':'finite_1d_reciprocal_sum_and_transverse_integral_at_nonzero_open_separation',
            'spectral_cache':self.mode_cache.stats(),
            'shells':shells,'transverse_cutoff_rad_per_m':transverse_cutoff,
            'quadrature_order':quadrature_order,'open_normal':normal.detach().cpu().tolist(),
            'transverse_map':transverse_map,'transverse_map_scale_rad_per_m':k0 if transverse_map=='sinh' else None,
            'spectral_normalization':'dq_transverse_over_2pi_times_lattice_length',
            'pole_relative_tolerance':self.pole_relative_tolerance,
            'primary_singularity_separated':False,'contact_distribution_included':False,
            'global_resolvent_verified':False,'cutoff_converged':False,'physical_qualification':False}

    def estimate_line(self,displacement,*,normal,shells,transverse_cutoff,quadrature_order,
                      relative_tolerance=1e-6,absolute_tolerance=1e-10,transverse_map='linear'):
        """Check quadrature and both spectral cutoffs separately, then together.

        Successive differences are estimates, not certified tail bounds. The
        returned value is the original requested-budget result, matching the
        other cutoff estimators. No zero-loss or source-plane limit is inferred.
        """
        _validate_cutoff_tolerances(relative_tolerance,absolute_tolerance)
        if transverse_map not in ('linear','sinh'):
            raise ValueError('Transverse map must be linear or sinh')
        mapping={} if transverse_map=='linear' else {'transverse_map':transverse_map}
        baseline,receipt=self.evaluate_line(displacement,normal=normal,shells=shells,
            transverse_cutoff=transverse_cutoff,quadrature_order=quadrature_order,**mapping)
        steps=(('quadrature',shells,transverse_cutoff,2*quadrature_order),
               ('transverse_tail',shells,2*transverse_cutoff,4*quadrature_order),
               ('reciprocal_tail',max(1,2*shells),2*transverse_cutoff,4*quadrature_order),
               ('refined_quadrature',max(1,2*shells),2*transverse_cutoff,8*quadrature_order))
        receipt.update(cutoff_checks={},compared_budgets=[
            {'shells':shells,'transverse_cutoff':transverse_cutoff,'quadrature_order':quadrature_order}],
            certified_error_bound=False)
        previous=baseline
        for name,depth,cutoff,order in steps:
            current,_=self.evaluate_line(displacement,normal=normal,shells=depth,
                transverse_cutoff=cutoff,quadrature_order=order,**mapping)
            check=_cutoff_check(previous,current,relative_tolerance,absolute_tolerance)
            receipt['cutoff_checks'][name]=check
            receipt['compared_budgets'].append(
                {'shells':depth,'transverse_cutoff':cutoff,'quadrature_order':order})
            if not check['passed']:
                receipt['failed_axis']=name
                raise PeriodicDyadicConvergenceError(receipt)
            previous=current
        check=_cutoff_check(baseline,previous,relative_tolerance,absolute_tolerance)
        receipt['cutoff_checks']['combined']=check
        if not check['passed']:
            receipt['failed_axis']='combined'
            raise PeriodicDyadicConvergenceError(receipt)
        receipt['cutoff_converged']=True
        return baseline,receipt

    def estimate_stacked(self, displacement, *, shells, relative_tolerance=1e-6,
                         absolute_tolerance=1e-10):
        """Check a doubled transverse cutoff in the unfiltered 3D layer sum.

        The third direction is already summed analytically. This per-query
        successive difference is an estimate, not a global tail bound, and does
        not certify omitted spectral poles or a limiting-absorption procedure.
        """
        _validate_cutoff_tolerances(relative_tolerance, absolute_tolerance)
        baseline, receipt = self.evaluate_stacked(displacement, shells=shells)
        deeper = max(1, 2*shells)
        refined, refined_receipt = self.evaluate_stacked(displacement, shells=deeper)
        check = _cutoff_check(baseline, refined, relative_tolerance, absolute_tolerance)
        receipt.update(cutoff_converged=check['passed'], cutoff_check=check,
            compared_shells=[shells, deeper], refined_evaluation=refined_receipt,
            certified_error_bound=False, limiting_absorption_verified=False)
        if not check['passed']:
            receipt['failed_axis'] = 'transverse_reciprocal_tail'
            raise PeriodicDyadicConvergenceError(receipt)
        return baseline, receipt

    def estimate_open(self,displacement,*,shells,relative_tolerance=1e-6,absolute_tolerance=1e-10):
        """Double the reciprocal range, checking every balanced tensor block."""
        _validate_cutoff_tolerances(relative_tolerance,absolute_tolerance)
        baseline,receipt=self.evaluate_open(displacement,shells=shells)
        deeper=max(1,2*shells)
        refined,_=self.evaluate_open(displacement,shells=deeper)
        check=_cutoff_check(baseline,refined,relative_tolerance,absolute_tolerance)
        receipt.update(cutoff_converged=check['passed'],cutoff_check=check,
            compared_shells=[shells,deeper],certified_error_bound=False)
        if not check['passed']:
            raise PeriodicDyadicConvergenceError(receipt)
        return baseline,receipt

    def estimate_open_plane_regular(self,displacement,*,height,shells,max_shells,
                                    max_halvings,absolute_tolerance,integration=None,primary_cache_max_bytes=256*1024**2):
        """Estimate the 2D nonprimary kernel on the source plane by two-sided limits.

        At each nonzero height subtract the full free-space primary from the
        outgoing reciprocal sum. Each sign has its own absolute spectral gate;
        neither a large primary norm nor cancellation between signs can admit
        an unresolved reciprocal tail. Symmetric averages have even height
        errors; two consecutive Richardson differences must converge. This is
        an uncertified limit estimator, not an Ewald formula.
        """
        return self._estimate_plane_regular(displacement,height=height,shells=shells,
            max_shells=max_shells,max_halvings=max_halvings,
            absolute_tolerance=absolute_tolerance,integration=integration,primary_cache_max_bytes=primary_cache_max_bytes)

    def estimate_stacked_plane_regular(self, displacement, *, height, shells, max_shells,
                                       max_halvings, absolute_tolerance, integration=None, primary_cache_max_bytes=256*1024**2):
        """Primary-excluded 3D remainder on the plane spanned by lattice rows1/2.

        Symmetric height probes stay inside half the third-axis plane spacing.
        The shared checked primary, transverse-tail, cancellation and Richardson
        gates all remain mandatory. This is an estimated smooth remainder, not
        contact-distribution evaluation or a globally certified resolvent.
        """
        return self._estimate_plane_regular(displacement, height=height, shells=shells,
            max_shells=max_shells, max_halvings=max_halvings,
            absolute_tolerance=absolute_tolerance, integration=integration, _stacked=True,
            primary_cache_max_bytes=primary_cache_max_bytes)

    def estimate_line_plane_regular(self,displacement,*,normal,height,shells,
                                    transverse_cutoff,quadrature_order,max_halvings,
                                    absolute_tolerance,integration=None,transverse_map='linear',primary_cache_max_bytes=256*1024**2):
        """1D primary-excluded plane limit with separately checked Fourier budgets.

        Each height uses the line kernel's quadrature, transverse-tail,
        reciprocal-tail and combined checks before the shared two-sided limit.
        Cutoffs are explicit and fixed; an insufficient component budget raises.
        The chosen plane normal must be unit and perpendicular to the lattice.
        """
        if (type(transverse_cutoff) not in (int,float) or not math.isfinite(transverse_cutoff)
                or transverse_cutoff<=0 or type(quadrature_order) is not int or quadrature_order<1):
            raise ValueError('Line limits require finite positive transverse cutoff and quadrature order')
        if transverse_map not in ('linear','sinh'):
            raise ValueError('Transverse map must be linear or sinh')
        return self._estimate_plane_regular(displacement,height=height,shells=shells,
            max_shells=2*shells if type(shells) is int else 0,max_halvings=max_halvings,
            absolute_tolerance=absolute_tolerance,integration=integration,
            primary_cache_max_bytes=primary_cache_max_bytes,
            _normal=normal,_line={'transverse_cutoff':transverse_cutoff,'quadrature_order':quadrature_order,
                                  'transverse_map':transverse_map})

    def _estimate_plane_regular(self,displacement,*,height,shells,max_shells,
                                max_halvings,absolute_tolerance,integration,_normal=None,_line=None,_stacked=False,
                                primary_cache_max_bytes=256*1024**2):
        from .dyadic_green import DyadicGreenKernel,DyadicIntegrationConfig,DyadicConvergenceError
        dimension=3 if _stacked else 2 if _line is None else 1
        if self.lattice.dimension!=dimension:
            raise ValueError(f'Source-plane regular limits require a {dimension}D lattice')
        self.medium.assert_current();self.lattice.assert_current()
        sample=self.lattice.vectors
        if (not isinstance(displacement,torch.Tensor) or displacement.ndim!=2 or displacement.shape[1]!=3
                or displacement.device!=sample.device or displacement.dtype!=sample.dtype
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Source-plane queries require finite matching real CUDA [N,3] coordinates')
        if (type(height) not in (int,float) or not math.isfinite(height) or height<=0
                or type(absolute_tolerance) not in (int,float)
                or not math.isfinite(absolute_tolerance) or absolute_tolerance<=0):
            raise ValueError('Height and absolute tolerance must be finite and positive')
        if (type(shells) is not int or shells<1 or type(max_shells) is not int or max_shells<2*shells
                or type(max_halvings) is not int or max_halvings<3):
            raise ValueError('Limits require positive shells, a doubling budget and at least three halvings')
        normal=torch.linalg.cross(sample[0],sample[1])/self.lattice.cell_measure if _line is None else _normal
        if _stacked:
            normal=torch.linalg.cross(sample[0],sample[1])
            normal=normal/normal.norm()
            spacing=(sample[2]@normal).abs()
            if bool(height>=spacing/2):
                raise ValueError('3D limit height must be below half the lattice-plane spacing')
        if _line is not None:
            tolerance=512*torch.finfo(sample.dtype).eps
            if (not isinstance(normal,torch.Tensor) or normal.shape!=(3,) or normal.device!=sample.device
                    or normal.dtype!=sample.dtype or not bool(torch.isfinite(normal).all())
                    or bool((normal.norm()-1).abs()>tolerance)
                    or bool((normal@(sample[0]/self.lattice.cell_measure)).abs()>tolerance)):
                raise ValueError('Open normal must be a matching unit vector perpendicular to the lattice')
        scale=torch.maximum(displacement.norm(dim=-1),sample.norm(dim=-1).max())
        roundoff=128*torch.finfo(sample.dtype).eps*scale
        if bool(((displacement@normal).abs()>roundoff).any()):
            raise ValueError('Regular plane limits require source-plane coordinates')
        nearest=(displacement@self.lattice.reciprocal.T/(2*math.pi)).round()
        if bool((((displacement-nearest@sample).norm(dim=-1)<=roundoff)
                 &(nearest!=0).any(-1)).any()):
            raise ValueError('Nonprimary lattice singularities cannot be removed by primary subtraction')
        config=DyadicIntegrationConfig() if integration is None else integration
        if not isinstance(config,DyadicIntegrationConfig):
            raise TypeError('Primary integration must be DyadicIntegrationConfig')
        # Absolute component budgets remain meaningful after primary cancellation.
        component=absolute_tolerance/8
        if component==0:
            raise ValueError('Absolute component budget underflow')
        config=replace(config,relative_tolerance=0.,absolute_tolerance=component)
        free=DyadicGreenKernel(self.medium,config,primary_cache_max_bytes)
        receipt={'kernel_id':'quasiperiodic_dyadic_maxwell',
            'version':'two_sided_primary_subtracted_plane_limit_v1',
            'quantity':'nonprimary_source_plane_regular_part',
            'primary_image_excluded':True,'primary_singularity_separated':True,
            'contact_distribution_included':False,'absolute_tolerance_ohm_per_m2':absolute_tolerance,
            'component_absolute_budget':component,'primary_cache_max_bytes':primary_cache_max_bytes,
            'levels':[],'limit_checks':[],
            'converged':False,'certified_error_bound':False,'physical_qualification':False,
            'global_resolvent_verified':False,'limit_estimator':'two_sided_even_Richardson',
            'periodic_dimension':dimension,'open_normal':normal.detach().cpu().tolist()}
        previous=None;previous_extrapolated=None;consecutive=0
        for level in range(max_halvings+1):
            distance=height/(2**level)
            if distance==0 or bool((distance<=roundoff).any()):
                receipt.update(failure_reason='height_below_coordinate_resolution')
                raise PeriodicDyadicConvergenceError(receipt)
            points=torch.cat((displacement+distance*normal,displacement-distance*normal))
            record={'height':distance,'spectral_checks':[]}
            receipt['levels'].append(record)
            try:
                primary,primary_receipt=free.evaluate(points)
            except DyadicConvergenceError as error:
                receipt.update(failure_reason='primary_spectral_budget_exhausted',primary_failure=error.receipt)
                raise PeriodicDyadicConvergenceError(receipt) from error
            record['primary']=primary_receipt
            def check_subtraction(refined):
                eta0=math.sqrt(MU_0/EPSILON_0)
                units=sample.new_tensor([1.,1.,1.,eta0,eta0,eta0])
                def balanced_norm(value):
                    return torch.linalg.matrix_norm(units[:,None]*value*units[None,:],ord='fro')
                resolution=64*torch.finfo(sample.dtype).eps*(balanced_norm(primary)+balanced_norm(refined))
                record['subtraction_resolution_estimate']=resolution.detach().cpu().tolist()
                if not bool((torch.isfinite(resolution)&(resolution<=component)).all()):
                    receipt.update(failure_reason='primary_subtraction_precision_budget_exhausted')
                    raise PeriodicDyadicConvergenceError(receipt)
            if _line is not None:
                try:
                    refined,line_receipt=self.estimate_line(points,normal=normal,shells=shells,
                        relative_tolerance=0.,absolute_tolerance=component,**_line)
                except PeriodicDyadicConvergenceError as error:
                    receipt.update(failure_reason='line_spectral_budget_exhausted',line_failure=error.receipt)
                    raise PeriodicDyadicConvergenceError(receipt) from error
                except (ValueError,RuntimeError) as error:
                    receipt.update(failure_reason='periodic_spectral_evaluation_failed',detail=str(error))
                    raise PeriodicDyadicConvergenceError(receipt) from error
                record['line_spectrum']=line_receipt
                check_subtraction(refined)
                regular=refined-primary
            else:
                order=shells
                def spectral(count):
                    try:
                        evaluate=self.evaluate_stacked if _stacked else self.evaluate_open
                        return evaluate(points,shells=count)[0]
                    except (ValueError,RuntimeError) as error:
                        receipt.update(failure_reason='periodic_spectral_evaluation_failed',
                                       failed_shells=count,detail=str(error))
                        raise PeriodicDyadicConvergenceError(receipt) from error
                baseline=spectral(order)
                while True:
                    deeper=2*order
                    if deeper>max_shells:
                        receipt.update(failure_reason='reciprocal_budget_exhausted')
                        raise PeriodicDyadicConvergenceError(receipt)
                    refined=spectral(deeper)
                    check_subtraction(refined)
                    # Every point and sign passes before averaging the smooth remainder.
                    check=_cutoff_check(baseline-primary,refined-primary,0.,component,
                                        increment=refined-baseline)
                    record['spectral_checks'].append({'shells':[order,deeper],**check})
                    if check['passed']:
                        regular=refined-primary
                        break
                    order=deeper;baseline=refined
            positive,negative=regular.split(len(displacement)) if len(displacement) else (regular,regular)
            current=(positive+negative)/2
            if previous is not None:
                extrapolated=current+(current-previous)/3
                if previous_extrapolated is not None:
                    check=_cutoff_check(previous_extrapolated,extrapolated,0.,absolute_tolerance/2)
                    receipt['limit_checks'].append(check)
                    consecutive=consecutive+1 if check['passed'] else 0
                    if consecutive==2:
                        receipt.update(converged=True,final_height=distance,failure_reason=None)
                        return extrapolated,receipt
                previous_extrapolated=extrapolated
            previous=current
        receipt.update(failure_reason='source_plane_limit_budget_exhausted')
        raise PeriodicDyadicConvergenceError(receipt)


@dataclass(frozen=True)
class LossyDyadicImageReference:
    """Independent finite direct-image reference with the primary image removed.

    Both constitutive loss forms must be strictly positive. Every retained
    free-space image uses its own checked dyadic spatial integration. The image
    cutoff is explicit and uncertified; this is not an Ewald implementation or
    a supported lossless limit.
    """
    lattice: LatticeSpec
    medium: TensorMedium
    shells: int
    integration: object = None
    image_batch: int = 4

    @property
    def kernel_id(self):
        return 'lossy_periodic_dyadic_image_reference_v1'

    def __post_init__(self):
        from .dyadic_green import DyadicIntegrationConfig
        if not isinstance(self.lattice,LatticeSpec) or not isinstance(self.medium,TensorMedium):
            raise TypeError('Dyadic image reference requires a lattice and TensorMedium')
        if (self.lattice.vectors.device!=self.medium.epsilon_r.device
                or self.lattice.vectors.dtype!=self.medium.epsilon_r.real.dtype):
            raise ValueError('Dyadic image lattice and medium must share precision/device')
        if type(self.shells) is not int or self.shells<1 or type(self.image_batch) is not int or self.image_batch<1:
            raise ValueError('Image shells and batch must be positive integers')
        config=DyadicIntegrationConfig() if self.integration is None else self.integration
        if not isinstance(config,DyadicIntegrationConfig):
            raise TypeError('Dyadic image quadrature requires DyadicIntegrationConfig')
        object.__setattr__(self,'integration',config)
        self._validate()

    def _validate(self):
        self.lattice.assert_current()
        self.medium.assert_current()
        for tensor in (self.medium.epsilon_r,self.medium.mu_r):
            loss=(tensor-tensor.mH)/(2j)
            threshold=64*torch.finfo(tensor.real.dtype).eps*torch.linalg.matrix_norm(tensor,ord=2)
            if bool(torch.linalg.eigvalsh(loss).amin()<=threshold):
                raise ValueError('Dyadic image reference requires strictly positive electric and magnetic loss')

    def evaluate(self,displacement,*,exclude_primary=False):
        return self._evaluate_image_band(displacement,exclude_primary=exclude_primary)

    def _evaluate_image_band(self,displacement,*,exclude_primary,minimum_shell=None):
        from .dyadic_green import DyadicGreenKernel,DyadicConvergenceError
        self._validate()
        if type(exclude_primary) is not bool:
            raise TypeError('exclude_primary must be boolean')
        sample=self.lattice.vectors
        if (not isinstance(displacement,torch.Tensor) or displacement.ndim!=2 or displacement.shape[1]!=3
                or displacement.device!=sample.device or displacement.dtype!=sample.dtype
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Image queries require finite matching CUDA real [N,3] coordinates')
        # Reject lattice singularities independently of the finite image budget.
        fractional=displacement@self.lattice.reciprocal.T/(2*math.pi)
        nearest=fractional.round()
        residual=displacement-nearest@sample
        scale=torch.maximum(displacement.norm(dim=-1),sample.norm(dim=-1).max())
        coincident=residual.norm(dim=-1)<=128*torch.finfo(sample.dtype).eps*scale
        retained=(nearest!=0).any(-1) if exclude_primary else torch.ones_like(coincident)
        if bool((coincident&retained).any()):
            raise ValueError('Coincident retained dyadic image requires singular integration')
        free=DyadicGreenKernel(self.medium,self.integration)
        output=self.medium.epsilon_r.new_zeros(len(displacement),6,6)
        width=2*self.shells+1
        count=width**self.lattice.dimension
        receipts=[]
        for start in range(0,count,self.image_batch):
            flat=torch.arange(start,min(start+self.image_batch,count),device=sample.device)
            indices=torch.stack([(flat//width**p)%width for p in reversed(range(self.lattice.dimension))],-1)-self.shells
            if exclude_primary:
                indices=indices[(indices!=0).any(-1)]
            if minimum_shell is not None:
                indices=indices[indices.abs().amax(-1)>minimum_shell]
            if not len(indices) or not len(displacement):
                continue
            translation=indices.to(sample.dtype)@sample
            phase=self.lattice.phase(indices)
            difference=displacement[:,None,:]-translation[None]
            try:
                value,receipt=free.evaluate(difference.reshape(-1,3))
            except DyadicConvergenceError as error:
                error.receipt['periodic_image_reference']={
                    'indices':indices.detach().cpu().tolist(),'query_count':len(displacement),
                    'primary_image_excluded':exclude_primary,'shells':self.shells}
                raise
            output=output+(value.reshape(len(displacement),len(indices),6,6)*phase[None,:,None,None]).sum(1)
            receipts.append(receipt)
        return output,{'quantity':'finite_nonzero_image_regular_part' if exclude_primary else 'finite_full_image_sum',
            'shells':self.shells,'primary_image_excluded':exclude_primary,
            'spatial_integration_receipts':receipts,'image_cutoff_converged':False,
            'physical_qualification':False,'support':'strictly_positive_tensor_loss_reference_only'}

    def estimate_regular(self,displacement,*,relative_tolerance=1e-6,absolute_tolerance=1e-10):
        """Estimate the nonprimary image tail without reintegrating inner images.

        Each image retains its checked free-space quadrature receipt. The image
        difference is a successive-cutoff estimate, not a bound on total error
        or an extrapolation to zero material loss.
        """
        _validate_cutoff_tolerances(relative_tolerance,absolute_tolerance)
        baseline,receipt=self.evaluate(displacement,exclude_primary=True)
        outer=replace(self,shells=2*self.shells)
        tail,outer_receipt=outer._evaluate_image_band(displacement,exclude_primary=True,
            minimum_shell=self.shells)
        check=_cutoff_check(baseline,baseline+tail,relative_tolerance,absolute_tolerance,increment=tail)
        receipt.update(image_cutoff_converged=check['passed'],cutoff_check=check,
            compared_shells=[self.shells,outer.shells],inner_images_reused=True,
            outer_spatial_integration_receipts=outer_receipt['spatial_integration_receipts'],
            certified_error_bound=False)
        if not check['passed']:
            raise PeriodicDyadicConvergenceError(receipt)
        return baseline,receipt


@dataclass(frozen=True)
class QuasiperiodicDyadicRegularKernel:
    """Checked 1D/2D/3D nonprimary values for weak assembly and field convolution.

    All budgets are explicit. Off-plane subtraction and source-plane limits are
    independent estimates, not certified global bounds. No 3D heat-filtered
    distribution is substituted for a primary-excluded spatial kernel.
    """
    kernel: QuasiperiodicDyadicGreenKernel
    shells: int
    height: float
    max_halvings: int
    absolute_tolerance: float
    max_shells: int
    normal: object = None
    transverse_cutoff: float = 64.
    quadrature_order: int = 64
    transverse_map: str = 'linear'
    integration: object = None
    primary_cache_max_bytes: int = 256*1024**2

    @property
    def medium(self):
        return self.kernel.medium

    @property
    def lattice(self):
        return self.kernel.lattice

    @property
    def kernel_id(self):
        return 'checked_quasiperiodic_dyadic_regular_v1'

    def __post_init__(self):
        from .dyadic_green import DyadicIntegrationConfig
        if type(self.primary_cache_max_bytes) is not int or self.primary_cache_max_bytes < 0:
            raise ValueError('Regular primary cache budget must be a nonnegative integer')
        if not isinstance(self.kernel, QuasiperiodicDyadicGreenKernel):
            raise TypeError('Regular spectral composition requires a periodic dyadic kernel')
        if self.lattice.dimension not in (1, 2, 3):
            raise ValueError('Regular spectral composition requires a 1D, 2D or 3D lattice')
        if (type(self.shells) is not int or self.shells < 1 or type(self.max_shells) is not int
                or self.max_shells < 2*self.shells or type(self.max_halvings) is not int or self.max_halvings < 3):
            raise ValueError('Regular spectra require positive shells, doubling budget and three halvings')
        if (type(self.height) not in (int, float) or not math.isfinite(self.height) or self.height <= 0):
            raise ValueError('Regular spectra require a finite positive initial height')
        _validate_cutoff_tolerances(0., self.absolute_tolerance)
        config = DyadicIntegrationConfig() if self.integration is None else self.integration
        if not isinstance(config, DyadicIntegrationConfig):
            raise TypeError('Regular spectral primary integration requires DyadicIntegrationConfig')
        if config.spatial_rule in ('factored_spd_tail_v1', 'factored_spd_pair_tail_v1'):
            raise NotImplementedError('Factored SPD tails do not support periodic primary composition')
        object.__setattr__(self, 'integration', replace(config, relative_tolerance=0.,
            absolute_tolerance=self.absolute_tolerance/8))
        sample = self.lattice.vectors
        if self.lattice.dimension == 1:
            normal = self.normal
            tolerance = 512*torch.finfo(sample.dtype).eps
            if (not isinstance(normal, torch.Tensor) or normal.shape != (3,)
                    or normal.device != sample.device or normal.dtype != sample.dtype
                    or not bool(torch.isfinite(normal).all()) or bool((normal.norm()-1).abs() > tolerance)
                    or bool((normal@(sample[0]/self.lattice.cell_measure)).abs() > tolerance)):
                raise ValueError('Regular line normal must be unit, matching and perpendicular to the lattice')
            if (type(self.transverse_cutoff) not in (int, float) or not math.isfinite(self.transverse_cutoff)
                    or self.transverse_cutoff <= 0 or type(self.quadrature_order) is not int or self.quadrature_order < 1
                    or self.transverse_map not in ('linear', 'sinh')):
                raise ValueError('Regular line transverse budgets or mapping are invalid')
            if self.max_shells != 2*self.shells:
                raise ValueError('Regular line spectra use an explicit two-level reciprocal budget')
            normal = normal.clone()
        else:
            normal = torch.linalg.cross(sample[0], sample[1])
            normal = normal/(normal.norm() if self.lattice.dimension == 3 else self.lattice.cell_measure)
            if self.lattice.dimension == 3 and bool(self.height >= (sample[2]@normal).abs()/2):
                raise ValueError('3D limit height must be below half the lattice-plane spacing')
            if self.normal is not None and (not isinstance(self.normal, torch.Tensor)
                    or self.normal.device != sample.device or self.normal.dtype != sample.dtype
                    or not torch.equal(self.normal, normal)):
                raise ValueError('Regular plane normal is determined by the first two lattice rows')
        object.__setattr__(self, 'normal', normal)
        object.__setattr__(self, '_normal_version', (id(normal), normal._version))
        self._validate()

    def _validate(self):
        self.medium.assert_current()
        self.lattice.assert_current()
        if (id(self.normal), self.normal._version) != self._normal_version:
            raise ValueError('Regular spectral normal changed after validation')

    def payload(self):
        """Save exact fixed configuration, without cached spectra or admission claims."""
        self._validate()
        return {'version': 'periodic_dyadic_regular_config_v2', 'kernel_id': self.kernel_id,
            'binding': self._payload_binding(self.medium, self.lattice),
            'spectral': {name: getattr(self.kernel, name) for name in
                ('pole_relative_tolerance', 'mode_batch', 'query_batch', 'cache_max_bytes')},
            'regular': {name: getattr(self, name) for name in
                ('shells', 'height', 'max_halvings', 'absolute_tolerance', 'max_shells',
                 'transverse_cutoff', 'quadrature_order', 'transverse_map', 'primary_cache_max_bytes')},
            'normal': self.normal.detach().cpu().tolist(), 'integration': self.integration.payload()}

    @staticmethod
    def _payload_binding(medium, lattice):
        if not isinstance(medium, TensorMedium) or not isinstance(lattice, LatticeSpec):
            raise TypeError('Regular restoration requires TensorMedium and LatticeSpec bindings')
        medium.assert_current()
        lattice.assert_current()
        def data(value):
            return torch.view_as_real(value.detach().resolve_conj()).cpu().tolist()
        return {'lattice': lattice.receipt(), 'dtype': str(medium.epsilon_r.dtype),
            'frequency_Hz': medium.frequency, 'reciprocal': medium.reciprocal,
            'epsilon_r': data(medium.epsilon_r), 'mu_r': data(medium.mu_r)}

    @classmethod
    def from_payload(cls, payload, *, medium, lattice):
        """Rebind exact settings to owned CUDA material/lattice objects.

        No Green evaluation or historical convergence certificate is restored.
        Every subsequent evaluation must pass the configured numerical gates.
        """
        from .dyadic_green import DyadicIntegrationConfig
        keys = {'version', 'kernel_id', 'binding', 'spectral', 'regular', 'normal', 'integration'}
        if type(payload) is not dict or set(payload) != keys:
            raise ValueError('Regular kernel payload has missing or unknown fields')
        if (payload['version'] != 'periodic_dyadic_regular_config_v2'
                or payload['kernel_id'] != 'checked_quasiperiodic_dyadic_regular_v1'):
            raise ValueError('Unsupported regular kernel payload version or kernel')
        def canonical(value):
            try:
                return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError('Regular kernel payload requires finite JSON data') from error
        saved = canonical(payload)
        if canonical(payload['binding']) != canonical(cls._payload_binding(medium, lattice)):
            raise ValueError('Regular kernel payload does not match material/lattice bindings')
        fields = {
            'spectral': {'pole_relative_tolerance', 'mode_batch', 'query_batch', 'cache_max_bytes'},
            'regular': {'shells', 'height', 'max_halvings', 'absolute_tolerance', 'max_shells',
                        'transverse_cutoff', 'quadrature_order', 'transverse_map', 'primary_cache_max_bytes'},
            'integration': set(DyadicIntegrationConfig().payload())}
        for name, expected in fields.items():
            if name == 'integration' and type(payload[name]) is dict and 'radial_map' in payload[name]:
                expected = expected | {'radial_map'}
            if name == 'integration' and type(payload[name]) is dict and 'spatial_rule' in payload[name]:
                expected = expected | {'spatial_rule'}
            if type(payload[name]) is not dict or set(payload[name]) != expected:
                raise ValueError(f'Regular kernel payload has invalid {name} fields')
            for key, value in payload[name].items():
                if key == 'transverse_map':
                    valid = type(value) is str and value in ('linear', 'sinh')
                elif key == 'radial_map':
                    valid = type(value) is str and value in ('material_split_v1','matched_spd_branch_v1','proportional_spd_branch_v1','spd_reference_branch_v1')
                elif key == 'spatial_rule':
                    valid = type(value) is str and value == 'proportional_spd_closed_form_v1'
                else:
                    valid = type(value) in (int, float) and math.isfinite(value)
                if not valid:
                    raise ValueError(f'Regular kernel payload has invalid {key}')
        normal = payload['normal']
        if (type(normal) is not list or len(normal) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in normal)):
            raise ValueError('Regular kernel payload has invalid normal')
        restored = cls(QuasiperiodicDyadicGreenKernel(lattice, medium, **payload['spectral']),
            **payload['regular'], normal=lattice.vectors.new_tensor(normal),
            integration=DyadicIntegrationConfig(**payload['integration']))
        # Constructors derive normals and component budgets; never silently
        # replace saved settings that disagree with those mandatory policies.
        if canonical(restored.payload()) != saved:
            raise ValueError('Regular kernel payload disagrees with derived configuration')
        return restored

    @property
    def identity(self):
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def evaluate(self, displacement, *, exclude_primary=False):
        from .dyadic_green import DyadicGreenKernel, DyadicConvergenceError
        self._validate()
        if exclude_primary is not True:
            raise ValueError('Regular spectral evaluation requires explicit primary exclusion')
        sample = self.lattice.vectors
        if (not isinstance(displacement, torch.Tensor) or displacement.ndim != 2 or displacement.shape[1] != 3
                or displacement.device != sample.device or displacement.dtype != sample.dtype
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Regular spectral queries require finite matching real CUDA [N,3] points')
        scale = torch.maximum(displacement.norm(dim=-1), sample.norm(dim=-1).max())
        roundoff = 128*torch.finfo(sample.dtype).eps*scale
        nearest = (displacement@self.lattice.reciprocal.T/(2*math.pi)).round()
        coincident = (displacement-nearest@sample).norm(dim=-1) <= roundoff
        if bool((coincident & (nearest != 0).any(-1)).any()):
            raise ValueError('Nonprimary lattice singularity requires explicit singular integration')
        on_plane = (displacement@self.normal).abs() <= roundoff
        if self.lattice.dimension == 3:
            layer = ((displacement@self.normal)/(sample[2]@self.normal)).round()
            wrapped = displacement-layer[:, None]*sample[2]
            on_plane = (wrapped@self.normal).abs() <= roundoff
        output = self.medium.epsilon_r.new_zeros(len(displacement), 6, 6)
        receipt = {'kernel_id': self.kernel_id, 'primary_image_excluded': True,
            'quantity': 'checked_nonprimary_regular_part', 'periodic_dimension': self.lattice.dimension,
            'absolute_tolerance_ohm_per_m2': self.absolute_tolerance,
            'query_count': len(displacement), 'source_plane_count': int(on_plane.sum()),
            'converged': False, 'certified_error_bound': False, 'physical_qualification': False}
        line = dict(normal=self.normal, transverse_cutoff=self.transverse_cutoff,
                    quadrature_order=self.quadrature_order, transverse_map=self.transverse_map)
        try:
            if bool(on_plane.any()):
                settings = dict(height=self.height, shells=self.shells, max_halvings=self.max_halvings,
                    absolute_tolerance=self.absolute_tolerance, integration=self.integration,
                    primary_cache_max_bytes=self.primary_cache_max_bytes)
                if self.lattice.dimension == 1:
                    value, record = self.kernel.estimate_line_plane_regular(displacement[on_plane], **settings, **line)
                elif self.lattice.dimension == 2:
                    value, record = self.kernel.estimate_open_plane_regular(displacement[on_plane],
                        max_shells=self.max_shells, **settings)
                else:
                    # G(r+m*a3) is Bloch-periodic; G(r)-G_free(r) is not.
                    # Keep primary ownership at the original physical query.
                    plane_points = displacement[on_plane]
                    plane_layers = layer[on_plane]
                    shifted = plane_layers != 0
                    value = output.new_zeros(len(plane_points), 6, 6)
                    record = {'translated_query_count': int(shifted.sum()),
                        'converged': False, 'primary_image_excluded': True}
                    if bool((~shifted).any()):
                        value[~shifted], record['principal_plane'] = self.kernel.estimate_stacked_plane_regular(
                            plane_points[~shifted], max_shells=self.max_shells, **settings)
                    if bool(shifted.any()):
                        local = wrapped[on_plane][shifted]
                        translated_settings = dict(settings, absolute_tolerance=self.absolute_tolerance/2)
                        local_regular, record['translated_plane'] = self.kernel.estimate_stacked_plane_regular(
                            local, max_shells=self.max_shells, **translated_settings)
                        free = DyadicGreenKernel(self.medium, self.integration, self.primary_cache_max_bytes)
                        local_primary, record['local_primary'] = free.evaluate(local)
                        original_primary, record['original_primary'] = free.evaluate(plane_points[shifted])
                        phase = torch.exp(1j*plane_layers[shifted]*(self.lattice.bloch@sample[2]))[:, None, None]
                        eta0 = math.sqrt(MU_0/EPSILON_0)
                        units = sample.new_tensor([1., 1., 1., eta0, eta0, eta0])
                        def norm(value):
                            return torch.linalg.matrix_norm(units[:, None]*value*units[None, :], ord='fro')
                        resolution = 64*torch.finfo(sample.dtype).eps*(
                            norm(local_regular)+norm(local_primary)+norm(original_primary))
                        record['translation_resolution_estimate'] = resolution.detach().cpu().tolist()
                        receipt['source_plane'] = record
                        if not bool((torch.isfinite(resolution) & (resolution <= self.absolute_tolerance/8)).all()):
                            receipt['failure_reason'] = 'primary_translation_precision_budget_exhausted'
                            raise PeriodicDyadicConvergenceError(receipt)
                        value[shifted] = phase*(local_regular+local_primary)-original_primary
                    record['converged'] = True
                output[on_plane] = value
                receipt['source_plane'] = record
            if bool((~on_plane).any()):
                points = displacement[~on_plane]
                component = self.absolute_tolerance/8
                settings = dict(shells=self.shells, relative_tolerance=0., absolute_tolerance=component)
                if self.lattice.dimension == 1:
                    periodic, record = self.kernel.estimate_line(points, **settings, **line)
                else:
                    estimate = (self.kernel.estimate_open if self.lattice.dimension == 2
                        else self.kernel.estimate_stacked)
                    attempts = receipt['off_plane_spectral_attempts'] = []
                    while True:
                        try:
                            periodic, record = estimate(points, **settings)
                        except PeriodicDyadicConvergenceError as error:
                            attempts.append(error.receipt)
                            # Only a failed successive-cutoff estimate can be
                            # refined. Pole, precision and other failures remain
                            # fatal, and the doubled comparison stays in budget.
                            check = error.receipt.get('cutoff_check', {})
                            if (error.receipt.get('cutoff_converged') is not False
                                    or check.get('passed') is not False):
                                raise
                            deeper = min(2*settings['shells'], self.max_shells//2)
                            if deeper <= settings['shells']:
                                raise
                            settings['shells'] = deeper
                        else:
                            attempts.append(record)
                            break
                receipt['off_plane_spectrum'] = record
                primary, record = DyadicGreenKernel(self.medium, self.integration, self.primary_cache_max_bytes).evaluate(points)
                receipt['primary'] = record
                eta0 = math.sqrt(MU_0/EPSILON_0)
                units = sample.new_tensor([1., 1., 1., eta0, eta0, eta0])
                def norm(value):
                    return torch.linalg.matrix_norm(units[:, None]*value*units[None, :], ord='fro')
                resolution = 64*torch.finfo(sample.dtype).eps*(norm(periodic)+norm(primary))
                receipt['subtraction_resolution_estimate'] = resolution.detach().cpu().tolist()
                if not bool((torch.isfinite(resolution) & (resolution <= component)).all()):
                    receipt['failure_reason'] = 'primary_subtraction_precision_budget_exhausted'
                    raise PeriodicDyadicConvergenceError(receipt)
                output[~on_plane] = periodic-primary
        except (PeriodicDyadicConvergenceError, DyadicConvergenceError) as error:
            if error.receipt is receipt:
                raise
            receipt.update(failure_reason='component_budget_exhausted', component_failure=error.receipt)
            raise PeriodicDyadicConvergenceError(receipt) from error
        except (ValueError, RuntimeError) as error:
            receipt.update(failure_reason='component_evaluation_failed', detail=str(error))
            raise PeriodicDyadicConvergenceError(receipt) from error
        if not bool(torch.isfinite(output).all()):
            receipt['failure_reason'] = 'nonfinite_regular_value'
            raise PeriodicDyadicConvergenceError(receipt)
        receipt['converged'] = True
        return output, receipt


REGULAR_DYADIC_KERNEL_TYPES = (LossyDyadicImageReference, QuasiperiodicDyadicRegularKernel)
