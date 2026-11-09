"""Versioned Bravais geometry and a lossy direct-image reference kernel."""
from dataclasses import dataclass, field
import math
import torch
from .kernels import FREE_SPACE


@dataclass(frozen=True, eq=False)
class LatticeSpec:
    """CUDA row lattice vectors in meters; real Bloch vector in radians/meter.

    With exp(-i omega t), image weights are exp(+i beta.R), so the infinite
    outgoing sum obeys G(r+R)=exp(+i beta.R)G(r). Open-axis components of beta
    are excluded: the Bloch vector describes translations in the lattice span.
    Geometry gradients are outside this fixed-geometry contract.
    """
    vectors: torch.Tensor
    bloch: torch.Tensor
    version: str = field(default='bravais_bloch_v1', init=False)

    def __post_init__(self):
        a,b = self.vectors,self.bloch
        if (not isinstance(a,torch.Tensor) or a.device.type != 'cuda'
                or a.dtype not in (torch.float32,torch.float64) or a.ndim != 2
                or a.shape[1] != 3 or not 1 <= len(a) <= 3
                or not isinstance(b,torch.Tensor) or b.shape != (3,)
                or b.device != a.device or b.dtype != a.dtype
                or not bool(torch.isfinite(a).all() and torch.isfinite(b).all())):
            raise ValueError('Lattice requires finite matching real CUDA [D,3] vectors and [3] Bloch vector')
        if a.requires_grad or b.requires_grad:
            raise NotImplementedError('Trainable lattice/Bloch compilation is not implemented')
        singular = torch.linalg.svdvals(a)
        if bool(singular[-1] <= 128*torch.finfo(a.dtype).eps*singular[0]):
            raise ValueError('Lattice vectors must be linearly independent and numerically resolved')
        a,b = a.clone(),b.clone()
        q,r = torch.linalg.qr(a.T,mode='reduced')
        reciprocal = 2*math.pi*torch.linalg.solve_triangular(r,q.T,upper=True)
        projection = (b@a.T)@reciprocal/(2*math.pi)
        tolerance = 512*torch.finfo(a.dtype).eps*b.norm()
        if bool((b-projection).norm() > tolerance):
            raise ValueError('Bloch vector must lie in the periodic lattice span')
        object.__setattr__(self,'vectors',a)
        object.__setattr__(self,'bloch',b)
        object.__setattr__(self,'reciprocal',reciprocal)
        object.__setattr__(self,'cell_measure',singular.prod())
        object.__setattr__(self,'_versions',self._state())

    def _state(self):
        return tuple((id(t),t._version) for t in
            (self.vectors,self.bloch,self.reciprocal,self.cell_measure))

    def assert_current(self):
        if self._state() != self._versions:
            raise RuntimeError('Lattice numerical state was mutated')

    @property
    def dimension(self):
        return len(self.vectors)

    def translations(self,indices):
        self.assert_current()
        if (not isinstance(indices,torch.Tensor) or indices.device != self.vectors.device
                or indices.dtype != torch.int64 or indices.ndim < 1
                or indices.shape[-1] != self.dimension):
            raise ValueError('Lattice indices require matching CUDA int64 [...,D] values')
        return indices.to(self.vectors.dtype)@self.vectors

    def phase(self,indices):
        return torch.exp(1j*(self.translations(indices)@self.bloch))

    def matches_incident(self,wavevector,*,phase_tolerance=None):
        """Match tangential wavevector modulo reciprocal translations."""
        self.assert_current()
        if (not isinstance(wavevector,torch.Tensor) or wavevector.shape != (3,)
                or wavevector.device != self.bloch.device or wavevector.dtype != self.bloch.dtype
                or not bool(torch.isfinite(wavevector).all())):
            raise ValueError('Incident wavevector must match the finite real CUDA Bloch vector')
        if phase_tolerance is None:
            phase_tolerance = max(1e-10,128*torch.finfo(self.bloch.dtype).eps)
        if not math.isfinite(phase_tolerance) or phase_tolerance <= 0:
            raise ValueError('Phase tolerance must be finite and positive')
        cycles = (wavevector-self.bloch)@self.vectors.T/(2*math.pi)
        return bool(((cycles-cycles.round()).abs()*2*math.pi <= phase_tolerance).all())

    def receipt(self):
        self.assert_current()
        return {'version':self.version,'vectors_m':self.vectors.detach().cpu().tolist(),
                'bloch_rad_per_m':self.bloch.detach().cpu().tolist(),
                'dimension':self.dimension,'translation_phase':'exp(+i beta.R)',
                'phasor_convention':'exp(-i omega t)'}


@dataclass(frozen=True)
class LossyImageReference:
    """Finite direct sum, independent reference for future Ewald convergence.

    This is restricted to positive absorption and carries no truncation
    certificate. It is not admitted as a production periodic operator kernel.
    """
    lattice: LatticeSpec
    shells: int
    image_batch: int = 64
    kernel_id: str = field(default='lossy_quasiperiodic_image_reference',init=False)
    version: str = field(default='direct_image_reference_v1',init=False)
    local_singularity: str = field(default='coulomb_1_over_4pi_r',init=False)

    def __post_init__(self):
        if not isinstance(self.lattice,LatticeSpec):
            raise TypeError('Image reference requires LatticeSpec')
        if type(self.shells) is not int or self.shells < 1 or type(self.image_batch) is not int or self.image_batch < 1:
            raise ValueError('Image shell and batch counts must be positive integers')

    def evaluate(self,displacement,k,*,exclude_zero=False):
        self.lattice.assert_current()
        if (not isinstance(displacement,torch.Tensor) or displacement.ndim < 1
                or displacement.shape[-1] != 3 or displacement.dtype != self.lattice.vectors.dtype
                or displacement.device != self.lattice.vectors.device
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Displacements must match finite real CUDA lattice coordinates')
        k = complex(k)
        if not math.isfinite(k.real) or not math.isfinite(k.imag) or k.imag <= 0:
            raise ValueError('Direct-image reference requires positive absorption')
        width = 2*self.shells+1
        dimension = self.lattice.dimension
        count = width**dimension
        dtype = torch.complex128 if displacement.dtype == torch.float64 else torch.complex64
        value = torch.zeros(displacement.shape[:-1],device=displacement.device,dtype=dtype)
        gradient = torch.zeros(displacement.shape,device=displacement.device,dtype=dtype)
        for start in range(0,count,self.image_batch):
            flat = torch.arange(start,min(start+self.image_batch,count),device=displacement.device)
            chunk = torch.stack([(flat//width**power)%width for power in reversed(range(dimension))],-1)-self.shells
            if exclude_zero:
                chunk = chunk[(chunk != 0).any(-1)]
            vectors = self.lattice.translations(chunk)
            phase = self.lattice.phase(chunk)
            g,dg = FREE_SPACE.evaluate(displacement[...,None,:]-vectors,k)
            value = value+(g*phase).sum(-1)
            gradient = gradient+(dg*phase[:,None]).sum(-2)
        return value,gradient

    def receipt(self):
        return {'kernel_id':self.kernel_id,'version':self.version,'lattice':self.lattice.receipt(),
                'shells':self.shells,'image_batch':self.image_batch,
                'physical_qualification':False,'truncation_certified':False,
                'support':'strictly_positive_absorption_reference_only'}


class PeriodicConvergenceError(RuntimeError):
    def __init__(self,receipt):
        self.receipt = receipt
        super().__init__('Periodic Ewald estimate failed: '+receipt['failure_reason'])


@dataclass(frozen=True)
class EwaldHelmholtzKernel:
    """Quasiperiodic scalar kernel with native real and spectral heat integrals.

    Real part: (2 pi**1.5)^-1 integral_eta^infinity
    exp(-r^2 t^2+k^2/(4 t^2)) dt. The Coulomb heat tail is analytic;
    its entire-in-k correction uses CUDA Gauss quadrature. Reciprocal part:
    exp(i q.r) exp(-(q^2-k^2)/(4 eta^2))/(V (q^2-k^2)), q=beta+B n.
    True resolvent poles reject. No limiting-absorption claim is implicit.
    """
    lattice: LatticeSpec
    eta: float
    real_shells: int = 3
    reciprocal_shells: int = 3
    quadrature_order: int = 64
    image_batch: int = 32
    query_batch: int = 256
    pole_relative_tolerance: float = 1e-10
    kernel_id: str = field(default='quasiperiodic_helmholtz',init=False)
    version: str = field(default='ewald_heat_v1',init=False)
    local_singularity: str = field(default='coulomb_1_over_4pi_r',init=False)

    def __post_init__(self):
        if not isinstance(self.lattice,LatticeSpec):
            raise ValueError('Ewald requires a Bravais lattice')
        for name in ('eta','pole_relative_tolerance'):
            value = getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be finite and positive')
        for name in ('real_shells','reciprocal_shells','quadrature_order','image_batch','query_batch'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(name+' must be a positive integer')

    def _indices(self,shells,device):
        width=2*shells+1
        count=width**self.lattice.dimension
        for start in range(0,count,self.image_batch):
            flat=torch.arange(start,min(start+self.image_batch,count),device=device)
            yield torch.stack([(flat//width**power)%width for power in reversed(range(self.lattice.dimension))],-1)-shells

    def _guard(self,k):
        self.lattice.assert_current()
        if not math.isfinite(k.real) or not math.isfinite(k.imag) or k.imag < 0:
            raise ValueError('Periodic passive resolvent requires finite k with nonnegative absorption')
        smallest=float(torch.linalg.svdvals(self.lattice.reciprocal)[-1])
        radius=(abs(k)+float(self.lattice.bloch.norm()))/smallest
        # Every potentially resonant reciprocal vector must be inside the scan.
        if self.reciprocal_shells < math.ceil(radius):
            raise ValueError('Reciprocal cutoff does not cover the complete spectral pole search')
        for indices in self._indices(self.reciprocal_shells,self.lattice.vectors.device):
            q=indices.to(self.lattice.vectors.dtype)@self.lattice.reciprocal+self.lattice.bloch
            q2=q.square().sum(-1)
            scale=torch.maximum(q2,q2.new_full(q2.shape,max(abs(k)**2,smallest**2)))
            if bool(((q2-k*k).abs() <= self.pole_relative_tolerance*scale).any()):
                raise ValueError('Periodic resolvent is at or unresolved near a spectral pole')

    def _real(self,points,k,*,images="all"):
        from .quadrature import gauss_rule
        u,w=gauss_rule(self.quadrature_order,device=points.device,dtype=points.dtype)
        eta=self.eta
        t=eta/u
        a=(k/(2*eta))**2
        z=a*u.square()
        # Integrate the first three heat moments analytically. Otherwise the
        # near-origin gradient needs quadrature nodes at u comparable to eta*r.
        polynomial=z+z.square()/2+z**3/6
        tail=z**4/24
        series=tail
        for power in range(5,19):
            tail=tail*z/power
            series=series+tail
        correction=torch.where(z.abs()<.5,series,torch.expm1(z)-polynomial)
        factor=eta*w/u.square()/(2*math.pi**1.5)
        value=points.new_zeros(len(points),dtype=torch.complex128 if points.dtype==torch.float64 else torch.complex64)
        gradient=value.new_zeros(len(points),3)
        chunks=(torch.zeros((1,self.lattice.dimension),device=points.device,dtype=torch.int64),) if images=='zero' else self._indices(self.real_shells,points.device)
        for indices in chunks:
            if images=='nonzero':
                indices=indices[(indices!=0).any(-1)]
            delta=points[:,None,:]-self.lattice.translations(indices)
            radius=torch.linalg.vector_norm(delta,dim=-1)
            if bool((radius == 0).any()):
                raise ValueError('Coincident periodic nodes require singular quadrature')
            heat=torch.exp(-(radius[...,None]*t)**2)
            remainder=(heat*correction*factor).sum(-1)
            radial=(-2*t.square()*heat*correction*factor).sum(-1)
            erfc=torch.erfc(eta*radius)
            x=eta*radius
            exponential=torch.exp(-x.square())
            moments=[math.sqrt(math.pi)*erfc/(2*x),exponential-math.sqrt(math.pi)*x*erfc]
            for power in (1,2):
                moments.append((exponential-2*x.square()*moments[-1])/(2*power+1))
            coefficient=eta/(2*math.pi**1.5)
            for power in (1,2,3):
                weight=a**power/math.factorial(power)
                remainder=remainder+coefficient*weight*moments[power]
                radial=radial-2*eta**2*coefficient*weight*moments[power-1]
            base=erfc/(4*math.pi*radius)
            base_gradient=(-erfc/radius**3-2*eta/math.sqrt(math.pi)*torch.exp(-(eta*radius)**2)/radius**2)/(4*math.pi)
            phase=self.lattice.phase(indices)
            value=value+((base+remainder)*phase).sum(-1)
            gradient=gradient+((base_gradient+radial)[...,None]*delta*phase[:,None]).sum(-2)
        return value,gradient

    def _spectral(self,points,k):
        if self.lattice.dimension < 3:
            return self._open_spectral(points,k)
        value=points.new_zeros(len(points),dtype=torch.complex128 if points.dtype==torch.float64 else torch.complex64)
        gradient=value.new_zeros(len(points),3)
        for indices in self._indices(self.reciprocal_shells,points.device):
            q=indices.to(points.dtype)@self.lattice.reciprocal+self.lattice.bloch
            denominator=q.square().sum(-1)-k*k
            amplitude=torch.exp(-denominator/(4*self.eta**2))/(self.lattice.cell_measure*denominator)
            term=torch.exp(1j*(points@q.T))*amplitude
            value=value+term.sum(-1)
            gradient=gradient+(term[...,None]*(1j*q)).sum(-2)
        return value,gradient

    def _open_spectral(self,points,k):
        """Deform the low-heat contour to the outgoing sheet, mode by mode.

        The radial leg starts at zero where a/t^2 is negative real. The arc
        returns to positive eta. This analytically continues propagating modes
        without divergent real-axis quadrature or numerical complex erfc.
        """
        from .quadrature import gauss_rule
        dimension=self.lattice.dimension
        u,w=gauss_rule(self.quadrature_order,device=points.device,dtype=points.dtype)
        parallel=(points@self.lattice.reciprocal.T/(2*math.pi))@self.lattice.vectors
        perpendicular=points-parallel
        rho2=perpendicular.square().sum(-1)
        dtype=torch.complex128 if points.dtype==torch.float64 else torch.complex64
        value=points.new_zeros(len(points),dtype=dtype)
        gradient=value.new_zeros(len(points),3)
        normalization=1/(2*math.pi**((3-dimension)/2)*self.lattice.cell_measure)
        for indices in self._indices(self.reciprocal_shells,points.device):
            q=indices.to(points.dtype)@self.lattice.reciprocal+self.lattice.bloch
            a=k*k-q.square().sum(-1)
            angle=torch.angle(a)
            if k.real < 0:
                angle=torch.where(angle>0,angle-2*math.pi,angle)
                theta=(angle+math.pi)/2
            else:
                theta=(angle-math.pi)/2
            radial_end=self.eta*torch.exp(1j*theta)
            legs=((radial_end[:,None]*u,radial_end[:,None].expand(-1,len(u))),
                  (self.eta*torch.exp(1j*theta[:,None]*(1-u)),None))
            integral=value.new_zeros(len(points),len(q))
            radial=integral.clone()
            for t,dt in legs:
                if dt is None:
                    dt=-1j*theta[:,None]*t
                heat=torch.exp(-rho2[:,None,None]*t.square()+a[None,:,None]/(4*t.square()))
                weighted=heat*(t**(-dimension)*dt*w)
                integral=integral+weighted.sum(-1)
                radial=radial+(weighted*(-2*t.square())).sum(-1)
            phase=torch.exp(1j*(points@q.T))*normalization
            value=value+(phase*integral).sum(-1)
            gradient=gradient+(phase[...,None]*(1j*q[None]*integral[...,None]
                +perpendicular[:,None,:]*radial[...,None])).sum(-2)
        return value,gradient

    def evaluate(self,displacement,k):
        k=complex(k)
        self._guard(k)
        if (not isinstance(displacement,torch.Tensor) or displacement.ndim < 1
                or displacement.shape[-1] != 3 or displacement.dtype != self.lattice.vectors.dtype
                or displacement.device != self.lattice.vectors.device
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Displacements must match finite real CUDA lattice coordinates')
        points=displacement.reshape(-1,3)
        shifts=torch.floor(points@self.lattice.reciprocal.T/(2*math.pi)+0.5)
        translations=shifts@self.lattice.vectors
        wrapped=points-translations
        phase=torch.exp(1j*(translations@self.lattice.bloch))
        values,gradients=[],[]
        for start in range(0,len(points),self.query_batch):
            query=wrapped[start:start+self.query_batch]
            real=self._real(query,k)
            spectral=self._spectral(query,k)
            values.append(real[0]+spectral[0])
            gradients.append(real[1]+spectral[1])
        if not values:
            dtype=torch.complex128 if points.dtype==torch.float64 else torch.complex64
            return (points.new_empty(displacement.shape[:-1],dtype=dtype),points.new_empty(displacement.shape,dtype=dtype))
        value=torch.cat(values)*phase
        gradient=torch.cat(gradients)*phase[:,None]
        if not bool(torch.isfinite(value).all() and torch.isfinite(gradient).all()):
            raise ValueError('Ewald split overflowed or produced nonfinite values; change the splitting parameter')
        return value.reshape(displacement.shape[:-1]),gradient.reshape_as(displacement)

    def _regular_radial_coefficients(self,points,k,count):
        """Coefficients of the entire primary heat remainder in powers of r^2."""
        from .quadrature import gauss_rule
        u,w=gauss_rule(self.quadrature_order,device=points.device,dtype=points.dtype)
        eta=self.eta
        a=(k/(2*eta))**2
        integral=(torch.expm1(a*u.square())*w/u.square()).sum()
        c=eta/(2*math.pi**1.5)*(integral-1)-1j*k/(4*math.pi)
        source=c.new_tensor(eta**3/math.pi**1.5)*torch.exp(c.new_tensor(a))
        coefficients=[c]
        for power in range(count-1):
            c=(source-k*k*c)/(2*(power+1)*(2*power+3))
            coefficients.append(c)
            source=source*(-eta**2)/(power+1)
        return tuple(coefficients)

    def _regular_zero(self,points,k):
        """Smooth zero-image heat remainder via its radial Helmholtz series.

        F=G_real,zero-G_free obeys (Delta+k^2)F=S exp(-eta^2 r^2).
        Its entire even radial series avoids subtracting singular quantities,
        including at r=0. Outside the local series disk direct subtraction is
        well scaled. This is the primary zero image, not a wrapped nearest one.
        """
        coefficients=self._regular_radial_coefficients(points,k,32)
        c=coefficients[0]
        eta=self.eta
        radius2=points.square().sum(-1)
        local=radius2*max(eta**2,abs(k)**2)<=1
        output=c.new_zeros(len(points))
        gradient=c.new_zeros(len(points),3)
        x=radius2[local]
        value=torch.zeros_like(x,dtype=c.dtype)
        radial=torch.zeros_like(value)
        for power in reversed(range(32)):
            value=value*x+coefficients[power]
            if power:
                radial=radial*x+2*power*coefficients[power]
        output[local]=value
        gradient[local]=radial[:,None]*points[local]
        distant=points[~local]
        if len(distant):
            real=self._real(distant,k,images='zero')
            free=FREE_SPACE.evaluate(distant,k)
            output[~local]=real[0]-free[0]
            gradient[~local]=real[1]-free[1]
        return output,gradient

    def regular_part(self,displacement,k):
        """Return G_periodic-G_free and its gradient, finite at the origin.

        The removed free-space singularity always uses the original primary
        image. Distant periodic queries are wrapped before subtraction; near
        the primary origin a stable series avoids singular cancellation.
        Coincidences with any other image remain singular and reject.
        """
        k=complex(k)
        self._guard(k)
        if (not isinstance(displacement,torch.Tensor) or displacement.ndim<1
                or displacement.shape[-1]!=3 or displacement.dtype!=self.lattice.vectors.dtype
                or displacement.device!=self.lattice.vectors.device
                or not bool(torch.isfinite(displacement).all())):
            raise ValueError('Displacements must match finite real CUDA lattice coordinates')
        points=displacement.reshape(-1,3)
        values,gradients=[],[]
        for query in points.split(self.query_batch):
            fractional=query@self.lattice.reciprocal.T/(2*math.pi)
            translated=(fractional.abs()>.5).any(-1)
            dtype=torch.complex128 if query.dtype==torch.float64 else torch.complex64
            value=query.new_empty(len(query),dtype=dtype)
            gradient=query.new_empty(query.shape,dtype=dtype)
            if bool(translated.any()):
                # Far from the primary singularity subtraction is safe. Wrapping
                # retains the nearest image even beyond the real-space cutoff.
                periodic=self.evaluate(query[translated],k)
                primary=FREE_SPACE.evaluate(query[translated],k)
                value[translated]=periodic[0]-primary[0]
                gradient[translated]=periodic[1]-primary[1]
            if bool((~translated).any()):
                local=query[~translated]
                images=self._real(local,k,images='nonzero')
                spectral=self._spectral(local,k)
                zero=self._regular_zero(local,k)
                value[~translated]=images[0]+spectral[0]+zero[0]
                gradient[~translated]=images[1]+spectral[1]+zero[1]
            values.append(value)
            gradients.append(gradient)
        if not values:
            dtype=torch.complex128 if points.dtype==torch.float64 else torch.complex64
            return points.new_empty(displacement.shape[:-1],dtype=dtype),points.new_empty(displacement.shape,dtype=dtype)
        value,gradient=torch.cat(values),torch.cat(gradients)
        if not bool(torch.isfinite(value).all() and torch.isfinite(gradient).all()):
            raise ValueError('Periodic regular part produced nonfinite values')
        return value.reshape(displacement.shape[:-1]),gradient.reshape_as(displacement)

    def estimate(self,displacement,k,*,relative_tolerance=1e-9,absolute_tolerance=1e-11,regular=False):
        """Independently deepen each cutoff; successive estimates are not bounds."""
        from dataclasses import replace
        if (not all(math.isfinite(v) and v>=0 for v in (relative_tolerance,absolute_tolerance))
                or relative_tolerance+absolute_tolerance==0):
            raise ValueError('Ewald estimate tolerances must be finite and nonzero')
        method="regular_part" if regular else "evaluate"
        baseline=getattr(self,method)(displacement,k)
        receipt={**self.receipt(),'checks':[],'converged':False,'failure_reason':None}
        receipt['quantity']='periodic_minus_primary_free_space' if regular else 'periodic_green'
        for axis in ('real_shells','reciprocal_shells','quadrature_order'):
            refined=getattr(replace(self,**{axis:2*getattr(self,axis)}),method)(displacement,k)
            checks=[]
            for name,current,previous in zip(('G','gradient'),refined,baseline):
                error=(current-previous).abs()
                threshold=absolute_tolerance+relative_tolerance*current.abs()
                passed=bool((error<=threshold).all())
                checks.append({'quantity':name,'passed':passed,
                    'maximum_absolute_difference':float(error.detach().max()) if error.numel() else 0.})
            receipt['checks'].append({'axis':axis,'checks':checks})
        receipt['converged']=all(c['passed'] for row in receipt['checks'] for c in row['checks'])
        if not receipt['converged']:
            receipt['failure_reason']='independent_cutoff_estimate_failed'
            raise PeriodicConvergenceError(receipt)
        return baseline,receipt

    def receipt(self):
        return {'kernel_id':self.kernel_id,'version':self.version,'lattice':self.lattice.receipt(),
                'eta_per_m':self.eta,'real_shells':self.real_shells,'reciprocal_shells':self.reciprocal_shells,
                'quadrature_order':self.quadrature_order,'image_batch':self.image_batch,'query_batch':self.query_batch,
                'pole_relative_tolerance':self.pole_relative_tolerance,
                'physical_qualification':False,'certified_error_bound':False,
                'support':'passive_outgoing_off_anomalies_and_spectral_poles',
                'local_singularity':self.local_singularity,'phasor_convention':'exp(-i omega t)'}
