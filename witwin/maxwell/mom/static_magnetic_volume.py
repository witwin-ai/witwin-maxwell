"""Internal P0 magnetic contrast VIE with analytic tensor static face layers.

H=H_J+H_background-grad(G_ext * -div(P)), P=(mu-mu_ext)H.
The constitutive law is collocated at cell centroids. Surface transmission and
pointwise constitutive accuracy require refinement; this is not a certificate.
"""
import torch
from .static_tensor_green import StaticTensorGreen
from .static_tensor_polarization import StaticTensorPolarization
from .magnetostatic import cell_tensor_magnetostatic_current_fields,cell_current_continuity
from .fields import classify_queries


class MagneticVolumeContrastResponse:
    """Owned dense regional response to induced currents and uniform H.

    Every body must have an exactly bound tetrahedral volume. The exterior is
    homogeneous real SPD. No cut surfaces or scalar-potential periods are used;
    currents enter through their global particular field. Nonconservative
    currents retain the particular field's Ampere defect.
    """
    def __init__(self,system,bindings,currents,exterior,interiors,*,background_h=None,
                 pair_batch=4096,max_matrix_bytes=256*1024**2):
        system.assert_current()
        if type(pair_batch) is not int or pair_batch<1 or type(max_matrix_bytes) is not int or max_matrix_bytes<1:
            raise ValueError('Magnetic volume budgets must be positive integers')
        self.system=system;self.pair_batch=pair_batch
        self.bindings=tuple(bindings)
        currents=tuple(currents);interiors=tuple(interiors)
        if len(self.bindings)!=len(system.bodies) or len(currents)!=len(self.bindings) or len(interiors)!=len(self.bindings):
            raise ValueError('Magnetic contrast requires one volume, current and permeability per body')
        for region,binding in enumerate(self.bindings,1):
            binding.assert_current()
            if binding.surface_system is not system or binding.volume.region_id!=region:
                raise ValueError('Magnetic volumes must bind the exact system in region order')
        self.exterior=StaticTensorGreen(exterior)
        self.interiors=tuple(StaticTensorGreen(mu) for mu in interiors)
        for kernel in (self.exterior,*self.interiors):
            if kernel.coefficient.device!=system.device or kernel.coefficient.dtype!=system.dtype:
                raise ValueError('Magnetic coefficients must match surface geometry')
        self.currents=tuple(current.clone() for current in currents)
        for binding,current in zip(self.bindings,self.currents):
            self.exterior._validate(current,(len(binding.volume.cells),3))
        self.background=exterior.new_zeros(3) if background_h is None else background_h.clone()
        self.exterior._validate(self.background,(3,))
        self.counts=tuple(len(b.volume.cells) for b in self.bindings)
        count=sum(self.counts);required=(3*count)**2*exterior.element_size()
        if required>max_matrix_bytes:
            raise ValueError('Magnetic contrast retained matrix exceeds its explicit budget')
        self.centers=torch.cat([b.volume.vertices[b.volume.cells].mean(1) for b in self.bindings])
        from .static_tensor_volume import cell_tensor_static_potentials
        action=torch.cat([cell_tensor_static_potentials(binding.volume,self.centers,self.exterior,
            pair_batch=pair_batch)[2] for binding in self.bindings],dim=1)
        self.contrast=torch.cat([(k.coefficient-self.exterior.coefficient).expand(n,3,3)
                                 for k,n in zip(self.interiors,self.counts)])
        blocks=torch.einsum('qtij,tjk->qitk',action.reshape(count,count,3,3),self.contrast)
        self.matrix=torch.eye(3*count,device=system.device,dtype=system.dtype)-blocks.reshape(3*count,3*count)
        rhs=self._source(self.centers)[0]
        self.source_field=rhs.clone()
        if not bool(torch.isfinite(self.matrix).all() & torch.isfinite(rhs).all()):
            raise ValueError('Magnetic contrast matrix is not representable')
        self.cell_fields=torch.linalg.solve(self.matrix,rhs.flatten()).reshape(count,3)
        residual=self.matrix@self.cell_fields.flatten()-rhs.flatten()
        if not bool(torch.isfinite(self.cell_fields).all() & torch.isfinite(residual).all()):
            raise ValueError('Magnetic contrast solution is not finite')
        def norm(value):
            scale=value.abs().max()
            return scale*(value/scale).norm() if bool(scale>0) else scale
        error=norm(residual)
        threshold=512*torch.finfo(system.dtype).eps*(norm(self.matrix)*norm(self.cell_fields)+norm(rhs))
        jumps=torch.stack([cell_current_continuity(b.volume,j)[
            'area_weighted_jump_rms_A_per_m2'] for b,j in zip(self.bindings,self.currents)])
        if not bool(torch.isfinite(error) & torch.isfinite(threshold) & torch.isfinite(jumps).all()):
            raise ValueError('Magnetic contrast residual diagnostics are not representable')
        if not bool(error<=threshold):
            raise RuntimeError('Magnetic contrast solve failed its true residual check')
        polarization=torch.einsum('tij,tj->ti',self.contrast,self.cell_fields)
        self.sources=tuple(StaticTensorPolarization(b,self.exterior.coefficient,p,pair_batch=pair_batch)
                           for b,p in zip(self.bindings,polarization.split(self.counts)))
        self.receipt=dict(formulation='regional_magnetic_p0_contrast_collocation_v1',
            algebraic_residual_norm=float(error.detach()),threshold=float(threshold.detach()),
            retained_matrix_bytes=required,pair_batch=pair_batch,physical_qualification=False,
            pointwise_constitutive_accuracy_verified=False,interface_continuity_verified=False,
            source_current_jump_rms_A_per_m2=jumps.detach().cpu().tolist())
        self._binding=self._state()

    def _state(self):
        return (id(self.system),self.pair_batch,self.counts,tuple(map(id,self.bindings)),
                tuple(map(id,(self.exterior,*self.interiors,*self.sources))),tuple((id(t),t._version) for t in
                (*self.currents,self.background,self.centers,self.contrast,self.matrix,self.cell_fields,self.source_field)))

    def assert_current(self):
        self.system.assert_current()
        if self._binding!=self._state():raise ValueError('Magnetic volume response changed')
        for binding in self.bindings:binding.assert_current()
        for kernel in (self.exterior,*self.interiors):kernel.assert_current()
        for source in self.sources:source.assert_current()

    def _source(self,points):
        field=self.background.expand_as(points).clone();defect=torch.zeros_like(points)
        for binding,current in zip(self.bindings,self.currents):
            h,_,receipt=cell_tensor_magnetostatic_current_fields(binding.volume,current,points,
                self.exterior.coefficient,pair_batch=self.pair_batch)
            field=field+h;defect=defect+receipt['ampere_defect_A_per_m2']
        if not bool(torch.isfinite(field).all() & torch.isfinite(defect).all()):
            raise ValueError('Magnetic current source sum is not representable')
        return field,defect

    def fields(self,points,region_id):
        """Off-face H and constitutive B=mu_region H, with retained source defect.

        P0 magnetization enforces the constitutive equation at collocation points
        only. These fields are not asserted to satisfy exact interface traces.
        """
        self.assert_current();self.exterior._validate(points,(None,3))
        region=self.system.region(region_id)
        for body in self.system.bodies:
            inside,_=classify_queries(body.mesh,points)
            if bool((inside!=(body.region_id==region)).any()):
                raise ValueError('Magnetic query does not lie in its declared region')
        h,defect=self._source(points)
        for source in self.sources:h=h+source.fields(points)[1]
        mu=self.exterior.coefficient if region==0 else self.interiors[region-1].coefficient
        b=h@mu.T
        if not bool(torch.isfinite(h).all() & torch.isfinite(b).all()):
            raise ValueError('Magnetic volume field is not representable')
        return h,b,dict(ampere_defect_A_per_m2=defect,physical_qualification=False,
                       pointwise_constitutive_accuracy_verified=False)
