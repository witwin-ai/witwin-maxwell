"""Physical-panel AC static integrals for elliptic complex constitutive samples.

This is the static singular part of an AC Green function, not DC material
admission. Real-SPD panels retain their analytic implementation. Complex panels
use the existing passive point-kernel branch and explicit odd-kernel face PV.
"""
import math
import torch
from ..constants import EPSILON_0, MU_0
from .tensor_medium import TensorMedium
from .dyadic_green import _electrostatic_components
from .static_tensor_green import _static_panel_geometry
from .quadrature import distance_scaled_duffy_fan,distance_scaled_line_rule


class ComplexStaticPanelConvergenceError(RuntimeError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('Complex static panel integration exhausted its order budget')


class ComplexStaticTensorGreen:
    """Physical electric/magnetic coefficient with an owned AC medium.

Admission requires a resolved definite real part (either sign) or strictly
positive imaginary part of the transpose-symmetric relative coefficient.
These sufficient global ellipticity conditions exclude real characteristic
cones; finite kernel samples alone would not establish this. Other passive
samples require a separate limiting formulation and remain unsupported.

Relative tolerance compares successive orders per potential and gradient
vector. Absolute tolerance has units of integrated potential; the gradient
test divides it by the physical panel diameter. This estimator is not a bound.
"""
    def __init__(self,medium,*,constitutive='electric',relative_tolerance=1e-8,
                 absolute_tolerance=0.,initial_order=8,max_order=128,pair_batch=8):
        if type(medium) is not TensorMedium:
            raise TypeError('Complex static integration requires an owned TensorMedium')
        medium.assert_current()
        if constitutive not in ('electric','magnetic'):
            raise ValueError('constitutive must be electric or magnetic')
        for value in (relative_tolerance,absolute_tolerance):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Static panel tolerances must be finite and nonnegative')
        if relative_tolerance==0 and absolute_tolerance==0:
            raise ValueError('Static panel integration needs a nonzero tolerance')
        if (type(initial_order) is not int or initial_order<2 or type(max_order) is not int
                or max_order<=initial_order or type(pair_batch) is not int or pair_batch<1):
            raise ValueError('Static panel integration needs increasing integer orders and a positive batch')
        self.medium=medium;self.constitutive=constitutive
        self.units=EPSILON_0 if constitutive=='electric' else MU_0
        relative=medium.epsilon_r if constitutive=='electric' else medium.mu_r
        symmetric=(relative+relative.mT)/2
        scale=torch.linalg.matrix_norm(symmetric)
        margin=64*torch.finfo(relative.real.dtype).eps*scale
        lower=torch.stack(tuple(torch.linalg.eigvalsh(part).amin()
            for part in (symmetric.real,-symmetric.real,symmetric.imag)))
        if not bool(torch.isfinite(scale) & (lower>margin).any()):
            raise NotImplementedError('Complex static panels require resolved definite real or strictly lossy symmetric parts')
        self.coefficient=self.units*symmetric
        self.relative_tolerance=float(relative_tolerance);self.absolute_tolerance=float(absolute_tolerance)
        self.initial_order=initial_order;self.max_order=max_order;self.pair_batch=pair_batch
        self._binding=self._state()

    def _state(self):
        return (id(self.medium),self.constitutive,self.units,id(self.coefficient),self.coefficient._version,
                self.relative_tolerance,self.absolute_tolerance,self.initial_order,self.max_order,self.pair_batch)

    def assert_current(self):
        self.medium.assert_current()
        if self._state()!=self._binding:
            raise ValueError('Complex static kernel state changed')

    def _validate(self,value,shape):
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.device!=self.coefficient.device
                or value.dtype!=self.coefficient.real.dtype or value.ndim!=len(shape)
                or any(n is not None and value.shape[i]!=n for i,n in enumerate(shape))
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Complex static queries require finite matching real CUDA coordinates')

    def evaluate(self,displacements):
        self._validate(displacements,(None,3))
        potential,v,squared,_=_electrostatic_components(displacements,self.medium,constitutive=self.constitutive)
        if not bool(torch.isfinite(squared).all() & torch.isfinite(v).all() & (potential.abs()>0).all()):
            raise ValueError('Complex static point geometry is not representable')
        potential=potential/self.units
        gradient=-potential[:,None]*v/squared[:,None]
        if not bool(torch.isfinite(potential).all() & torch.isfinite(gradient).all()):
            raise ValueError('Complex static point evaluation is not representable')
        return potential,gradient

    def panel_integrals(self,observers,triangles,*,principal_value=None):
        potential,gradient,_=self.integrate_panels(observers,triangles,principal_value=principal_value)
        return potential,gradient

    def integrate_panels(self,observers,triangles,*,principal_value=None):
        """Return physical single layers, observer gradients and an estimator receipt.

        On-face potential and PV gradient follow boundary line potentials
        and the zero conormal derivative of a coplanar homogeneous kernel.
        Off-face queries use radial and edge distance-scaled signed fans.
        No offset, cutoff, fitted contact or complex geometry is used.
        """
        self._validate(observers,(None,3));self._validate(triangles,(len(observers),3,3))
        observers,triangles,_,_,pv=_static_panel_geometry(observers,triangles,principal_value)
        diameter=(triangles-triangles.roll(1,dims=1)).norm(dim=-1).amax(-1)
        receipt=dict(representation='complex_static_physical_panel_v2',constitutive=self.constitutive,
            panel_rule='edge_radial_grading_and_face_boundary_lines',
            initial_order=self.initial_order,max_order=self.max_order,
            ellipticity_admission='definite_real_or_strictly_positive_imaginary_symmetric_part',
            estimator='successive_order_per_potential_and_gradient',relative_tolerance=self.relative_tolerance,
            absolute_tolerance_potential=self.absolute_tolerance,gradient_absolute_scale='inverse_panel_diameter',
            pair_batch=self.pair_batch,principal_value_count=int(pv.sum()),converged=False,
            error_certified=False,physical_qualification=False,checks=[])
        if not len(observers):
            receipt.update(converged=True,final_order=0,failure_reason=None)
            return self.coefficient.new_empty(0),self.coefficient.new_empty(0,3),receipt

        def integrate(order):
            potentials=[];gradients=[]
            for start in range(0,len(observers),self.pair_batch):
                x=observers[start:start+self.pair_batch];panels=triangles[start:start+self.pair_batch]
                mask=pv[start:start+self.pair_batch]
                value,gradient=self._integrate_panel_rule(x,panels,mask,order)
                potentials.append(value);gradients.append(gradient)
            return torch.cat(potentials),torch.cat(gradients)

        def norm(value):
            if value.ndim==1:return value.abs()
            scale=value.abs().amax(-1)
            safe=torch.where(scale==0,torch.ones_like(scale),scale)
            return scale*torch.linalg.vector_norm(value/safe[:,None],dim=-1)

        order=self.initial_order;previous=integrate(order)
        while order<self.max_order:
            order=min(2*order,self.max_order);current=integrate(order)
            check={'order':order};passed=True
            for name,value,old,absolute in zip(('potential','gradient'),current,previous,
                    (self.absolute_tolerance,self.absolute_tolerance/diameter)):
                error=norm(value.detach()-old.detach());size=norm(value.detach())
                threshold=absolute+self.relative_tolerance*size
                accepted=torch.isfinite(error)&torch.isfinite(size)&torch.isfinite(threshold)&(error<=threshold)
                check[name]={'max_absolute_difference':float(error.max()),'failed_pairs':int((~accepted).sum())}
                passed=passed and bool(accepted.all())
            receipt['checks'].append(check);receipt['final_order']=order
            if passed:
                self.assert_current();receipt.update(converged=True,failure_reason=None)
                return *current,receipt
            previous=tuple(value.detach() for value in current)
        receipt['failure_reason']='panel_order_budget_exhausted'
        raise ComplexStaticPanelConvergenceError(receipt)

    def _integrate_panel_rule(self,x,panels,mask,order):
        """One rule on already validated, centered physical panel pairs.

        Shared with cell refinement so the latter checks the summed volume
        quantities directly, rather than accepting separate panel estimates.
        """
        potential=self.coefficient.new_zeros(len(x));integrated=self.coefficient.new_zeros(len(x),3)
        if bool((~mask).any()):
            points=x[~mask];faces=panels[~mask]
            nodes,weights=distance_scaled_duffy_fan(points,faces,order,edge_grading=True)
            value,gradient=self.evaluate((points[:,None]-nodes).reshape(-1,3))
            ids=torch.nonzero(~mask).flatten()
            potential=potential.index_copy(0,ids,(value.reshape_as(weights)*weights).sum(-1))
            integrated=integrated.index_copy(0,ids,(gradient.reshape(*weights.shape,3)*weights[:,:,None]).sum(1))
        if bool(mask.any()):
            value,gradient=self._face_boundary_integrals(x[mask],panels[mask],order)
            ids=torch.nonzero(mask).flatten()
            potential=potential.index_copy(0,ids,value);integrated=integrated.index_copy(0,ids,gradient)
        return potential,integrated

    def _face_boundary_integrals(self,x,panels,order):
        """Face PV from graded line single layers, without a singular area rule.

        On the plane, div_y((y-x)G)=G. Tangential observer gradients are
        minus the outward edge-conormal single layers. For symmetric S,
        S grad G is parallel to x-y, so n.S.grad G=0 in the face PV.
        This fixes the normal component; it adds no trace jump or contact.
        """
        edge=panels.roll(-1,1)-panels;length=edge.norm(dim=-1);unit=edge/length[:,:,None]
        normal=torch.linalg.cross(edge[:,0],edge[:,1]);normal/=normal.norm(dim=-1,keepdim=True)
        x=x-((x-panels[:,0])*normal).sum(-1)[:,None]*normal
        relative=x[:,None]-panels;along=(relative*unit).sum(-1)
        perpendicular=relative-along[:,:,None]*unit
        position,weights=distance_scaled_line_rule(along,perpendicular.norm(dim=-1),length,order)
        delta=relative[:,:,None]-position[:,:,:,None]*unit[:,:,None]
        value,_=self.evaluate(delta.reshape(-1,3))
        single=(value.reshape_as(weights)*weights).sum(-1)
        conormal=torch.linalg.cross(unit,normal[:,None].expand_as(unit))
        height=(-relative*conormal).sum(-1)
        potential=(height*single).sum(-1)
        tangent=-(single[:,:,None]*conormal).sum(1)
        n=normal.to(self.coefficient.dtype)
        ns=n@self.coefficient
        gradient=tangent-n*((ns*tangent).sum(-1)/(ns*n).sum(-1))[:,None]
        if not bool(torch.isfinite(potential).all() & torch.isfinite(gradient).all()):
            raise ValueError('Complex static face integral is not representable')
        return potential,gradient
