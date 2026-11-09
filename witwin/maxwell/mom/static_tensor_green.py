"""Homogeneous real SPD static Green kernel and paired triangle integrals.

For constant A, div(A grad G) = -delta and
G(r) = 1/(4*pi*sqrt(det(A))*sqrt(r.T @ inv(A) @ r)).
A = L L.T maps physical coordinates to y = inv(L) x. Triangle
integration additionally retains the physical-to-transformed area ratio.
This internal primitive does not infer DC samples from AC materials or enable HB.
"""
import math
import torch
from .fields import _static_panel_gradient, _static_panel_potential, _pair_triangle_distance, _static_edge_log


def _static_panel_geometry(observers,triangles,principal_value):
    """Shared physical face admission for real and complex static kernels."""
    # The homogeneous kernel depends only on relative coordinates. Center
    # before geometric predicates and the tensor transform, so neither
    # projection nor transformed edges subtract large absolute positions.
    origin=triangles[:,0]
    observers=observers-origin
    triangles=triangles-origin[:,None]
    a,b,c=triangles.unbind(1)
    normal=torch.linalg.cross(b-a,c-a)
    area2=normal.norm(dim=-1)
    scale=torch.maximum((b-a).norm(dim=-1),torch.maximum((c-a).norm(dim=-1),(c-b).norm(dim=-1)))
    tolerance=128*torch.finfo(triangles.dtype).eps*scale
    if bool((area2<=tolerance*scale).any()):
        raise ValueError('Static panels require resolved nondegenerate triangles')
    normal=normal/area2[:,None]
    on_face=_pair_triangle_distance(observers,triangles)<=tolerance
    if principal_value is None:
        principal_value=torch.zeros(len(observers),device=observers.device,dtype=torch.bool)
    if (not isinstance(principal_value,torch.Tensor) or principal_value.device!=observers.device
            or principal_value.dtype!=torch.bool or principal_value.shape!=(len(observers),)
            or not torch.equal(principal_value,on_face)):
        raise ValueError('Explicit principal values must match face-interior queries')
    for start,end in ((a,b),(b,c),(c,a)):
        direction=end-start
        fraction=(((observers-start)*direction).sum(-1)/direction.square().sum(-1)).clamp(0,1)
        distance=(observers-start-fraction[:,None]*direction).norm(dim=-1)
        if bool((distance<=tolerance).any()):
            raise ValueError('Static panel gradients are singular or unresolved at edges')
    return observers,triangles,normal,area2,principal_value



class StaticTensorGreen:
    """Owned CUDA coefficient A, with the units supplied explicitly by callers.

    A must be a real symmetric positive-definite constitutive coefficient, not
    a complex AC sample. Native autograd is retained for A and observations.
    Panel inputs are paired [N,3] observers and [N,3,3] physical triangles.
    Surface values require an explicit face-interior principal-value mask.
    """
    def __init__(self, coefficient):
        if (not isinstance(coefficient,torch.Tensor) or coefficient.device.type!='cuda'
                or coefficient.dtype not in (torch.float32,torch.float64)
                or coefficient.shape!=(3,3) or not bool(torch.isfinite(coefficient).all())
                or not torch.equal(coefficient,coefficient.T)):
            raise ValueError('Static coefficient must be a finite symmetric real CUDA 3x3 tensor')
        self.coefficient=coefficient.clone()
        self.factor,info=torch.linalg.cholesky_ex(self.coefficient)
        if bool(info!=0):
            raise ValueError('Static coefficient must be positive definite')
        self.root_determinant=self.factor.diagonal().prod()
        if not bool(torch.isfinite(self.root_determinant)) or not bool(self.root_determinant>0):
            raise ValueError('Static coefficient determinant is not representable')
        self._binding=self._state()

    def _state(self):
        return tuple((id(v),v._version) for v in (self.coefficient,self.factor,self.root_determinant))

    def assert_current(self):
        if self._state()!=self._binding:
            raise ValueError('Static tensor kernel state changed')

    def _validate(self,value,shape):
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.device!=self.coefficient.device
                or value.dtype!=self.coefficient.dtype or value.ndim!=len(shape)
                or any(size is not None and value.shape[i]!=size for i,size in enumerate(shape))
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Static queries require finite matching real CUDA coordinates')

    def _transform(self,value):
        return torch.linalg.solve_triangular(self.factor,value.reshape(-1,3).T,upper=False).T.reshape(value.shape)

    def evaluate(self,displacements):
        """Return point Green values and physical observer gradients off source."""
        self._validate(displacements,(None,3))
        y=self._transform(displacements)
        scale=y.abs().amax(dim=-1)
        if bool((scale==0).any()):
            raise ValueError('Point Green queries cannot include the source singularity')
        # Unit-scale norms avoid squaring physical distances near dtype limits.
        unit=y/scale[:,None]
        unit_radius=unit.norm(dim=-1)
        inverse_radius=(1/scale)/unit_radius
        potential=(inverse_radius/self.root_determinant)/(4*math.pi)
        transformed_gradient=-(unit/unit_radius[:,None])*potential[:,None]*inverse_radius[:,None]
        gradient=torch.linalg.solve_triangular(self.factor.T,transformed_gradient.T,upper=True).T
        if not bool(torch.isfinite(potential).all() & torch.isfinite(gradient).all()):
            raise ValueError('Static point evaluation is not representable')
        return potential,gradient

    def evaluate_with_hessian(self,displacements):
        """Native point derivatives with scaled Cholesky distances."""
        potential,gradient=self.evaluate(displacements)
        y=self._transform(displacements)
        scale=y.abs().amax(-1)
        unit=y/scale[:,None]
        radius=unit.norm(dim=-1)
        unit=unit/radius[:,None]
        inverse_radius=(1/scale)/radius
        identity=torch.eye(3,device=y.device,dtype=y.dtype)
        # Combine inverse distance with the physical coordinate transform
        # before products: a local Hessian can underflow while its physical
        # counterpart remains representable for small constitutive scales.
        scaled_inverse=torch.linalg.solve_triangular(self.factor.T,
            inverse_radius[:,None,None]*identity,upper=True)
        direction=(scaled_inverse@unit[:,:,None]).squeeze(-1)
        gradient=-potential[:,None]*direction
        hessian=3*(potential[:,None]*direction)[:,:,None]*direction[:,None,:]-(
            potential[:,None,None]*scaled_inverse)@scaled_inverse.transpose(-1,-2)
        if not bool(torch.isfinite(hessian).all() & torch.isfinite(gradient).all()) or bool((potential==0).any()):
            raise ValueError('Static point derivatives are not representable')
        return potential,gradient,hessian

    def panel_integrals(self,observers,triangles,*,principal_value=None):
        """Integrate G and grad G over each physical triangle, including face PV.

        Edges/vertices have singular gradients and reject. The principal-value
        mask may only select on-face interior points, never off-surface queries.
        """
        self._validate(observers,(None,3));self._validate(triangles,(len(observers),3,3))
        observers,triangles,normal,area2,principal_value=_static_panel_geometry(
            observers,triangles,principal_value)
        y=self._transform(observers)
        panel=self._transform(triangles)
        transformed_normal=torch.linalg.cross(panel[:,1]-panel[:,0],panel[:,2]-panel[:,0])
        transformed_area2=transformed_normal.norm(dim=-1)
        if (not bool(torch.isfinite(y).all() & torch.isfinite(panel).all()
                     & torch.isfinite(transformed_area2).all())
                or bool((transformed_area2<=0).any())):
            raise ValueError('Transformed static panel geometry is not representable')
        transformed_normal=transformed_normal/transformed_area2[:,None]
        gradient_y=_static_panel_gradient(y,panel,principal_value=principal_value)
        value_y=_static_panel_potential(y,panel,transformed_normal,gradient_y)
        weight=(area2/transformed_area2)/self.root_determinant
        if not bool(torch.isfinite(weight).all()) or bool((weight<=0).any()):
            raise ValueError('Static panel area Jacobian is not representable')
        potential=weight*value_y
        gradient=weight[:,None]*torch.linalg.solve_triangular(self.factor.T,gradient_y.T,upper=True).T
        if not bool(torch.isfinite(potential).all() & torch.isfinite(gradient).all()):
            raise ValueError('Transformed static panel evaluation is not representable')
        return potential,gradient

    def panel_linear_double_layer(self,points,triangles,*,principal_value=None):
        """Integrate each physical barycentric basis against source conormal G.

        For r=x-y, mu grad_x G is parallel to -r. On a flat panel its normal
        component has constant height d. Thus the linear moments follow from
        the exact constant double layer and vector single-layer gradient:
        K[lambda_a]=lambda_a(x_parallel) K[1]+d grad(lambda_a).mu grad(S).
        Returned [pairs,3] columns follow the supplied triangle vertex order.
        No far-field quadrature or fitted interpolation is used. Face/edge
        admission is exactly that of panel_integrals; no trace jump is added.
        """
        _,gradient=self.panel_integrals(points,triangles,principal_value=principal_value)
        edge1=triangles[:,1]-triangles[:,0];edge2=triangles[:,2]-triangles[:,0]
        cross=torch.linalg.cross(edge1,edge2);area2=cross.norm(dim=-1)
        normal=cross/area2[:,None]
        g1=torch.linalg.cross(edge2,normal)/area2[:,None]
        g2=torch.linalg.cross(normal,edge1)/area2[:,None]
        basis_gradient=torch.stack((-g1-g2,g1,g2),dim=1)
        relative=points-triangles[:,0]
        barycentric=torch.einsum('qi,qai->qa',relative,basis_gradient)+points.new_tensor([1.,0.,0.])
        conormal_gradient=gradient@self.coefficient.T
        double=-(normal*conormal_gradient).sum(-1)
        height=(relative*normal).sum(-1)
        moments=barycentric*double[:,None]+height[:,None]*torch.einsum('qai,qi->qa',basis_gradient,conormal_gradient)
        if not bool(torch.isfinite(moments).all()):
            raise ValueError('Linear static double layer is not representable')
        return moments

    def _higher_layer_far_mask(self,points,triangles):
        local=triangles-triangles[:,:1]
        center=local.mean(1)
        radius=self._transform(local-center[:,None]).norm(dim=-1).amax(-1)
        distance=self._transform(points-triangles[:,0]-center).norm(dim=-1)
        return radius<=distance/32

    def _higher_layer_far(self,points,triangles,*,quadratic,with_gradient=False,conormal=False):
        """Stable separated moments with a positive order16 Duffy product rule.

        Admission is transformed source radius / center distance <= 1/32.
        For s=-2 Rhat.u/R+|u/R|^2, |s|<=65/1024. The binomial series
        of (1+s)^(-1/2) or (-3/2), through k=14, times the P1/P2
        basis and the P1 gradient's linear numerator has degree <=30.
        A source-conormal numerator is constant on the source plane, so the
        P2 conormal also has degree <=30. These terms are integrated exactly;
        their binomial tail is <=t^15*(16-15*t)/(1-t)^2 < 2.1e-17.
        The P2 observer gradient has a linear numerator: only terms through
        k=13 (degree <=29) are guaranteed exact by this rule. Its binomial
        tail is <=t^14*(15-14*t)/(1-t)^2 < 3.2e-16, with t=65/1024.
        Duffy's Jacobian adds one radial degree; order16 is exact through 31.
        Positive quadrature adds at most the integral and quadrature remainder
        contributions, before floating point error. This is an absolute bound
        scaled by kernel/basis magnitudes, not a relative bound near zero.
        """
        from .quadrature import gauss_rule
        u,w=gauss_rule(16,device=points.device,dtype=points.dtype)
        radial,angular=torch.meshgrid(u,u,indexing='ij')
        radial=radial.flatten();angular=angular.flatten()
        bary=torch.stack((1-radial,radial*(1-angular),radial*angular),-1)
        weights=(w[:,None]*w[None,:]).flatten()*radial
        basis=(torch.cat((bary*(2*bary-1),4*torch.stack((bary[:,0]*bary[:,1],
            bary[:,1]*bary[:,2],bary[:,2]*bary[:,0]),-1)),-1) if quadratic else bary)
        result=[];gradient_result=[]
        for start in range(0,len(points),256):
            panels=triangles[start:start+256]
            local=panels-panels[:,:1];observers=points[start:start+256]-panels[:,0]
            cross=torch.linalg.cross(local[:,1],local[:,2]);area2=cross.norm(dim=-1)
            nodes=torch.einsum('qa,pac->pqc',bary,local)
            transformed=self._transform(observers[:,None]-nodes)
            radius=transformed.norm(dim=-1)
            density=(1/radius)/(4*math.pi*self.root_determinant)
            if with_gradient:
                transformed_gradient=-transformed*(density/radius.square())[:,:,None]
                physical_gradient=torch.linalg.solve_triangular(self.factor.T,
                    transformed_gradient.reshape(-1,3).T,upper=True).T.reshape_as(transformed)
                gradient_result.append(torch.einsum('pqi,qa,q,p->pai',physical_gradient,basis,weights,area2))
            if conormal:
                height=(observers*(cross/area2[:,None])).sum(-1)
                density=density*(height[:,None]/radius)/radius
            result.append(torch.einsum('pq,qa,q,p->pa',density,basis,weights,area2))
        value=torch.cat(result)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Separated static layer moments are not representable')
        if with_gradient:
            gradient=torch.cat(gradient_result)
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError('Separated static layer gradients are not representable')
            return value,gradient
        return value

    def panel_linear_single_layer(self,points,triangles,*,principal_value=None):
        """Return P1 single-layer values in physical vertex order."""
        return self.panel_linear_integrals(points,triangles,principal_value=principal_value)[0]

    def panel_linear_integrals(self,points,triangles,*,principal_value=None):
        """Integrate G and its observer gradient against physical P1 bases.

        In isotropic transformed coordinates, the tangential first moment of
        r/R equals minus the outward edge integral of R. For a barycentric
        gradient g, this gives S[lambda]=lambda(x_parallel)S[1]
        + g.dot(sum_edges m integral_edge R)/(4*pi), followed by the physical
        area/determinant Jacobian. Separated panels use a bounded positive
        quadrature to avoid cancellation. No jump is added; panel_integrals supplies
        the shared face/edge admission. Columns follow physical vertex order.
        """
        single,constant_gradient=self.panel_integrals(points,triangles,principal_value=principal_value)
        far=self._higher_layer_far_mask(points,triangles)
        if bool(far.any()):
            result=single.new_zeros(len(points),3)
            derivative=single.new_zeros(len(points),3,3)
            result[far],derivative[far]=self._higher_layer_far(points[far],triangles[far],quadratic=False,with_gradient=True)
            if bool((~far).any()):
                pv=None if principal_value is None else principal_value[~far]
                result[~far],derivative[~far]=self.panel_linear_integrals(points[~far],triangles[~far],principal_value=pv)
            return result,derivative
        origin=triangles[:,0]
        panel=self._transform(triangles-origin[:,None])
        observer=self._transform(points-origin)
        e1=panel[:,1]-panel[:,0];e2=panel[:,2]-panel[:,0]
        cross=torch.linalg.cross(e1,e2);area2=cross.norm(dim=-1)
        normal=cross/area2[:,None]
        g1=torch.linalg.cross(e2,normal)/area2[:,None]
        g2=torch.linalg.cross(normal,e1)/area2[:,None]
        gradients=torch.stack((-g1-g2,g1,g2),1)
        barycentric=torch.einsum('qi,qai->qa',observer-panel[:,0],gradients)+points.new_tensor([1.,0.,0.])
        boundary=torch.zeros_like(observer)
        boundary_gradient=observer.new_zeros(len(points),3,3)
        for first,last in ((0,1),(1,2),(2,0)):
            edge=panel[:,last]-panel[:,first];length=edge.norm(dim=-1)
            tangent=edge/length[:,None]
            relative=panel[:,first]-observer
            t0=(relative*tangent).sum(-1);t1=t0+length
            perpendicular=relative-t0[:,None]*tangent
            distance2=perpendicular.square().sum(-1)
            r0=relative.norm(dim=-1);r1=(panel[:,last]-observer).norm(dim=-1)
            # Definite integral of sqrt(t^2+b^2). Difference r1-r0 is
            # rationalized to avoid subtracting nearly equal endpoint norms.
            logarithm=_static_edge_log(observer,panel[:,first],panel[:,last])
            integral=.5*(length*(r1+t0*(t0+t1)/(r0+r1))+distance2*logarithm)
            outward=torch.linalg.cross(tangent,normal)
            boundary=boundary+outward*integral[:,None]
            radial_difference=length*(t0+t1)/(r0+r1)
            edge_gradient=-perpendicular*logarithm[:,None]-tangent*radial_difference[:,None]
            projection=(gradients*outward[:,None]).sum(-1)
            boundary_gradient=boundary_gradient+projection[:,:,None]*edge_gradient[:,None]
        physical_area2=torch.linalg.cross(triangles[:,1]-origin,triangles[:,2]-origin).norm(dim=-1)
        weight=(physical_area2/area2)/self.root_determinant
        moments=barycentric*single[:,None]+(weight/(4*math.pi))[:,None]*torch.einsum('qai,qi->qa',gradients,boundary)
        transformed_derivative=gradients*single[:,None,None]+(weight/(4*math.pi))[:,None,None]*boundary_gradient
        derivative=torch.linalg.solve_triangular(self.factor.T,transformed_derivative.reshape(-1,3).T,upper=True).T.reshape(-1,3,3)
        derivative=derivative+barycentric[:,:,None]*constant_gradient[:,None]
        if not bool(torch.isfinite(moments).all() & torch.isfinite(derivative).all()):
            raise ValueError('Linear static single layer is not representable')
        return moments,derivative

    def panel_quadratic_single_layer(self,points,triangles,*,principal_value=None):
        """Return P2 single-layer values at vertices 0/1/2 and edges 01/12/20."""
        return self.panel_quadratic_integrals(points,triangles,principal_value=principal_value)[0]

    def panel_quadratic_integrals(self,points,triangles,*,principal_value=None):
        """Integrate G and its observer gradient against six P2 Lagrange bases.

        In the transformed plane let y be source minus projected observer,
        h the signed height, R=sqrt(y.y+h*h), and T its tangent projector.
        Integrating d_j(y_i R) gives the second moment
        Q_ij=boundary(y_i R m_j)-T_ij*integral(R), with
        integral(R)=(boundary(y.m R)+h*h*integral(1/R))/3.
        Tangential gradients use integration by parts on the source polynomial;
        the existing P2 source-conormal moments give the normal component.
        A positive quadrature handles separated panels to avoid cancellation.
        No internal backward calls or trace jumps are added. Shared panel
        admission applies, including explicit face principal values.
        """
        single,_=self.panel_integrals(points,triangles,principal_value=principal_value)
        far=self._higher_layer_far_mask(points,triangles)
        if bool(far.any()):
            result=single.new_zeros(len(points),6)
            derivative=single.new_zeros(len(points),6,3)
            result[far],derivative[far]=self._higher_layer_far(
                points[far],triangles[far],quadratic=True,with_gradient=True)
            if bool((~far).any()):
                pv=None if principal_value is None else principal_value[~far]
                result[~far],derivative[~far]=self.panel_quadratic_integrals(
                    points[~far],triangles[~far],principal_value=pv)
            return result,derivative
        linear,linear_gradient=self.panel_linear_integrals(points,triangles,principal_value=principal_value)
        origin=triangles[:,0]
        panel=self._transform(triangles-origin[:,None]);x=self._transform(points-origin)
        e1=panel[:,1];e2=panel[:,2]
        cross=torch.linalg.cross(e1,e2);area2=cross.norm(dim=-1);normal=cross/area2[:,None]
        g1=torch.linalg.cross(e2,normal)/area2[:,None];g2=torch.linalg.cross(normal,e1)/area2[:,None]
        g=torch.stack((-g1-g2,g1,g2),1)
        bary=torch.einsum('pi,pai->pa',x,g)+x.new_tensor([1.,0,0])
        height=(x*normal).sum(-1)
        physical_cross=torch.linalg.cross(triangles[:,1]-origin,triangles[:,2]-origin)
        physical_area2=physical_cross.norm(dim=-1)
        weight=physical_area2/area2/self.root_determinant/(4*math.pi)
        first=torch.zeros_like(x);second_boundary=x.new_zeros(len(x),3,3)
        area_radial=x.new_zeros(len(x));boundary_gradient=x.new_zeros(len(x),6,3)
        for a,b in ((0,1),(1,2),(2,0)):
            edge=panel[:,b]-panel[:,a];length=edge.norm(dim=-1);tangent=edge/length[:,None]
            relative=panel[:,a]-x;t0=(relative*tangent).sum(-1);t1=t0+length
            perp=relative-t0[:,None]*tangent;b2=perp.square().sum(-1)
            r0=relative.norm(dim=-1);r1=(panel[:,b]-x).norm(dim=-1)
            log=_static_edge_log(x,panel[:,a],panel[:,b])
            dr=length*(t0+t1)/(r0+r1)
            edge_r=.5*(length*(r1+t0*(t0+t1)/(r0+r1))+b2*log)
            edge_tr=dr*(r1.square()+r1*r0+r0.square())/3
            outward=torch.linalg.cross(tangent,normal)
            planar_perp=perp+height[:,None]*normal
            first+=outward*edge_r[:,None]
            second_boundary+=(planar_perp*edge_r[:,None]+tangent*edge_tr[:,None])[:,:,None]*outward[:,None,:]
            area_radial+=(planar_perp*outward).sum(-1)*edge_r
            # Barycentric coordinates along the edge, expressed as c+d*t.
            unit=torch.eye(3,device=x.device,dtype=x.dtype)
            d=(unit[b]-unit[a])[None]/length[:,None]
            c=unit[a][None]-t0[:,None]*d
            edge_linear=c*log[:,None]+d*dr[:,None]
            edge_second=(c[:,:,None]*c[:,None,:]*log[:,None,None]
                +(c[:,:,None]*d[:,None,:]+d[:,:,None]*c[:,None,:])*dr[:,None,None]
                +d[:,:,None]*d[:,None,:]*(edge_r-b2*log)[:,None,None])
            edge_p2=torch.cat((2*edge_second.diagonal(dim1=-2,dim2=-1)-edge_linear,
                4*torch.stack((edge_second[:,0,1],edge_second[:,1,2],edge_second[:,2,0]),-1)),1)
            boundary_gradient+=edge_p2[:,:,None]*outward[:,None]
        area_radial=(area_radial+height.square()*single/weight)/3
        projector=torch.eye(3,device=x.device,dtype=x.dtype)[None]-normal[:,:,None]*normal[:,None,:]
        q=second_boundary-projector*area_radial[:,None,None]
        q=(q+q.mT)/2
        first_basis=torch.einsum('pai,pi->pa',g,first)*weight[:,None]
        moments=bary[:,:,None]*bary[:,None,:]*single[:,None,None]
        moments+=bary[:,:,None]*first_basis[:,None,:]+first_basis[:,:,None]*bary[:,None,:]
        moments+=torch.einsum('pai,pij,pbj->pab',g,q,g)*weight[:,None,None]
        values=torch.cat((2*moments.diagonal(dim1=-2,dim2=-1)-linear,
            4*torch.stack((moments[:,0,1],moments[:,1,2],moments[:,2,0]),-1)),1)
        source_gradient=torch.cat((g*(4*linear-single[:,None])[:,:,None],
            4*torch.stack((g[:,0]*linear[:,1,None]+g[:,1]*linear[:,0,None],
                           g[:,1]*linear[:,2,None]+g[:,2]*linear[:,1,None],
                           g[:,2]*linear[:,0,None]+g[:,0]*linear[:,2,None]),1)),1)
        transformed_gradient=source_gradient-weight[:,None,None]*boundary_gradient
        physical_normal=physical_cross/physical_area2[:,None]
        conormal_factor=(physical_normal@self.factor).norm(dim=-1)
        # Reuse the already evaluated P1 gradient. Any affine extension of
        # barycentrics matching the source plane satisfies this identity; here
        # it is the transformed-plane extension used for the single moments.
        linear_conormal=-(linear_gradient@self.coefficient.T*physical_normal[:,None]).sum(-1)
        physical_height=((points-origin)*physical_normal).sum(-1)
        quadratic_conormal=(linear_conormal[:,:,None]*bary[:,None,:]
            +physical_height[:,None,None]*torch.einsum('pai,pbi->pab',linear_gradient@self.factor,g))
        quadratic_conormal=(quadratic_conormal+quadratic_conormal.mT)/2
        double=torch.cat((2*quadratic_conormal.diagonal(dim1=-2,dim2=-1)-linear_conormal,
            4*torch.stack((quadratic_conormal[:,0,1],quadratic_conormal[:,1,2],quadratic_conormal[:,2,0]),-1)),1)
        transformed_gradient-=normal[:,None,:]*(double/conormal_factor[:,None])[:,:,None]
        gradients=torch.linalg.solve_triangular(self.factor.T,transformed_gradient.reshape(-1,3).T,upper=True).T.reshape(-1,6,3)
        if not bool(torch.isfinite(values).all() & torch.isfinite(gradients).all()):
            raise ValueError('Quadratic static single layer is not representable')
        return values,gradients

    def panel_quadratic_double_layer(self,points,triangles,*,principal_value=None):
        """Integrate six P2 Lagrange basis functions against source conormal G.

        Columns are vertices 0,1,2 and edge midpoints 01,12,20. Differentiate
        the analytic P1 single layers to form quadratic conormal moments via
        K[lambda_a lambda_b]=lambda_b(x_parallel) K[lambda_a]
        +d grad(lambda_b).A grad(S[lambda_a]). Analytic observer gradients avoid
        internal backward calls while retaining native material/observer graphs.
        """
        self._validate(points,(None,3));self._validate(triangles,(len(points),3,3))
        # Reuse the shared physical geometry and exact PV/edge admission even
        # when the separated path bypasses the cancellation-prone identity.
        self.panel_integrals(points,triangles,principal_value=principal_value)
        far=self._higher_layer_far_mask(points,triangles)
        if bool(far.any()):
            result=points.new_zeros(len(points),6)
            result[far]=self._higher_layer_far(points[far],triangles[far],quadratic=True,conormal=True)
            if bool((~far).any()):
                pv=None if principal_value is None else principal_value[~far]
                result[~far]=self.panel_quadratic_double_layer(points[~far],triangles[~far],principal_value=pv)
            return result
        _,gradients=self.panel_linear_integrals(points,triangles,principal_value=principal_value)
        linear=self.panel_linear_double_layer(points,triangles,principal_value=principal_value)
        e1=triangles[:,1]-triangles[:,0];e2=triangles[:,2]-triangles[:,0]
        cross=torch.linalg.cross(e1,e2);area2=cross.norm(dim=-1)
        normal=cross/area2[:,None]
        g1=torch.linalg.cross(e2,normal)/area2[:,None]
        g2=torch.linalg.cross(normal,e1)/area2[:,None]
        basis_gradient=torch.stack((-g1-g2,g1,g2),1)
        relative=points-triangles[:,0]
        barycentric=torch.einsum('qi,qai->qa',relative,basis_gradient)+points.new_tensor([1.,0.,0.])
        height=(relative*normal).sum(-1)
        second=linear[:,:,None]*barycentric[:,None,:]+height[:,None,None]*torch.einsum(
            'qai,qbi->qab',gradients@self.coefficient.T,basis_gradient)
        second=(second+second.mT)/2
        result=torch.cat((2*second.diagonal(dim1=-2,dim2=-1)-linear,
                          4*torch.stack((second[:,0,1],second[:,1,2],second[:,2,0]),-1)),1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Quadratic static double layer is not representable')
        return result

    @staticmethod
    def _weak_polynomial_basis(bary,quadratic):
        if not quadratic:return bary
        return torch.cat((bary*(2*bary-1),4*torch.stack((bary[...,0]*bary[...,1],
            bary[...,1]*bary[...,2],bary[...,2]*bary[...,0]),-1)),-1)

    def touching_linear_weak_layers(self,left,right,left_ids,right_ids,category,*,order=8,max_nodes=262144):
        """P1 weak layers; see the shared touching implementation's contract."""
        return self._touching_weak_layers(left,right,left_ids,right_ids,category,order=order,max_nodes=max_nodes,quadratic=False)

    def touching_quadratic_weak_layers(self,left,right,left_ids,right_ids,category,*,order=8,max_nodes=262144):
        """P2 weak layers in vertex0/1/2, edge01/12/20 order.

        Uses the same touching geometry, quadrature and budget contract as P1.
        The returned conormal block excludes the mass-matrix half jump.
        """
        return self._touching_weak_layers(left,right,left_ids,right_ids,category,order=order,max_nodes=max_nodes,quadratic=True)

    def _touching_weak_layers(self,left,right,left_ids,right_ids,category,*,order=8,max_nodes=262144,quadratic):
        """Paired P1/P2 weak single and target-conormal layers on touching faces.

        Returns [pairs, target_basis, source_basis] integrals. Categories are
        0=self, 1=edge, 2=vertex. Joint Duffy rules remove the touching singularity;
        the caller must check order convergence. The conormal result is K', with
        the left outward normal, and excludes the mass-matrix half jump. Self K'
        is exactly zero on a flat face. max_nodes bounds quadrature nodes per
        generated batch, not peak memory or retained autograd graphs. The faces
        must come from a conforming mesh with no additional intersections; the
        local id checks do not certify that global geometric condition.
        """
        self._validate(left,(None,3,3));self._validate(right,(len(left),3,3))
        if type(category) is not int or category not in (0,1,2):
            raise ValueError('Weak touching layers require self, edge or vertex category')
        if type(order) is not int or order<2 or type(max_nodes) is not int or max_nodes<1:
            raise ValueError('Weak touching layers require order >=2 and positive node budget')
        nodes_per_pair=(2 if category==2 else 6)*order**4
        if nodes_per_pair>max_nodes:
            raise ValueError('Weak touching rule exceeds its quadrature node budget')
        for ids in (left_ids,right_ids):
            if (not isinstance(ids,torch.Tensor) or ids.device!=left.device or ids.dtype!=torch.int64
                    or ids.shape!=(len(left),3) or bool((ids<0).any())
                    or bool((ids.sort(-1).values[:,1:]==ids.sort(-1).values[:,:-1]).any())):
                raise ValueError('Weak touching layers require distinct matching CUDA vertex ids')
        shared=left_ids[:,:,None]==right_ids[:,None,:]
        if bool((shared.sum((1,2))!=(3-category)).any()):
            raise ValueError('Weak touching topology differs from its declared category')
        if bool((shared & (left[:,:,None,:]!=right[:,None,:,:]).any(-1)).any()):
            raise ValueError('Shared weak vertices must have identical coordinates')
        origin=left[:,0]
        left=left-origin[:,None];right=right-origin[:,None]
        def geometry(triangles):
            e1=triangles[:,1]-triangles[:,0];e2=triangles[:,2]-triangles[:,0]
            cross=torch.linalg.cross(e1,e2);area2=cross.norm(dim=-1)
            scale=torch.stack((e1.norm(dim=-1),e2.norm(dim=-1),(e2-e1).norm(dim=-1)),-1).amax(-1)
            if bool((area2<=128*torch.finfo(left.dtype).eps*scale.square()).any()) or not bool(torch.isfinite(area2).all()):
                raise ValueError('Weak touching panels must be resolved and nondegenerate')
            normal=cross/area2[:,None]
            g1=torch.linalg.cross(e2,normal)/area2[:,None]
            g2=torch.linalg.cross(normal,e1)/area2[:,None]
            return normal,torch.stack((-g1-g2,g1,g2),1)
        normals,left_gradient=geometry(left);_,right_gradient=geometry(right)
        nb=6 if quadratic else 3
        if len(left)==0:return left.new_empty(0,nb,nb),left.new_empty(0,nb,nb)
        from .quadrature import touching_pair_rule
        single=[];conormal=[];chunk=max_nodes//nodes_per_pair
        for start in range(0,len(left),chunk):
            stop=min(start+chunk,len(left))
            x,y,weight=touching_pair_rule(left[start:stop],right[start:stop],
                left_ids[start:stop],right_ids[start:stop],category,order)
            lb=torch.einsum('pqi,pai->pqa',x-left[start:stop,:1],left_gradient[start:stop])+left.new_tensor([1.,0.,0.])
            rb=torch.einsum('pqi,pai->pqa',y-right[start:stop,:1],right_gradient[start:stop])+left.new_tensor([1.,0.,0.])
            lb=self._weak_polynomial_basis(lb,quadratic)
            rb=self._weak_polynomial_basis(rb,quadratic)
            value,gradient=self.evaluate((x-y).reshape(-1,3))
            value=value.reshape_as(weight)
            single.append(torch.einsum('pq,pqa,pqb->pab',weight*value,lb,rb))
            if category==0:
                conormal.append(single[-1]*0)
            else:
                density=(gradient.reshape_as(x)@self.coefficient.T*normals[start:stop,None]).sum(-1)
                conormal.append(torch.einsum('pq,pqa,pqb->pab',weight*density,lb,rb))
        single=torch.cat(single);conormal=torch.cat(conormal)
        if not bool(torch.isfinite(single).all() & torch.isfinite(conormal).all()):
            raise ValueError('Weak touching static layers are not representable')
        return single,conormal

    def disjoint_linear_weak_layers(self,left,right,*,order=8,max_points=4096):
        """P1 weak layers; see the shared disjoint implementation's contract."""
        return self._disjoint_weak_layers(left,right,order=order,max_points=max_points,quadratic=False)

    def disjoint_quadratic_weak_layers(self,left,right,*,order=8,max_points=4096):
        """P2 weak layers in vertex0/1/2, edge01/12/20 order.

        Uses the same disjoint geometry, quadrature and budget contract as P1.
        The returned conormal block excludes the mass-matrix half jump.
        """
        return self._disjoint_weak_layers(left,right,order=order,max_points=max_points,quadratic=True)

    def _disjoint_weak_layers(self,left,right,*,order=8,max_points=4096,quadratic):
        """P1/P2 weak V/K' with exact source layers and target Gauss integration.

        Physical triangle separation is validated. This fixed-order primitive
        does not estimate its own integration error. max_points bounds target
        evaluation pairs per batch, not peak workspace or retained graphs.
        """
        self._validate(left,(None,3,3));self._validate(right,(len(left),3,3))
        if type(order) is not int or order<2 or type(max_points) is not int or max_points<1:
            raise ValueError('Disjoint weak layers require order >=2 and positive point budget')
        if order**2>max_points:
            raise ValueError('Disjoint weak rule exceeds its target point budget')
        origin=left[:,0];left=left-origin[:,None];right=right-origin[:,None]
        normals=[];scales=[]
        for triangles in (left,right):
            e1=triangles[:,1]-triangles[:,0];e2=triangles[:,2]-triangles[:,0]
            cross=torch.linalg.cross(e1,e2);area2=cross.norm(dim=-1)
            scale=torch.stack((e1.norm(dim=-1),e2.norm(dim=-1),(e2-e1).norm(dim=-1)),-1).amax(-1)
            if (not bool(torch.isfinite(area2).all()) or
                    bool((area2<=128*torch.finfo(left.dtype).eps*scale.square()).any())):
                raise ValueError('Disjoint weak panels must be resolved and nondegenerate')
            normals.append(cross/area2[:,None]);scales.append(scale)
        from .mesh import triangle_pair_distances
        gap=triangle_pair_distances(left,right)
        if bool((gap<=128*torch.finfo(left.dtype).eps*torch.maximum(*scales)).any()) or not bool(torch.isfinite(gap).all()):
            raise ValueError('Disjoint weak panels must have a resolved positive gap')
        nb=6 if quadratic else 3
        if len(left)==0:return left.new_empty(0,nb,nb),left.new_empty(0,nb,nb)
        from .quadrature import triangle_rule,gauss_rule
        u,_=gauss_rule(order,device=left.device,dtype=left.dtype)
        radial,angular=torch.meshgrid(u,u,indexing='ij')
        radial=radial.flatten();angular=angular.flatten()
        basis=self._weak_polynomial_basis(torch.stack((1-radial,radial*(1-angular),radial*angular),-1),quadratic)
        layers=self.panel_quadratic_integrals if quadratic else self.panel_linear_integrals
        single=[];conormal=[];chunk=max_points//order**2
        for start in range(0,len(left),chunk):
            stop=min(start+chunk,len(left));count=stop-start
            points,weight=triangle_rule(left[start:stop],order)
            panels=right[start:stop,None].expand(-1,order**2,-1,-1).reshape(-1,3,3)
            value,gradient=layers(points.reshape(-1,3),panels)
            value=value.reshape(count,order**2,nb);gradient=gradient.reshape(count,order**2,nb,3)
            density=(gradient@self.coefficient.T*normals[0][start:stop,None,None]).sum(-1)
            single.append(torch.einsum('pq,qa,pqb->pab',weight,basis,value))
            conormal.append(torch.einsum('pq,qa,pqb->pab',weight,basis,density))
        result=torch.cat(single),torch.cat(conormal)
        if not all(bool(torch.isfinite(value).all()) for value in result):
            raise ValueError('Disjoint weak static layers are not representable')
        return result

    def adaptive_disjoint_linear_weak_layers(self,left,right,*,orders=(4,8,16,32),rtol=1e-6,max_points=4096):
        """P1 weak layers; see the shared adaptive_disjoint implementation's contract."""
        return self._adaptive_disjoint_weak_layers(left,right,orders=orders,rtol=rtol,max_points=max_points,quadratic=False)

    def adaptive_disjoint_quadratic_weak_layers(self,left,right,*,orders=(4,8,16,32),rtol=1e-6,max_points=4096):
        """P2 weak layers in vertex0/1/2, edge01/12/20 order.

        Uses the same adaptive_disjoint geometry, quadrature and budget contract as P1.
        The returned conormal block excludes the mass-matrix half jump.
        """
        return self._adaptive_disjoint_weak_layers(left,right,orders=orders,rtol=rtol,max_points=max_points,quadratic=True)

    def _adaptive_disjoint_weak_layers(self,left,right,*,orders=(4,8,16,32),rtol=1e-6,max_points=4096,quadratic):
        """Accept per-pair successive-order changes, or reject at the order cap.

        V and K' are compared jointly after scaling K' by length/mean(A). The
        returned change is an estimator, not a rigorous quadrature/physics bound.
        No pair is accepted solely because a quadrature budget was exhausted.
        """
        if (not isinstance(orders,tuple) or len(orders)<2 or
                any(type(order) is not int or order<2 for order in orders) or
                any(b<=a for a,b in zip(orders,orders[1:]))):
            raise ValueError('Adaptive weak orders must be a strictly increasing tuple')
        if type(rtol) not in (int,float) or not math.isfinite(rtol) or rtol<=0:
            raise ValueError('Adaptive weak tolerance must be finite and positive')
        # Validate both full geometry and the maximum point request up front.
        if type(max_points) is not int or max_points<orders[-1]**2:
            raise ValueError('Adaptive weak maximum order exceeds its point budget')
        self._validate(left,(None,3,3));self._validate(right,(len(left),3,3))
        count=len(left)
        length=torch.maximum((left[:,1:]-left[:,:1]).norm(dim=-1).amax(-1),
                             (right[:,1:]-right[:,:1]).norm(dim=-1).amax(-1))
        if not bool(torch.isfinite(length).all()) or bool((length<=0).any()):
            raise ValueError('Adaptive weak comparison length is not representable')
        flux_scale=self.coefficient.diagonal().mean()/length
        if not bool(torch.isfinite(flux_scale).all()) or bool((flux_scale<=0).any()):
            raise ValueError('Adaptive weak comparison flux scale is not representable')
        previous=self._disjoint_weak_layers(left,right,order=orders[0],max_points=max_points,quadratic=quadratic)
        pending=torch.arange(count,device=left.device)
        nb=6 if quadratic else 3
        outputs=[left.new_zeros(count,nb,nb),left.new_zeros(count,nb,nb)]
        used=torch.zeros(count,device=left.device,dtype=torch.int64)
        changes=left.new_zeros(count)
        def normalized(blocks,scale):
            return torch.cat((blocks[0].flatten(1),(blocks[1]/scale[:,None,None]).flatten(1)),1)
        for order in orders[1:]:
            if len(pending)==0:break
            current=self._disjoint_weak_layers(left[pending],right[pending],order=order,max_points=max_points,quadratic=quadratic)
            high=normalized(current,flux_scale[pending]);low=normalized(previous,flux_scale[pending])
            magnitude=high.abs().amax(-1)
            if bool((magnitude==0).any()) or not bool(torch.isfinite(magnitude).all()):
                raise ValueError('Adaptive weak comparison scale is not representable')
            error=((high-low)/magnitude[:,None]).norm(dim=-1)/(high/magnitude[:,None]).norm(dim=-1)
            if not bool(torch.isfinite(error).all()):
                raise ValueError('Adaptive weak change estimate is not representable')
            accepted=error<=rtol;indices=pending[accepted]
            for output,value in zip(outputs,current):output[indices]=value[accepted]
            used[indices]=order;changes[indices]=error[accepted]
            pending=pending[~accepted];previous=tuple(value[~accepted] for value in current)
        if len(pending):
            raise ValueError(f'Disjoint weak integration did not converge for {len(pending)} pairs by order {orders[-1]}')
        return (*outputs,dict(orders=used,relative_change=changes,rtol=rtol,
            estimator='successive_order_scaled_V_Kprime',certified_error_bound=False,physical_qualification=False))
