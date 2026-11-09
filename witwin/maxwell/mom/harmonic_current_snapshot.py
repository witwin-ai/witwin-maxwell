"""Exact current-DC preparation records and decode-only restoration."""
import math
import torch
from .harmonic_current import HarmonicCurrentClosure
from .static_current import FaceCurrentBalance
from .static_floating import FloatingPotentialOperator

FLOATING_ARRAYS = ('triangles','normals','areas','indicator','body_areas','charge_test','lengths',
                   'single_layer','mean_gradient','matrix','lu','pivots')


def encode(static):
    closure,floating = static['current_closure'],static['floating_operator']
    closure.assert_current()
    floating._assert_current()
    return {**{k:v for k,v in static.items() if k not in ('current_closure','floating_operator')},
        'current_closure':encode_current_closure(closure),
        'floating_operator':{**{k:getattr(floating,k) for k in FLOATING_ARRAYS},
                             'testing_order':floating.testing_order,'body_faces':floating.body_faces}}


def encode_current_closure(closure):
    """Canonical numerical record shared by scalar and tensor mixed owners."""
    closure.assert_current()
    return {'conductivities':closure.conductivities,'counts':closure.counts,
            **closure.numerical_policy(),
            'relative_tolerance':closure.relative_tolerance,'absolute_tolerance':closure.absolute_tolerance,
            'potential_basis':'crouzeix_raviart_face_mean',
            'blocks':[{'gradients':b.gradients,'nodal_volume':b.nodal_volume,
                       'faces':b.faces,'face_dofs':b.face_dofs} for b in closure.blocks]}


def restore(static,bindings,materials,testing_order):
    """Rebind saved numerical arrays; never assemble or factor a DC operator."""
    system=bindings[0].surface_system
    channels = tuple((m.electric_dc.kind,m.electric_dc.finite_relative_part,m.electric_dc.conductivity,
                      m.electric_dc.plasma_angular_frequency_squared) for m in materials)
    if static.get('material_channels') != channels:
        raise ValueError('Current DC snapshot material channels mismatch')
    from ..constants import MU_0
    from .harmonic_material import compile_harmonic_material
    from ..media import Material
    exterior = compile_harmonic_material(Material(),materials[0].fundamental_frequency,len(materials[0].linear_samples))
    expected_mu = (exterior.magnetic_dc.dielectric_sample(MU_0),tuple(m.magnetic_dc.dielectric_sample(MU_0) for m in materials))
    if static.get('magnetic_permeabilities') != expected_mu:
        raise ValueError('Current snapshot magnetic material mismatch')
    closure=restore_current_closure(static['current_closure'],bindings,expected_conductivities={
        b.volume.region_id:materials[b.volume.region_id-1].electric_dc.conductivity for b in bindings})
    floating = restore_floating_operator(static['floating_operator'],system,testing_order)
    static.update(current_closure=closure,floating_operator=floating)
    return static


def restore_current_closure(raw,bindings,*,expected_conductivities=None):
    """Restore native face blocks against geometry; the caller owns record identity."""
    bindings=tuple(bindings)
    if not bindings:raise ValueError('Current closure restoration requires volumes')
    for binding in bindings:binding.assert_current()
    system = bindings[0].surface_system
    sample = system.bodies[0].mesh.vertices
    def array(value,shape,dtype=None):
        if (not isinstance(value,torch.Tensor) or value.shape != shape or value.device != sample.device
                or value.dtype != (sample.dtype if dtype is None else dtype) or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid current DC snapshot array')
    def close(actual,expected):
        scale = expected.abs().max()
        if not bool((actual-expected).abs().max() <= 512*torch.finfo(sample.dtype).eps*scale):
            raise ValueError('Inconsistent current DC snapshot geometry or operator')
    active=isinstance(raw,dict) and 'tangent_rule' in raw
    policy_keys={'tangent_rule','tangent_column_batch'} if active else set()
    if (not isinstance(raw,dict) or set(raw) != {'conductivities','counts','relative_tolerance','absolute_tolerance','blocks','potential_basis',*policy_keys}
            or raw['potential_basis'] != 'crouzeix_raviart_face_mean'):
        raise ValueError('Invalid current closure snapshot fields')
    counts = tuple(len(b.volume.cells) for b in bindings)
    if raw['counts'] != counts or len(raw['blocks']) != len(bindings) or len(raw['conductivities']) != len(bindings):
        raise ValueError('Current closure snapshot volume count mismatch')
    for key in ('relative_tolerance','absolute_tolerance'):
        if type(raw[key]) not in (int,float) or not math.isfinite(raw[key]) or raw[key] < 0:
            raise ValueError('Invalid current closure snapshot tolerance')
    blocks = []
    for binding,record,sigma in zip(bindings,raw['blocks'],raw['conductivities']):
        volume = binding.volume
        if not isinstance(record,dict) or set(record) != {'gradients','nodal_volume','faces','face_dofs'}:
            raise ValueError('Invalid nodal current snapshot fields')
        local = volume.cells.new_tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
        faces,inverse = torch.unique(volume.cells[:,local].reshape(-1,3).sort(-1).values,dim=0,return_inverse=True)
        array(record['faces'],faces.shape,torch.int64)
        array(record['face_dofs'],volume.cells.shape,torch.int64)
        if not torch.equal(record['faces'],faces) or not torch.equal(record['face_dofs'],inverse.reshape(-1,4)):
            raise ValueError('Current snapshot face topology mismatch')
        array(record['gradients'],(len(volume.cells),4,3))
        array(record['nodal_volume'],(len(faces),))
        shape = (len(volume.cells),3,3) if isinstance(sigma,torch.Tensor) and sigma.ndim == 3 else (len(volume.cells),)
        array(sigma,shape)
        validator = FaceCurrentBalance.__new__(FaceCurrentBalance)
        validator.volume = volume
        canonical_sigma = validator._conductivity(sigma)
        if not torch.equal(sigma,canonical_sigma):
            raise ValueError('Stored current conductivity must be canonical')
        if expected_conductivities is not None:
            expected_sigma=validator._conductivity(expected_conductivities[volume.region_id])
            if not torch.equal(sigma,expected_sigma):
                raise ValueError('Current closure snapshot conductivity mismatch')
        gradient = record['gradients'].clone()
        tetrahedra = volume.vertices[volume.cells]
        affine = torch.einsum('tij,taj->tia',tetrahedra[:,1:]-tetrahedra[:,:1],-gradient[:,1:]/3)
        close(affine,torch.eye(3,device=sample.device,dtype=sample.dtype).expand_as(affine))
        close(gradient[:,0],-gradient[:,1:].sum(1))
        weights = sample.new_zeros(len(faces)).index_add(0,inverse,
            (volume.volumes[:,None].expand(-1,4)/4).flatten())
        close(record['nodal_volume'],weights)
        block = FaceCurrentBalance.__new__(FaceCurrentBalance)
        block.volume,block.gradients,block.nodal_volume = volume,gradient,record['nodal_volume'].clone()
        block.faces,block.face_dofs = record['faces'].clone(),record['face_dofs'].clone()
        block._face_versions = tuple((id(t),t._version) for t in (block.faces,block.face_dofs))
        block._identity = (id(volume),id(gradient),gradient._version,id(block.nodal_volume),block.nodal_volume._version)
        blocks.append(block)
    closure = HarmonicCurrentClosure.__new__(HarmonicCurrentClosure)
    closure.blocks,closure.counts,closure.conductivities = tuple(blocks),counts,tuple(v.clone() for v in raw['conductivities'])
    closure.relative_tolerance,closure.absolute_tolerance = raw['relative_tolerance'],raw['absolute_tolerance']
    closure.tangent_rule,closure.tangent_column_batch=None,8
    if active:
        from .linalg import StoppingRule
        from dataclasses import asdict
        if not isinstance(raw['tangent_rule'],dict):raise ValueError('Invalid stored DC tangent rule')
        try:rule=StoppingRule(**raw['tangent_rule'])
        except (TypeError,ValueError) as error:raise ValueError('Invalid stored DC tangent rule') from error
        if (asdict(rule)!=raw['tangent_rule'] or rule.method!='gmres' or rule.preconditioner!='none'
                or rule.recycle_size or rule.fallback!='none'
                or type(raw['tangent_column_batch']) is not int or raw['tangent_column_batch']<1):
            raise ValueError('Invalid stored DC tangent policy')
        closure.tangent_rule,closure.tangent_column_batch=rule,raw['tangent_column_batch']
    closure._versions = closure._state()
    closure.assert_current()
    return closure


def restore_floating_operator(raw,system,testing_order):
    """Rebind and validate saved Laplace arrays without quadrature or solves."""
    system.assert_current()
    sample=system.bodies[0].mesh.vertices
    def array(value,shape,dtype=None):
        if (not isinstance(value,torch.Tensor) or value.shape!=shape or value.device!=sample.device
                or value.dtype!=(sample.dtype if dtype is None else dtype) or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid floating snapshot array')
    def close(actual,expected):
        if not bool((actual-expected).abs().max()<=512*torch.finfo(sample.dtype).eps*expected.abs().max()):
            raise ValueError('Inconsistent floating snapshot geometry or operator')
    if not isinstance(raw,dict) or set(raw) != {*FLOATING_ARRAYS,'testing_order','body_faces'}:
        raise ValueError('Invalid floating operator snapshot fields')
    faces = tuple(len(b.mesh.faces) for b in system.bodies)
    if raw['body_faces'] != faces or raw['testing_order'] != testing_order:
        raise ValueError('Floating snapshot discretization mismatch')
    n,b = sum(faces),len(faces)
    shapes = {'triangles':(n,3,3),'normals':(n,3),'areas':(n,),'indicator':(n,b),'body_areas':(b,),
        'charge_test':(b,n),'lengths':(b,),'single_layer':(n,n),'mean_gradient':(n,n,3),
        'matrix':(n+b,n+b),'lu':(n+b,n+b),'pivots':(n+b,)}
    for key,shape in shapes.items():
        array(raw[key],shape,torch.int32 if key == 'pivots' else None)
    for key in ('triangles','normals','areas'):
        close(raw[key],torch.cat([getattr(body.mesh,key) for body in system.bodies]))
    indicator = torch.repeat_interleave(torch.eye(b,device=sample.device,dtype=sample.dtype),
                                       torch.tensor(faces,device=sample.device),dim=0)
    close(raw['indicator'],indicator)
    close(raw['charge_test'],indicator.T*raw['areas'][None])
    close(raw['body_areas'],indicator.T@raw['areas'])
    close(raw['lengths'],raw['body_areas'].sqrt())
    expected_matrix = torch.cat((torch.cat((raw['single_layer'],-indicator),1),
        torch.cat((raw['charge_test']/raw['lengths'][:,None],sample.new_zeros(b,b)),1)))
    close(raw['matrix'],expected_matrix)
    if bool(((raw['pivots'] < 1) | (raw['pivots'] > n+b)).any()):
        raise ValueError('Invalid floating LU pivot index')
    permutation,lower,upper = torch.lu_unpack(raw['lu'],raw['pivots'])
    close(permutation@lower@upper,raw['matrix'])
    floating = FloatingPotentialOperator.__new__(FloatingPotentialOperator)
    floating.system,floating.testing_order,floating.body_faces = system,testing_order,faces
    for key in FLOATING_ARRAYS:
        setattr(floating,key,raw[key])
    floating._versions = floating._state_versions()
    return floating


def validate_state(static,state,bindings,threshold):
    """Check nodal/current/gauge and floating equations directly, without solves."""
    closure,op = static['current_closure'],static['floating_operator']
    boundary = [None]*len(op.system.bodies)
    electric = []
    for binding,block,potential,current in zip(bindings,closure.blocks,state.dc_face_potentials,
                                              state.total_volume_current[0].real.split(closure.counts)):
        electric.append(block.electric(potential))
        gauge = (potential*block.nodal_volume).sum()
        bound = 512*torch.finfo(potential.dtype).eps*(potential.abs()*block.nodal_volume).sum()
        if not bool(gauge.abs() <= bound):
            raise ValueError('Stored current potential violates its volume gauge')
        residual = torch.linalg.vector_norm(block.weak_current(current))
        records = state.receipt.get('dc_current',{}).get('current',[])
        index = len(electric)-1
        closure.validate_receipt(records[index] if index<len(records) else None,block)
        allowed = records[index].get('threshold_A_l2') if index < len(records) else None
        if type(allowed) not in (int,float) or not math.isfinite(allowed) or allowed < 0 or not bool(residual <= allowed):
            raise ValueError('Stored DC state violates current conservation')
        from .harmonic_current import surface_face_potentials
        boundary[binding.volume.region_id-1] = surface_face_potentials(block,binding,potential)
    reference = torch.cat(electric)
    if not bool(torch.linalg.vector_norm(reference-state.cell_electric[0].real) <= threshold):
        raise ValueError('Stored DC potential violates the harmonic electric state')
    boundary = torch.cat(boundary)
    density,offsets = state.dc_surface_charge,state.dc_potential_offsets
    residual = op.single_layer@density-op.indicator@offsets-boundary
    tolerance = 512*torch.finfo(boundary.dtype).eps*(op.single_layer.abs()@density.abs()+op.indicator@offsets.abs()+boundary.abs())
    charge = op.charge_test@density
    charge_tolerance = 512*torch.finfo(boundary.dtype).eps*(op.charge_test@density.abs())
    if not bool((residual.abs() <= tolerance).all()) or not bool((charge.abs() <= charge_tolerance).all()):
        raise ValueError('Stored floating state violates boundary potential or neutral charge')
    validate_magnetic(static,state,bindings)
    return reference.flatten()


def validate_magnetic(static,state,bindings):
    """Validate recovered magnetic source means and interface equations only."""
    op=static['floating_operator']
    sample=op.triangles
    counts=tuple(len(binding.volume.cells) for binding in bindings)
    from .static_magnetic import MagnetostaticInterfaceResponse
    from .quadrature import triangle_rule
    mu_e,mu_i = static['magnetic_permeabilities']
    magnetic_receipt = state.receipt.get('dc_current',{}).get('magnetic',{})
    if (magnetic_receipt.get('formulation') != 'scalar_mu_biot_savart_single_layer_v1'
            or magnetic_receipt.get('physical_qualification') is not False):
        raise ValueError('Invalid stored magnetic response receipt')
    response = MagnetostaticInterfaceResponse.from_state(op,bindings,
        state.total_volume_current[0].real.split(counts),mu_e,mu_i,
        state.dc_magnetic_density,state.dc_magnetic_source_mean,magnetic_receipt)
    # Verify the source field, not a newly assembled interface matrix.
    nodes,weights = triangle_rule(op.triangles,op.testing_order)
    weights = weights/weights.sum(-1,keepdim=True)
    actual_mean = (response._source(nodes.reshape(-1,3),principal_value=True).reshape_as(nodes)*weights[...,None]).sum(1)
    roundoff = 512*torch.finfo(sample.dtype).eps
    if not bool((actual_mean-state.dc_magnetic_source_mean).norm() <= roundoff*actual_mean.norm()):
        raise ValueError('Stored magnetic source mean violates its volume current')
    mu_face = torch.cat([sample.new_full((size,),mu) for size,mu in zip(op.body_faces,mu_i)])
    contrast = 2*(mu_face-mu_e)/(mu_face+mu_e)
    density = state.dc_magnetic_density
    normal = (torch.einsum('qti,t->qi',op.mean_gradient,density)*op.normals).sum(-1)
    source = (state.dc_magnetic_source_mean*op.normals).sum(-1)
    residual = density+contrast*(normal-source)
    tolerance = roundoff*(density.abs()+contrast.abs()*(normal.abs()+source.abs()))
    if not bool((residual.abs() <= tolerance).all()):
        raise ValueError('Stored magnetic density violates its interface equation')
