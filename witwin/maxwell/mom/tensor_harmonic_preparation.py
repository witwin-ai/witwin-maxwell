"""Mixed scalar/tensor volume harmonic preparation.

Tensor AC volumes use their actual regional dyadic kernel and admitted static
parts. Analytic real SPD DC retains its separate potential/conormal maps.
Public tensor-source admission is controlled by the runtime independently.
"""
import math
import torch
from ..constants import EPSILON_0, MU_0
from .physics import Medium
from .tensor_medium import sampled_medium_payload
from .harmonic_material import HarmonicMaterial, StaticChannel, cell_nonlinear_coefficients
from .static_tensor_material import StaticTensorChannel
from .static_tensor_interface import StaticTensorInterfaceOperator
from .static_tensor_preparation import prepare_static_tensor_cell_response
from .volume_mesh import bind_volume_region
from .volume_response_matrix import prepare_cell_field_response
from .harmonic_balance import DenseHarmonicBalance



class MixedHarmonicDCAdmissionError(NotImplementedError):
    """All unsupported analytic DC channels, without changing material data."""
    def __init__(self,failures):
        self.failures=tuple(dict(row) for row in failures)
        super().__init__('Mixed harmonic DC admission failed: '+'; '.join(
            f"region {row['region']} {row['channel']}: {row['reason']}" for row in self.failures))


def admit_mixed_dc_channels(materials,exterior_material,regions,*,device,dtype,tensor_current=False):
    """Admit every analytic channel before assembly; return actual electric data."""
    if type(tensor_current) is not bool:raise TypeError('tensor_current must be an explicit boolean')
    eye=torch.eye(3,device=device,dtype=dtype)
    electric=[];conductivities={};failures=[]
    for region,material in enumerate((exterior_material,*materials)):
        for name,scale in (('electric',EPSILON_0),('magnetic',MU_0)):
            channel=getattr(material,name+'_dc')
            try:
                current=region>0 and name=='electric' and (material.tpa_scale!=0 or channel.kind=='conductive_limit')
                if current:
                    if region not in regions:raise ValueError('Every mixed current body requires an explicit volume')
                    if isinstance(channel,StaticTensorChannel):
                        if not tensor_current:
                            raise NotImplementedError('Mixed tensor conductivity requires explicit internal tensor-current admission')
                        if channel.absolute_scale!=scale:
                            raise ValueError('Mixed DC constitutive channel has incompatible units')
                        conductivity=channel.conductivity_sample()
                        if conductivity.device!=eye.device or conductivity.dtype!=dtype:
                            raise ValueError('Mixed DC conductivity must match device and precision')
                    elif isinstance(channel,StaticChannel):conductivity=channel.conductivity
                    else:raise TypeError('Mixed harmonic current channel is not supported')
                    if channel.kind not in ('conductive_limit','finite_positive_dielectric'):
                        raise NotImplementedError('Mixed current requires its actual finite dielectric or conductive DC channel')
                    conductivities[region]=conductivity;value=None
                elif isinstance(channel,StaticTensorChannel):
                    value=channel.dielectric_sample()
                    if channel.absolute_scale!=scale:
                        raise ValueError('Mixed DC constitutive channel has incompatible units')
                elif isinstance(channel,StaticChannel):value=channel.dielectric_sample(scale)*eye
                else:raise TypeError('Mixed harmonic DC channel is not supported')
                if name=='electric':electric.append(value)
            except NotImplementedError as error:
                failures.append(dict(region=region,channel=name,kind=channel.kind,reason=str(error)))
    if failures:raise MixedHarmonicDCAdmissionError(failures)
    return electric,conductivities


def prepare_mixed_tensor_harmonic_balance(operators, volumes, body_materials, exterior_material, *,
        query_config=None, static_pair_batch=4096, max_static_matrix_bytes=256*1024**2,
        max_static_response_bytes=256*1024**2, dc_policy=None, dc_magnetic_volumes=None, reuse_prepared=None,
        compressed_settings=None, tensor_current=False, **matrix_settings):
    """Prepare retained HB equations from actual AC maps and analytic SPD DC.

    DC admission is shared with public preparation. Retained DC data use their
    regional potential/conormal representation and optional current closure.
    Reuse requires the exact operator/volume prefix and unchanged static physics,
    nonlinear coefficients and DC policy. AC integration settings apply only to
    newly appended frequencies; old responses and their receipts are retained.
    Explicit compressed settings select admitted AC actions while retaining the
    same analytic DC preparation. Public admission and bundle persistence have
    separate contracts.
    """
    operators,volumes,materials=tuple(operators),tuple(volumes),tuple(body_materials)
    if not operators or not volumes or not all(isinstance(m,HarmonicMaterial) for m in (*materials,exterior_material)):
        raise ValueError('Mixed tensor HB requires operators, volumes and sampled materials')
    compressed=compressed_settings is not None
    if compressed and (not isinstance(compressed_settings,dict) or matrix_settings):
        raise ValueError('Compressed mixed HB requires separate compressed settings, not dense matrix settings')
    backend='compressed' if compressed else 'dense'
    system=operators[0].system;system.assert_current()
    if len(materials)!=len(system.bodies):
        raise ValueError('Every mixed surface body requires a sampled material')
    count=len(operators);frequency=exterior_material.fundamental_frequency
    if isinstance(frequency,bool) or not math.isfinite(frequency) or frequency<=0:
        raise ValueError('Mixed harmonic fundamental frequency must be finite and positive')
    if any(len(m.linear_samples)!=count or m.fundamental_frequency!=frequency for m in (*materials,exterior_material)):
        raise ValueError('Mixed harmonic material axes must match the operators')
    if any(sample.frequency!=(n+1)*frequency for m in (*materials,exterior_material)
           for n,sample in enumerate(m.linear_samples)):
        raise ValueError('Mixed harmonic samples must match the actual integer frequency axis')
    regions=tuple(v.region_id for v in volumes)
    if len(set(regions))!=len(regions) or any(r<1 or r>len(materials) for r in regions):
        raise ValueError('Mixed harmonic volume regions must be unique existing bodies')
    if any(m.is_nonlinear and i+1 not in regions for i,m in enumerate(materials)):
        raise ValueError('Every nonlinear body needs an explicit volume')
    if exterior_material.is_nonlinear:
        raise ValueError('Mixed harmonic exterior must be linear')
    from .tensor_volume import tensor_volume_static_kernels
    for region in regions:
        for sample in materials[region-1].linear_samples:
            if not isinstance(sample,Medium):
                tensor_volume_static_kernels(sample)
    electric,conductivities=admit_mixed_dc_channels(materials,exterior_material,regions,
        device=system.device,dtype=system.dtype,tensor_current=tensor_current)
    if dc_policy is not None and not conductivities:
        raise ValueError('Mixed DC current policy requires current-balance regions')
    static_policy=dict(pair_batch=static_pair_batch,max_matrix_bytes=max_static_matrix_bytes,
                       max_response_bytes=max_static_response_bytes)
    reuse_count=0
    if reuse_prepared is not None:
        from .harmonic_state import preparation_signature
        if (type(reuse_prepared) is not dict or
                preparation_signature(reuse_prepared)!=reuse_prepared.get('_signature')):
            raise ValueError('Reused mixed harmonic preparation changed after validation')
        if reuse_prepared['receipt'].get('static_preparation_policy')!=static_policy:
            raise ValueError('Mixed harmonic extension must preserve static preparation policy and budgets')
        reuse_count=len(reuse_prepared['operators'])
        if (reuse_prepared.get('response_backend')!=backend
                or reuse_prepared['fundamental_frequency']!=frequency
                or not 0<reuse_count<count or any(old is not new for old,new in
                    zip(reuse_prepared['operators'],operators[:reuse_count]))):
            raise ValueError('Mixed harmonic extension requires the exact operator prefix and response backend')
        if any(len(group)!=len(volumes) or any(binding.volume is not volume
                for binding,volume in zip(group,volumes)) for group in reuse_prepared['bindings']):
            raise ValueError('Mixed harmonic extension requires the exact prepared volume meshes')
        for group in reuse_prepared['bindings']:
            for binding in group:binding.assert_current()
        coefficients=cell_nonlinear_coefficients(materials,volumes,reuse_prepared['harmonics'][0]['background_electric'])
        if (set(coefficients)!=set(reuse_prepared['nonlinear_coefficients']) or
                any(not torch.equal(value,reuse_prepared['nonlinear_coefficients'][name]) for name,value in coefficients.items())):
            raise ValueError('Mixed harmonic extension must preserve nonlinear coefficients')
        old_static=reuse_prepared['static']
        old_closure=old_static.get('tensor_current_closure')
        if bool(conductivities)!=(old_closure is not None):
            raise ValueError('Mixed harmonic extension must preserve DC current regions')
        old_operator=old_static['tensor_floating_operator' if old_closure is not None else 'tensor_operator']
        for value,kernel in zip(electric,(old_operator.exterior,*old_operator.interiors)):
            if (value is None)!=(kernel is None) or (value is not None and not torch.equal(value,kernel.coefficient)):
                raise ValueError('Mixed harmonic extension must preserve DC electric samples')
        if old_closure is not None:
            if tuple(conductivities)!=old_operator.current_regions:
                raise ValueError('Mixed harmonic extension must preserve DC current regions')
            for binding,block,sigma in zip(old_closure.current_bindings,old_closure.current.blocks,old_closure.current.conductivities):
                if not torch.equal(sigma,block._conductivity(conductivities[binding.volume.region_id])):
                    raise ValueError('Mixed harmonic extension must preserve DC conductivity')
            policy=dc_policy or {'tangent_rule':None,'tangent_column_batch':8}
            if (old_closure.current.tangent_rule!=policy['tangent_rule'] or
                    old_closure.current.tangent_column_batch!=policy['tangent_column_batch']):
                raise ValueError('Mixed harmonic extension must preserve DC tangent policy')
        if (dc_magnetic_volumes is not None)!=('magnetic_volume' in old_static):
            raise ValueError('Mixed harmonic extension must preserve magnetic DC recovery')
    magnetic=None
    if dc_magnetic_volumes is not None:
        if not conductivities:
            raise ValueError('Magnetic volume recovery requires mixed DC current regions')
        magnetic_volumes=tuple(dc_magnetic_volumes)
        if tuple(v.region_id for v in magnetic_volumes)!=tuple(range(1,len(materials)+1)):
            raise ValueError('Magnetic DC volumes must cover every body in region order')
        selected=dict(zip(regions,volumes))
        if any(v.region_id in selected and v is not selected[v.region_id] for v in magnetic_volumes):
            raise ValueError('Magnetic DC current volumes must be the exact selected HB volumes')
        if reuse_count:
            magnetic_bindings=old_static['magnetic_volume']['bindings']
            if any(binding.volume is not volume for binding,volume in zip(magnetic_bindings,magnetic_volumes)):
                raise ValueError('Mixed harmonic extension requires exact magnetic volume meshes')
        else:magnetic_bindings=tuple(bind_volume_region(v,system) for v in magnetic_volumes)
        eye=torch.eye(3,device=system.device,dtype=system.dtype)
        samples=[]
        for material in (exterior_material,*materials):
            channel=material.magnetic_dc
            value=channel.dielectric_sample() if isinstance(channel,StaticTensorChannel) else channel.dielectric_sample(MU_0)*eye
            samples.append(value.clone())
        if reuse_count:
            old_magnetic=old_static['magnetic_volume']
            if any(not torch.equal(a,b) for a,b in zip(samples,(old_magnetic['exterior'],*old_magnetic['interiors']))):
                raise ValueError('Mixed harmonic extension must preserve DC magnetic samples')
        magnetic=dict(bindings=magnetic_bindings,exterior=samples[0],interiors=tuple(samples[1:]),
                      pair_batch=static_pair_batch,max_matrix_bytes=max_static_matrix_bytes)
    groups=[]
    for harmonic,operator in enumerate(operators):
        if harmonic<reuse_count and not hasattr(operator,'refresh'):
            operator.system.assert_current();operator.validate_tensor_state()
        else:
            operator.refresh()
        if len(operator.system.bodies)!=len(system.bodies) or operator.dtype!=operators[0].dtype:
            raise ValueError('Mixed operators must use a common body table and precision')
        if sampled_medium_payload(operator.exterior)!=sampled_medium_payload(exterior_material.linear_samples[harmonic]):
            raise ValueError('Mixed harmonic exterior sample differs')
        if not isinstance(operator.exterior,Medium):
            raise NotImplementedError('Mixed tensor HB currently requires a scalar exterior carrier')
        for index,body in enumerate(operator.system.bodies):
            reference=system.bodies[index]
            if (body.name!=reference.name or body.structure_index!=reference.structure_index
                    or not torch.equal(body.mesh.vertices,reference.mesh.vertices)
                    or not torch.equal(body.mesh.faces,reference.mesh.faces)
                    or sampled_medium_payload(body.interior)!=sampled_medium_payload(materials[index].linear_samples[harmonic])):
                raise ValueError('Mixed harmonic body geometry or sampled material differs')
        groups.append(reuse_prepared['bindings'][harmonic] if harmonic<reuse_count else
                      tuple(bind_volume_region(v,operator.system) for v in volumes))
    closure=None
    if reuse_count:
        static=old_static;closure=old_closure
    elif conductivities:
        from .static_tensor_floating import StaticTensorFloatingOperator
        from .harmonic_mixed import MixedHarmonicClosure
        static_operator=StaticTensorFloatingOperator(system,electric[0],electric[1:],
            pair_batch=static_pair_batch,max_matrix_bytes=max_static_matrix_bytes)
        closure=MixedHarmonicClosure(static_operator,groups[0],conductivities,dc_policy=dc_policy,
                                    max_response_bytes=max_static_response_bytes)
        static=dict(tensor_current_closure=closure,tensor_floating_operator=static_operator,
                    maps=closure.maps)
    else:
        static_operator=StaticTensorInterfaceOperator(system,electric[0],electric[1:],
            pair_batch=static_pair_batch,max_matrix_bytes=max_static_matrix_bytes)
        static=prepare_static_tensor_cell_response(static_operator,groups[0],max_response_bytes=max_static_response_bytes)
        static['tensor_operator']=static_operator
    if magnetic is not None and not reuse_count:static['magnetic_volume']=magnetic
    from .harmonic_preparation import _prepare_compressed_harmonic
    dynamic=tuple(reuse_prepared['harmonics'] if reuse_count else ())+tuple(
        (_prepare_compressed_harmonic(op,bindings,query_config,compressed_settings) if compressed else
         prepare_cell_field_response(op,bindings,query_config=query_config,**matrix_settings))
        for op,bindings in zip(operators[reuse_count:],groups[reuse_count:]))
    cells=sum(len(v.cells) for v in volumes)
    background=torch.stack([dynamic[0]['background_electric'].new_zeros(cells,3)]+
                            [p['background_electric'] for p in dynamic])
    coefficients=cell_nonlinear_coefficients(materials,volumes,background)
    if compressed:
        from .action_harmonic_balance import ActionHarmonicBalance
        dc={} if closure is not None else dict(
            dc_polarization_response=static['electric_response'].to(background.dtype),
            dc_current_response=background.new_zeros(3*cells,3*cells))
        model=ActionHarmonicBalance(background,[p['electric_action'] for p in dynamic],
            dynamic[0]['cell_volumes'],frequency,dc_current_closure=closure,**dc,**coefficients)
    else:
        current=torch.stack([dynamic[0]['electric_response'].new_zeros(3*cells,3*cells)]+
                             [p['electric_response'] for p in dynamic])
        polarization=torch.stack([torch.zeros_like(current[0]) if closure is not None else static['electric_response'].to(current.dtype)]+
            [-1j*op.exterior.omega*p['electric_response'] for op,p in zip(operators,dynamic)])
        model=DenseHarmonicBalance(background,polarization,current,dynamic[0]['cell_volumes'],frequency,
                                  **coefficients,dc_current_closure=closure)
    prepared=dict(response_backend=backend,static=static,harmonics=dynamic,bindings=tuple(groups),
        operators=operators,_model_identity=id(model),fundamental_frequency=frequency,
        nonlinear_coefficients={name:getattr(model,name).clone() for name in coefficients},receipt=dict(
            formulation=('mixed_tensor_scalar_hb_current_dc_preparation_v1' if closure is not None else
                         'mixed_tensor_scalar_hb_spd_dc_preparation_v1'),physical_qualification=False,
            dc_surface_representation='regional_potential_exterior_conormal_v1',
            harmonic_truncation_converged=False,public_runtime_supported=False,
            static_preparation_policy=static_policy,reused_positive_harmonics=reuse_count,reused_static_response=bool(reuse_count)))
    from .harmonic_state import preparation_signature
    prepared['_signature']=preparation_signature(prepared)
    return model,prepared
