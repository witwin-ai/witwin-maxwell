"""Public mixed tensor/scalar harmonic runtime and exact Result bridge."""
import copy
import math
from dataclasses import replace
import torch
from ..media import Material,GyromagneticFerrite
from .harmonic_runtime import HarmonicRuntime,HarmonicData,_harmonic_operator,_response_settings
from .harmonic_material import compile_harmonic_material,compile_tensor_harmonic_material,material_signature,cell_nonlinear_coefficients
from .harmonic_config import HarmonicBalanceConfig,HarmonicCutoffConfig
from .physics import validate_scene_structure
from .runtime import _compile_basis,_signature,_same_metadata,_numerical_identity,_validate_surface_queries
from .mesh import SurfaceSystem
from .tensor_medium import sampled_medium_payload
from .tensor_harmonic_preparation import prepare_mixed_tensor_harmonic_balance
from .harmonic_state import preparation_signature
from .static_tensor_material import StaticTensorChannel
from ..constants import EPSILON_0,MU_0

SCHEMA=21
KIND='coupled_sie_vie_hb_mixed_tensor_spd'
CURRENT_SCHEMA=22
CURRENT_KIND='coupled_sie_vie_hb_mixed_tensor_current'
MAGNETIC_SCHEMA=23
MAGNETIC_KIND='coupled_sie_vie_hb_mixed_tensor_current_magnetic'


def _materials(scene,frequency,count,dtype):
    structures=validate_scene_structure(scene,frequency)
    return tuple(compile_tensor_harmonic_material(s.material,frequency,count,device=scene.device,dtype=dtype,tensor_sources=True)
        if isinstance(s.material,GyromagneticFerrite) or getattr(s.material,'is_anisotropic',False)
        else compile_harmonic_material(s.material,frequency,count,device=scene.device,dtype=dtype,
            tensor_sources=True) for s in structures)


def _admit(config,materials,exterior):
    hb=config.harmonic_balance
    current_regions=tuple(i+1 for i,m in enumerate(materials) if m.tpa_scale!=0 or m.electric_dc.kind=='conductive_limit')
    if config.operator_backend not in ('dense','hmatrix','h2') or config.periodic is not None:
        raise NotImplementedError('Mixed tensor HB requires free-space dense or H/H2 preparation')
    if (hb.dc_tangent_rule is not None and not current_regions) or hb.dc_testing_order!=4:
        raise NotImplementedError('Mixed tensor HB uses regional collocation; tangent policy requires current regions')
    if config.operator_backend=='dense' and (config.stopping.method!='direct' or config.stopping.direct_storage!='resident'):
        raise NotImplementedError('Mixed tensor HB preparation requires resident direct linear solves')
    if config.operator_backend!='dense' and (config.stopping.method not in ('gmres','block_gmres')
            or config.stopping.fallback!='none' or config.stopping.recycle_size
            or config.stopping.preconditioner not in ('none','row_column_norm')):
        raise ValueError('Mixed compressed HB requires unrecycled Krylov with none or row-column scaling')
    from .tensor_harmonic_preparation import admit_mixed_dc_channels
    sample=hb.volumes[0].vertices
    admit_mixed_dc_channels(materials,exterior,tuple(v.region_id for v in hb.volumes),
                           device=sample.device,dtype=sample.dtype)
    from .physics import Medium
    for volume in hb.volumes:
        if not 1<=volume.region_id<=len(materials):raise ValueError('Mixed volume region is outside the body table')
        if any(not isinstance(m,Medium) for m in materials[volume.region_id-1].linear_samples):
            raise NotImplementedError('Mixed tensor HB volume sources require scalar regional AC kernels')
    if hb.dc_magnetic_volumes is not None:
        if not current_regions or len(hb.dc_magnetic_volumes)!=len(materials):
            raise ValueError('DC magnetic volumes require current closure and every body')
        selected={v.region_id:v for v in hb.volumes}
        if any(v.region_id in selected and v is not selected[v.region_id] for v in hb.dc_magnetic_volumes):
            raise ValueError('DC magnetic volumes must reuse selected HB volume objects')
    return current_regions


class MixedTensorHarmonicRuntime(HarmonicRuntime):
    def __init__(self,simulation):
        self.simulation=simulation;self.config=config=simulation.config
        if simulation.has_trainable_parameters:
            raise NotImplementedError('Public harmonic material/geometry parameter compilation is not differentiable yet')
        settings=config.harmonic_balance
        _validate_surface_queries(simulation.scene)
        self.materials=_materials(simulation.scene,simulation.frequency,settings.harmonics,config.dtype)
        self.exterior=compile_harmonic_material(Material(),simulation.frequency,settings.harmonics)
        current_regions=_admit(config,self.materials,self.exterior)
        enabled=[(i,s) for i,s in enumerate(simulation.scene.structures) if s.enabled]
        bases=tuple(_compile_basis(simulation.scene,s.geometry,config) for _,s in enabled)
        linear=replace(config,harmonic_balance=None)
        operators=tuple(_harmonic_operator(SurfaceSystem.create([
            (index,s.name,basis,material.linear_samples[n])
            for (index,s),basis,material in zip(enabled,bases,self.materials)]),
            self.exterior.linear_samples[n],simulation.scene.sources if n==0 else [],linear)
            for n in range(settings.harmonics))
        self.model,self.prepared=prepare_mixed_tensor_harmonic_balance(operators,settings.volumes,
            self.materials,self.exterior,query_config=config.query_config,
            dc_policy=settings.dc_policy() if current_regions else None,dc_magnetic_volumes=settings.dc_magnetic_volumes,
            **_response_settings(config))
        self.prepared['receipt'].update(public_runtime_supported=True,public_harmonic_policy=settings.numerical_payload())
        self.prepared['_signature']=preparation_signature(self.prepared)
        self.signature=self._signature()

    def _cutoff_materials(self,count):
        return _materials(self.simulation.scene,self.simulation.frequency,count,self.config.dtype)

    def _prepare_cutoff_response(self,operators,materials,exterior,config,previous):
        current_regions=_admit(config,materials,exterior)
        settings=config.harmonic_balance
        model,prepared=prepare_mixed_tensor_harmonic_balance(operators,settings.volumes,materials,exterior,
            query_config=config.query_config,reuse_prepared=previous,
            dc_policy=settings.dc_policy() if current_regions else None,
            dc_magnetic_volumes=settings.dc_magnetic_volumes,**_response_settings(config))
        prepared['receipt'].update(public_runtime_supported=True,public_harmonic_policy=settings.numerical_payload())
        prepared['_signature']=preparation_signature(prepared)
        return model,prepared

    def _extend_cutoff(self,data,policy):
        data,receipt=super()._extend_cutoff(data,policy)
        # Bind the cutoff record into native persistence without mutating a
        # previous Result or the original single-use preparation.
        prepared=dict(data.decoder.prepared)
        prepared['receipt']=dict(prepared['receipt'],cutoff=copy.deepcopy(receipt))
        prepared['_signature']=preparation_signature(prepared)
        state=replace(data.state,preparation_identity=id(prepared))
        return HarmonicData(prepared,state,data.config,data.scene_signature),receipt


def snapshot(data):
    from .tensor_harmonic_bundle import snapshot_mixed_tensor_harmonic_bundle
    data.decoder._assert_current()
    bundle,binding=snapshot_mixed_tensor_harmonic_bundle(data.state,data.decoder.prepared)
    current=bool(data.state.dc_current_regions)
    magnetic=data.state.dc_magnetic_volume is not None
    return dict(schema_version=MAGNETIC_SCHEMA if magnetic else CURRENT_SCHEMA if current else SCHEMA,payload_kind=MAGNETIC_KIND if magnetic else CURRENT_KIND if current else KIND,bundle=bundle,binding=binding,
        numerical=_numerical_identity(data.config),harmonic_config=data.config.harmonic_balance.numerical_payload(),
        scene_signature=copy.deepcopy(data.scene_signature))


def validate_result_metadata(data,metadata,solver_stats=None):
    """Bind public representation and qualification claims to native state."""
    cutoff=data.decoder.prepared['receipt'].get('cutoff')
    if cutoff is not None and (type(solver_stats) is not dict or not _same_metadata(solver_stats.get('cutoff'),cutoff)):
        raise ValueError('Mixed harmonic cutoff statistics differ from native record')
    settings=data.config.harmonic_balance.numerical_payload()
    if settings['cutoff'] is None:
        if cutoff is not None or (isinstance(solver_stats,dict) and solver_stats.get('cutoff') is not None):
            raise ValueError('Fixed mixed harmonic state unexpectedly contains adaptive cutoff execution')
    else:
        if not isinstance(cutoff,dict) or type(cutoff.get('initial_harmonics')) is not int:
            raise ValueError('Mixed harmonic adaptive cutoff record is missing or invalid')
        from .harmonic_cutoff_receipt import validate_harmonic_cutoff
        initial={**settings,'harmonics':cutoff['initial_harmonics']}
        validate_harmonic_cutoff(solver_stats,initial,settings,data.state.cell_electric,
            data.decoder.prepared['harmonics'][0]['cell_volumes'],
            volumes=[dict(region_id=v.region_id,cells=v.cells) for v in data.config.harmonic_balance.volumes],
            dc_current_regions=data.state.dc_current_regions)
    expected={**data.decoder.prepared['receipt'],'native_data':'coupled_sie_vie_hb',
        'fundamental_frequency_hz':data.decoder.prepared['fundamental_frequency'],
        'solver_converged':True,'physical_qualification':False,
        'harmonic_truncation_converged':False,'cell_cutoff_estimate_passed':None if cutoff is None else cutoff['cell_truncation_estimate_passed']}
    if type(metadata) is not dict or any(k not in metadata or not _same_metadata(metadata[k],v)
            for k,v in expected.items()):
        raise ValueError('Mixed harmonic Result metadata differs from native identity')


def restore(payload,scene):
    from .tensor_harmonic_bundle import restore_mixed_tensor_harmonic_bundle,_config
    from .linalg import StoppingRule
    from .carrier import IncidentCarrier
    magnetic=type(payload) is dict and payload.get('schema_version')==MAGNETIC_SCHEMA
    current=magnetic or (type(payload) is dict and payload.get('schema_version')==CURRENT_SCHEMA)
    schema,kind=(MAGNETIC_SCHEMA,MAGNETIC_KIND) if magnetic else (CURRENT_SCHEMA,CURRENT_KIND) if current else (SCHEMA,KIND)
    if (type(payload) is not dict or set(payload)!={'schema_version','payload_kind','bundle','binding','numerical','harmonic_config','scene_signature'}
            or type(payload['schema_version']) is not int or payload['schema_version']!=schema or payload['payload_kind']!=kind):
        raise ValueError('Invalid public mixed tensor harmonic payload')
    if torch.device(scene.device).type!='cuda':raise ValueError('Mixed harmonic Result requires CUDA Scene')
    model,prepared,state=restore_mixed_tensor_harmonic_bundle(payload['bundle'],expected_binding=payload['binding'],device=scene.device)
    settings=copy.deepcopy(payload['harmonic_config'])
    if type(settings) is not dict:raise ValueError('Invalid public mixed harmonic policy')
    if settings.get('cutoff') is not None:settings['cutoff']=HarmonicCutoffConfig(**settings['cutoff'])
    for name in ('tangent_rule','dc_tangent_rule'):
        if settings.get(name) is not None:settings[name]=StoppingRule(**settings[name])
    if magnetic!=(state.dc_magnetic_volume is not None):raise ValueError('Mixed magnetic schema differs from state')
    magnetic_volumes=tuple(b.volume for b in prepared['static']['magnetic_volume']['bindings']) if magnetic else None
    hb=HarmonicBalanceConfig(volumes=tuple(b.volume for b in prepared['bindings'][0]),dc_magnetic_volumes=magnetic_volumes,**settings)
    numerical=payload['numerical']
    compressed=isinstance(numerical,dict) and numerical.get('operator_backend') in ('hmatrix','h2')
    if compressed:
        from .physical_surface_snapshot import _restore_config
        config=replace(_restore_config(numerical),harmonic_balance=hb)
    else:config=replace(_config(numerical),harmonic_balance=hb)
    if prepared['response_backend']!=('compressed' if compressed else 'dense'):
        raise ValueError('Public mixed harmonic response backend differs from preparation')
    if (not _same_metadata(hb.numerical_payload(),payload['harmonic_config'])
            or not _same_metadata(prepared['receipt'].get('public_harmonic_policy'),payload['harmonic_config'])
            or len(prepared['operators'])!=hb.harmonics):
        raise ValueError('Stored public mixed harmonic policy differs from preparation')
    from .harmonic_runtime import _validate_dense_response_policy,_validate_response_policy
    for part,operator,bindings in zip(prepared['harmonics'],prepared['operators'],prepared['bindings']):
        if compressed:_validate_response_policy(part,config)
        else:_validate_dense_response_policy(part,config,operator=operator,bindings=bindings)
    frequency=prepared['fundamental_frequency']
    materials=_materials(scene,frequency,hb.harmonics,config.dtype)
    exterior=compile_harmonic_material(Material(),frequency,hb.harmonics)
    current_regions=_admit(config,materials,exterior);_validate_surface_queries(scene)
    if current!=bool(current_regions) or current!=bool(state.dc_current_regions):
        raise ValueError('Mixed harmonic current schema differs from Scene or state')
    signature=(_signature(scene,exterior.linear_samples[0],tuple(m.linear_samples[0] for m in materials)),
        tuple(material_signature(s.material) for s in scene.structures if s.enabled),frequency)
    if not _same_metadata(signature,payload['scene_signature']):raise ValueError('Mixed harmonic Scene identity differs')
    enabled=[(i,s) for i,s in enumerate(scene.structures) if s.enabled]
    bases=tuple(_compile_basis(scene,s.geometry,config) for _,s in enabled)
    for n,operator in enumerate(prepared['operators']):
        if (_numerical_identity(operator._public_config)!=_numerical_identity(config)
                or len(operator.system.bodies)!=len(materials)
                or sampled_medium_payload(operator.exterior)!=sampled_medium_payload(exterior.linear_samples[n])):
            raise ValueError('Mixed harmonic operator policy or exterior differs')
        for body,(index,structure),basis,material in zip(operator.system.bodies,enabled,bases,materials):
            if (body.structure_index!=index or body.name!=structure.name
                    or not torch.equal(body.mesh.vertices,basis.mesh.vertices) or not torch.equal(body.mesh.faces,basis.mesh.faces)
                    or sampled_medium_payload(body.interior)!=sampled_medium_payload(material.linear_samples[n])):
                raise ValueError('Mixed harmonic body geometry or material differs from Scene')
        expected=IncidentCarrier.compile(scene.sources if n==0 else [],operator.exterior)
        if expected.identity!=operator.incident_carrier.identity:raise ValueError('Mixed harmonic carrier differs from Scene')
    coefficients=cell_nonlinear_coefficients(materials,hb.volumes,state.cell_electric)
    if (set(coefficients)!=set(prepared['nonlinear_coefficients']) or
            any(not torch.equal(coefficients[k],prepared['nonlinear_coefficients'][k]) for k in coefficients)):
        raise ValueError('Mixed harmonic nonlinear coefficients differ from Scene')
    static=prepared['static']
    operator=static['tensor_floating_operator' if current else 'tensor_operator']
    if current:
        closure=static['tensor_current_closure']
        if operator.current_regions!=current_regions:
            raise ValueError('Mixed harmonic DC current regions differ from Scene')
        for binding,block,sigma in zip(closure.current_bindings,closure.current.blocks,closure.current.conductivities):
            expected=block._conductivity(materials[binding.volume.region_id-1].electric_dc.conductivity)
            if not torch.equal(sigma,expected):raise ValueError('Mixed harmonic DC conductivity differs from Scene')
        from .harmonic_current import HarmonicCurrentClosure
        if (closure.current.tangent_rule!=hb.dc_tangent_rule
                or closure.current.tangent_column_batch!=hb.dc_tangent_column_batch
                or closure.current.relative_tolerance!=HarmonicCurrentClosure.relative_tolerance_default
                or closure.current.absolute_tolerance!=HarmonicCurrentClosure.absolute_tolerance_default):
            raise ValueError('Mixed harmonic DC current policy differs from public configuration')
    for region,(kernel,material) in enumerate(zip((operator.exterior,*operator.interiors),(exterior,*materials))):
        if region in current_regions:
            if kernel is not None:raise ValueError('Mixed current body cannot own a dielectric DC kernel')
            continue
        channel=material.electric_dc
        coefficient=(channel.dielectric_sample() if isinstance(channel,StaticTensorChannel) else
                     channel.dielectric_sample(EPSILON_0)*torch.eye(3,device=scene.device,dtype=kernel.coefficient.dtype))
        if not torch.equal(kernel.coefficient,coefficient):raise ValueError('Mixed harmonic DC coefficient differs from Scene')
    if magnetic:
        response=state.dc_magnetic_volume
        for kernel,material in zip((response.exterior,*response.interiors),(exterior,*materials)):
            channel=material.magnetic_dc
            coefficient=(channel.dielectric_sample() if isinstance(channel,StaticTensorChannel) else
                channel.dielectric_sample(MU_0)*torch.eye(3,device=scene.device,dtype=kernel.coefficient.dtype))
            if not torch.equal(coefficient,kernel.coefficient):
                raise ValueError('Mixed magnetic DC coefficient differs from Scene')
    if (state.receipt['pump_scale']!=hb.pump_scales[-1]
            or state.receipt['solver'].get('branch_id')!=hb.branch_id):
        raise ValueError('Mixed harmonic branch or pump policy differs')
    from .harmonic_snapshot import validate_solver_policy
    solver=state.receipt['solver']
    packed=model._pack(hb.pump_scales[-1]*model.background)
    expected_threshold=hb.absolute_tolerance+hb.relative_tolerance*float(torch.linalg.vector_norm(packed))
    if not math.isfinite(expected_threshold) or solver.get('threshold_V_per_m_l2')!=expected_threshold:
        raise ValueError('Stored harmonic solver threshold disagrees with configured tolerances')
    validate_solver_policy(solver,hb.numerical_payload(),harmonics=hb.harmonics,
        cell_count=sum(len(v.cells) for v in hb.volumes),real_dtype=state.cell_electric.real.dtype)
    return HarmonicData(prepared,state,config,signature)
