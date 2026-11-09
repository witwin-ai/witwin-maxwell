"""Physical AC response preparation with dielectric or isolated-current DC."""
import torch
from ..constants import EPSILON_0,MU_0
from .static_coupling import StaticVolumeResponse
from .volume_mesh import bind_volume_region
from .volume_response_matrix import prepare_cell_field_response
from .harmonic_material import HarmonicMaterial,cell_nonlinear_coefficients
from .harmonic_balance import DenseHarmonicBalance


def prepare_static_cell_response(system,bindings,epsilon_exterior,epsilon_interiors,*,testing_order=4):
    """Differentiate a fixed linear DC discretization once to obtain all columns.

    Unlike adaptive zero-source probing, the DC Galerkin order is fixed and
    explicit; zero input does not select or certify any integration tolerance.
    """
    bindings = tuple(bindings)
    if not bindings:
        raise ValueError('DC response preparation requires volume bindings')
    counts = tuple(len(binding.volume.cells) for binding in bindings)
    points = tuple(binding.volume.vertices[binding.volume.cells].mean(1) for binding in bindings)
    sample = points[0]
    count = sum(counts)
    last = {}
    def evaluate(polarization):
        response = StaticVolumeResponse(system,epsilon_exterior,epsilon_interiors,bindings,
            polarization.split(counts),testing_order=testing_order)
        fields = [response.fields(query,binding.volume.region_id)[1]
                  for binding,query in zip(bindings,points)]
        last['receipt'] = response.receipt
        return torch.cat((torch.cat(fields).flatten(),response.charge))
    zero = sample.new_zeros(count,3)
    jacobian = torch.autograd.functional.jacobian(evaluate,zero,vectorize=False).reshape(-1,3*count)
    return {'electric_response':jacobian[:3*count],'surface_response':jacobian[3*count:],
        'points':torch.cat(points),'receipt':{**last['receipt'],
            'preparation':'native_jacobian_of_fixed_linear_dc_operator',
            'quadrature_convergence_certified':False,'physical_qualification':False}}


def prepare_harmonic_balance(operators,volumes,body_materials,exterior_material,*,
                                       query_config=None,dc_testing_order=4,reuse_prepared=None,
                                       compressed_settings=None,dc_policy=None,**matrix_settings):
    """Bind actual scalar SIE/VIE maps to the retained-harmonic nonlinear solve.

    DC uses dielectric displacement or isolated face current balance with a
    floating exterior. Mixed DC materials use dielectric rows and floating current rows; plasma limits remain unresolved.
    reuse_prepared extends an exact existing operator/volume prefix; requested
    query/matrix settings apply only to newly appended frequencies.
    compressed_settings explicitly selects admitted H/H2 responses and a guarded
    configurable inverse_rule; public H/H2 dispatch supplies the matching policy.
    """
    operators,volumes,materials = tuple(operators),tuple(volumes),tuple(body_materials)
    if not operators or not volumes or not all(isinstance(m,HarmonicMaterial) for m in (*materials,exterior_material)):
        raise ValueError('Physical HB requires operators, volumes and sampled harmonic materials')
    if any(m.chi2_tensor is not None or m.chi3_tensor is not None for m in (*materials,exterior_material)):
        raise NotImplementedError('Tensor susceptibility sources require mixed harmonic preparation')
    compressed = compressed_settings is not None
    if compressed and (not isinstance(compressed_settings,dict) or matrix_settings):
        raise ValueError('Compressed HB requires separate compressed settings, not dense matrix settings')
    system = operators[0].system
    if len(materials) != len(system.bodies):
        raise ValueError('Every surface body requires a harmonic material')
    harmonics = len(operators)
    if any(len(m.linear_samples) != harmonics or m.fundamental_frequency != exterior_material.fundamental_frequency
           for m in (*materials,exterior_material)):
        raise ValueError('Harmonic material axes must match all supplied operators')
    eps_e = exterior_material.electric_dc.dielectric_sample(EPSILON_0)
    current_flags = tuple(m.tpa_scale != 0 or m.electric_dc.kind == 'conductive_limit' for m in materials)
    from .harmonic_dual_bridge import dual_materials
    dual_dc = dual_materials(materials)
    current_dc = any(current_flags)
    mixed_dc = current_dc and not all(current_flags)
    if dc_policy and dc_policy.get('tangent_rule') is not None and not current_dc:
        raise NotImplementedError('DC tangent policy requires electric current or mixed current closure')
    if current_dc and any(m.electric_dc.kind not in ('finite_positive_dielectric','conductive_limit') for m in materials):
        raise NotImplementedError('Current DC preparation does not resolve plasma or complex static channels')
    eps_i = (tuple(None if flag else m.electric_dc.dielectric_sample(EPSILON_0)
                   for m,flag in zip(materials,current_flags)) if mixed_dc else
             None if current_dc else tuple(m.electric_dc.dielectric_sample(EPSILON_0) for m in materials))
    magnetic_dc = ((exterior_material.magnetic_dc.dielectric_sample(MU_0),
                    tuple(m.magnetic_dc.dielectric_sample(MU_0) for m in materials)) if current_dc and not dual_dc else None)
    channels = tuple((m.electric_dc.kind,m.electric_dc.finite_relative_part,m.electric_dc.conductivity,
                      m.electric_dc.plasma_angular_frequency_squared) for m in materials)
    regions = [volume.region_id for volume in volumes]
    if len(set(regions)) != len(regions) or any(region < 1 or region > len(materials) for region in regions):
        raise ValueError('Physical HB volume regions must be unique existing bodies')
    if any(m.is_nonlinear and i+1 not in regions for i,m in enumerate(materials)):
        raise ValueError('Every nonlinear body requires an explicit volume discretization')
    if current_dc and sorted(regions) != list(range(1,len(materials)+1)):
        raise ValueError('Every current DC body requires an explicit volume discretization')
    reuse_count = 0
    if reuse_prepared is not None:
        from .harmonic_state import preparation_signature
        if (not isinstance(reuse_prepared,dict) or
                preparation_signature(reuse_prepared) != reuse_prepared.get('_signature')):
            raise ValueError('Reused harmonic preparation changed after validation')
        if reuse_prepared.get('response_backend','dense') != ('compressed' if compressed else 'dense'):
            raise ValueError('Harmonic extension must preserve its response backend')
        reuse_count = len(reuse_prepared['operators'])
        if (not 0 < reuse_count < harmonics or any(old is not new for old,new in
                zip(reuse_prepared['operators'],operators[:reuse_count]))):
            raise ValueError('Harmonic extension requires the exact prepared operator prefix')
        if (reuse_prepared['dc_permittivities'] != (eps_e,eps_i)
                or reuse_prepared['dc_testing_order'] != dc_testing_order):
            raise ValueError('Harmonic extension must preserve the DC material and testing rule')
        if current_dc and reuse_prepared['static'].get('material_channels') != channels:
            raise ValueError('Harmonic extension must preserve the DC current material channels')
        if current_dc and reuse_prepared['static'].get('magnetic_permeabilities') != magnetic_dc:
            raise ValueError('Harmonic extension must preserve DC magnetic material samples')
        if any(len(group) != len(volumes) or any(binding.volume is not volume
                for binding,volume in zip(group,volumes)) for group in reuse_prepared['bindings']):
            raise ValueError('Harmonic extension requires the exact prepared volume meshes')
        for group in reuse_prepared['bindings']:
            for binding in group:
                binding.assert_current()
    all_bindings = []
    for harmonic,operator in enumerate(operators):
        if len(operator.system.bodies) != len(system.bodies):
            raise ValueError('Harmonic operators must use the same body table')
        expected = exterior_material.linear_samples[harmonic]
        if (operator.exterior.frequency,operator.exterior.epsilon_r,operator.exterior.mu_r) != (
                expected.frequency,expected.epsilon_r,expected.mu_r):
            raise ValueError('Harmonic exterior does not match the sampled linear background')
        for index,body in enumerate(operator.system.bodies):
            reference = system.bodies[index]
            expected = materials[index].linear_samples[harmonic]
            if (body.name != reference.name or body.structure_index != reference.structure_index
                    or not torch.equal(body.mesh.vertices,reference.mesh.vertices)
                    or not torch.equal(body.mesh.faces,reference.mesh.faces)
                    or (body.interior.frequency,body.interior.epsilon_r,body.interior.mu_r) != (
                        expected.frequency,expected.epsilon_r,expected.mu_r)):
                raise ValueError('Harmonic body geometry/material does not match the common discretization')
        all_bindings.append(reuse_prepared['bindings'][harmonic] if harmonic < reuse_count else
                            tuple(bind_volume_region(volume,operator.system) for volume in volumes))
    if reuse_count:
        static = reuse_prepared['static']
        if dual_dc:
            selected=static['current_closure']
            policy=dc_policy or {'tangent_rule':None,'tangent_column_batch':8}
            if selected.streamed_rule!=policy['tangent_rule'] or selected.streamed_column_batch!=policy['tangent_column_batch']:
                raise ValueError('Harmonic extension must preserve dual DC tangent policy')
        elif current_dc:
            selected=static['current_closure'].current if mixed_dc else static['current_closure']
            policy=dc_policy or {'tangent_rule':None,'tangent_column_batch':8}
            if selected.tangent_rule!=policy['tangent_rule'] or selected.tangent_column_batch!=policy['tangent_column_batch']:
                raise ValueError('Harmonic extension must preserve DC tangent policy')
    elif dual_dc:
        from .harmonic_dual_current import DualHarmonicCurrentClosure
        from .static_floating import FloatingPotentialOperator
        closure=DualHarmonicCurrentClosure(volumes,
            [materials[v.region_id-1].electric_dc.conductivity for v in volumes],
            [materials[v.region_id-1].magnetic_dc.conductivity for v in volumes],
            streamed_rule=(dc_policy or {}).get('tangent_rule'),
            streamed_column_batch=(dc_policy or {}).get('tangent_column_batch',8))
        static={'current_closure':closure,'floating_operator':FloatingPotentialOperator(system,testing_order=dc_testing_order),
                'material_channels':channels,'magnetic_permeabilities':None,
                'points':torch.cat([v.vertices[v.cells].mean(1) for v in volumes]),
                'receipt':{'formulation':'dual_conductive_floating_dc_v1','physical_qualification':False}}
    elif mixed_dc:
        from .static_floating import FloatingPotentialOperator
        from .static_mixed import MixedStaticOperator
        from .harmonic_mixed import MixedHarmonicClosure
        floating = FloatingPotentialOperator(system,testing_order=dc_testing_order)
        mixed = MixedStaticOperator(floating,eps_e,eps_i)
        closure = MixedHarmonicClosure(mixed,all_bindings[0],
            {i+1:m.electric_dc.conductivity for i,m in enumerate(materials) if current_flags[i]},dc_policy=dc_policy)
        static = {'current_closure':closure,'floating_operator':floating,'mixed_operator':mixed,
                  'material_channels':channels,'magnetic_permeabilities':magnetic_dc,
                  'points':torch.cat([v.vertices[v.cells].mean(1) for v in volumes]),
                  'receipt':{'formulation':'mixed_dielectric_floating_current_dc_v1','physical_qualification':False}}
    elif current_dc:
        from .harmonic_current import HarmonicCurrentClosure
        from .static_floating import FloatingPotentialOperator
        closure = HarmonicCurrentClosure(volumes,[materials[v.region_id-1].electric_dc.conductivity for v in volumes],**(dc_policy or {}))
        static = {'current_closure':closure,'floating_operator':FloatingPotentialOperator(system,testing_order=dc_testing_order),
                  'material_channels':channels,'magnetic_permeabilities':magnetic_dc,
                  'points':torch.cat([v.vertices[v.cells].mean(1) for v in volumes]),
                  'receipt':{'formulation':'isolated_current_with_floating_exterior_v1','physical_qualification':False}}
    else:
        static = prepare_static_cell_response(system,all_bindings[0],eps_e,eps_i,testing_order=dc_testing_order)
    # New integration settings apply only to appended frequencies. Existing
    # responses retain their original empirical error receipts unchanged.
    dynamic = tuple(reuse_prepared['harmonics'] if reuse_count else ())+tuple(
        (_prepare_compressed_harmonic(operator,bindings,query_config,compressed_settings) if compressed else
         prepare_cell_field_response(operator,bindings,query_config=query_config,**matrix_settings))
        for operator,bindings in zip(operators[reuse_count:],all_bindings[reuse_count:]))
    dtype = operators[0].dtype
    count = sum(len(volume.cells) for volume in volumes)
    background = torch.stack([dynamic[0]['background_electric'].new_zeros(count,3)]
                             +[part['background_electric'] for part in dynamic])
    coefficients = cell_nonlinear_coefficients(materials,volumes,background)
    if compressed:
        from .action_harmonic_balance import ActionHarmonicBalance
        dc = {} if current_dc else dict(dc_polarization_response=static['electric_response'].to(dtype),
            dc_current_response=background.new_zeros(3*count,3*count))
        model = ActionHarmonicBalance(background,[part['electric_action'] for part in dynamic],
            dynamic[0]['cell_volumes'],exterior_material.fundamental_frequency,
            dc_current_closure=static.get('current_closure'),**dc,**coefficients)
    else:
        current = torch.stack([dynamic[0]['electric_response'].new_zeros(3*count,3*count)]
                              +[part['electric_response'] for part in dynamic])
        polarization = torch.stack([torch.zeros_like(current[0]) if current_dc else static['electric_response'].to(dtype)]
            +[-1j*operator.exterior.omega*part['electric_response'] for operator,part in zip(operators,dynamic)])
        model = DenseHarmonicBalance(background,polarization,current,dynamic[0]['cell_volumes'],
            exterior_material.fundamental_frequency,dc_current_closure=static.get('current_closure'),**coefficients)
    prepared = {'response_backend':'compressed' if compressed else 'dense','static':static,'harmonics':dynamic,'bindings':tuple(all_bindings),
        'operators':operators,'_model_identity':id(model),
        'dc_permittivities':(eps_e,eps_i),'dc_testing_order':dc_testing_order,
        'receipt':{'formulation':'coupled_sie_vie_hb_dual_current_candidate_v1' if dual_dc else
                       'coupled_sie_vie_hb_mixed_candidate_v1' if mixed_dc else
                       'coupled_sie_vie_hb_current_candidate_v1' if current_dc else 'coupled_sie_vie_hb_dielectric_candidate_v1',
                   'volume_sampling':'cell_centroid_collocation','physical_qualification':False,
                   'reused_positive_harmonics':reuse_count,'reused_static_response':bool(reuse_count),
                   'mean_current_dc_supported':current_dc,'dc_magnetic_fields_supported':current_dc,
                   'mixed_dc_state_recovery_supported':mixed_dc,
                   'harmonic_truncation_converged':False}}

    from .harmonic_state import preparation_signature
    prepared['_signature'] = preparation_signature(prepared)
    return model,prepared


def _prepare_compressed_harmonic(operator,bindings,query_config,settings):
    """Admit induced/background actions with an explicitly configurable inner rule."""
    from .compressed_volume_preparation import prepare_compressed_cell_response
    from .implicit_surface_solve import FixedSurfaceInverse
    from .fields import FieldQueryConfig
    options=dict(settings)
    from .linalg import StoppingRule
    rule=options.pop('inverse_rule',StoppingRule(method='block_gmres',
        preconditioner='row_column_norm',rtol=1e-11,atol=1e-20,stagnation_window=48))
    inverse=FixedSurfaceInverse(operator,rule)
    result=prepare_compressed_cell_response(inverse,bindings,
        query_config=FieldQueryConfig() if query_config is None else query_config,
        incident_sources=operator.sources,**options)
    bank=result['background_bank']
    result.update(bank.combine(torch.ones(len(bank.carriers),device=inverse.device,dtype=inverse.dtype)))
    result['cell_volumes']=torch.cat([b.volume.volumes for b in bindings])
    result['points']=torch.cat([b.volume.vertices[b.volume.cells].mean(1) for b in bindings])
    return result
