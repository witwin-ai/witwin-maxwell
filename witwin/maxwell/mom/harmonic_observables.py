"""Full-field power diagnostics for recovered finite-dielectric harmonics."""
import copy
import torch
import math
from .mesh import SurfaceMesh
from .quadrature import triangle_rule
from .observables import integrate_volume_dissipation


def harmonic_power_balance(data,control_surface,*,surface_order=3,volume_order=3,query_config=None,query_batch_size=None,_cache=None):
    """Compare exterior flux with separately integrated regional work in watts.

    A convex closed surface must strictly enclose every physical body. The
    source-free incident flux is removed analytically; scattering and incident
    interference are integrated explicitly. Fixed integration orders do not
    certify quadrature convergence or a production power-balance threshold.
    """
    if query_batch_size is not None and (type(query_batch_size) is not int or query_batch_size<1):
        raise ValueError('Power query batch size must be a positive integer or None')
    def ac_fields(points,region,harmonic):
        batches=(points,) if query_batch_size is None else points.split(query_batch_size)
        electric,magnetic,logs=[],[],[]
        for batch in batches:
            e,h=data.query_fields(batch,region,'total',harmonic=harmonic,query_config=config)
            electric.append(e);magnetic.append(h)
            logs.append(copy.deepcopy(data.decoder.last_query_receipt))
        return torch.cat(electric),torch.cat(magnetic),(logs[0] if len(logs)==1 else
            {'query_count':len(points),'batch_size':query_batch_size,'batches':logs})
    cache = {} if _cache is None else _cache
    data.decoder._assert_current()
    system = data.decoder.prepared['operators'][0].system
    if not isinstance(control_surface,SurfaceMesh):
        raise TypeError('Power control surface must be a closed SurfaceMesh')
    control_surface = SurfaceMesh(control_surface.vertices.clone(),control_surface.faces.clone())
    if control_surface.device != system.device or control_surface.dtype != system.dtype:
        raise ValueError('Power control surface must match the harmonic CUDA geometry')
    if any(type(order) is not int or order < 2 for order in (surface_order,volume_order)):
        raise ValueError('Power integration orders must be integers >= 2')
    normals = control_surface.normals
    origins = control_surface.triangles[:,0]
    scale = (control_surface.vertices.amax(0)-control_surface.vertices.amin(0)).norm()
    tolerance = 128*torch.finfo(system.dtype).eps*scale
    planes = lambda points: torch.einsum('vfi,fi->vf',points[:,None]-origins,normals)
    if bool((planes(control_surface.vertices) > tolerance).any()):
        raise ValueError('Power control surface must be convex')
    for body in system.bodies:
        if not bool((planes(body.mesh.vertices) < -tolerance).all()):
            raise ValueError('Power control surface must strictly enclose every body')
    state = data.state
    dual_dc = state.dc_dual_state is not None
    current_dc = dual_dc or state.dc_potential_offsets is not None
    if bool(state.total_volume_current[0].any()) and not current_dc:
        raise NotImplementedError('DC current power requires the static current formulation')
    points,weights = triangle_rule(control_surface.triangles,surface_order)
    points,weights = points.reshape(-1,3),weights.flatten()
    normals = normals[:,None].expand(-1,len(weights)//len(normals),-1).reshape(-1,3)
    count = len(state.frequencies)
    flux,absorption,polarization,dissipation = [state.cell_electric.real.new_zeros(count) for _ in range(4)]
    scattered,interference = [torch.zeros_like(flux) for _ in range(2)]
    regions = {}
    query_logs = {}
    config = data.query_config if query_config is None else query_config
    if current_dc:
        surface_key = ('surface',0,surface_order)
        if surface_key not in cache:
            _,electric = data.query_dc(points,0)
            electric_log = data.decoder.last_query_receipt
            magnetic,_ = data.query_dc_magnetic(points,0)
            dc_flux = ((torch.linalg.cross(electric,magnetic)*normals).sum(-1)*weights).sum()
            cache[surface_key] = (dc_flux,dc_flux.new_zeros(()),
                {'electric':electric_log,'magnetic':data.decoder.last_query_receipt})
        scattered[0],interference[0],query_logs[(0,0)] = cache[surface_key]
        flux[0] = scattered[0]
        closure = data.decoder.prepared['static']['current_closure']
        if dual_dc:
            linear_closure=closure.current
            conductivity={v.region_id:sigma for v,sigma in zip(closure.volumes,
                linear_closure.electric.reshape(-1,3)[:,0].split(closure.counts))}
            magnetic_conductivity={v.region_id:sigma for v,sigma in zip(closure.volumes,
                linear_closure.magnetic.reshape(-1,3)[:,0].split(closure.counts))}
        else:
            selected = closure.current if state.dc_current_regions else closure
            conductivity = {block.volume.region_id:sigma for block,sigma in
                            zip(selected.blocks,selected.conductivities)}
        offset = 0
        for binding in data.decoder.prepared['bindings'][0]:
            volume,region = binding.volume,binding.volume.region_id
            cells = len(volume.cells)
            sigma = conductivity.get(region)
            if sigma is None:
                sigma = volume.vertices.new_zeros(cells)
            volume_key = ('volume',0,region,volume_order)
            if volume_key not in cache:
                cell_points,cell_weights = volume.quadrature(volume_order)
                if dual_dc:
                    dual_fields=data.query_dc_dual(cell_points.reshape(-1,3),region)
                    electric=dual_fields['electric_V_per_m']
                else:
                    _,electric = data.query_dc(cell_points.reshape(-1,3),region)
                local = electric.reshape(cells,-1,3)
                linear = (sigma[:,None]*local.square().sum(-1)*cell_weights).sum()
                if dual_dc:
                    magnetic=dual_fields['magnetic_A_per_m']
                    magnetic=magnetic.reshape(cells,-1,3)
                    linear=linear+(magnetic_conductivity[region][:,None]*magnetic.square().sum(-1)*cell_weights).sum()
                nonlinear = state.dissipative_current[0,offset:offset+cells].real
                work = ((local*nonlinear[:,None]).sum(-1)*cell_weights).sum()
                cache[volume_key] = (linear,linear.new_zeros(()),work,data.decoder.last_query_receipt)
            linear,p,d,query_logs[(0,region)] = cache[volume_key]
            absorption[0] += linear
            dissipation[0] += d
            regions[(0,region)] = {'linear_absorption_W':linear,'polarization_work_W':p,'dissipative_work_W':d}
            offset += cells
    for harmonic in range(1,count):
        operator = data.decoder.prepared['operators'][harmonic-1]
        surface_key = ('surface',harmonic,surface_order)
        if surface_key not in cache:
            total_e,total_h,field_log = ac_fields(points,0,harmonic)
            incident_e,incident_h = (state.receipt['pump_scale']*value for value in
                                    operator.incident_carrier.evaluate(points,operator.exterior))
            scattered_e,scattered_h = total_e-incident_e,total_h-incident_h
            scatter_vector = .5*torch.linalg.cross(scattered_e,scattered_h.conj()).real
            interference_vector = .5*(torch.linalg.cross(scattered_e,incident_h.conj())+
                                      torch.linalg.cross(incident_e,scattered_h.conj())).real
            integrate = lambda vector: ((vector*normals).sum(-1)*weights).sum()
            cache[surface_key] = (integrate(scatter_vector),integrate(interference_vector),field_log)
        scattered[harmonic],interference[harmonic],query_logs[(harmonic,0)] = cache[surface_key]
        flux[harmonic] = scattered[harmonic]+interference[harmonic]
        offset = 0
        for binding in data.decoder.prepared['bindings'][harmonic-1]:
            volume = binding.volume
            region = volume.region_id
            cells = len(volume.cells)
            volume_key = ('volume',harmonic,region,volume_order)
            if volume_key not in cache:
                cell_points,cell_weights = volume.quadrature(volume_order)
                electric,magnetic,field_log = ac_fields(cell_points.reshape(-1,3),region,harmonic)
                query_logs[(harmonic,region)] = field_log
                linear = integrate_volume_dissipation(operator.system.bodies[region-1].interior,
                                                      electric,magnetic,cell_weights.flatten())
                local_e = electric.reshape(cells,-1,3)
                omega = operator.exterior.omega
                current_p = -1j*omega*state.polarization[harmonic,offset:offset+cells]
                current_d = state.dissipative_current[harmonic,offset:offset+cells]
                work = lambda current: (.5*(local_e.conj()*current[:,None]).sum(-1).real*cell_weights).sum()
                p,d = work(current_p),work(current_d)
                cache[volume_key] = (linear,p,d,query_logs[(harmonic,region)])
            linear,p,d,query_logs[(harmonic,region)] = cache[volume_key]
            absorption[harmonic] += linear
            polarization[harmonic] += p
            dissipation[harmonic] += d
            regions[(harmonic,region)] = {'linear_absorption_W':linear,'polarization_work_W':p,'dissipative_work_W':d}
            offset += cells
        if len(data.decoder.prepared['bindings'][harmonic-1]) != len(system.bodies):
            raise ValueError('Full power balance requires a volume rule for every body')
    residual = flux+absorption+polarization+dissipation
    balance_scale = scattered.abs()+interference.abs()+absorption.abs()+polarization.abs()+dissipation.abs()
    relative = residual.abs()/balance_scale.clamp_min(torch.finfo(balance_scale.dtype).tiny)
    return {'harmonics':tuple(range(count)),'frequencies_hz':state.frequencies.clone(),
        'scattered_power_W':scattered,'interference_power_W':interference,
        'balance_scale_W':balance_scale,'relative_balance_residual':relative,
        'outward_flux_W':flux,'linear_absorption_W':absorption,'polarization_work_W':polarization,
        'nonlinear_dissipation_W':dissipation,'balance_residual_W':residual,
        'total_balance_residual_W':residual.sum(),'per_region':regions,'queries':query_logs,
        'surface_order':surface_order,'volume_order':volume_order,'query_batch_size':query_batch_size,
        'quadrature_convergence_verified':False,
        'global_power_balance_verified':False,'physical_qualification':False,
        'dc_flux':'queried_real_E_cross_H_without_half_factor' if current_dc else 'zero_for_dielectric_polarization_without_DC_current',
        'incident_flux':'source_free_closed_surface_exact_zero',
        'volume_work':('queried_E_H_dual_ohmic_plus_E_against_cell_constant_nonlinear_current' if dual_dc else
                       'queried_E_against_cell_constant_induced_currents')}


def harmonic_power_refinement(data,control_surface,*,surface_orders=(2,4,8),volume_orders=(2,4,8),
                              relative_tolerance=1e-4,absolute_tolerance=1e-20,query_config=None,
                              query_batch_size=None,progress_callback=None):
    """Refine surface/volume rules independently, reusing each covered term.

    Successive-order differences are empirical estimates. The returned physical
    residual has a separate scale; neither local refinement nor cancellation
    implies that the discrete electromagnetic solution conserves power.
    """
    for orders in (surface_orders,volume_orders):
        if (not isinstance(orders,tuple) or len(orders) < 2 or
                any(type(order) is not int or order < 2 for order in orders) or
                any(b <= a for a,b in zip(orders,orders[1:]))):
            raise ValueError('Power refinement requires strictly increasing integer order tuples >= 2')
    if (not all(type(v) in (int,float) and math.isfinite(v) and v >= 0
                for v in (relative_tolerance,absolute_tolerance)) or relative_tolerance+absolute_tolerance == 0):
        raise ValueError('Power refinement requires finite nonnegative, nonzero tolerances')
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError('Power progress callback must be callable')
    def publish(axis,report):
        if progress_callback is not None:
            from .harmonic_snapshot import tensors
            progress_callback({'axis':axis,'report':tensors(report,'cpu')})
    cache = {}
    surface_order,volume_order = surface_orders[0],volume_orders[0]
    current = harmonic_power_balance(data,control_surface,surface_order=surface_order,
        volume_order=volume_order,query_config=query_config,query_batch_size=query_batch_size,_cache=cache)
    publish('initial',current)
    histories = {'surface':[],'volume':[]}
    successes = {}
    for axis,orders,names in (('surface',surface_orders,('scattered_power_W','interference_power_W')),
            ('volume',volume_orders,('linear_absorption_W','polarization_work_W','nonlinear_dissipation_W'))):
        for order in orders[1:]:
            previous = current
            if axis == 'surface':
                surface_order = order
            else:
                volume_order = order
            current = harmonic_power_balance(data,control_surface,surface_order=surface_order,
                volume_order=volume_order,query_config=query_config,query_batch_size=query_batch_size,_cache=cache)
            record = {'order':order,'terms':{}}
            passed = True
            for name in names:
                difference = (current[name]-previous[name]).abs()
                threshold = absolute_tolerance+relative_tolerance*current[name].abs()
                accepted = bool((difference <= threshold).all())
                passed = passed and accepted
                record['terms'][name] = {'difference_W':difference,'threshold_W':threshold,'passed':accepted}
            record['passed'] = passed
            histories[axis].append(record)
            publish(axis,{**current,'integration_refinement':histories})
            if passed:
                break
        successes[axis] = passed
    current.update(integration_refinement=histories,integration_refinement_passed=successes,
                   quadrature_convergence_verified=all(successes.values()),
                   quadrature_estimator='independent_successive_order_differences',
                   integration_relative_tolerance=relative_tolerance,integration_absolute_tolerance_W=absolute_tolerance)
    return current
