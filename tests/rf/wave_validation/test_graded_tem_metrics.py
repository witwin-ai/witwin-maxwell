"""Independent finite-volume electrostatic checks on unequal Yee intervals."""
import pytest
import torch

from witwin.maxwell.fdtd.excitation.modes.tem import _quasistatic_laplace_energy_torch


def boundary(shape):
    mask=torch.zeros(shape,dtype=torch.bool)
    mask[0,:]=mask[-1,:]=True
    mask[:,0]=mask[:,-1]=True
    return mask


def test_graded_native_affine_potential_and_exact_physical_area_energy():
    u=torch.tensor([-.020,-.010,-.009,-.002,0.,.001,.012,.020],dtype=torch.float64)
    v=torch.tensor([-.030,-.013,-.004,0.,.002,.017,.030],dtype=torch.float64)
    epsilon=2.3;expected=3*u[:,None]-7*v[None,:]+.2
    phi,energy=_quasistatic_laplace_energy_torch(torch.full_like(expected,epsilon),
        boundary(expected.shape),expected,du=None,dv=None,coords_u=u,coords_v=v)
    torch.testing.assert_close(phi,expected,rtol=0,atol=2e-11)
    exact=epsilon*(3**2+7**2)*float((u[-1]-u[0])*(v[-1]-v[0]))
    assert float(energy)==pytest.approx(exact,rel=1e-10)


def test_graded_native_dielectric_series_flux_and_work():
    u=torch.tensor([0.,.001,.003,.004,.010,.016,.020],dtype=torch.float64)
    v=torch.tensor([0.,.002,.009,.010,.017,.030],dtype=torch.float64)
    eps_u=torch.tensor([1.,1.,2.,4.,4.,4.,4.],dtype=torch.float64)
    # Independent one-dimensional series-resistance solution of the face flux.
    resistance=torch.diff(u)/(.5*(eps_u[:-1]+eps_u[1:]))
    potential=torch.cat((torch.zeros(1),torch.cumsum(resistance,0)))/resistance.sum()
    expected=potential[:,None].expand(-1,len(v))
    phi,energy=_quasistatic_laplace_energy_torch(eps_u[:,None].expand_as(expected),
        boundary(expected.shape),expected,du=None,dv=None,coords_u=u,coords_v=v)
    torch.testing.assert_close(phi,expected,rtol=0,atol=2e-11)
    assert float(energy)==pytest.approx(float((v[-1]-v[0])/resistance.sum()),rel=1e-10)


def test_uniform_native_coordinate_input_retains_legacy_potential_bitwise():
    u=torch.arange(9,dtype=torch.float64)*.002
    v=torch.arange(7,dtype=torch.float64)*.003
    eps=torch.ones((len(u),len(v)),dtype=torch.float64)
    mask=boundary(eps.shape);mask[4,3]=True
    value=torch.zeros_like(eps);value[4,3]=1.
    legacy=_quasistatic_laplace_energy_torch(eps,mask,value,du=.002,dv=.003)
    native=_quasistatic_laplace_energy_torch(eps,mask,value,du=.002,dv=.003,coords_u=u,coords_v=v)
    assert torch.equal(legacy[0],native[0]) and torch.equal(legacy[1],native[1])


@pytest.mark.parametrize("u",[[0.,.001,.001],[0.,.001,float('nan')],[.001,0.,-.001]])
def test_native_transverse_coordinate_invalidity_is_explicit(u):
    with pytest.raises(ValueError,match="finite and strictly increasing"):
        _quasistatic_laplace_energy_torch(torch.ones((3,4),dtype=torch.float64),
            boundary((3,4)),torch.zeros((3,4)),du=None,dv=None,
            coords_u=u,coords_v=[0.,.001,.003,.010])


def cyclic_scene(scene,permutation):
    import witwin.maxwell as mw
    def vector(value):return tuple(value[index] for index in permutation)
    def box(item):return mw.Box(position=torch.tensor(vector(item.position),dtype=torch.float64),
                                size=torch.tensor(vector(item.size),dtype=torch.float64))
    structures=[]
    for item in scene.structures:
        geometry=item.geometry
        if isinstance(geometry,mw.PlanarPEC):
            normal='xyz'.index(geometry.normal_axis)
            points=torch.zeros((len(geometry.vertices),3),dtype=torch.float64)
            points[:,normal]=geometry.coordinate
            points[:,[i for i in range(3) if i!=normal]]=geometry.vertices
            rotated=points[:,permutation];new_normal=permutation.index(normal)
            geometry=mw.PlanarPEC(rotated[:,[i for i in range(3) if i!=new_normal]],
                normal_axis='xyz'[new_normal],coordinate=geometry.coordinate,
                physical_thickness=geometry.physical_thickness)
        else:geometry=box(geometry)
        structures.append(mw.Structure(geometry,item.material,name=item.name))
    ports=[]
    for port in scene.ports:
        modes=[mw.WaveModeSpec(mode.family,
            polarization='E'+'xyz'[permutation.index('xyz'.index(mode.polarization[-1].lower()))],
            voltage_path=tuple(vector(point) for point in mode.voltage_path),
            current_contour=box(mode.current_contour)) for mode in port.modes]
        axis=vector(port.size).index(0.)
        ports.append(mw.WavePort(port.name,position=vector(port.position),size=vector(port.size),
            direction=port.direction,reference_plane=vector(port.position)[axis],modes=modes))
    return scene.clone(domain=mw.Domain(bounds=vector(scene.domain.bounds)),
        grid=mw.GridSpec.custom(*vector((scene.grid.x_coords,scene.grid.y_coords,scene.grid.z_coords))),
        structures=structures,ports=ports)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Native graded TEM launch requires CUDA')
@pytest.mark.parametrize('direction',['+','-'])
@pytest.mark.parametrize('kind',['stripline','microstrip'])
def test_graded_native_launch_rotation_and_graph_fields(direction,kind):
    import witwin.maxwell as mw
    from benchmark.scenes.rf.printed_lines import printed_line_scene
    scene=printed_line_scene(kind,dx=.000625,longitudinal_dl=.0025,
        transverse_refinement_radius=.020,sigma_e=.01 if kind=='stripline' else 0.,device='cuda:0')
    selected=scene.ports[0] if direction=='+' else scene.ports[1]
    scene=scene.clone(ports=(selected,))
    reference=None;reference_projection=None
    for permutation,graph in [((0,1,2),False),((0,1,2),True),((2,0,1),True),((1,2,0),True)]:
        rotated=cyclic_scene(scene,permutation)
        port=rotated.ports[0];position=list(port.position)
        position['xyz'.index(port.normal_axis)]=0.
        observer=mw.ModeMonitor('graded_observer',position=position,size=port.size,
            direction=port.direction,polarization=port.modes[0].polarization,frequencies=(1e9,))
        object.__setattr__(observer,'_wave_family','tem')
        rotated=rotated.clone(monitors=(observer,))
        result=mw.Simulation.fdtd(rotated,frequencies=(1e9,),excitations=mw.PortExcitation(selected.name),
            run_time=mw.TimeConfig(time_steps=96),spectral_sampler=mw.SpectralSampler(window='none'),
            full_field_dft=False,cuda_graph=graph).run()
        projection=result.monitor('graded_observer')
        measured={key:complex(projection[key]) for key in
            ('amplitude_forward','amplitude_backward','total_power')}
        if kind=='stripline':assert result.port(selected.name).beta.imag.item()>0
        if reference is None:
            reference=result
            reference_projection=measured
            assert max(float(getattr(result.solver,'E'+a).abs().max()) for a in 'xyz')>0
            assert_native_projection_basis(result)
            continue
        for key,expected in reference_projection.items():
            assert abs(measured[key]-expected)/max(abs(expected),1e-9)<3e-4
        for index,axis in enumerate('xyz'):
            mapped='xyz'[permutation.index(index)]
            for family in 'EH':
                actual=getattr(result.solver,family+mapped)
                expected=getattr(reference.solver,family+axis).permute(permutation)
                assert float((actual-expected).abs().max()/expected.abs().max().clamp_min(1e-7))<2e-4


def assert_native_projection_basis(result):
    """Actual native observer geometry and CUDA mode fields form a wave basis."""
    from witwin.maxwell.postprocess import modal
    name='graded_observer'
    payload=result.raw_monitor(name)
    assert 'cell_widths' in payload
    source=modal._resolve_mode_source(result.scene,payload,mode_source=None,frequency=1e9)
    _,_,electric,magnetic,_=modal._reference_mode_fields(modal._mode_overlap_context(result),
        source,payload,frequency=1e9,direction=None)
    assert electric.is_cuda and magnetic.is_cuda
    for forward,backward in ((1.+0j,0j),(0j,1.+0j),(.4+.7j,-.2+.1j)):
        supplied=dict(payload)
        for index,axis in enumerate('xyz'):
            supplied['E'+axis]=(forward+backward)*electric[...,index]
            supplied['H'+axis]=(forward-backward)*magnetic[...,index]
        overlap=modal._compute_mode_overlap_from_payload(result,name,supplied)
        assert complex(overlap['amplitude_forward'])==pytest.approx(forward,abs=2e-5)
        assert complex(overlap['amplitude_backward'])==pytest.approx(backward,abs=2e-5)
        tangents=[axis for axis in 'xyz' if axis!=payload['axis']]
        widths=[torch.as_tensor(payload['cell_widths'][axis],device=electric.device,dtype=torch.float64) for axis in tangents]
        area=widths[0][:,None]*widths[1][None,:]
        normal_sign=1 if payload['normal_direction']=='+' else -1
        expected=(.5*torch.cross(((forward+backward)*electric).to(torch.complex128),
            ((forward-backward)*magnetic).to(torch.complex128).conj(),dim=-1)[...,'xyz'.index(payload['axis'])].real*area*normal_sign).sum()
        torch.testing.assert_close(overlap['total_power'].real,expected.real,rtol=3e-5,atol=1e-7)
