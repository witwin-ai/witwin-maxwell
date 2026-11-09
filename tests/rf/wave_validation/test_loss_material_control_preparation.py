"""CPU physical contracts only; no field propagation or CUDA test stand-ins."""
from dataclasses import replace
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase, build_case, _require_physical_metal_basis
from scripts.dev.fdtd.printed_line_fixed_vi import fixed_physical_probe_monitors
from scripts.dev.fdtd.printed_line_broadband import load_helper
from scripts.dev.fdtd.printed_line_loss_basis import LossBasis, SpectralContract
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.ports import compile_ports, compile_port_geometry
from witwin.maxwell.fdtd.lumped import prepare_lumped_runtime
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.constants import EPSILON_0
from pathlib import Path


@pytest.fixture(scope='module')
def contour_helpers():
    # The printed-line builders load their contour helpers from this evidence directory.
    if not (Path(__file__).resolve().parents[3]/'docs/assessments/printed-lines-2026-10-22').exists():
        pytest.skip('repository evidence not in this checkout: docs/assessments/printed-lines-2026-10-22')


@pytest.fixture(scope='module')
def baseline(contour_helpers): return build_case(PilotCase())


@pytest.mark.parametrize('changes', [{'metal':'pec'}, {'sigma':0.}])
def test_fixed_mesh_material_controls_keep_actual_source_apertures_grid_exterior_and_VI(changes,baseline):
    original, groups, bounds, schema, rows, expected, _ = baseline
    scene, other_groups, other_bounds, other_schema, other_rows, other_expected, _ = build_case(replace(PilotCase(),**changes))
    a,b=prepare_scene(original),prepare_scene(scene)
    assert original.domain == scene.domain and original.boundary == scene.boundary
    for axis in 'xyz':np.testing.assert_array_equal(getattr(a,axis+'_nodes64'),getattr(b,axis+'_nodes64'))
    assert groups == other_groups and bounds == other_bounds and rows == other_rows and expected == other_expected
    for before,after in zip(compile_ports(a,device='cpu'),compile_ports(b,device='cpu')):
        assert before.voltage_component == after.voltage_component
        torch.testing.assert_close(before.voltage_indices,after.voltage_indices,rtol=0,atol=0)
        torch.testing.assert_close(before.voltage_weights,after.voltage_weights,rtol=0,atol=0)
    for before,after in zip(original.ports,scene.ports):
        assert before.positive == after.positive and before.negative == after.negative
        torch.testing.assert_close(before.voltage_aperture.position,after.voltage_aperture.position,rtol=0,atol=0)
        torch.testing.assert_close(before.voltage_aperture.size,after.voltage_aperture.size,rtol=0,atol=0)


def _rows_by_native_key(rows,expected):
    output=[]
    for pair in rows:
        pair_output=[]
        for row in pair:
            values={}
            for name,component,weight in row:
                key=(component,expected[name][1]);values[key]=values.get(key,0.)+weight
            pair_output.append({key:value for key,value in values.items() if value!=0})
        output.append(pair_output)
    return output


def test_fixed_functional_coarse_mesh_is_exactly_the_original_native_integrals(baseline):
    scene,_,_,_,rows,expected,_=baseline
    fixed=build_case(replace(PilotCase(),probe_reference_dx=.0025))
    assert _rows_by_native_key(rows,expected)==_rows_by_native_key(fixed[4],fixed[5])


@pytest.mark.usefixtures('contour_helpers')
@pytest.mark.parametrize('dx',[.00125,.000625])
@pytest.mark.parametrize('affine',[False,True])
def test_fixed_physical_curve_and_quadrature_reproduce_independent_constant_affine_fields(dx,affine):
    scene,_,_,_,rows,expected,_=build_case(replace(PilotCase(),dx=dx,longitudinal_dl=.0025,probe_reference_dx=.0025))
    monitors={m.name:m for m in scene.monitors};prepared=prepare_scene(scene)
    def field(component,position):
        x,y,z=position
        if not affine:return {'Ez':2.,'Hy':3.,'Hz':7.}[component]
        return {'Ez':2+3*x+4*y+5*z,'Hy':2+3*z+4*x+5*y,'Hz':7+11*y-13*z+2*x}[component]
    for name,(component,index) in expected.items():
        axes=component_coords(prepared,component)
        assert monitors[name].position == tuple(float(axis[i]) for axis,i in zip(axes,index))
    for station,pair in zip(PilotCase().positions,rows):
        values=[sum(weight*field(component,monitors[name].position) for name,component,weight in row) for row in pair]
        V=-(2+3*station)*.005+5*.005**2/2 if affine else -.01
        assert values[0] == pytest.approx(V,rel=2e-14,abs=2e-16)
        assert abs(values[1]) == pytest.approx(.0125*.0075*8 if affine else 0.,rel=5e-13,abs=2e-16)


@pytest.mark.usefixtures('contour_helpers')
@pytest.mark.parametrize('metal',['pec','copper'])
def test_fixed_H_stencil_cannot_cross_conductor_interface(metal):
    scene=build_case(replace(PilotCase(),metal=metal))[0]
    monitor=mw.PointMonitor('on_interface',(0.,0.,0.),fields=('Hy',))
    with pytest.raises(ValueError,match='interface|trace_side'):
        fixed_physical_probe_monitors(scene,scene,[monitor],[([('on_interface','Hy',1.)],[])])


def test_fixed_physical_stencil_cannot_cross_periodic_seam(baseline):
    scene=baseline[0].clone(boundary=mw.BoundarySpec.periodic())
    with pytest.raises(ValueError,match='seams'):
        fixed_physical_probe_monitors(scene,scene,[],[])


def test_empty_copper_basis_is_permitted_only_for_declared_PEC():
    basis=LossBasis((1e9,),1e-12,SpectralContract('a'*64,'b'*64,'c'*64,'d'*64,10,'none'),(),{})
    _require_physical_metal_basis(replace(PilotCase(),metal='pec'),basis)
    with pytest.raises(RuntimeError,match='nonempty'):_require_physical_metal_basis(PilotCase(),basis)


def test_alternative_actual_compiled_source_H_measurement_does_not_change_Norton_field_stamp(baseline):
    scene=baseline[0];prepared=prepare_scene(scene);port=scene.ports[0]
    position=port.current_surface.position.clone();position[2]+=.0025
    other=mw.LumpedPort(port.name,positive=port.positive,negative=port.negative,voltage_path=port.voltage_path,
        voltage_aperture=port.voltage_aperture,current_surface=mw.Box(position=position,size=port.current_surface.size),reference_impedance=50.)
    a,b=[compile_port_geometry(prepared,p,device='cpu') for p in (port,other)]
    assert any(not torch.equal(x,y) for x,y in zip(a.current_indices,b.current_indices))
    torch.testing.assert_close(a.voltage_indices,b.voltage_indices,rtol=0,atol=0)
    torch.testing.assert_close(a.voltage_weights,b.voltage_weights,rtol=0,atol=0)
    shape=tuple(len(axis) for axis in component_coords(prepared,a.voltage_component))
    eps=torch.full(shape,2.25*EPSILON_0,dtype=torch.float64)
    coefficient=1e-12/(eps+.5e-12*.01)
    runtimes=[prepare_lumped_runtime(g,dt=1e-12,eps_edge=eps,electric_update_coefficient=coefficient,
        yee_control_volume=.0025**3,resistance=50.) for g in (a,b)]
    assert vars(runtimes[0]).keys()==vars(runtimes[1]).keys()
    for key,value in vars(runtimes[0]).items():
        other_value=vars(runtimes[1])[key]
        if key in ('edge_buffer','correction_buffer'):
            # Scratch is deliberately empty/uninitialized until field sampling.
            assert value.shape==other_value.shape and value.dtype==other_value.dtype
        elif torch.is_tensor(value):torch.testing.assert_close(value,other_value,rtol=0,atol=0)
        else:assert value==other_value


@pytest.mark.parametrize('dx',[.00125,.000625])
def test_transverse_refinement_holds_axial_source_aperture_and_exterior_nodes_exactly(dx,baseline):
    original=baseline[0];scene=build_case(replace(PilotCase(),dx=dx,longitudinal_dl=.0025,probe_reference_dx=.0025))[0]
    a,b=prepare_scene(original),prepare_scene(scene)
    assert original.domain==scene.domain and original.boundary==scene.boundary
    np.testing.assert_array_equal(a.x_nodes64,b.x_nodes64)
    for axis in 'yz':
        old,new=[np.asarray(getattr(owner,axis+'_nodes64')) for owner in (a,b)]
        np.testing.assert_array_equal(old[abs(old)>.02000000000001],new[abs(new)>.02000000000001])
    for old,new in zip(original.ports,scene.ports):
        assert old.positive==new.positive and old.negative==new.negative
        torch.testing.assert_close(old.voltage_aperture.position,new.voltage_aperture.position,rtol=0,atol=0)
        torch.testing.assert_close(old.voltage_aperture.size,new.voltage_aperture.size,rtol=0,atol=0)
    old_geometry,new_geometry=[compile_ports(owner,device='cpu') for owner in (a,b)]
    for old,new in zip(old_geometry,new_geometry):
        assert float(old.voltage_weights.sum())==pytest.approx(float(new.voltage_weights.sum()),rel=2e-15)
    assert scene.metadata['fixed_VI_observation_contract']['scope'].startswith('Fixed physical reference quadrature')


@pytest.fixture(autouse=True)
def _cpu_preparation_keeps_CUDA_initialization_state():
    before = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() == before
