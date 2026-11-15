"""Public-only native FP64 constructor/compiler/persistence contracts."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.fdtd.runtime.precision import validate_fp64_forward
from tests.core.scene.fp64_public_scene import scene,simulation
from witwin.maxwell.compiler import compile_materials


def test_direct_public_scene_and_typed_precision_contract():
    value=scene();model=simulation(value)
    assert model.config.field_dtype==torch.float64
    assert model.config.spectral_sampler.observer_dtype==torch.float64
    validate_fp64_forward(model,value)
    assert mw.Simulation.fdtd(value,frequencies=3e9).config.field_dtype==torch.float32
    low=prepare_scene(value, device="cpu");high=prepare_scene(value,material_dtype=torch.float64, device="cpu")
    assert low.x.dtype==torch.float32 and high.x.dtype==torch.float64
    for family in compile_materials(high)['edge_components'].values():
        assert all(item.dtype==torch.float64 for item in family.values())


@pytest.mark.parametrize('axis',tuple('xyz'))
@pytest.mark.parametrize('side',('low','high'))
@pytest.mark.parametrize('kind',('none','pec','pmc'))
def test_each_mixed_boundary_face_is_rejected(axis,side,kind):
    value=scene(boundary=mw.BoundarySpec.faces(default='pml',num_layers=4,**{axis+'_'+side:kind}))
    model=simulation(value)
    with pytest.raises(ValueError,match='all six faces'):validate_fp64_forward(model,value)


def test_zero_layer_boundary_cannot_enter_unqualified_standard_kernels():
    with pytest.raises(ValueError,match='num_layers > 0'):
        scene(boundary=mw.BoundarySpec.pml(num_layers=0))


@pytest.mark.parametrize('option,value,message',(('full_field_dft',True,'full-field'),('shutoff',.01,'fixed-duration')))
def test_public_precision_rejects_unqualified_execution_options(option,value,message):
    value_scene=scene();model=simulation(value_scene,**{option:value})
    with pytest.raises(ValueError,match=message):validate_fp64_forward(model,value_scene)


@pytest.mark.parametrize('kind',('modulation','breakdown'))
def test_dynamic_materials_reject_before_preparing_native_fields(kind):
    if kind=='modulation':material=mw.Material(eps_r=2.,modulation=mw.media.ModulationSpec(frequency=1e9,amplitude=.1))
    else:material=mw.Material(eps_r=2.,breakdown=mw.breakdown.DielectricBreakdown(critical_field=1e6,post_breakdown_conductivity=1.))
    # Cartesian declaration avoids the independent existing mapped-physics guard.
    value=scene(curvature=None,material=material);model=simulation(value)
    with pytest.raises(ValueError,match='static materials'):validate_fp64_forward(model,value)


def test_trainable_cartesian_inputs_reject_without_adjoint_downcast():
    parameter=torch.tensor(2.,requires_grad=True)
    value=scene(curvature=None,material=mw.Material(eps_r=parameter));model=simulation(value)
    with pytest.raises(ValueError,match='detached'):validate_fp64_forward(model,value)


def test_result_reconstruction_preserves_actual_material_precision(tmp_path):
    value=scene();prepared=prepare_scene(value,material_dtype=torch.float64, device="cpu")
    n=(prepared.Nx,prepared.Ny,prepared.Nz)
    fields={name:torch.zeros(tuple(size-(dim==axis)for dim,size in enumerate(n)),dtype=torch.float64)
        for axis,name in enumerate(('Ex','Ey','Ez'))}
    result=mw.Result(method='fdtd',scene=value,frequencies=(3e9,),fields=fields,
        solver_stats={'field_dtype':'torch.float64','observer_dtype':'torch.float64','full_field_dft':False})
    assert result.prepared_scene.x.dtype==torch.float64
    path=tmp_path/'precision.pt';result.save(path)
    restored=mw.Result.load(path,scene=value,map_location='cpu')
    assert restored.prepared_scene.x.dtype==torch.float64
    with pytest.raises(ValueError,match='frequency-domain volume fields'):
        restored.monitor('bulk')


@pytest.mark.parametrize('curvature',(None,-3.))
def test_external_config_tensor_rejected_without_scene_adjoint_detection(curvature):
    value=scene(curvature=curvature);model=simulation(value)
    assert not model.has_trainable_parameters
    object.__setattr__(model.config.cpml,'kappa_max',torch.tensor(3.,requires_grad=True))
    with pytest.raises(ValueError,match='detached declarative'):
        validate_fp64_forward(model,value)
    with pytest.raises(NotImplementedError,match='fixed'):
        model.prepare()



def test_auto_grid_is_outside_qualified_native_precision_contract():
    value=scene().updated(grid=mw.GridSpec.auto(wavelength=.1));model=simulation(value)
    with pytest.raises(ValueError,match='fixed declared grid'):validate_fp64_forward(model,value)


def test_unqualified_modal_observation_does_not_enter_double_native_dispatch():
    value=scene(curvature=None).updated(monitors=(mw.ModeMonitor('mode',size=(0.,.02,.02)),));model=simulation(value)
    with pytest.raises(ValueError,match='declarations only'):validate_fp64_forward(model,value)


def test_wire_bound_port_is_outside_qualified_norton_precision_contract():
    port=mw.LumpedPort('feed',wire_binding=mw.rf.WirePortBinding.nodes(negative=mw.rf.WireNodeRef('w',0),positive=mw.rf.WireNodeRef('w',1)))
    value=scene().updated(ports=(port,));model=simulation(value)
    with pytest.raises(ValueError,match='direct LumpedPort'):validate_fp64_forward(model,value)


def test_optical_phase_screen_is_not_silently_treated_as_scalar_bulk_material():
    from dataclasses import replace
    from witwin.core import PhaseScreen
    value=scene();screen=PhaseScreen(height=torch.zeros(2,2),height_scale_m=1e-6);model=simulation(value)
    value=value.updated(structures=(replace(value.structures[0],phase_screen=screen),value.structures[1]))
    with pytest.raises(NotImplementedError,match='optical phase_screen or uv/face_uv assignment'):simulation(value)
    # The FP64 admission keeps its own refusal for scenes that reach it directly.
    with pytest.raises(ValueError,match='optical phase screens'):validate_fp64_forward(model,value)
