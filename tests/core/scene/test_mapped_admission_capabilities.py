"""Genuine fixed-template capability negatives, separate from numerical gates."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.core import PhysicalMaterial
from tests.core.scene.fp64_public_scene import scene as fixed_scene
from tests.core.scene.test_mapped_admission_lifecycle import LifecycleModule,simulation


def leaf():return torch.tensor(2.,dtype=torch.float64,requires_grad=True)


def test_fixed_sweep_does_not_bypass_single_norton_admission():
    with pytest.raises(NotImplementedError,match='direct fixed Norton'):
        mw.Simulation.fdtd(fixed_scene(material=mw.Material(eps_r=leaf())),frequency=3e9,
            run_time=mw.TimeConfig(dt=5e-12,time_steps=96),excitations=mw.PortSweep(ports=('feed',)))


def test_standalone_finite_face_requires_independent_admission_qualification():
    value=fixed_scene(material=mw.Material(eps_r=leaf())).clone(monitors=(
        mw.FinitePlaneMonitor('finite',position=(.025,0.,0.),size=(0.,.02,.02),fields=('Ey',)),))
    with pytest.raises(NotImplementedError,match='fixed point, closed-surface'):
        simulation(value)


def test_fixed_sphere_material_geometry_is_outside_initial_box_class():
    value=fixed_scene(material=mw.Material(eps_r=leaf()))
    value=value.clone(structures=(replace(value.structures[0],geometry=mw.Sphere(radius=.01)),value.structures[1]))
    with pytest.raises(NotImplementedError,match='fixed Box'):
        simulation(value)


def test_remote_fixed_pec_remains_outside_initial_material_class():
    value=fixed_scene(material=mw.Material(eps_r=leaf()))
    remote=mw.Structure(mw.Box(position=(.022,.022,0.),size=(.005,.005,.005)),mw.Material.pec())
    with pytest.raises(NotImplementedError,match='PEC material combinations'):
        simulation(value.clone(structures=(*value.structures,remote)))


def test_core_physical_material_protocol_is_not_silently_admitted_as_maxwell_material():
    value=fixed_scene(material=PhysicalMaterial(eps_r=leaf()))
    with pytest.raises(NotImplementedError,match='scalar Material or CopperFoil'):
        simulation(value)


class ChangedMaterialModule(LifecycleModule):
    def to_scene(self):
        options={}
        if self.kind=='ade':options['debye_poles']=(mw.DebyePole(delta_eps=1.,tau=1e-10),)
        if self.kind=='modulation':options['modulation']=mw.ModulationSpec(frequency=1e9,amplitude=.1)
        return fixed_scene(material=mw.Material(eps_r=self.epsilon.exp(),**options))


@pytest.mark.parametrize('kind,reason',(('ade','scalar nondispersive'),('modulation','static constitutive')))
def test_refresh_does_not_admit_new_constitutive_states(kind,reason):
    module=ChangedMaterialModule();value=simulation(module);module.kind=kind
    with pytest.raises(NotImplementedError,match=reason):
        value._refresh_scene()


def test_fixed_auto_order_copper_is_explicitly_outside_initial_fixed_mode_contract():
    value=fixed_scene(material=mw.Material(eps_r=leaf()))
    fixed=mw.CopperFoil(thickness=1e-6,conductivity=1e6,frequency_range=(2e9,4e9))
    value=value.clone(structures=(value.structures[0],replace(value.structures[1],material=fixed)))
    with pytest.raises(NotImplementedError,match='fixed mode_count'):
        simulation(value)


def test_fixed_reactive_load_remains_outside_direct_norton_scope():
    value=fixed_scene(material=mw.Material(eps_r=leaf()))
    old=value.ports[0]
    port=mw.LumpedPort(old.name,positive=old.positive,negative=old.negative,voltage_path=old.voltage_path,
        current_surface=old.current_surface,voltage_aperture=old.voltage_aperture,termination=mw.SeriesRLC(r=50.,l=1e-9))
    value=value.clone(ports=(port,))
    with pytest.raises(NotImplementedError,match='LumpedPort Norton'):
        simulation(value)



def test_internal_mapped_lowering_helper_rejects_legitimate_cartesian_scene():
    from witwin.maxwell.fdtd.adjoint.mapped_admission import assess_mapped_materials
    from witwin.maxwell.simulation import FDTDConfig
    value=fixed_scene(curvature=None,material=mw.Material(eps_r=leaf()))
    with pytest.raises(NotImplementedError,match='fixed BendCoordinates'):
        assess_mapped_materials(value,module=None,config=FDTDConfig(run_time=mw.TimeConfig(dt=5e-12,time_steps=96)),
            excitations=(),port_sweep=None,frequencies=(3e9,))
