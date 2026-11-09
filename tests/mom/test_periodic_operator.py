"""Periodic exterior ownership controls, not scattering qualification."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec,EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_operator import assemble_periodic_system
from witwin.maxwell.mom.operator import assemble_system,IntegrationConfig
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
import witwin.maxwell as mw

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_periodic_correction_is_independent_of_every_interior_material():
    operator,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    original=operator.system
    sample=original.bodies[0].mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[2.,0.,0.]]),sample.new_tensor([.2,0.,0.]))
    kernel=EwaldHelmholtzKernel(lattice,2.,real_shells=2,reciprocal_shells=4,quadrature_order=48)
    config=IntegrationConfig(order=2,singular_order=3)
    # Loss suppresses distant images; only the shared exterior sees the lattice.
    exterior=compile_medium(mw.Material(eps_r=1.,sigma_e=.02),3e8)
    changed=SurfaceSystem.create([(b.structure_index,b.name,b.basis,
        compile_medium(mw.Material(eps_r=4.+b.region_id,mu_r=1.+.2*b.region_id),3e8)) for b in original.bodies])
    corrections=[]
    for system in (original,changed):
        free,_=assemble_system(system,exterior,config)
        periodic,receipt=assemble_periodic_system(system,exterior,config,kernel=kernel)
        corrections.append(periodic-free)
        assert not receipt['interiors_periodic'] and not receipt['physical_qualification']
        assert len(receipt['regular_blocks'])==4
        for i in range(2):
            for j in range(2):
                assert corrections[-1][system.J_slices[i],system.M_slices[j]].norm()>0
    torch.testing.assert_close(corrections[0],corrections[1],rtol=1e-9,atol=1e-14)


def test_periodic_seam_rejects_before_free_space_assembly(monkeypatch):
    from witwin.maxwell.mom import periodic_operator
    operator,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample=operator.system.bodies[0].mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[.2,0.,0.]]),sample.new_zeros(3))
    kernel=EwaldHelmholtzKernel(lattice,2.)
    monkeypatch.setattr(periodic_operator,'assemble_system',lambda *a,**k:pytest.fail('Invalid seam reached assembly'))
    with pytest.raises(ValueError,match='seam'):
        assemble_periodic_system(operator.system,operator.exterior,kernel=kernel)


def test_periodic_regular_magnetic_self_panels_are_not_forced_to_zero():
    from witwin.maxwell.mom.periodic_operator import _RegularKernel
    from witwin.maxwell.mom.operator import _pair_integrals
    operator,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    basis=operator.system.bodies[0].basis
    sample=basis.mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[2.,0.,0.]]),sample.new_tensor([.2,0.,0.]))
    kernel=_RegularKernel(EwaldHelmholtzKernel(lattice,2.,real_shells=2,reciprocal_shells=4))
    medium=compile_medium(mw.Material(eps_r=1.,sigma_e=.02),3e8)
    faces=torch.arange(len(basis.mesh.faces),device=sample.device)
    _,actual=_pair_integrals(basis,faces,faces,medium,2,2,False,kernel=kernel,zero_planar_self=False)
    _,suppressed=_pair_integrals(basis,faces,faces,medium,2,2,False,kernel=kernel)
    assert actual.norm()>1e-12
    assert torch.count_nonzero(suppressed)==0


@pytest.mark.parametrize('order,shells,accepted',[(2,4,False),(4,8,True)],ids=['coarse','refined'])
def test_periodic_assembly_checks_four_axes_and_reuses_primary_matrix(monkeypatch,order,shells,accepted):
    from witwin.maxwell.mom import periodic_operator
    from witwin.maxwell.mom.periodic_operator import PeriodicAssemblyCheck,PeriodicAssemblyConvergenceError
    operator,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    body=operator.system.bodies[0]
    system=SurfaceSystem.create([(body.structure_index,body.name,body.basis,body.interior)])
    sample=body.mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[2.,0.,0.]]),sample.new_tensor([.2,0.,0.]))
    kernel=EwaldHelmholtzKernel(lattice,2.,real_shells=2,reciprocal_shells=4,quadrature_order=48)
    exterior=compile_medium(mw.Material(eps_r=1.,sigma_e=.02),3e8)
    original=periodic_operator.assemble_system
    calls=[]
    def tracked(*args,**kwargs):
        calls.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(periodic_operator,'assemble_system',tracked)
    from dataclasses import replace
    kernel=replace(kernel,reciprocal_shells=shells)
    if not accepted:
        with pytest.raises(PeriodicAssemblyConvergenceError) as failure:
            assemble_periodic_system(system,exterior,IntegrationConfig(order=order,singular_order=3),
                kernel=kernel,check=PeriodicAssemblyCheck())
        assert len(calls)==1
        assert not failure.value.receipt['converged']
        assert any(not row['passed'] for row in failure.value.receipt['checks'])
        return
    try:
        _,receipt=assemble_periodic_system(system,exterior,IntegrationConfig(order=order,singular_order=3),
            kernel=kernel,check=PeriodicAssemblyCheck())
    except PeriodicAssemblyConvergenceError as error:
        print({'refined_failure':error.receipt})
        raise
    assert len(calls)==1
    checks=receipt['regular_convergence']
    assert checks['converged'] and len(checks['checks'])==4
    assert not checks['certified_error_bound'] and not receipt['physical_qualification']
    print(checks)


def test_incident_waves_share_one_bloch_sector_including_reciprocal_aliases():
    import math
    from witwin.maxwell.mom.periodic_operator import validate_periodic_sources
    medium=compile_medium(mw.Material(),3e8)
    vectors=torch.tensor([[2*math.pi/medium.k.real,0.,0.]],device='cuda',dtype=torch.float64)
    lattice=LatticeSpec(vectors,vectors.new_zeros(3))
    sources=[mw.PlaneWave(),mw.PlaneWave(direction=(1.,0.,0.),polarization=(0.,1.,0.))]
    assert validate_periodic_sources(sources,lattice,medium)['source_count']==2
    with pytest.raises(ValueError,match='Bloch vector'):
        validate_periodic_sources([mw.PlaneWave(direction=(1.,0.,1.),polarization=(0.,1.,0.))],lattice,medium)
    lossy=compile_medium(mw.Material(sigma_e=.02),3e8)
    assert validate_periodic_sources(sources[:1],lattice,lossy)['incident_phase_match']
    with pytest.raises(ValueError,match='attenuation'):
        validate_periodic_sources(sources[1:],lattice,lossy)
