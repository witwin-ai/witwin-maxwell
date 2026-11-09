"""Actual AC response maps coupled to the nonlinear DC current closure."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
from witwin.maxwell.mom.harmonic_current import HarmonicCurrentClosure,surface_face_potentials
from witwin.maxwell.mom.static_floating import FloatingPotentialOperator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_actual_ac_maps_close_tpa_mean_current_and_recover_neutral_exterior():
    setup = runpy.run_path('tests/mom/test_harmonic_public.py')['setup']
    scene,config,_ = setup()
    runtime = mw.Simulation.mom(scene,frequency=3e8,config=config).prepare().solver
    base = runtime.model
    prepared = runtime.prepared
    volume = config.harmonic_balance.volumes[0]
    closure = HarmonicCurrentClosure([volume],[0.])
    polarization = base.polarization_response.clone()
    polarization[0] = 0
    tpa = base.background.real.new_tensor(1e-4,requires_grad=True)
    model = DenseHarmonicBalance(base.background,polarization,base.current_response,base.cell_volumes,
        base.frequency,chi2=base.chi2,chi3=base.chi3,tpa_scale=tpa,dc_current_closure=closure)
    result = model.solve(relative_tolerance=1e-9,absolute_tolerance=1e-10)
    assert result.receipt['converged']
    assert result.receipt['dc_formulation'] == 'isolated_face_current_balance'
    assert result.electric[2].norm() > 0
    assert result.electric[0].norm() > 0
    dc = closure.solve(result.electric[1:],model.tpa_scale)[0]
    torch.testing.assert_close(result.electric[0].real,dc.electric,rtol=1e-7,atol=1e-10)
    # Check the original constitutive DC current, independently of elimination.
    assert closure.blocks[0].weak_current(result.sources.dissipative_current[0].real).norm() < 1e-14
    assert torch.count_nonzero(result.sources.polarization_current(base.frequency)[0]) == 0
    floating = FloatingPotentialOperator(prepared['operators'][0].system,testing_order=4)
    exterior = floating.solve_face_means(surface_face_potentials(closure.blocks[0],prepared['bindings'][0][0],dc.potential),epsilon_exterior=prepared['dc_permittivities'][0])
    potential,electric = exterior.fields(volume.vertices.new_tensor([[.16,.1,.09]]))
    assert torch.isfinite(electric).all() and potential.norm() > 0
    gradient, = torch.autograd.grad(result.electric[1].abs().square().sum(),tpa)
    assert torch.isfinite(gradient) and gradient.abs() > 0
    values = []
    delta = 1e-6
    for coefficient in (tpa.detach()-delta,tpa.detach()+delta):
        perturbed = DenseHarmonicBalance(base.background,polarization,base.current_response,base.cell_volumes,
            base.frequency,chi2=base.chi2,chi3=base.chi3,tpa_scale=coefficient,dc_current_closure=closure)
        solved = perturbed.solve(result.electric.detach(),relative_tolerance=1e-9,absolute_tolerance=1e-10)
        values.append(solved.electric[1].abs().square().sum())
    torch.testing.assert_close(gradient,(values[1]-values[0])/(2*delta),rtol=1e-5,atol=1e-8)
    with pytest.raises(ValueError,match='empty DC'):
        DenseHarmonicBalance(base.background,base.polarization_response,base.current_response,
            base.cell_volumes,base.frequency,dc_current_closure=closure)
    print({'residual':result.receipt['final_residual'],'dc_norm':float(dc.electric.norm().detach()),
           'harmonic_2_norm':float(result.electric[2].norm().detach()),'tpa_gradient':float(gradient),
           'physical_qualification':False})
