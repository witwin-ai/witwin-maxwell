"""Real baseline rejection contracts: intentionally fail until constructors are fixed.

These are not xfailed or relaxed. Run separately and preserve actual failures.
"""
import pytest
import torch
import witwin.maxwell as mw


@pytest.mark.parametrize('kind',('CW_phase','CW_frequency','CW_amplitude','sigma_m','domain','grid_spacing','grid_nodes','point_observer'))
def test_fixed_physical_constructor_must_reject_trainable_input_before_coercion(kind):
    theta=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    with pytest.raises(NotImplementedError):
        if kind=='CW_phase':mw.CW(3e9,phase=theta)
        elif kind=='CW_frequency':mw.CW(theta*1e9)
        elif kind=='CW_amplitude':mw.CW(3e9,amplitude=theta)
        elif kind=='sigma_m':mw.Material(sigma_m=theta)
        elif kind=='domain':mw.Domain(bounds=((-theta,theta),(-.02,.02),(-.02,.02)))
        elif kind=='grid_spacing':mw.GridSpec.uniform(theta)
        elif kind=='grid_nodes':mw.GridSpec.custom(theta*torch.tensor((-1.,0.,1.)),(-.02,0.,.02),(-.02,0.,.02))
        elif kind=='point_observer':mw.PointMonitor('point',position=(theta,0.,0.))
