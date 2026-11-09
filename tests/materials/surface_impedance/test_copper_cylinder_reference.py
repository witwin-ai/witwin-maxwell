"""Independent cylinder physical oracle; no production mapped solver support."""
import math
import numpy as np
import pytest

from scripts.dev.fdtd.copper_cylinder_reference import shell_scattering,collapsed_shell_scattering,MU_0


@pytest.mark.parametrize("polarization",("te","tm"))
@pytest.mark.parametrize("order",(0,1,3))
def test_finite_vacuum_annulus_transmits_both_hankel_ports_without_reflection(order,polarization):
    actual=shell_scattering(.3,.01,1e9,0.,order=order,polarization=polarization)
    np.testing.assert_allclose(actual,((0,1),(1,0)),rtol=3e-14,atol=3e-14)


@pytest.mark.parametrize("polarization",("te","tm"))
@pytest.mark.parametrize("order",(0,1,3))
@pytest.mark.parametrize("skin_depths",(.01,1.,10.,30.))
@pytest.mark.parametrize("radius",(.03,.3))
def test_copper_cylinder_port_passivity_reciprocity_and_thin_layer_approximation(order,polarization,skin_depths,radius):
    frequency,sigma=1e9,5.8e7
    delta=math.sqrt(2/(2*math.pi*frequency*MU_0*sigma))
    exact=shell_scattering(radius,skin_depths*delta,frequency,sigma,order=order,polarization=polarization)
    assert np.isfinite(exact).all()
    np.testing.assert_allclose(exact[0,1],exact[1,0],rtol=3e-11,atol=2e-15)
    absorption=np.eye(2)-exact.conj().T@exact
    assert np.linalg.eigvalsh(absorption).min()>0
    collapsed=collapsed_shell_scattering(radius,skin_depths*delta,frequency,sigma,order=order,polarization=polarization)
    # Same physical radius and thickness, no fitted phase/reference plane. This
    # is the declared curvature omission bound, separate from numerical errors.
    assert np.linalg.norm(collapsed-exact)/np.linalg.norm(exact)<.002
    if skin_depths<=10:
        assert abs(collapsed[0,1]-exact[0,1])/abs(exact[0,1])<.002
    else:
        assert abs(collapsed[0,1]-exact[0,1])<2e-12
