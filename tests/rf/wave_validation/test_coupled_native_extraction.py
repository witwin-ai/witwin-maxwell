"""Independent four-drive physical transmission matrices constrain odd/even VI."""
import numpy as np
import pytest
from scripts.dev.fdtd.printed_line_physical_modes import coupled_physical_state,coupled_transfer


def test_conductor_to_mixed_physical_variables_preserve_complex_power():
    voltage=np.array([2+3j,-1+.7j]);current=np.array([.04-.02j,-.03+.01j])
    state=coupled_physical_state(voltage[0],voltage[0]-voltage[1],current.sum(),current[0])
    assert state[0]==pytest.approx(voltage.mean())
    assert state[1]==pytest.approx(voltage[0]-voltage[1])
    assert state[2]==pytest.approx(current.sum())
    assert state[3]==pytest.approx((current[0]-current[1])/2)
    assert state[:2]@state[2:].conj()==pytest.approx(voltage@current.conj())


@pytest.mark.parametrize('loss',[0.,.17])
def test_four_independent_physical_drives_recover_known_coupled_line(loss):
    positions=[-.021,0.,.019]
    rng=np.random.default_rng(13)
    initial=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    states=[];known=((30.-.3j,24.),(110.-.7j,31.))
    for position in positions:
        transfer=np.zeros((4,4),complex)
        for index,(z,beta) in enumerate(known):
            g=1j*beta-loss*(index+1);c,s=np.cosh(g*position),np.sinh(g*position)
            transfer[np.ix_([index,index+2],[index,index+2])]=[[c,z/50*s],[50/z*s,c]]
        states.append(transfer@initial)
    states=np.asarray(states)
    measured=coupled_transfer(states[:,:2],states[:,2:]/50,positions)
    for segment in measured:
        for index,name in enumerate(('common','differential')):
            assert segment['modes'][name]['characteristic_impedance_ohm']==pytest.approx(known[index][0],rel=3e-14)
            assert segment['modes'][name]['beta_rad_per_m']==pytest.approx(known[index][1],rel=2e-14)
            assert segment['modes'][name]['alpha_np_per_m']==pytest.approx(loss*(index+1),abs=2e-12)
