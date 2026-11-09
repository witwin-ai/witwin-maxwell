"""Reference probes retain native addresses and magnetic half-step phase."""
import numpy as np
import pytest
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from scripts.dev.fdtd.printed_line_external_reference import native_addresses,read_native_spectra


def test_external_no_interpolation_probe_addresses_are_distinct_from_component_positions():
    scene=prepare_scene(mw.Scene(domain=mw.Domain(bounds=((-1.,2.),(-2.,3.),(-3.,4.))),
        grid=mw.GridSpec.custom((-1.,0.,2.),(-2.,0.,3.),(-3.,0.,4.)),device="cpu"))
    records=native_addresses(scene,{"electric":("Ez",(1,1,1)),"magnetic":("Hy",(1,1,1))})
    assert records["electric"]["address_m"]==[0.,0.,0.]
    assert records["electric"]["physical_component_position_m"]==[0.,0.,2.]
    assert records["magnetic"]["address_m"]==[1.,1.5,2.]
    assert records["magnetic"]["physical_component_position_m"]==[1.,0.,2.]


def test_raw_native_half_step_timestamps_recover_complex_physical_impedance(tmp_path):
    frequency=1e9;dt=1e-11;impedance=43.-2j
    electric=3.+4j;magnetic=electric/impedance
    records={}
    for name,component,amplitude,offset in (("electric","Ez",electric,0.),("magnetic","Hy",magnetic,.5*dt)):
        times=np.arange(0.,10e-9,dt)+offset
        envelope=np.exp(-.5*((times-5e-9)/.6e-9)**2)
        samples=envelope*np.real(amplitude*np.exp(-2j*np.pi*frequency*times))
        values=np.zeros((len(times),4));values[:,0]=times
        values[:,1+"xyz".index(component[-1].lower())]=samples
        path=tmp_path/name
        np.savetxt(path,values,header="start-coordinates: [1,1,1]",comments="% ")
        records[name]={"component":component,"index":[1,1,1]}
    spectra,tails=read_native_spectra(tmp_path,records,(frequency,))
    np.testing.assert_allclose(spectra["electric"]/spectra["magnetic"],impedance,rtol=1e-10)
    assert max(tails.values())<1e-8
    records["electric"]["index"]=[2,1,1]
    with pytest.raises(ValueError,match="different grid address"):
        read_native_spectra(tmp_path,records,(frequency,))


@pytest.mark.parametrize('axial',[.0025,.00125])
def test_independent_reference_graded_grid_keeps_local_cells_physical_domain_and_pml(axial):
    from types import SimpleNamespace
    from scripts.dev.fdtd.printed_line_external_reference import reference_line_scene
    common=dict(kind='microstrip',dx=.000625,longitudinal_dl=axial,padding=.02,separation=.4,axial_half_length=.3,sigma=0.)
    uniform=reference_line_scene(SimpleNamespace(**common,transverse_refinement_radius=None))
    graded=reference_line_scene(SimpleNamespace(**common,transverse_refinement_radius=.02))
    assert uniform.domain.bounds==graded.domain.bounds
    a,b=prepare_scene(uniform),prepare_scene(graded)
    for axis in 'xyz':
        old,new=[np.asarray(getattr(s,axis+'_nodes64')) for s in (a,b)]
        if axis=='x':np.testing.assert_array_equal(old,new)
        else:
            np.testing.assert_array_equal(old[abs(old)<=.02+1e-12],new[abs(new)<=.02+1e-12])
            n=uniform.boundary.num_layers
            np.testing.assert_array_equal(old[:n+1],new[:n+1])
            np.testing.assert_array_equal(old[-n-1:],new[-n-1:])
    for actual,original in zip(graded.structures,uniform.structures):
        for attribute in ('position','size','vertices'):
            if hasattr(actual.geometry,attribute):
                import torch
                torch.testing.assert_close(getattr(actual.geometry,attribute),getattr(original.geometry,attribute),rtol=0,atol=0)


@pytest.mark.parametrize('kind',['stripline','cpw'])
def test_balanced_external_feed_arrows_drive_signal_against_both_ground_gaps(kind):
    from scripts.dev.fdtd.printed_line_external_reference import reference_feeds
    feeds,groups=reference_feeds(kind,.4,True)
    assert groups==[[0,1],[2,3]]
    for column,indices in enumerate(groups):
        a,b=[feeds[index] for index in indices]
        assert a[2]==b[2]
        normal='xyz'.index(a[2]);transverse=[axis for axis in range(3) if axis!=normal]
        da=np.array(a[1])-a[0];db=np.array(b[1])-b[0]
        assert da[normal]==-db[normal]
        assert a[1][normal]==-b[1][normal]
        assert a[0][0]==b[0][0]==(-.2 if column==0 else .2)
        np.testing.assert_array_equal(da[transverse],db[transverse])


def test_external_coupled_feeds_are_four_independent_conductor_gap_drives():
    from scripts.dev.fdtd.printed_line_external_reference import reference_feeds
    from benchmark.scenes.rf.printed_lines import PAIR_OFFSET,PAIR_WIDTH,HEIGHT
    feeds,groups=reference_feeds('differential_pair',.4)
    assert groups==[[0],[1],[2],[3]]
    for feed,(x,y) in zip(feeds,[(x,y) for x in (-.2,.2) for y in (PAIR_OFFSET,-PAIR_OFFSET)]):
        start,stop,axis=feed
        assert axis=='z'
        np.testing.assert_array_equal(start,(x,y-PAIR_WIDTH/2,-HEIGHT))
        np.testing.assert_array_equal(stop,(x,y+PAIR_WIDTH/2,0.))


def test_default_external_two_conductor_source_geometry_is_unchanged():
    from scripts.dev.fdtd.printed_line_external_reference import reference_feeds
    from benchmark.scenes.rf.printed_lines import WIDTH,HEIGHT,GAP
    for kind in ('microstrip','stripline','cpw'):
        feeds,groups=reference_feeds(kind,.4)
        assert groups==[[0],[1]]
        for (start,stop,axis),x in zip(feeds,(-.2,.2)):
            if kind=='cpw':
                assert start==(x,WIDTH/2,0.) and stop==(x,WIDTH/2+GAP,0.) and axis=='y'
            else:
                assert start==(x,-WIDTH/2,-HEIGHT) and stop==(x,WIDTH/2,0.) and axis=='z'
