"""Retain full declared frequencies and independent physical feed declarations."""
import json
import math
from pathlib import Path
import pytest
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd import pcb_application_campaign as campaign
from witwin.maxwell.adapters.pcb import import_pcb_scene
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.ports import compile_port_geometry
from scripts.dev.fdtd.pcb_application_campaign import NAMES,declared_frequencies,direct_scene,power_observations,simulation
from scripts.dev.fdtd.pcb_application_refinement import refined_definition

ROOT=Path(__file__).resolve().parents[3]
INPUTS=ROOT/'docs/assessments/pcb-application-campaign-2026-10-22/fixed-reference-spatial-inputs'


@pytest.fixture
def fixed_reference_inputs():
    if not INPUTS.exists():
        pytest.skip('repository evidence not in this checkout: docs/assessments/pcb-application-campaign-2026-10-22/fixed-reference-spatial-inputs')


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.usefixtures('fixed_reference_inputs')
def test_complete_declared_band_reimports_each_loss_reference_frequency(name):
    definition=json.loads((INPUTS/f'{name}-z1.json').read_text())
    frequencies=declared_frequencies(definition,2.45e9,full_band=True)
    assert frequencies==tuple(definition['frequencies_hz'])
    assert len(frequencies)==(121 if name.startswith('metamaterial-') else 11)
    material_records={record['id']:record for record in definition['materials']}
    dielectric_records={record['id']:record for record in definition['dielectrics']}
    for frequency in frequencies:
        scene=import_pcb_scene(definition,frequency=frequency,device='cpu')
        assert scene.metadata['pcb_import']['loss_reference_frequency_hz']==frequency
        for structure in scene.structures:
            if structure.name in dielectric_records:
                record=material_records[dielectric_records[structure.name]['material_id']]
                expected=2*math.pi*frequency*8.8541878128e-12*record['epsilon_r']*record.get('loss',{}).get('tan_delta',0)
                assert float(structure.material.sigma_e)==pytest.approx(expected,rel=1e-6,abs=1e-20)
        job=simulation(scene,frequency,with_power=True)
        assert job.frequencies==(frequency,)
        assert job.excitations[0].source_time.frequency==frequency
    for frequency in (frequencies[0],frequencies[len(frequencies)//2],frequencies[-1]):
        scene=import_pcb_scene(definition,frequency=frequency,device='cpu')
        monitors=power_observations(definition,scene,frequency)
        assert all(monitor.frequencies==(frequency,) for monitor in monitors)
    assert declared_frequencies(definition,2.45e9,full_band=False)==(2.45e9,)


@pytest.mark.usefixtures('fixed_reference_inputs')
def test_single_reference_tangent_loss_cannot_be_relabelled_as_broadband():
    definition=json.loads((INPUTS/'pcb-flexible-z1.json').read_text())
    scene=import_pcb_scene(definition,frequency=2.45e9,device='cpu')
    with pytest.raises(ValueError,match='reimport for each frequency'):
        mw.Simulation.fdtd(scene,frequencies=tuple(definition['frequencies_hz']))


@pytest.mark.usefixtures('fixed_reference_inputs')
def test_per_target_import_recalibrates_loss_and_still_rejects_a_frequency_mismatch():
    definition=json.loads((INPUTS/'pcb-flexible-z1.json').read_text())
    first,last=definition['frequencies_hz'][0],definition['frequencies_hz'][-1]
    initial=campaign.application_scenes(definition,first,definition['initial_conductor_ids'],with_power=True,device='cpu')
    scenes=campaign.application_scenes(definition,last,definition['initial_conductor_ids'],with_power=True,device='cpu')
    direct,adapted=scenes
    assert adapted.structures[0].geometry is not initial[1].structures[0].geometry
    assert adapted.metadata['pcb_import']['loss_reference_frequency_hz']==last
    assert initial[1].metadata['pcb_import']['loss_reference_frequency_hz']==first
    for scene in scenes:
        assert all(monitor.frequencies==(last,) for monitor in scene.monitors)
        assert simulation(prepare_scene(scene),last,with_power=True).frequency==last
    assert float(direct.structures[0].material.sigma_e)==float(adapted.structures[0].material.sigma_e)
    with pytest.raises(ValueError,match='reimport for each frequency'):
        simulation(prepare_scene(adapted),first,with_power=True)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('axes',('z','xy','xyz'))
@pytest.mark.usefixtures('fixed_reference_inputs')
def test_refining_a_fixed_reference_preserves_declared_port_bounds_exactly(name,axes):
    definition=json.loads((INPUTS/f'{name}-z1.json').read_text())
    original=json.dumps(definition,sort_keys=True)
    refined=refined_definition(definition,axes=axes,factor=3)
    assert refined['ports']==definition['ports']
    assert json.dumps(definition,sort_keys=True)==original


@pytest.mark.usefixtures('fixed_reference_inputs')
def test_direct_fixed_reference_loop_is_independent_of_an_altered_adapter_result():
    definition=json.loads((INPUTS/'pcb-flexible-z1.json').read_text())
    scene=import_pcb_scene(definition,frequency=2.45e9,device='cpu')
    port=scene.ports[0]
    corrupted=mw.LumpedPort(port.name,positive=port.positive,negative=port.negative,
        voltage_path=port.voltage_path,voltage_aperture=port.voltage_aperture,
        current_surface=mw.Box(position=(0.,0.,0.),size=(.01,0.,.01)),reference_impedance=port.reference_impedance)
    direct=direct_scene(definition,scene.clone(ports=[corrupted]),2.45e9,definition['initial_conductor_ids'])
    assert direct.ports[0].current_surface is not port.current_surface
    actual=compile_port_geometry(prepare_scene(direct),direct.ports[0],device='cpu')
    expected=compile_port_geometry(prepare_scene(scene),port,device='cpu')
    for left,right in zip(actual.current_indices+actual.current_weights,expected.current_indices+expected.current_weights):
        torch.testing.assert_close(left,right,rtol=0,atol=0)


@pytest.mark.parametrize('values',([],[-1.],[float('nan')],[2.,1.],[1.,1.]))
def test_invalid_declared_band_is_rejected_instead_of_partially_observed(values):
    with pytest.raises(ValueError):declared_frequencies({'frequencies_hz':values},2.45e9,full_band=True)


def test_campaign_preserves_prior_evidence_before_any_gpu_work(tmp_path,monkeypatch):
    evidence=tmp_path/'report.json';evidence.write_text('prior evidence')
    monkeypatch.setattr('sys.argv',['campaign','--output',str(tmp_path)])
    with pytest.raises(ValueError,match='Preserve prior evidence'):
        campaign.main()
    assert evidence.read_text()=='prior evidence'


@pytest.mark.usefixtures('fixed_reference_inputs')
def test_cli_schedules_every_original_frequency_in_separate_output(tmp_path,monkeypatch):
    input_dir=tmp_path/'inputs';input_dir.mkdir();output=tmp_path/'output'
    for name in NAMES:
        (input_dir/f'metahz-scene-{name}.json').write_bytes((INPUTS/f'{name}-z1.json').read_bytes())
    calls=[]
    def record_run(name,**kwargs):
        calls.append((name,kwargs['frequency'],kwargs['output']))
        return {'family':name,'frequency_hz':kwargs['frequency'],'steps':0,
            'run_seconds':[],'outputs_bitwise_equal':True,'input_unchanged':True}
    monkeypatch.setattr(campaign,'run',record_run)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'No device execution in scheduling test')
    monkeypatch.setattr('sys.argv',['campaign','--output',str(output),'--input-dir',str(input_dir),'--declared-band'])
    campaign.main()
    expected=[(name,frequency) for name in NAMES for frequency in
        json.loads((INPUTS/f'{name}-z1.json').read_text())['frequencies_hz']]
    assert [(name,frequency) for name,frequency,_ in calls]==expected
    assert len({path for _,_,path in calls})==len(expected)==385
    report=json.loads((output/'report.json').read_text())
    assert report['expected_runs']==385 and report['inputs_unchanged']
    assert report['failures']==[]


def test_cached_gate_rechecks_recover_only_when_both_pass(tmp_path,monkeypatch):
    input_dir=tmp_path/'inputs';input_dir.mkdir()
    (input_dir/'metahz-scene-metamaterial-solid.json').write_text(json.dumps({'frequencies_hz':[1e9,2e9]}))
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'CPU workflow test')
    monkeypatch.setattr(campaign,'package_identity',lambda *args:{'source_sha256':'unchanged'})
    value=torch.ones(1,dtype=torch.float64)
    different=torch.nextafter(value,torch.full_like(value,float('inf')))
    max_relative=float(((value-different).abs()/different.abs()).max())
    campaign._assert_bitwise_gate([{'voltage':value},{'voltage':value}],[])
    cases=[(True,True,True,False,False),
           (False,True,True,False,False),(False,False,True,False,False),
           (False,True,False,False,False),(False,False,False,False,False),
           (False,True,True,True,False),(False,True,True,False,True)]
    for case,(first,uncached,second,no_cache,runtime_error) in enumerate(cases):
        output=tmp_path/f'case-{case}';cache=object();calls=[]
        monkeypatch.setattr(campaign,'FamilyPreparation',lambda:cache)

        def run_once(name,**kwargs):
            label=('first','uncached','cached_second')[len(calls)] if kwargs['frequency']==1e9 else 'next'
            calls.append((label,kwargs))
            for path in ('direct','adapter'):
                (kwargs['output']/f'{name}-{path}.pt').write_text(label)
            if runtime_error:
                raise RuntimeError('preparation failed')
            passed={'first':first,'uncached':uncached,'cached_second':second,'next':True}[label]
            ports=[{'voltage':value},{'voltage':value}]
            powers=[{'radiated_w':value},{'radiated_w':value}]
            if not passed:
                (powers if case%2 else ports)[1][('radiated_w' if case%2 else 'voltage')]=different
            campaign._assert_bitwise_gate(ports,powers)
            return {'family':name,'frequency_hz':kwargs['frequency'],'steps':0,'run_seconds':[],
                    'outputs_bitwise_equal':True,'input_unchanged':True,
                    'prep_cache':kwargs['prep_cache'] is not None,'saved_attempt':label}

        monkeypatch.setattr(campaign,'_run_once',run_once)
        argv=['campaign','--output',str(output),'--input-dir',str(input_dir),'--names','metamaterial-solid',
              '--declared-band','--power','--fail-fast']
        if no_cache:argv.append('--no-prep-cache')
        monkeypatch.setattr('sys.argv',argv)
        recovered=not first and uncached and second and not no_cache and not runtime_error
        if first or recovered:
            campaign.main()
        else:
            with pytest.raises(SystemExit) as error:campaign.main()
            assert error.value.code==1
        report=json.loads((output/'report.json').read_text())
        target=output/'metamaterial-solid'/'frequency-000-1000000000'
        rechecked=not first and not no_cache and not runtime_error
        assert [label for label,_ in calls]==(['first','uncached','cached_second'] if rechecked else ['first'])+(
            ['next'] if first or recovered else [])
        if rechecked:
            assert [options['prep_cache'] for _,options in calls[:3]]==[cache,None,cache]
            row=report['runs'][0] if recovered else report['failures'][0]
            assert row['gate_recheck']=={'cached_first':False,'uncached':uncached,'cached_second':second,
                                         'max_rel_diff_first':max_relative}
        if recovered:
            assert report['failures']==[] and len(report['runs'])==2
            assert row['recovered_after_recheck'] is True and row['prep_cache'] is False
            assert calls[-1][1]['prep_cache'] is cache
        elif not first:
            assert report['stopped_early'] and not report['runs'] and len(report['failures'])==1
            if not rechecked:assert 'gate_recheck' not in report['failures'][0]
        else:
            assert 'gate_recheck' not in report['runs'][0]
        for path in ('direct','adapter'):
            assert (target/f'metamaterial-solid-{path}.pt').read_text()==('uncached' if recovered else 'first')
