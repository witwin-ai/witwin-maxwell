"""Internal cutoff consistency preserves the public single-use preparation contract."""
from dataclasses import replace
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.harmonic_runtime import HarmonicCutoffError

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('failure',['record','raise'])
def test_cutoff_keeps_internal_initial_model_and_public_single_use(failure,tmp_path):
    original,config,_=runpy.run_path('tests/mom/test_harmonic_public.py')['setup']()
    scene=mw.Scene(structures=[mw.Structure(geometry=original.structures[0].geometry,
        material=mw.Material(eps_r=2.))],sources=original.sources)
    policy=mw.HarmonicCutoffConfig(max_harmonics=2,successive_passes=1 if failure=='record' else 2,failure=failure)
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,harmonics=1,pump_scales=(1.,),cutoff=policy))
    prepared=mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
    runtime=prepared.solver;model,initial=runtime.model,runtime.prepared
    seed=.3*model.background
    for repeat in range(2):
        run=prepared.run if repeat==0 else runtime.run
        if failure=='raise':
            with pytest.raises(HarmonicCutoffError,match='budget exhausted'):
                run(initial_harmonic_electric=seed)
        else:
            result=run(initial_harmonic_electric=seed)
            assert result.frequencies==(0.,3e8,6e8)
            assert result.solver_stats['cutoff']['initial_harmonics']==1
            assert result.solver_stats['continuation_harmonics']==1
            assert result._harmonic_data.config.harmonic_balance.harmonics==2
            assert result._metadata['reused_positive_harmonics']==1
            result.save(tmp_path/f'repeat{repeat}.pt')
            restored=mw.Result.load(tmp_path/f'repeat{repeat}.pt',scene=scene)
            assert restored.frequencies==result.frequencies
        with pytest.raises(RuntimeError,match='execute only once'):
            prepared.run(initial_harmonic_electric=seed)
        assert runtime.model is model and runtime.prepared is initial
        assert len(runtime.prepared['harmonics'])==1 and model.background.shape[0]==2
