"""Multi-frequency FDFD preparation and transactional local design updates."""
from __future__ import annotations
from dataclasses import replace
import torch
from ..result import Result


def combine_results(results):
    frequencies = tuple(r.frequency for r in results)
    names = tuple(results[0].monitors)
    fields = {key: torch.stack([r.fields[key] for r in results]) for key in results[0].fields}
    monitors = {name: {'value': torch.stack([r.monitor(name)['value'] for r in results]),
                       'frequencies': frequencies, 'kind': 'linear_projection'} for name in names}
    raw = {}
    if isinstance(results[0].raw_output, dict):
        raw = {key: torch.stack([r.raw_output[key] for r in results])
               for key in results[0].raw_output if isinstance(results[0].raw_output[key], torch.Tensor)}
    return Result(method='fdfd',scene=results[0].scene,prepared_scene=results[0]._prepared_scene,frequencies=frequencies,
                  fields=fields,monitors=monitors,raw_output=raw,
                  solver_stats={'converged': all(r.solver_stats.get('converged',False) for r in results),
                                'frequency_runs': tuple(r.solver_stats for r in results)})


class PreparedFDFDSweep:
    def __init__(self, simulation, *, response_cache=None):
        from ..simulation import Simulation
        self.simulation = simulation
        self.responses = []
        try:
            for frequency in simulation.frequencies:
                child = Simulation(scene=simulation.scene_input, method=simulation.method,
                                   frequencies=(frequency,),config=replace(simulation.config),metadata=simulation.metadata)
                from .response_cache import frequency_cache_path
                cache = None if response_cache is None else frequency_cache_path(response_cache,frequency)
                self.responses.append(child.prepare(response_cache=cache))
        except Exception:
            self.close()
            raise

    @property
    def design_indices(self):
        return self.responses[0].design_indices

    @property
    def device(self):
        return self.responses[0].device

    def _kwargs(self, kwargs, index):
        result = dict(kwargs)
        delta = result.get('delta')
        if isinstance(delta, torch.Tensor) and delta.ndim == 2:
            if delta.shape[0] != len(self.responses):
                raise ValueError("Sweep contrast needs one row per frequency.")
            result['delta'] = delta[index]
        return result

    def run(self, **kwargs):
        return combine_results([response.run(**self._kwargs(kwargs,index))
                                for index,response in enumerate(self.responses)])

    def trial(self, **kwargs):
        return SweepTrial(self,tuple(response.trial(**self._kwargs(kwargs,index))
                                   for index,response in enumerate(self.responses)))

    def trial_updates_batch(self, indices, *, delta=None, pec=None):
        batches=[]
        if delta is not None and delta.ndim==3 and delta.shape[0]!=len(self.responses):
            raise ValueError('Sweep batch contrast needs one leading row per frequency.')
        for i,response in enumerate(self.responses):
            batches.append(response.trial_updates_batch(indices,
                delta=delta[i] if delta is not None and delta.ndim==3 else delta,pec=pec))
        return tuple(SweepTrial(self,tuple(batch[j] for batch in batches)) for j in range(indices.shape[0]))

    def accept(self, trial):
        return self._accept(trial,materialize=True)

    def _accept(self, trial, *, materialize=False):
        if not isinstance(trial,SweepTrial) or trial.owner is not self:
            raise ValueError("Sweep trial belongs to a different context.")
        names = ('_generation','_inverse','_accepted_matrix','_accepted_rhs','_accepted_current',
                 '_accepted_projection','_projected_inverse','_accepted_delta','_accepted_pec',
                 '_accepted_residual','_rebuild_count')
        before = [{name:getattr(response,name) for name in names} for response in self.responses]
        try:
            results = [response._accept(candidate,materialize=materialize) for response,candidate in zip(self.responses,trial.trials)]
            return combine_results(results) if materialize else None
        except Exception:
            for response,state in zip(self.responses,before):
                for name,value in state.items():
                    setattr(response,name,value)
            raise

    def _accepted_result(self):
        return combine_results([r._accepted_result() for r in self.responses])

    def optimize_binary(self, groups, objective, **kwargs):
        from .optimization import optimize_binary
        return optimize_binary(self,groups,objective,**kwargs)

    def save_cache(self, directory):
        from .response_cache import frequency_cache_path
        if self.simulation.config.local_response is None:
            raise ValueError("Response caches require local-response FDFD.")
        return tuple(response.save_cache(frequency_cache_path(directory,response.frequency))
                     for response in self.responses)

    def close(self):
        for response in self.responses:
            if hasattr(response,'close'):
                response.close()
            else:
                response.solver._release_direct_solver()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()


class SweepTrial:
    def __init__(self,owner,trials):
        self.owner,self.trials = owner,trials

    @property
    def projections(self):
        return torch.stack([trial.projections for trial in self.trials])

    @property
    def rank(self):
        return max(trial.rank for trial in self.trials)

    def result(self):
        return combine_results([trial.result() for trial in self.trials])
