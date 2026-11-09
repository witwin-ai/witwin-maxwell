"""Public zero-induced-source degeneration with explicit owned volume state."""
import copy
from dataclasses import replace
from .runtime import MoMRuntime, SurfaceData, _same_metadata
from .linear_volume_state import LinearVolumeState, LinearCoupledConfig, PARTITION
from .volume_h_snapshot import _digest


class LinearSurfaceData(SurfaceData):
    """Regional fields plus an explicit zero additional P/J state.

    ``config`` describes the underlying surface solve. ``coupled_config`` owns
    the volume binding. The complete representation is preserved by schema20.
    """
    def __init__(self, surface, volume_state):
        super().__init__(surface.scattered_coefficients, surface.handle,
                         surface.config, surface.scene_signature)
        if type(volume_state) is not LinearVolumeState or volume_state.system is not self.handle.system:
            raise ValueError('Linear data must own the exact solved surface system')
        if self.config.harmonic_balance is not None:
            raise ValueError('Linear volume degeneration cannot wrap harmonic surface data')
        self.linear_volume_state = volume_state
        self.coupled_config = LinearCoupledConfig(tuple(b.volume for b in volume_state.bindings))
        self._volume_binding = volume_state
        self._coupled_config_binding = self.coupled_config
        self._volume_identity = _digest(volume_state.snapshot())

    def _validate_carrier_binding(self):
        super()._validate_carrier_binding()
        if (self.linear_volume_state is not self._volume_binding
                or self.coupled_config is not self._coupled_config_binding
                or self.linear_volume_state.system is not self.handle.system):
            raise ValueError('Linear volume result binding changed')
        self.linear_volume_state.assert_current()

    def representation_metadata(self):
        self._validate_carrier_binding()
        return dict(self.handle.metadata, formulation_id='coupled_sie_vie_linear',
            decoder_version='linear_zero_induced_volume_fields_v1',
            material_partition=PARTITION, volume_state_identity=self._volume_identity,
            volume_state_role='zero_additional_induced_source', physical_qualification=False)

    def snapshot(self):
        self._validate_carrier_binding()
        body = dict(schema_version=20, payload_kind='linear_coupled_zero_induced_solution',
            executable=True, surface=super().snapshot(), volume_state=self.linear_volume_state.snapshot(),
            representation=self.representation_metadata())
        return dict(body, sha256=_digest(body))

    @classmethod
    def restore(cls, payload, scene, frequency):
        keys = {'schema_version', 'payload_kind', 'executable', 'surface',
                'volume_state', 'representation', 'sha256'}
        if (type(payload) is not dict or set(payload) != keys
                or type(payload['schema_version']) is not int or payload['schema_version'] != 20
                or payload['payload_kind'] != 'linear_coupled_zero_induced_solution'
                or payload['executable'] is not True
                or payload['sha256'] != _digest({k:v for k,v in payload.items() if k != 'sha256'})):
            raise ValueError('Invalid linear coupled snapshot identity')
        if type(payload['surface']) is not dict or payload['surface'].get('schema_version') == 20:
            raise ValueError('Linear state requires one underlying surface solution')
        surface = SurfaceData.restore(payload['surface'], scene, frequency)
        state = LinearVolumeState.restore(payload['volume_state'], surface.handle.system,
                                          dtype=surface.config.dtype)
        result = cls(surface, state)
        if not _same_metadata(payload['representation'], result.representation_metadata()):
            raise ValueError('Linear representation differs from its surface/volume identity')
        return result


class LinearCoupledRuntime:
    """Reuse the regional linear solve with an explicit zero induced closure.

    Homogeneous linear response is already in the regional Green function.
    The additional volume source is exactly zero; no source quadrature or
    constitutive contrast is added. This is the frozen PMCHWT degeneration.
    """
    def __init__(self, simulation):
        self.simulation = simulation
        self.prepared_config = simulation.config
        base = copy.copy(simulation)
        base.config = replace(simulation.config, linear_coupled=None)
        self.surface_runtime = MoMRuntime(base)
        self.handle = self.surface_runtime.handle
        self.volume_state = LinearVolumeState(self.handle.system, simulation.config.linear_coupled,
                                              dtype=simulation.config.dtype)
        self._volume_state_binding = self.volume_state
        self._surface_runtime_binding = self.surface_runtime
        self.scene_signature = self.surface_runtime.scene_signature

    def run(self):
        if (self.volume_state is not self._volume_state_binding
                or self.surface_runtime is not self._surface_runtime_binding
                or self.simulation.config is not self.prepared_config
                or self.simulation.frequency != self.surface_runtime.prepared_frequency
                or self.simulation.scene is not self.surface_runtime.simulation.scene):
            raise ValueError('Linear coupled simulation changed after preparation')
        self.volume_state.assert_current()
        result = self.surface_runtime.run()
        data = LinearSurfaceData(result.surface_currents, self.volume_state)
        result._surface_data = data
        result.solver = self
        result._metadata.update(data.representation_metadata())
        return result
