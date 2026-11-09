"""Periodic FMM Krylov adapter with matching original periodic residual checks."""
from .direct_operator import DirectHierarchicalOperator, DirectSolveConfig
from .fmm_periodic_system import PeriodicMultipoleSystemActions
from .periodic_nearfield import PeriodicImageSystemBlocks
from .ewald_nearfield import EwaldSystemBlocks


class PeriodicMultipoleOperator(DirectHierarchicalOperator):
    @staticmethod
    def _build_actions(provider, **construction):
        if isinstance(provider, EwaldSystemBlocks):
            return PeriodicMultipoleSystemActions(provider, provider.kernel.lattice,
                ewald_kernel=provider.kernel, **construction)
        if not isinstance(provider, PeriodicImageSystemBlocks):
            raise TypeError('Periodic FMM solves require original periodic image blocks')
        return PeriodicMultipoleSystemActions(provider, provider.lattice, shells=provider.shells, **construction)

    def __init__(self, provider, *, solve_config=DirectSolveConfig(), **construction):
        super().__init__(provider, solve_config=solve_config, **construction)
        self.metadata.update(version=('ewald_cartesian_multipole_krylov_v1' if isinstance(self.provider, EwaldSystemBlocks)
                                      else 'absorbing_periodic_image_multipole_krylov_v1'),
            operator_backend='periodic_fmm_internal',
            original_residual_scope=self.actions.receipt()['original_reference_scope'],
            infinite_periodic_certified=False, physical_qualification=False,
            complexity_qualification=False)
