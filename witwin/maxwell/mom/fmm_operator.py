"""Internal true multipole backend for the shared Riesz/Krylov protocol."""
from .direct_operator import DirectHierarchicalOperator, DirectSolveConfig
from .fmm_system import MultipoleSystemActions


class MultipoleOperator(DirectHierarchicalOperator):
    """Fixed-order scalar FMM solve; adaptive and production gates remain open."""
    @staticmethod
    def _build_actions(provider, **construction):
        return MultipoleSystemActions(provider, **construction)

    def __init__(self, provider, *, solve_config=DirectSolveConfig(), **construction):
        super().__init__(provider, solve_config=solve_config, **construction)
        self.metadata.update(version='scalar_cartesian_multipole_krylov_v1',
            operator_backend='fmm_internal', interpolation_construction=False,
            complexity_qualification=False, truncation_certified=False)


class DirectionalMultipoleOperator(DirectHierarchicalOperator):
    """Shared Riesz/Krylov adapter for a complete prescribed directional system."""
    @staticmethod
    def _build_actions(provider, *, actions):
        from .fmm_system import DirectionalSystemActions
        if not isinstance(actions, DirectionalSystemActions):
            raise TypeError('Directional solves require complete prescribed system actions')
        actions.assert_current()
        if actions.provider is not provider:
            raise ValueError('Directional solve provider must be the actual admitted provider')
        return actions

    def __init__(self, provider, actions, *, solve_config=DirectSolveConfig()):
        super().__init__(provider, solve_config=solve_config, actions=actions)
        from .fmm_order import AutomaticDirectionalDAGPlan
        from .fmm_directional_moments import DirectionalRWGDAGProjection
        automatic = all(isinstance(r.test, DirectionalRWGDAGProjection)
            and isinstance(r.source, DirectionalRWGDAGProjection)
            and isinstance(r.test.plan_guard, AutomaticDirectionalDAGPlan)
            and r.source.plan_guard is r.test.plan_guard for r in (actions.exterior, *actions.interiors))
        self.metadata.update(version=('automatic_directional_scalar_krylov_v1' if automatic
                                     else 'prescribed_directional_scalar_krylov_v1'),
            operator_backend='directional_fmm_internal', interpolation_construction=False,
            prescribed_complete_system=not automatic, automatic_cone_configuration=automatic,
            ideal_truncation_is_residual_certificate=False,
            implemented_operator_error_certified=False, truncation_certified=False,
            floating_point_certified=False, quadrature_certified=False,
            complexity_qualification=False, production_qualification=False)
