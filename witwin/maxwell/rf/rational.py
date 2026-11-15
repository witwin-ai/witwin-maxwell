"""Public namespace ``mw.rf.rational``: rational (vector) fitting, state-space networks and
passivity checks."""
from .. import rational as _implementation
from .._stability import experimental

__all__ = [
    "DiscreteStateSpaceNetwork",
    "FitReport",
    "NetworkFitReport",
    "PassivityReport",
    "RationalFitConfig",
    "RationalModel",
    "StateSpaceNetwork",
    "bilinear_discretize",
    "check_sampled_passivity",
    "fit_rational",
]

# ``witwin/maxwell/rational.py`` is a shared numerical file that the consumer-metadata bridge
# pins byte for byte, so its objects are marked here, where they are published.
globals().update({name: experimental("rational")(getattr(_implementation, name)) for name in __all__})
