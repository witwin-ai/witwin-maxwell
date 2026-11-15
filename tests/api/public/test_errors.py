import warnings

import pytest

import witwin.maxwell as mw
from witwin.maxwell import errors
from tests.oracles.nonlinear_devices import NonlinearConvergenceError, NonlinearDeviceError
from witwin.maxwell.postprocess.waveports import ModeTrackingError


FAMILIES = {
    "UnsupportedFeatureError": NotImplementedError,
    "SetupError": ValueError,
    "ConvergenceError": RuntimeError,
    "SchemaError": ValueError,
    "CapacityError": MemoryError,
}

# Re-parented class -> (new domain base, built-in base it had before 0.7.0).
REPARENTED = [
    (mw.fdtd.FDTDPassivityError, mw.ConvergenceError, RuntimeError),
    (NonlinearDeviceError, mw.ConvergenceError, RuntimeError),
    (NonlinearConvergenceError, mw.ConvergenceError, RuntimeError),
    (mw.rf.TouchstoneParseError, mw.SchemaError, ValueError),
    (ModeTrackingError, mw.MaxwellError, RuntimeError),
    (mw.execution.DistributedFailure, mw.MaxwellError, Exception),
]

REPARENTED_WARNINGS = [mw.rf.TerminalWavePortApertureWarning, mw.ExperimentalWarning]


def test_top_level_exports_are_the_errors_module_types():
    for name in errors.__all__:
        assert name in mw.__all__
        assert getattr(mw, name) is getattr(errors, name)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_family_mro(name):
    family = getattr(mw, name)
    builtin = FAMILIES[name]
    assert family.__mro__[1:3] == (mw.MaxwellError, builtin)
    assert issubclass(family, Exception)


def test_root_and_warning_bases():
    assert mw.MaxwellError.__mro__[1] is Exception
    assert mw.MaxwellWarning.__mro__[1] is UserWarning
    assert mw.ExperimentalWarning.__mro__[1] is mw.MaxwellWarning
    assert not issubclass(mw.MaxwellWarning, mw.MaxwellError)


@pytest.mark.parametrize("cls,domain,builtin", REPARENTED, ids=lambda value: getattr(value, "__name__", None))
def test_reparented_classes_keep_their_builtin_base(cls, domain, builtin):
    assert issubclass(cls, domain)
    assert issubclass(cls, mw.MaxwellError)
    assert issubclass(cls, builtin)


@pytest.mark.parametrize("cls", REPARENTED_WARNINGS, ids=lambda value: value.__name__)
def test_reparented_warnings_keep_userwarning(cls):
    assert issubclass(cls, mw.MaxwellWarning)
    assert issubclass(cls, UserWarning)
    with pytest.warns(mw.MaxwellWarning):
        warnings.warn("probe", cls)


def test_touchstone_parse_error_instance_is_still_a_value_error(tmp_path):
    error = mw.rf.TouchstoneParseError("bad option line", path=tmp_path / "a.s2p", line_number=3)
    assert isinstance(error, ValueError) and isinstance(error, mw.SchemaError)
    assert error.line_number == 3


def test_unsupported_feature_error_fields():
    error = mw.UnsupportedFeatureError(
        "lossy metal is not supported by FDFD",
        feature="material.lossy_metal",
        method="fdfd",
        alternative="use Simulation.fdtd",
        items=["a", "b"],
    )
    assert str(error) == "lossy metal is not supported by FDFD"
    assert error.feature == "material.lossy_metal"
    assert error.method == "fdfd"
    assert error.alternative == "use Simulation.fdtd"
    assert error.items == ("a", "b")
    assert isinstance(error, NotImplementedError)

    bare = mw.UnsupportedFeatureError("deep guard", feature="guard:abc123", method=None)
    assert bare.method is None and bare.alternative is None and bare.items == ()

    with pytest.raises(TypeError):
        mw.UnsupportedFeatureError("missing keywords")


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_maxwell_error_catches_each_family(name):
    family = getattr(mw, name)
    error = family("x", feature="k", method=None) if family is mw.UnsupportedFeatureError else family("x")
    with pytest.raises(mw.MaxwellError):
        raise error
    with pytest.raises(FAMILIES[name]):
        raise error


def test_maxwell_error_catches_reparented_solver_errors():
    for error in (
        mw.fdtd.FDTDPassivityError("x"),
        NonlinearDeviceError("x"),
        NonlinearConvergenceError("x", stats=None),
        ModeTrackingError("x"),
    ):
        with pytest.raises(mw.MaxwellError):
            raise error
