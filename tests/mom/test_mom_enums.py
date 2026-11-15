"""MoM option Enums (0.7.0 Q25): parsed once at the MoMConfig boundary; payloads keep the value strings."""
import pickle

import pytest

import witwin.maxwell as mw
from witwin.maxwell.mom.runtime import _numerical_identity, _representation_identity


def test_members_and_exact_value_strings_are_accepted_and_stored_as_members():
    Backend, Storage = mw.mom.MoMOperatorBackend, mw.mom.RieszStorage
    assert [m.value for m in Backend] == ["dense", "hmatrix", "h2", "direct_h2", "fmm", "pfft"]
    assert [m.value for m in Storage] == ["dense", "compact_rwg_v1", "local_rwg_v1"]
    for backend in (Backend.HMATRIX, "hmatrix"):
        config = mw.mom.MoMConfig(operator_backend=backend, riesz_storage="local_rwg_v1",
                                  stopping=mw.mom.MoMConfig().stopping.__class__(method="gmres", preconditioner="none"))
        assert config.operator_backend is Backend.HMATRIX and config.operator_backend == "hmatrix"
        assert config.riesz_storage is Storage.LOCAL_RWG_V1
    default = mw.mom.MoMConfig()
    assert default.operator_backend is Backend.DENSE and default.riesz_storage is Storage.DENSE
    assert str(Backend.FMM) == "fmm" and repr(Storage.COMPACT_RWG_V1) == "'compact_rwg_v1'"


@pytest.mark.parametrize("field, value", [("operator_backend", "HMatrix"), ("operator_backend", " dense"),
                                          ("operator_backend", "pme"), ("riesz_storage", "Dense"),
                                          ("riesz_storage", "compact")])
def test_other_spellings_raise_setup_error(field, value):
    with pytest.raises(mw.SetupError, match=f"MoMConfig.{field} must be one of"):
        mw.mom.MoMConfig(**{field: value})


def test_identity_payloads_hold_plain_value_strings():
    configs = [mw.mom.MoMConfig(), mw.mom.MoMConfig(riesz_storage="compact_rwg_v1"),
               mw.mom.MoMConfig(operator_backend="fmm", riesz_storage="local_rwg_v1",
                                stopping=mw.mom.MoMConfig().stopping.__class__(method="gmres", preconditioner="none"))]
    for config in configs:
        for payload in (_numerical_identity(config), _representation_identity(config)):
            for key in ("operator_backend", "riesz_storage"):
                if key in payload:
                    assert type(payload[key]) is str
            data = pickle.dumps(payload)
            assert b"MoMOperatorBackend" not in data and b"RieszStorage" not in data
