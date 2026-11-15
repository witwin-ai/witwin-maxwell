"""MoM string options are ``str`` Enums parsed once at construction; payloads keep their value strings.

Each option accepts a member or exactly its value string, stores the member and refuses any other
value with ``SetupError``. Payloads, receipts and identities written from a config hold only plain
``str`` values (``_same_metadata`` compares types, snapshots are pickled and restore checks read
``type(value) is str``), and a config built from members writes the same payload as one built from
the value strings.
"""

import enum
import pickle
from dataclasses import fields

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.errors import SetupError, UnsupportedFeatureError
from witwin.maxwell.mom.linalg import DirectStorage, MoMSolveMethod, StoppingRule
from witwin.maxwell.mom.options import MoMOption, option_asdict
from witwin.maxwell.mom.runtime import _numerical_identity, _representation_identity

LAYERED = dict(bridge_width_m=1e-6, max_edge_length_m=1e-5)
ITERATIVE = dict(stopping=StoppingRule(method="gmres", preconditioner="none"), riesz_storage="local_rwg_v1")

# (config class, required arguments, option field, option Enum)
OPTIONS = [
    (mw.mom.HMatrixConfig, {}, "pivoting", mw.mom.ACAPivoting),
    (mw.mom.FMMConfig, {"near_storage": "full_matrix"}, "table_storage", mw.mom.FMMTableStorage),
    (mw.mom.FMMConfig, {}, "shift_storage", mw.mom.FMMShiftStorage),
    (mw.mom.FMMConfig, {"table_storage": "dense"}, "near_storage", mw.mom.FMMNearStorage),
    (mw.mom.FMMConfig, {}, "kernel_bound_rule", mw.mom.FMMKernelBoundRule),
    (mw.mom.FMMConfig, {}, "expansion_center_policy", mw.mom.FMMExpansionCenter),
    (mw.mom.FMMConfig, {}, "leaf_factor_kernel", mw.mom.LeafFactorKernel),
    (mw.mom.DirectionalFMMConfig, {}, "near_storage", mw.mom.FMMNearStorage),
    (mw.mom.DirectionalFMMConfig, {}, "leaf_factor_kernel", mw.mom.LeafFactorKernel),
    (mw.mom.PFFTConfig, {}, "grid_precision", mw.mom.PFFTGridPrecision),
    (mw.mom.PFFTConfig, {}, "exterior_grids", mw.mom.PFFTExteriorGrids),
    (mw.mom.HarmonicCutoffConfig, {"max_harmonics": 4}, "failure", mw.mom.CutoffFailure),
    (StoppingRule, {}, "method", MoMSolveMethod),
    (StoppingRule, {"method": "direct"}, "direct_storage", DirectStorage),
    (mw.mom.DyadicIntegrationConfig, {}, "radial_map", mw.mom.DyadicRadialMap),
    (mw.mom.DyadicIntegrationConfig, {}, "spatial_rule", mw.mom.DyadicSpatialRule),
    (mw.mom.MoMDyadicPeriodicConfig, dict(vectors_m=((1e-3, 0., 0.), (0., 1e-3, 0.)), shells=2, height_m=1e-4,
                                          max_halvings=3, absolute_tolerance=1e-6, max_shells=4),
     "transverse_map", mw.mom.TransverseMap),
    (mw.mom.LayeredSurfaceConfig, LAYERED, "operator_backend", mw.mom.LayeredOperatorBackend),
    (mw.mom.LayeredSurfaceConfig, LAYERED, "conductor_model", mw.mom.LayeredConductorModel),
    (mw.mom.LayeredSurfaceConfig, LAYERED, "planar_mesher", mw.mom.PlanarMesher),
]
IDS = [f"{cls.__name__}.{name}" for cls, _, name, _ in OPTIONS]


def assert_plain(value, path="payload"):
    assert not isinstance(value, enum.Enum), f"{path} holds {value!r}"
    if isinstance(value, dict):
        for key, item in value.items():
            assert_plain(key, f"{path}.<key>")
            assert_plain(item, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_plain(item, f"{path}[{index}]")


def build(cls, required, name, value):
    """The config, or None when this value is refused in combination with the required defaults."""
    try:
        return cls(**{**required, name: value})
    except (ValueError, TypeError, UnsupportedFeatureError) as error:
        if "must be one of" in str(error):
            raise
        return None


def records(config):
    """Every payload-like record a config writes."""
    if isinstance(config, StoppingRule):
        return [option_asdict(config)]
    if isinstance(config, mw.mom.HarmonicCutoffConfig):
        return [option_asdict(config)]
    out = [config.payload()]
    if isinstance(config, mw.mom.FMMConfig):
        out.append(config.construction())
    return out


@pytest.mark.parametrize("cls, required, name, option", OPTIONS, ids=IDS)
def test_option_parses_members_and_exact_strings_only(cls, required, name, option):
    assert issubclass(option, MoMOption) and issubclass(option, str)
    built = 0
    for member in option:
        from_string = build(cls, required, name, member.value)
        from_member = build(cls, required, name, member)
        assert (from_string is None) == (from_member is None)
        if from_string is None:
            continue
        built += 1
        assert getattr(from_string, name) is member and getattr(from_member, name) is member
        assert from_string == from_member and repr(from_string) == repr(from_member)
        for string_record, member_record in zip(records(from_string), records(from_member), strict=True):
            assert_plain(string_record)
            assert pickle.dumps(string_record) == pickle.dumps(member_record)
        if hasattr(cls, "restore"):
            again = cls.restore(from_string.payload())
            assert again == from_string and getattr(again, name) is member
    assert built
    first = next(iter(option)).value
    for bad in (first.upper(), " " + first, "unknown"):
        with pytest.raises(SetupError, match=f"must be one of .*got {bad!r}"):
            cls(**{**required, name: bad})


def test_every_option_field_is_typed_by_its_enum():
    for cls, _, name, option in OPTIONS:
        field = next(f for f in fields(cls) if f.name == name)
        assert field.type in (option, option.__name__), (cls, name)


def test_calderon_open_edge_rule():
    for member in mw.mom.OpenEdgeRule:
        for value in (member, member.value):
            assert mw.mom.CalderonPreconditioner(open_edge_rule=value).open_edge_rule is member
    with pytest.raises(SetupError, match="must be one of"):
        mw.mom.CalderonPreconditioner(open_edge_rule="Natural")


@pytest.mark.parametrize("config", [
    {},
    {**ITERATIVE, "operator_backend": "hmatrix", "hierarchical": mw.mom.HMatrixConfig(pivoting="adaptive")},
    {**ITERATIVE, "operator_backend": "fmm",
     "fmm": mw.mom.FMMConfig(near_storage="full_matrix", table_storage="dense", shift_storage="dense")},
    {**ITERATIVE, "operator_backend": "fmm", "fmm": mw.mom.FMMConfig(kernel_bound_rule="first_wave_min_v1",
                                                                     leaf_factor_kernel="original")},
    {**ITERATIVE, "operator_backend": "fmm",
     "fmm": mw.mom.DirectionalFMMConfig(near_storage="scalar_reciprocal_cached_dk_v1")},
    {**ITERATIVE, "operator_backend": "pfft",
     "pfft": mw.mom.PFFTConfig(grid_precision="complex128", exterior_grids="clusters")},
    {"stopping": StoppingRule(method="direct", direct_storage="sealed_host_restore_v1")},
    {"dyadic_integration": mw.mom.DyadicIntegrationConfig(radial_map="material_split_v1",
                                                          spatial_rule="factored_spd_tail_v1")},
], ids=["dense", "hmatrix", "fmm", "fmm_rules", "directional", "pfft", "sealed", "dyadic"])
def test_mom_config_identities_hold_plain_strings(config):
    mom = mw.mom.MoMConfig(**config)
    assert_plain(_numerical_identity(mom))
    assert_plain(_representation_identity(mom))


def test_options_are_exported_from_mw_mom():
    exported = {name for name in mw.mom.__all__ if isinstance(getattr(mw.mom, name), type)
                and issubclass(getattr(mw.mom, name), MoMOption)}
    assert exported == {option.__name__ for _, _, _, option in OPTIONS if option not in (MoMSolveMethod, DirectStorage)} | {
        "OpenEdgeRule", "HarmonicBalanceMethod", "TensorFaceRule", "HarmonicImplicitSolver"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TetrahedralVolumeMesh needs CUDA")
def test_harmonic_balance_options():
    tet = mw.mom.TetrahedralVolumeMesh(
        torch.tensor([[0., 0., 0.], [1e-3, 0., 0.], [0., 1e-3, 0.], [0., 0., 1e-3]], device="cuda"),
        torch.tensor([[0, 1, 2, 3]], device="cuda"), region_id=1)
    base = dict(harmonics=2, volumes=(tet,), branch_id="b")
    krylov = dict(base, method="newton_krylov")
    cases = [(base, "method", mw.mom.HarmonicBalanceMethod), (base, "tensor_face_rule", mw.mom.TensorFaceRule),
             (krylov, "implicit_solver", mw.mom.HarmonicImplicitSolver)]
    for required, name, option in cases:
        for member in option:
            from_string = mw.mom.HarmonicBalanceConfig(**{**required, name: member.value})
            from_member = mw.mom.HarmonicBalanceConfig(**{**required, name: member})
            assert getattr(from_string, name) is member and from_string == from_member
            for method in ("solve_settings", "numerical_payload", "dc_policy"):
                record = getattr(from_string, method)()
                if method == "solve_settings":
                    record = {k: v for k, v in record.items() if k != "tangent_rule"}
                assert_plain(record)
                assert record == {k: v for k, v in getattr(from_member, method)().items() if k in record}
        with pytest.raises(SetupError, match="must be one of"):
            mw.mom.HarmonicBalanceConfig(**{**required, name: "unknown"})
