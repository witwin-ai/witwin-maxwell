"""CPU contract of the typed CPML configuration (FDTDConfig.cpml)."""
from __future__ import annotations

import dataclasses

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.boundary.cpml import (
    DEFAULT_CPML_GRADING, STABLE_PML_GRADING, CPMLGrading, resolve_cpml_grading,
)


def _scene():
    return mw.Scene(domain=mw.Domain(bounds=((-0.1, 0.1),) * 3), grid=mw.GridSpec.uniform(0.05),
                    boundary=mw.BoundarySpec.pml(num_layers=2))


def test_defaults_leave_the_solver_defaults_unset():
    config = mw.fdtd.CPMLConfig()
    assert (config.grading_order, config.kappa_max, config.alpha_max, config.reflection) == (None, 1.0, None, None)
    assert config.memory_mode is mw.fdtd.CPMLMemoryMode.AUTO
    assert config.dense_memory_limit_mib == 64.0
    assert mw.FDTDConfig().cpml == config
    assert resolve_cpml_grading(config) == DEFAULT_CPML_GRADING == CPMLGrading(3.0, 1.0, 0.0, 1e-6)


def test_explicit_fields_override_only_their_own_default():
    config = mw.fdtd.CPMLConfig(grading_order=4, kappa_max=8, reflection=1e-5)
    assert resolve_cpml_grading(config) == CPMLGrading(4.0, 8.0, 0.0, 1e-5)
    assert type(config.kappa_max) is float and type(config.grading_order) is float
    assert STABLE_PML_GRADING == CPMLGrading(5.0, 5.0, 0.24, 1e-9)


@pytest.mark.parametrize("value,expected", [
    ("auto", mw.fdtd.CPMLMemoryMode.AUTO), ("dense", mw.fdtd.CPMLMemoryMode.DENSE), ("slab", mw.fdtd.CPMLMemoryMode.SLAB),
    (mw.fdtd.CPMLMemoryMode.SLAB, mw.fdtd.CPMLMemoryMode.SLAB),
])
def test_memory_mode_parses_exact_strings_to_the_enum(value, expected):
    assert mw.fdtd.CPMLConfig(memory_mode=value).memory_mode is expected


@pytest.mark.parametrize("value", ["Dense", " slab", "compressed", "", None, 1])
def test_memory_mode_refuses_other_values(value):
    with pytest.raises(mw.SetupError, match="memory_mode must be one of 'auto', 'dense', 'slab'"):
        mw.fdtd.CPMLConfig(memory_mode=value)


@pytest.mark.parametrize("field,value", [
    ("grading_order", 0.0), ("grading_order", float("nan")), ("kappa_max", 0.5), ("alpha_max", -1e-3),
    ("reflection", 0.0), ("reflection", 1.0), ("dense_memory_limit_mib", -1.0), ("kappa_max", "large"),
])
def test_out_of_range_values_raise_setup_error(field, value):
    with pytest.raises(mw.SetupError, match=f"CPMLConfig.{field}"):
        mw.fdtd.CPMLConfig(**{field: value})


def test_unknown_fields_are_impossible_and_the_config_is_frozen():
    with pytest.raises(TypeError, match="kapa_max"):
        mw.fdtd.CPMLConfig(kapa_max=5.0)
    config = mw.fdtd.CPMLConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.kappa_max = 2.0
    assert dataclasses.replace(config, kappa_max=2).kappa_max == 2.0


def test_the_removed_dict_form_is_refused():
    with pytest.raises(mw.SetupError, match="cpml_config dict was removed"):
        mw.FDTDConfig(cpml={"kappa_max": 5.0})
    with pytest.raises(TypeError, match="cpml_config"):
        mw.FDTDConfig(cpml_config={"kappa_max": 5.0})
    with pytest.raises(TypeError, match="cpml_config"):
        mw.Simulation.fdtd(_scene(), frequencies=1e9, cpml_config={"memory_mode": "slab"})


def test_simulation_fdtd_stores_the_typed_config():
    config = mw.fdtd.CPMLConfig(memory_mode="slab", alpha_max=0.0)
    assert mw.Simulation.fdtd(_scene(), frequencies=1e9, config=mw.FDTDConfig(cpml=config)).config.cpml is config
    assert mw.Simulation.fdtd(_scene(), frequencies=1e9).config.cpml == mw.fdtd.CPMLConfig()


def test_trainable_values_are_refused_before_conversion():
    with pytest.raises(NotImplementedError, match="fixed"):
        mw.fdtd.CPMLConfig(alpha_max=torch.tensor(0.1, requires_grad=True))
