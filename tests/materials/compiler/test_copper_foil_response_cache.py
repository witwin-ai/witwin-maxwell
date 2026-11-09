"""Automatic modal fits reuse exact inputs without sharing mutable outputs."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields
import math

import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler import copper_foil as compiler


def test_fit_cache_exact_keys_and_private_response_tensors(monkeypatch):
    parameters = dict(thickness=35e-6, conductivity=5.8e7, frequency_range=(.1e9, 10e9))
    expected = compiler.compile_copper_response(mw.CopperFoil(
        thickness=1e-6, frequency_range=(.5e9, 2e9), mode_count=16))
    calls = []

    def fit(*args):
        calls.append(args)
        return deepcopy(expected)

    monkeypatch.setattr(compiler, "_COPPER_FIT_CACHE", OrderedDict())
    monkeypatch.setattr(compiler, "_fit_copper_response", fit)
    first = compiler.compile_copper_response(mw.CopperFoil(**parameters))
    first.rates.fill_(-1)
    second = compiler.compile_copper_response(mw.CopperFoil(**parameters))
    assert len(calls) == 1
    assert torch.equal(second.rates, expected.rates)
    second.state_space.B.zero_()
    third = compiler.compile_copper_response(mw.CopperFoil(**parameters))
    assert torch.equal(third.state_space.B, expected.state_space.B)

    # One ULP changes are distinct, as are scalar/tensor arithmetic and fit options.
    changes = [dict(thickness=math.nextafter(parameters["thickness"], math.inf)),
               dict(conductivity=math.nextafter(parameters["conductivity"], math.inf)),
               dict(frequency_range=(math.nextafter(.1e9, math.inf), 10e9)),
               dict(frequency_range=(.1e9, math.nextafter(10e9, math.inf))),
               dict(thickness=torch.tensor(35e-6, dtype=torch.float64)),
               dict(thickness=torch.tensor(35e-6, dtype=torch.float32))]
    for changed in changes:
        compiler.compile_copper_response(mw.CopperFoil(**(parameters | changed)))
    compiler.compile_copper_response(mw.CopperFoil(**parameters),
                                    tolerance=math.nextafter(2e-4, math.inf))
    compiler.compile_copper_response(mw.CopperFoil(**parameters), max_modes=128)
    assert len(calls) == 1 + len(changes) + 2


def test_fit_cache_evicts_least_recently_used(monkeypatch):
    expected = compiler.compile_copper_response(mw.CopperFoil(
        thickness=1e-6, frequency_range=(.5e9, 2e9), mode_count=16))
    monkeypatch.setattr(compiler, "_COPPER_FIT_CACHE", OrderedDict())
    monkeypatch.setattr(compiler, "_fit_copper_response", lambda *args: deepcopy(expected))
    assert compiler._COPPER_FIT_CACHE_CAPACITY == 64
    materials = [mw.CopperFoil(thickness=(35 + i) * 1e-6, frequency_range=(.5e9, 2e9))
                 for i in range(65)]
    for i, material in enumerate(materials[:64]):
        compiler.compile_copper_response(material)
        assert len(compiler._COPPER_FIT_CACHE) == i + 1
    first_key, second_key = list(compiler._COPPER_FIT_CACHE)[:2]
    compiler.compile_copper_response(materials[0])
    assert list(compiler._COPPER_FIT_CACHE)[-1] == first_key
    compiler.compile_copper_response(materials[64])
    assert len(compiler._COPPER_FIT_CACHE) == 64
    assert first_key in compiler._COPPER_FIT_CACHE
    assert second_key not in compiler._COPPER_FIT_CACHE


def test_fit_cache_eviction_preserves_response_tensors(monkeypatch):
    monkeypatch.setattr(compiler, "_COPPER_FIT_CACHE", OrderedDict())
    monkeypatch.setattr(compiler, "_COPPER_FIT_CACHE_CAPACITY", 2)
    material = mw.CopperFoil(thickness=35e-6, frequency_range=(.1e9, 10e9))
    first = compiler.compile_copper_response(material)
    key = next(iter(compiler._COPPER_FIT_CACHE))
    for thickness in (36e-6, 37e-6):
        compiler.compile_copper_response(mw.CopperFoil(
            thickness=thickness, frequency_range=material.frequency_range))
    assert len(compiler._COPPER_FIT_CACHE) == 2
    assert key not in compiler._COPPER_FIT_CACHE
    recomputed = compiler.compile_copper_response(material)
    for before, after in ((first, recomputed), (first.rational, recomputed.rational),
                          (first.state_space, recomputed.state_space)):
        for field in fields(before):
            value = getattr(before, field.name)
            if torch.is_tensor(value):
                assert torch.equal(value, getattr(after, field.name)), field.name
