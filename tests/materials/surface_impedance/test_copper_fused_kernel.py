"""P-9(c) E1/E1b-E3 against the unchanged torch recurrence.

E1 applies elementwise relative <= 1e-13 to FP64 state, right_e and updated E
before casting, and <= 1 ulp to fp32 fields. E1b additionally requires normwise
relative <= 1e-13 per spectral real/imaginary accumulator, clock and normalization.
Elementwise relative error is ill-posed for spectral cancellation residuals:
an element can be 1e-27 while its tensor maximum is 3.5e-8. E1b measures the
whole tensor; an identically zero reference must remain exactly zero.
"""
import json
from collections import Counter

import pytest
import torch

import witwin.maxwell as mw
from scripts.dev.fdtd.copper_step_floor_profile import build_scene
from tests.materials.surface_impedance.test_copper_foil import DELTA, FREQUENCY
from witwin.maxwell.fdtd.copper_foil import (
    advance_copper_electric, begin_copper_step, correct_copper_magnetic, accumulate_copper_spectra,
)
from witwin.maxwell.fdtd.cuda import backend
from witwin.maxwell.fdtd.cuda.backend import prepare_copper_electric
from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
STATE = ("state", "right_e", "old_e", "cumulative_loss",
         "spectral_step", "spectral_norm", "spectral_real", "spectral_imag")
NATIVE = ("begin_copper_step", "correct_copper_magnetic", "advance_copper_electric",
          "accumulate_copper_spectra")


def _simulation(case, *, steps=2000, width=.01, graph=True):
    if case == "f3":
        scene, frequencies, excitations = build_scene("copper_port")
    else:
        scene = mw.Scene(domain=mw.Domain(((-.3, .3), (-width, width), (-width, width))),
            grid=mw.GridSpec.uniform(.005), device="cuda",
            boundary=mw.BoundarySpec.faces(default="pml", num_layers=12, strength=1., y="pmc", z="pec"),
            structures=[mw.Structure(mw.Box(position=(0., 0., 0.), size=(0., 8*width, 8*width)),
                mw.CopperFoil(thickness=DELTA, frequency_range=(.5e9, 2e9)), name="foil")],
            sources=[mw.PlaneWave(direction=(1., 0., 0.), polarization=(0., 0., 1.),
                source_time=mw.CW(FREQUENCY, amplitude=80.), name="pw")],
            monitors=[mw.PointMonitor("probe", position=(.1, 0., 0.), fields=("Ez",))])
        frequencies, excitations = (FREQUENCY,), ()
    # Closed-transverse PML (y = PMC, z = PEC) ran CPML before absorber="auto"; keep CPML (PCB ACK).
    return mw.Simulation.fdtd(scene, frequencies=frequencies, excitations=excitations, absorber="cpml",
        run_time=mw.TimeConfig(time_steps=steps, transient_cycles=0),
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
        full_field_dft=False, shutoff=0., cuda_graph=graph)


def _tensors(solver):
    return [getattr(solver, name) for name in FIELDS] + [
        entry[name] for entry in solver._copper_foils for name in STATE]


def _comparison_details(solver, actual, reference):
    names = list(FIELDS) + [f"entry{i}.{name}" for i in range(len(solver._copper_foils)) for name in STATE]
    rows = []
    for name, value, expected in zip(names, actual, reference):
        value, expected = value.double().cpu(), expected.double().cpu()
        difference = (value - expected).abs()
        relative = torch.where(expected != 0, difference / expected.abs(), difference)
        flat = int(relative.flatten().argmax())
        norm, diff_norm = float(expected.norm()), float(difference.norm())
        maximum = float(expected.abs().max())
        rows.append(dict(tensor=name, shape=list(expected.shape),
            index=([int(i) for i in torch.unravel_index(torch.tensor(flat), expected.shape)]
                   if expected.ndim else []),
            actual=float(value.flatten()[flat]), reference=float(expected.flatten()[flat]),
            max_relative=float(relative.flatten()[flat]), reference_max_abs=maximum,
            element_absolute_difference=float(difference.flatten()[flat]),
            max_absolute_difference=float(difference.max()),
            difference_relative_to_max=float(difference.max())/maximum if maximum else float(difference.max()),
            reference_norm=norm, difference_norm=diff_norm, relative_norm=diff_norm/norm if norm else diff_norm))
    return rows


def _restore(tensors, saved):
    for destination, source in zip(tensors, saved):
        destination.copy_(source)


def _relative(actual, expected):
    difference = (actual - expected).abs()
    return float(torch.where(expected != 0, difference / expected.abs(), difference).max())


def _ulp_difference(actual, expected):
    def ordered(value):
        bits = value.view(torch.int32).to(torch.int64)
        return torch.where(bits < 0, -(bits & 0x7fffffff), bits)
    return (ordered(actual)-ordered(expected)).abs()


@pytest.fixture
def fused_dispatches(monkeypatch):
    """Count completed native dispatch calls without instrumenting production code."""
    calls = Counter()
    for name in NATIVE:
        original = getattr(backend, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            _original(*args, **kwargs)
            calls[_name] += 1

        monkeypatch.setattr(backend, name, counted)
    return calls


def _step(solver, *, activity=1):
    begin_copper_step(solver)
    correct_copper_magnetic(solver, activity=activity)
    advance_copper_electric(solver, activity=activity)
    accumulate_copper_spectra(solver, activity=activity)


@pytest.mark.parametrize("case", ("f3", "g2"))
@pytest.mark.parametrize("dtype", (torch.float64, torch.float32))
def test_e1_one_step(case, dtype, copper_torch_kernel, fused_dispatches):
    prepared = _simulation(case).prepare()
    prepared.run_until(64 if case == "f3" else 256)
    solver = prepared.solver
    for name in FIELDS:
        setattr(solver, name, getattr(solver, name).to(dtype))
    for entry in solver._copper_foils:
        entry["right_e"] = entry["right_e"].to(dtype)
    assert any(bool(entry["state"].abs().max() > 0) for entry in solver._copper_foils)
    for entry in solver._copper_foils:
        entry["spectral_starts"].zero_()
    tensors = _tensors(solver)
    initial = [value.clone() for value in tensors]
    errors, max_ulp, e1b_errors = [], 0, []
    for window in ("none", "hanning", "ramp"):
        for entry in solver._copper_foils:
            entry.pop("fused_electric", None)
            entry["spectral_window"] = window
        fused_dispatches.clear()
        _restore(tensors, initial)
        _step(solver)
        assert not fused_dispatches
        reference = [value.clone() for value in tensors]
        _restore(tensors, initial)
        for entry in solver._copper_foils:
            assert not (entry["paired"] or entry["imaginary"] or entry["material_ade"])
            prepare_copper_electric(solver, entry)
        _step(solver)
        assert fused_dispatches == Counter({name: len(solver._copper_foils) for name in NATIVE})
        entries, spectral = [], []
        for i, entry in enumerate(solver._copper_foils):
            saved = dict(zip(STATE, reference[len(FIELDS)+i*len(STATE):len(FIELDS)+(i+1)*len(STATE)]))
            field = getattr(solver, entry["electric_name"])
            expected_field = reference[FIELDS.index(entry["electric_name"])]
            row = dict(entry=i, component=entry["electric_name"], state=_relative(entry["state"], saved["state"]))
            if dtype == torch.float64:
                # Double field storage exposes both FP64 trace updates before a field cast.
                row.update(right_e=_relative(entry["right_e"], saved["right_e"]),
                           updated_e=_relative(field[entry["index"]], expected_field[entry["index"]]))
            errors.extend(value for key, value in row.items() if key not in ("entry", "component"))
            entries.append(row)
            assert torch.equal(entry["old_e"], saved["old_e"])
            for name in ("spectral_real", "spectral_imag", "spectral_step", "spectral_norm"):
                norm = torch.linalg.vector_norm(saved[name])
                difference = torch.linalg.vector_norm(entry[name]-saved[name])
                if float(norm) == 0:
                    assert float(difference) == 0
                    error = 0.
                else:
                    error = float(difference/norm)
                spectral.append(dict(entry=i, tensor=name, relative_norm=error))
                e1b_errors.append(error)
                assert error <= 1e-13
        print(json.dumps(dict(gate="E1/E1b", case=case, dtype=str(dtype), window=window,
                              entries=entries, spectral=spectral)))
        ulp_rows = []
        names = list(FIELDS) + [f"entry{i}.{name}" for i in range(len(solver._copper_foils)) for name in STATE]
        for name, actual, expected in zip(names, tensors, reference):
            if actual.dtype == torch.float32:
                ulps = _ulp_difference(actual, expected)
                max_ulp = max(max_ulp, int(ulps.max()))
                flat = int(ulps.flatten().argmax())
                ulp_rows.append(dict(tensor=name, max_ulp=int(ulps.flatten()[flat]),
                    index=[int(i) for i in torch.unravel_index(torch.tensor(flat), actual.shape)],
                    actual=float(actual.flatten()[flat]), reference=float(expected.flatten()[flat])))
        print(json.dumps(dict(gate="E1", case=case, dtype=str(dtype), window=window,
                              max_relative=max(errors), max_ulp=max_ulp, fp32_tensors=ulp_rows)))
        if max(errors) > 1e-13 or max_ulp > 1:
            print(json.dumps(dict(comparisons=_comparison_details(solver, tensors, reference))))
        assert max(errors) <= 1e-13
        assert max_ulp <= 1
    print(json.dumps(dict(gate="E1", case=case, dtype=str(dtype), max_relative=max(errors), max_ulp=max_ulp,
                          e1b_max_relative_norm=max(e1b_errors),
                          torch_fused_dispatches=0, fused_dispatches=dict(fused_dispatches))))
    # A masked step matches the reference, including indexed H-add semantics.
    begin_copper_step(solver)
    before = [value.clone() for value in tensors]
    _step(solver, activity=torch.zeros((), dtype=torch.int32, device="cuda"))
    inactive = [value.clone() for value in tensors]
    _restore(tensors, before)
    for entry in solver._copper_foils:
        entry.pop("fused_electric")
    _step(solver, activity=torch.zeros((), dtype=torch.int32, device="cuda"))
    assert all(torch.equal(a, b) for a, b in zip(tensors, inactive))
    for name, actual, expected in zip(names, tensors, before):
        if name not in ("Hx", "Hy", "Hz"):
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("case", ("f3", "g2"))
def test_e2_trajectory(case, monkeypatch, fused_dispatches):
    outputs = []
    for kernel in ("torch", "fused"):
        fused_dispatches.clear()
        monkeypatch.setenv("WITWIN_MAXWELL_COPPER_KERNEL", kernel)
        result = _simulation(case).run()
        outputs.append((torch.cat([getattr(result.solver, name).flatten().double().cpu() for name in FIELDS]),
                        torch.stack([e["cumulative_loss"].cpu() for e in result.solver._copper_foils]),
                        torch.cat([e["state"].flatten().cpu() for e in result.solver._copper_foils]),
                        dict(fused_dispatches)))
        assert all(("fused_electric" in e) == (kernel == "fused") for e in result.solver._copper_foils)
        assert set(fused_dispatches) == (set(NATIVE) if kernel == "fused" else set())
        del result
    (reference, reference_loss, reference_state, torch_calls), (actual, loss, state, fused_calls) = outputs
    field_error = float(torch.linalg.vector_norm(actual-reference) / torch.linalg.vector_norm(reference))
    assert float(reference_loss.sum()) > 0
    assert torch.equal(loss[reference_loss == 0], reference_loss[reference_loss == 0])
    loss_error = _relative(loss, reference_loss)
    state_difference = float((state-reference_state).abs().max())
    state_relative = _relative(state, reference_state)
    state_relative_linf = state_difference / float(reference_state.abs().max())
    print(json.dumps(dict(gate="E2", case=case, steps=2000, field_relative_l2=field_error,
                          loss_relative=loss_error, state_max_absolute=state_difference,
                          state_max_relative=state_relative, state_relative_linf=state_relative_linf,
                          torch_fused_dispatches=torch_calls,
                          fused_dispatches=fused_calls)))
    assert field_error <= 1e-6
    assert loss_error <= 1e-10
    assert state_difference > 0


def test_e3_graph_equals_eager(monkeypatch):
    monkeypatch.delenv("WITWIN_MAXWELL_COPPER_KERNEL", raising=False)
    prepared = _simulation("g2", width=.05).prepare()
    prepared.run_until(256)
    solver = prepared.solver
    assert all("fused_electric" in e for e in solver._copper_foils)
    assert max(e["fused_electric"]["partials"].numel() for e in solver._copper_foils) > 1
    tensors = _tensors(solver)
    initial = [value.clone() for value in tensors]

    def step():
        _step(solver)

    replay = CudaGraphRunner(device=solver.device).capture(step)
    _restore(tensors, initial)
    for _ in range(100):
        replay()
    graph = [value.clone() for value in tensors]
    _restore(tensors, initial)
    for _ in range(100):
        step()
    assert all(torch.equal(a, b) for a, b in zip(tensors, graph))
    print(json.dumps(dict(gate="E3", steps=100, max_difference=0)))
