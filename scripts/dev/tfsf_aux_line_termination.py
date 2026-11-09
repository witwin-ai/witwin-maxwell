"""CPU qualification of appended TFSF incident-line terminations.

This is an offline numerical experiment, not a runtime CPU backend. The reduced
source-free operator is the downstream invariant block of the basis-probed
operator in coverage/tfsf-surviving-mode:scripts/dev/tfsf_surviving_mode.py:
entry E and terminal E are clamped, and H is expressed in electric units.

Run with the maxwell interpreter, hidden CUDA devices, and one BLAS thread.
The command exits 1 if no setting passes approved G1--G3 over the full record.
The former absolute-tail instruments remain as discrete-wake diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import csv
import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.linalg import eigvals, solve_banded

import witwin.maxwell as mw
from witwin.maxwell.sources import evaluate_source_time

DT = 4.7619e-12
CELLS = (20, 40, 80, 160, 320)
ORDERS = (2, 3, 4)
# Include weaker and stronger loss than the original continuum round-trip target.
TARGETS = (1e-2, 1e-4, 1e-8, 1e-12, 1e-16)


@dataclass(frozen=True)
class Configuration:
    name: str = "axis_pulse"
    h: float = .025
    half: float = .3
    layers: int = 4
    direction: tuple[float, float, float] = (0., 0., 1.)
    slab: bool = False
    cw: bool = False


CONFIGURATIONS = (
    Configuration(),
    Configuration(name="refined_pulse", h=.0125, half=.15),
    Configuration(name="oblique_pulse", direction=(.6, 0., .8)),
    Configuration(name="slab_pulse", slab=True),
    Configuration(name="special_pulse", direction=(1., 0., 0.)),
    Configuration(name="axis_cw", cw=True),
    Configuration(name="slab_cw", slab=True, cw=True),
    Configuration(name="special_cw", direction=(1., 0., 0.), cw=True),
)


def prepare_line(config: Configuration, dtype=torch.float64, *, production=False):
    """Materialize the surviving-mode scene and its 65,536-step time plan."""
    injection = (mw.TFSF.slab(axis="z", bounds=(-.1, .1)) if config.slab
                 else mw.TFSF(bounds=((- .1, .1),) * 3))
    source = mw.PlaneWave(direction=config.direction,
        polarization="Ez" if config.direction[0] == 1 else "Ey" if config.direction[0] else "Ex",
        source_time=mw.CW(2e9) if config.cw else mw.GaussianPulse(2e9, 1e9),
        injection=injection)
    scene = mw.Scene(domain=mw.Domain(bounds=((-config.half, config.half),) * 3),
        grid=mw.GridSpec.uniform(config.h), boundary=mw.BoundarySpec.pml(num_layers=config.layers),
        device="cpu", sources=[source])
    # Diagnostic-only precision admission bypass, as in the surviving-mode study.
    with patch("witwin.maxwell.fdtd.runtime.precision.validate_fp64_forward"):
        solver = mw.Simulation.fdtd(scene, frequency=2e9, backend="cpu_reference", cuda_graph=False,
            full_field_dft=False, field_dtype=dtype,
            run_time=mw.TimeConfig(time_steps=65536, dt=DT * config.h / .025, transient_cycles=0),
            spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False)).prepare().solver
    line = solver._tfsf_state["auxiliary_grid"]
    if not config.cw:
        from witwin.maxwell.fdtd.excitation.tfsf_common import projection_extrema
        state = solver._tfsf_state
        bounds64 = tuple((getattr(solver.scene, axis + "_nodes64")[lo],
                          getattr(solver.scene, axis + "_nodes64")[hi])
                         for axis, lo, hi in zip("xyz", state["lower"], state["upper"]))
        entry64, exit64 = projection_extrema(bounds64, config.direction)
        line.extent64 = exit64 - entry64 + (line.source_index + 6 + 20) * line.ds
        current_bounds = tuple((float(coords[lo]), float(coords[hi])) for coords, lo, hi in
                               zip((solver.scene.x, solver.scene.y, solver.scene.z), state["lower"], state["upper"]))
        _, current_exit = projection_extrema(current_bounds, config.direction)
        master_s_max = current_exit + 6 * line.ds + 20 * line.ds
    else:
        master_s_max = line.s_max
    line.record_steps = solver._time_plan["total_steps"] - solver._time_plan["spectral_start_step"]
    if not production:
        # Frozen pre-fix load for before/after instruments, never a runtime path.
        count = max(math.ceil((master_s_max - line.s_min) / line.ds) + 1, 4)
        line = _graded_line(line, count - 20, 20, 2, 1e-8)
        line.s_max = master_s_max
    return line, solver._tfsf_state


def snapped_cell_count(x: float) -> int:
    """Snap only roundoff within 1e-9 relative of an integer before ceiling."""
    nearest = round(x)
    return nearest if abs(x - nearest) <= 1e-9 * max(1., abs(x)) else math.ceil(x)


def appended_line(line, cells: int, order: int, target: float):
    """Candidate: retain the entire old extent, then append the graded loss.

    In particular no formerly sampled position or old interior coefficient is
    moved. The factory's existing 20-cell downstream margin is retained; this
    is the literal appended-absorber experiment requested for qualification.
    """
    sampled_nodes = max(snapped_cell_count(getattr(line, "extent64", line.s_max - line.s_min) / line.ds) + 1, 4)
    return _graded_line(line, sampled_nodes, cells, order, target)


def _graded_line(line, sampled_nodes, cells, order, target):
    result = copy.copy(line)
    ne = sampled_nodes + cells
    result.electric = torch.zeros(ne, dtype=line.dtype)
    result.magnetic = torch.zeros(ne - 1, dtype=line.dtype)
    result._incident_boundary_h = torch.zeros_like(line._incident_boundary_h)
    result._incident_values = torch.zeros_like(line._incident_values)
    result.electric[result.source_index] = result.source_value(-1)
    result.time_step = 0
    result.absorber_cells = cells
    result.absorber_order = order
    result.absorber_target = target
    result.sampled_nodes = sampled_nodes
    result.max_rate = -(order + 1) * line.wave_speed * math.log(target) / (2 * cells * line.ds)
    for field, offset, material in (("electric", 1., line.eps), ("magnetic", .5, line.mu)):
        rate = torch.zeros_like(getattr(result, field))
        rate[-cells:] = result.max_rate * ((torch.arange(cells, dtype=line.dtype) + offset) / cells) ** order
        loss = .5 * line.dt * rate
        setattr(result, field + "_decay", (1 - loss) / (1 + loss))
        setattr(result, field + "_curl", (line.dt / (material * line.ds)) / (1 + loss))
    return result


def _coefficients(line):
    """Normalize H by impedance; preserve all production coefficient values."""
    j = line.source_index
    return (line.electric_decay.detach().cpu().numpy()[j:].astype(np.float64),
            line.electric_curl.detach().cpu().numpy()[j:].astype(np.float64) / line.impedance,
            line.magnetic_decay.detach().cpu().numpy()[j:].astype(np.float64),
            line.magnetic_curl.detach().cpu().numpy()[j:].astype(np.float64) * line.impedance)


def source_free_update_operator(line) -> np.ndarray:
    """Return the source-clamped Yee step on all dynamic downstream states.

    Ordering is E[j+1:-1], Z*H[j:]. Overwritten entry, endpoint and upstream
    ghost states cannot feed this block; omitting them removes redundant zero
    eigenvalues, without changing any surviving pole. This vectorized basis
    probe applies H before E, just like the original diagnosis.
    """
    de, ce, dh, ch = _coefficients(line)
    ne, nh = len(de) - 2, len(dh)
    electric = np.zeros((ne + 2, ne + nh))
    magnetic = np.zeros((nh, ne + nh))
    electric[1:-1, :ne] = np.eye(ne)
    magnetic[:, ne:] = np.eye(nh)
    magnetic = dh[:, None] * magnetic - ch[:, None] * np.diff(electric, axis=0)
    electric = de[1:-1, None] * electric[1:-1] - ce[1:-1, None] * np.diff(magnetic, axis=0)
    return np.concatenate((electric, magnetic))


def slowest_pole(line) -> dict[str, float]:
    """Slowest oscillatory pole by per-step modulus, matching the diagnosis."""
    values = eigvals(source_free_update_operator(line), check_finite=False)
    positive = values[np.angle(values) > 1e-5]
    value = positive[np.argmax(abs(positive))]
    frequency = float(np.angle(value) / (2 * math.pi * line.dt))
    rate = float(math.log(abs(value)) / line.dt)
    return dict(frequency_Hz=frequency, per_step_modulus=float(abs(value)),
                per_period_modulus=math.exp(rate / frequency), rate_per_s=rate)


def reflection(line, frequency: float) -> float:
    """Discrete line reflection via the driven, tridiagonal harmonic recurrence.

    Solve the actual staggered update at a real frequency, with entry E=1
    and final E=0; decompose the first two interior lossless E samples into
    the two axial Yee waves. This removes hard-source multiple reflections
    and transient fitting errors from the load reflection measurement.
    """
    de, ce, dh, ch = _coefficients(line)
    q = np.exp(-2j * math.pi * frequency * line.dt)
    transport = ch / (q - dh)
    count = len(de) - 2
    diagonal = q - de[1:-1] + ce[1:-1] * q * (transport[1:] + transport[:-1])
    lower = -ce[2:-1] * q * transport[1:-1]
    upper = -ce[1:-2] * q * transport[1:-1]
    bands = np.zeros((3, count), dtype=np.complex128)
    bands[0, 1:] = upper
    bands[1] = diagonal
    bands[2, :-1] = lower
    rhs = np.zeros(count, dtype=np.complex128)
    rhs[0] = ce[1] * q * transport[0]
    field = solve_banded((1, 1), bands, rhs, check_finite=False)
    argument = math.sin(math.pi * frequency * line.dt) * line.ds / (line.wave_speed * line.dt)
    if abs(argument) >= 1:
        raise ValueError("Reflection frequency is outside the line propagating band")
    phase = 2 * math.asin(argument)
    # exp(-i omega t) and +z propagation: exp(+i k z).
    basis = np.exp(1j * np.outer(np.array([1., 2.]), np.array([phase, -phase])))
    forward, backward = np.linalg.solve(basis, field[:2])
    return float(abs(backward / forward))


def entry_history(line, stop: int) -> np.ndarray:
    """March the actual H-then-E recurrence and record dynamic entry samples."""
    de, ce, dh, ch = _coefficients(line)
    source = line.source_time
    electric = np.zeros(len(de))
    magnetic = np.zeros(len(dh))
    electric[0] = evaluate_source_time(source, 0.)
    entry = np.empty(stop)
    for n in range(stop):
        magnetic = dh * magnetic - ch * np.diff(electric)
        electric[1:-1] = de[1:-1] * electric[1:-1] - ce[1:-1] * np.diff(magnetic)
        electric[0] = evaluate_source_time(source, (n + 1) * line.dt)
        electric[-1] = 0
        entry[n] = max(abs(electric[1]), abs(magnetic[0])) / source["amplitude"]
    return entry


def pulse_tail(line, pole, *, extra_transits=1) -> dict[str, float | bool]:
    """Measure A0 and entry-neighbour tail after envelope decay + L/c.

    E at the entry itself is clamped and cannot diagnose a returning wave.
    Use E[j+1] and Z*H[j], the first dynamic entry-face samples. N includes
    both electric and magnetic state, a more permissive FP32 floor than E only.
    The windowed maxima span one source period and avoid a zero crossing.
    """
    de, ce, dh, ch = _coefficients(line)
    source = line.source_time
    count = len(line.electric) + len(line.magnetic)
    bound = float(max(1e-6, float(np.finfo(np.float32).eps) * math.sqrt(count)))
    # Peak is the declared envelope peak, 1 for every qualification pulse.
    sigma = 1 / (2 * math.pi * source["fwidth"])
    end = math.ceil((source["delay"] + sigma * math.sqrt(-2 * math.log(bound))) / line.dt)
    length = (len(de) - 1) * line.ds
    transit_steps = math.ceil(length / line.wave_speed / line.dt)
    window = math.ceil(1 / source["frequency"] / line.dt)
    gate = end + transit_steps
    stop = gate + max(window, extra_transits * transit_steps)
    entry = entry_history(line, stop)
    a0 = float(max(entry[end - 1:end - 1 + window]))
    at_gate = float(max(entry[gate - 1:gate - 1 + window]))
    after_gate = float(max(entry[gate - 1:]))
    periods = pole["frequency_Hz"] * transit_steps * line.dt
    prediction = a0 * pole["per_period_modulus"] ** periods
    return dict(state_size=count, relative_bound=bound, pulse_end_ns=end * line.dt * 1e9,
        transit_ns=transit_steps * line.dt * 1e9, available_periods=periods, A0=a0,
        pole_bound_prediction=prediction, entry_tail=at_gate, subsequent_tail=after_gate,
        pole_gate=prediction < bound, time_domain_gate=after_gate < bound)


def sampled_region_lossless(line, state):
    """Check actual face gathers, including both interpolation neighbours."""
    records = []
    for key in ("electric_terms", "magnetic_terms"):
        for term in state[key]:
            # Electric corrections consume H and magnetic corrections consume E.
            kind = term.get("sample_kind", "magnetic" if key == "electric_terms" else "electric")
            decay = getattr(line, kind + "_decay")
            if "sample_indices" in term:
                indices = term["sample_indices"].long()
            else:
                origin = line.s_min + (.5 * line.ds if kind == "magnetic" else 0.)
                coordinate = (term["sample_positions"].double() - origin) / line.ds
                lower = coordinate.floor().long()
                indices = torch.cat((lower.reshape(-1), (lower + 1).reshape(-1)))
            records.append(bool(torch.all((indices >= 0) & (indices < len(decay)))
                           and torch.all(decay[indices] == 1)))
    return all(records)


def outgoing_control(line, candidate_row):
    """Compare a candidate against a distant, entirely lossless line end.

    This distinguishes the source-driven dispersive wake from a returning
    termination reflection. Only the qualification experiment is lengthened.
    """
    candidate = appended_line(line, candidate_row["cells"], candidate_row["order"],
                              candidate_row["target"])
    reference = appended_line(line, 2048, 2, 1e-8)
    reference.electric_decay.fill_(1)
    reference.magnetic_decay.fill_(1)
    reference.electric_curl.fill_(line.dt / (line.eps * line.ds))
    reference.magnetic_curl.fill_(line.dt / (line.mu * line.ds))
    tail = candidate_row["tail"]
    end = round(tail["pulse_end_ns"] * 1e-9 / line.dt)
    transit = round(tail["transit_ns"] * 1e-9 / line.dt)
    gate = end + transit
    window = math.ceil(1 / line.source_time["frequency"] / line.dt)
    original = entry_history(reference, gate + transit)
    measured = entry_history(candidate, gate + transit)
    at_gate = slice(gate - 1, gate - 1 + window)
    return dict(electric_nodes=len(reference.electric),
        lossless_reference_entry_tail=float(max(original[at_gate])),
        candidate_entry_tail=float(max(measured[at_gate])),
        relative_bound=tail["relative_bound"],
        max_entry_history_difference_at_gate=float(max(abs(measured[at_gate] - original[at_gate]))),
        deadline_ns=gate * line.dt * 1e9,
        reference_roundtrip_ns=2 * (len(reference.electric) - line.source_index - 1)
                              * line.ds / line.wave_speed * 1e9)


# An offline compiled recurrence keeps the full grid search practical on CPU.
# No runtime backend, production kernel, or numerical fast-math is introduced.
_MARCH_LIBRARY = None
_SEED_CACHE = {}
_MARCH_SOURCE = r'''
#include <vector>
template<class T> void march(int ne, int steps, int j, const T* de, const T* ce,
 const T* dh, const T* ch, const T* seeds, int ns, const int* ei, int nh,
 const int* hi, T* out) {
 std::vector<T> e(ne,0), h(ne-1,0);
 e[j]=seeds[0];
 for(int n=0;n<steps;n++) {
  T left=0, outer=0;
  if(j==2) {
   T rh=h[j]-ch[j]*(e[j+1]-e[j]);
   T fh=h[j+1]-ch[j]*(e[j+2]-e[j+1]);
   T re=e[j+1]-ce[j]*(fh-rh);
   T nrh=rh-ch[j]*(re-seeds[n+1]);
   left=rh+(seeds[n+1]-e[j])/ce[j];
   T nl=nrh+(seeds[n+2]-seeds[n+1])/ce[j];
   T le=e[j]+(left-h[j-1])/ch[j];
   T nle=seeds[n+1]+(nl-left)/ch[j];
   outer=left+(nle-le)/ce[j];
   e[j-1]=le;
  }
  for(int k=0;k<ns;k++) out[(long long)n*(2*ns+nh)+k]=e[ei[k]];
  for(int i=0;i<ne-1;i++) h[i]=dh[i]*h[i]-ch[i]*(e[i+1]-e[i]);
  if(j==2) {h[j-1]=left;h[j-2]=outer;}
  for(int k=0;k<nh;k++) out[(long long)n*(2*ns+nh)+2*ns+k]=h[hi[k]];
  for(int i=1;i<ne-1;i++) e[i]=de[i]*e[i]-ce[i]*(h[i]-h[i-1]);
  e[j]=seeds[n+1];e[ne-1]=0;
  for(int k=0;k<ns;k++) out[(long long)n*(2*ns+nh)+ns+k]=e[ei[k]];
 }
}
extern "C" {
void march32(int a,int b,int c,const float*d,const float*e,const float*f,const float*g,
 const float*h,int i,const int*j,int k,const int*l,float*m) {march(a,b,c,d,e,f,g,h,i,j,k,l,m);}
void march64(int a,int b,int c,const double*d,const double*e,const double*f,const double*g,
 const double*h,int i,const int*j,int k,const int*l,double*m) {march(a,b,c,d,e,f,g,h,i,j,k,l,m);}
}
'''


def sampled_indices(line, state):
    """All raw samples, including both neighbours of each interpolated gather."""
    result = {"electric": set(), "magnetic": set()}
    for key in ("electric_terms", "magnetic_terms"):
        for term in state[key]:
            kind = term.get("sample_kind", "magnetic" if key == "electric_terms" else "electric")
            if "sample_indices" in term:
                indices = term["sample_indices"].reshape(-1).tolist()
            else:
                origin = line.s_min + (.5 * line.ds if kind == "magnetic" else 0.)
                # Match the production interpolation arithmetic, including dtype.
                coordinate = (term["sample_positions"].to(line.dtype) - origin) / line.ds
                lower = coordinate.floor().long().reshape(-1)
                indices = torch.cat((lower, lower + 1)).tolist()
            result[kind].update(indices)
    return tuple(np.array(sorted(result[kind]), dtype=np.int32) for kind in ("electric", "magnetic"))


def sampled_history(line, state, steps):
    """Actual physical-H recurrence, without fusion, at both face sample clocks."""
    global _MARCH_LIBRARY
    if _MARCH_LIBRARY is None:
        folder = Path(tempfile.mkdtemp(prefix="tfsf-line-instrument-"))
        (folder / "march.cpp").write_text(_MARCH_SOURCE)
        subprocess.run(["g++", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
                        str(folder / "march.cpp"), "-o", str(folder / "march.so")], check=True)
        _MARCH_LIBRARY = ctypes.CDLL(str(folder / "march.so"))
        import shutil
        shutil.rmtree(folder)
    assert line.source_index in (0, 2), "Instrument needs the actual entry ghost stencil"
    ei, hi = sampled_indices(line, state)
    arrays = [getattr(line, key).numpy() for key in
              ("electric_decay", "electric_curl", "magnetic_decay", "magnetic_curl")]
    dtype = arrays[0].dtype
    seed_key = (tuple(sorted(line.source_time.items())), line.dt, steps, dtype.str)
    if seed_key not in _SEED_CACHE:
        _SEED_CACHE[seed_key] = np.array([evaluate_source_time(line.source_time, n * line.dt)
                                        for n in range(steps + 2)], dtype=dtype)
    seeds = _SEED_CACHE[seed_key]
    history = np.empty((steps, 2 * len(ei) + len(hi)), dtype=dtype)
    ptr = lambda a: ctypes.c_void_p(a.ctypes.data)
    function = getattr(_MARCH_LIBRARY, "march32" if dtype == np.float32 else "march64")
    function(len(line.electric), steps, line.source_index, *map(ptr, arrays), ptr(seeds),
             len(ei), ptr(ei), len(hi), ptr(hi), ptr(history))
    # Scale only the recorded H, preserving physical-H stepping roundoff.
    history[:, 2 * len(ei):] *= line.impedance
    return history


def lossless_reference(line, steps):
    """Same coefficients/dtype; far-end round trip exceeds the entire record."""
    ne = max(len(line.electric), math.ceil(1.1 * steps * line.dt * line.wave_speed / (2 * line.ds))
             + line.source_index + 64)
    result = copy.copy(line)
    result.electric = torch.zeros(ne, dtype=line.dtype)
    result.magnetic = torch.zeros(ne - 1, dtype=line.dtype)
    for kind, material in (("electric", line.eps), ("magnetic", line.mu)):
        setattr(result, kind + "_decay", torch.ones_like(getattr(result, kind)))
        setattr(result, kind + "_curl", torch.full_like(getattr(result, kind), line.dt / (material * line.ds)))
    result.reference_roundtrip_steps = 2 * (ne - line.source_index - 1) * line.ds / (line.wave_speed * line.dt)
    assert result.reference_roundtrip_steps > steps
    return result


def history_gate(line, state, reference, steps):
    measured = sampled_history(line, state, steps)
    error = np.max(np.abs(measured.astype(np.float64) - reference), axis=1)
    peak = abs(line.source_time["amplitude"])
    bound = max(1e-6 * peak, float(np.finfo(np.float32).eps) * math.sqrt(steps))
    state_bound = max(1e-6 * peak, float(np.finfo(np.float32).eps) * math.sqrt(len(line.electric) + len(line.magnetic)))
    half, full = float(max(error[:steps // 2])), float(max(error))
    growth = full / half if half else (1. if not full else math.inf)
    return dict(steps=steps, bound=bound, half=half, full=full, growth=growth,
                half_over_B=half / bound, full_over_B=full / bound,
                state_bound=state_bound, state_margin=state_bound / full if full else math.inf,
                passed=full <= bound and growth <= 4)


def approved_sweep(output: Path):
    """Approved 2026-11-05 G1--G4, with no changes to production code."""
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    rows, baselines = [], {}
    for config in CONFIGURATIONS:
        line, state = prepare_line(config)
        edge = math.asin(line.wave_speed * line.dt / line.ds) / (math.pi * line.dt)
        frequencies = [2e9] if config.cw else np.linspace(1e9, min(3e9, .99 * edge), 17).tolist()
        old_reflection = [reflection(line, f) for f in frequencies]
        baseline = dict(frequencies_Hz=frequencies, reflection=old_reflection, dtypes={})
        configurations = {}
        for dtype in (torch.float32, torch.float64):
            old, faces = prepare_line(config, dtype)
            reference = None
            if not config.cw:
                ref_line = lossless_reference(old, old.record_steps)
                reference = sampled_history(ref_line, faces, old.record_steps)
                baseline["dtypes"][str(dtype)] = history_gate(old, faces, reference, old.record_steps)
                baseline["dtypes"][str(dtype)]["reference_roundtrip_steps"] = ref_line.reference_roundtrip_steps
            old_refl = [reflection(old, f) for f in frequencies]
            baseline.setdefault("reflection_by_dtype", {})[str(dtype)] = old_refl
            configurations[str(dtype)] = (old, faces, reference, old_refl)
        baselines[config.name] = baseline
        for cells in CELLS:
            for order in ORDERS:
                for target in TARGETS:
                    candidate = appended_line(line, cells, order, target)
                    reflected = [reflection(candidate, f) for f in frequencies]
                    worst_ratio = max(n / o for n, o in zip(reflected, old_reflection))
                    row = dict(configuration=config.name, cells=cells, order=order, target=target,
                               reflection_ratio=worst_ratio, reflection=reflected,
                               reflection_gate=worst_ratio <= 1, dtypes={})
                    for dtype, (old, faces, reference, old_refl) in configurations.items():
                        new = appended_line(old, cells, order, target)
                        assert sampled_region_lossless(new, faces)
                        ratio = max(reflection(new, f) / o for f, o in zip(frequencies, old_refl))
                        row.setdefault("reflection_ratio_by_dtype", {})[dtype] = ratio
                        row["reflection_ratio"] = max(row["reflection_ratio"], ratio)
                        if not config.cw:
                            row["dtypes"][dtype] = history_gate(new, faces, reference, old.record_steps)
                    row["reflection_gate"] = row["reflection_ratio"] <= 1
                    row["passed"] = row["reflection_gate"] and all(g["passed"] for g in row["dtypes"].values())
                    rows.append(row)
        print(config.name, "passing", sum(r["passed"] for r in rows if r["configuration"] == config.name),
              "before", baseline["dtypes"], flush=True)
        (output / "approved.json").write_text(json.dumps(dict(baselines=baselines, rows=rows), indent=2) + "\n")
    passing = []
    for cells in CELLS:
        for order in ORDERS:
            for target in TARGETS:
                group = [r for r in rows if (r["cells"], r["order"], r["target"]) == (cells, order, target)]
                if all(r["passed"] for r in group):
                    margin = min(1 / g["full_over_B"] for r in group for g in r["dtypes"].values())
                    passing.append((cells, order, target, margin))
    passing.sort(key=lambda s: (s[0], s[1], -s[3]))
    receipt = dict(baselines=baselines, rows=rows, passing_settings=passing,
                   selected=passing[0] if passing else None)
    write_sweep_csv(output / "approved.csv", rows)
    (output / "approved.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt



def write_sweep_csv(path, rows):
    with path.open("w") as handle:
        names = ["configuration", "M", "p", "R", "fp32_half/B", "fp32_full/B", "fp32_growth", "fp32_state_margin",
                 "fp64_half/B", "fp64_full/B", "fp64_growth", "fp64_state_margin", "G3_ratio", "passed"]
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        for row in rows:
            values = [row["configuration"], row["cells"], row["order"], row["target"]]
            for dtype in ("torch.float32", "torch.float64"):
                g = row["dtypes"].get(dtype, {})
                values.extend(format(g[k], ".8g") if k in g else "" for k in
                              ("half_over_B", "full_over_B", "growth", "state_margin"))
            writer.writerow([*values, format(row["reflection_ratio"], ".8g"), row["passed"]])


def selected_report(receipt, output):
    """Report 10x horizon and pole/layout diagnostics without extending the gate."""
    if receipt["selected"] is None:
        return
    cells, order, target, _ = receipt["selected"]
    reports = {}
    for config in CONFIGURATIONS:
        if config.cw:
            continue
        reports[config.name] = {}
        for dtype in (torch.float32, torch.float64):
            old, state = prepare_line(config, dtype)
            candidate = appended_line(old, cells, order, target)
            faces = state
            steps = old.record_steps * 10
            reference_line = lossless_reference(old, steps)
            reference = sampled_history(reference_line, state, steps)
            report = dict(candidate=history_gate(candidate, faces, reference, steps),
                          before=history_gate(old, state, reference, steps),
                          reference_roundtrip_steps=reference_line.reference_roundtrip_steps,
                          reference_nodes=len(reference_line.electric),
                          candidate_pole=slowest_pole(candidate), before_pole=slowest_pole(old))
            reports[config.name][str(dtype)] = report
            print("10x", config.name, str(dtype), report, flush=True)
        (output / "extended.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/tfsf-aux-line-termination"))
    parser.add_argument("--report-only", action="store_true", help="Use the completed approved.json sweep")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES='' for CPU qualification")
    receipt = (json.loads((args.out / "approved.json").read_text()) if args.report_only
               else approved_sweep(args.out))
    if receipt["passing_settings"]:
        selected_report(receipt, args.out)
    print("passing_settings", receipt["passing_settings"], flush=True)
    return 0 if receipt["passing_settings"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
