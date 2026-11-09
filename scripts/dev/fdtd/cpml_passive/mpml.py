"""CPU-only B0b-1 multi-axial CPML, spectra and exact discrete reflection.

Tangential sigma/alpha and kappa-1 are p times their normal values. Two
previously inactive derivative memories are required in each reduced block.
The normal profile is never modified. No native runtime is constructed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
from itertools import product
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import linalg
from scipy.linalg import solve_banded

from model import (Block, Cell, C0, DT, EPS0, ETA, MU0, native_matrix,
                   oscillatory_spectrum, profiles)
from reflection import layer_profiles, reflection, transfer

ROOT = Path(__file__).resolve().parent
P_GRID = (0., .001, .002, .003, .004, .005, .0075, .01, .015, .02,
          .025, .03, .04, .05, .06, .07, .085, .1, .125, .15, .175, .2, .25, .3)
THRESHOLD = 1+1e-12


def tangential_profile(nodes, positions, cell, p, *, magnetic=False, native=False,
                       one_sided=False):
    """Resample the *physical* scaled normal profile at the target Yee field.

    Scaling shift_rate together with alpha keeps the original sigma compensation
    factor exactly fixed; it is not applied a second time at the scaled alpha.
    The repository builder then applies its ordinary held-input or CFS map.
    """
    if p < 0:
        raise ValueError("p must be nonnegative")
    import torch
    from witwin.maxwell.fdtd.boundary.cpml import _cpml_profile_1d, DEFAULT_CPML_CONFIG
    config = dict(DEFAULT_CPML_CONFIG, grading_order=cell.grading,
                  kappa_max=cell.kappa, reflection=cell.reflection)
    rate, floor = None, 0.
    if cell.preset == "cfs":
        rate = 2*np.pi*cell.f_min
        fraction = np.sin(np.deg2rad(cell.theta))
        floor = min(1., 2*fraction)
        cosine = np.sqrt(1-min(fraction, np.sqrt(1-(4.2/8)**2))**2)
        config.update(alpha_max=EPS0*rate, grading_order=2., reflection=10**(-4.2/cosine))
    if cell.alpha_scale is not None:
        rate, floor = 2*np.pi*C0/cell.period, 1.
        config["alpha_max"] = cell.alpha_scale*EPS0*rate
    config["reflection"] = np.exp(p*np.log(config["reflection"]))
    config["kappa_max"] = 1+p*(config["kappa_max"]-1)
    config["alpha_max"] *= p
    if magnetic:
        config["alpha_max"] *= MU0/EPS0
    dtype = torch.float32 if native else torch.float64
    values = _cpml_profile_1d(positions, nodes, cell.layers, cell.dt,
        MU0 if magnetic else EPS0,
        torch.tensor(MU0/EPS0 if magnetic else 1., dtype=dtype), ETA, config,
        apply_low=not one_sided, shift_rate=None if rate is None else p*rate,
        shift_floor=floor)
    return tuple(v.numpy().copy() for v in values)


@lru_cache(maxsize=512)
def tangential_coefficients(cell, p, native=False):
    positions = cell.nodes if cell.polarization == "TE" else (cell.nodes[:-1]+cell.nodes[1:])/2
    values = tuple(tangential_profile(cell.nodes, positions, cell, p,
                    magnetic=magnetic, native=native) for magnetic in (False, True))
    if not native and cell.polarization == "TE":
        values = tuple(tuple(v[1:-1] for v in group) for group in values)
    return values


def block(cell, p, **kwargs):
    # p=0 uses precisely the B0 arithmetic and constrained state space.
    return Block(cell, tangential_coefficients=None if p == 0 else
                 tangential_coefficients(cell, p), **kwargs)


def native(cell, p, *, rounded=False):
    return native_matrix(cell, rounded=rounded, tangential_coefficients=None if p == 0 else
                         tangential_coefficients(cell, p, True))


def spectrum(b):
    indices = b.active_indices("current")
    return oscillatory_spectrum(b.matrix()[np.ix_(indices, indices)], b.cell.dt)


def layer_reflection(frequency, theta, p, pol, *, preset="real", evanescent=False):
    """Eliminate the actual H-before-E map; TE and TM differ when p is nonzero."""
    # At k_y=0 the tangential memories have zero drive and the normal stack is
    # exactly B0; retain its band scaling and roundoff as well as its physics.
    if p == 0 or (theta == 0 and not evanescent):
        return reflection(frequency, theta, "current", preset=preset, evanescent=evanescent)
    h = C0/(20*3e9)
    dt = h/(2*C0*np.sqrt(2))
    nodes, normal = layer_profiles(h, dt, preset, theta)
    cell = Cell(polarization=pol, dt=dt, preset=preset, theta=theta)
    positions = nodes if pol == "TE" else (nodes[:-1]+nodes[1:])/2
    tangent = [tangential_profile(nodes, positions, cell, p, magnetic=m,
                                one_sided=True) for m in (False, True)]
    omega = 2*np.pi*frequency
    lam = np.exp(-1j*omega*dt)
    c = 1/np.sqrt(EPS0*MU0)
    ky = omega/c*(1.25 if evanescent else np.sin(np.deg2rad(theta)))
    k = 2/h*np.sin(.5*ky*h)
    temporal = 2*np.sin(.5*omega*dt)/(c*dt)
    vacuum_K2 = temporal**2-k*k
    kz = 2/h*np.arcsin(np.sqrt(complex(vacuum_K2))*h/2)
    ge, gh = [transfer(g, lam, dt, "current") for g in normal]
    te, th = [transfer(g, lam, dt, "current") for g in tangent]
    K2 = temporal**2-k*k*te*th
    if pol == "TE":
        low = ge[1:-1]*gh[:-1]/h**2
        high = ge[1:-1]*gh[1:]/h**2
        diag = -low-high+K2[1:-1]
    else:
        low = ge[1:-1]*gh[:-1]/(K2[:-1]*h**2)
        high = ge[1:-1]*gh[1:]/(K2[1:]*h**2)
        diag = -low-high+1
    count = len(diag)
    band = np.zeros((3, count), complex)
    band[0, 1:], band[1], band[2, :-1] = high[:-1], diag, low[1:]
    rhs = np.zeros(count, complex)
    rhs[0] = -low[0]
    ratio = solve_banded((1, 1), band, rhs, check_finite=False)[0]
    r = (np.exp(1j*kz*h)-ratio)/(ratio-np.exp(-1j*kz*h))
    return float(abs(r*np.exp(2j*kz*nodes[0])))


def reflection_curve(ps):
    rows = []
    for p, preset, pol, metric, f in product(ps, ("real", "cfs", "alpha0"),
            ("TE", "TM"), ("normal", "60deg", "80deg", "evanescent"), (.3, .6, 1., 2., 3.)):
        angle = {"normal": 0., "60deg": 60., "80deg": 80., "evanescent": 0.}[metric]
        kw = dict(preset=preset, evanescent=metric == "evanescent")
        baseline = layer_reflection(f*1e9, angle, 0, pol, **kw)
        value = layer_reflection(f*1e9, angle, p, pol, **kw)
        limit = 2 if metric in ("normal", "60deg") else 1
        passed = value < baseline if metric == "evanescent" else value <= limit*baseline
        rows.append(dict(p=p, preset=preset, polarization=pol, metric=metric,
                         frequency_GHz=f, R=value, current_R=baseline,
                         ratio=value/baseline, passes=bool(passed)))
    summary = []
    for p in ps:
        chosen = [r for r in rows if r["p"] == p]
        summary.append(dict(p=p, passes=all(r["passes"] for r in chosen),
            failed_rows=sum(not r["passes"] for r in chosen),
            metrics={m:dict(max_ratio=max(r["ratio"] for r in chosen if r["metric"] == m),
                           max_R=max(r["R"] for r in chosen if r["metric"] == m),
                           failed_rows=sum(not r["passes"] for r in chosen if r["metric"] == m))
                     for m in ("normal", "60deg", "80deg", "evanescent")}))
    return dict(summary=summary, rows=rows, limits=dict(normal=2, **{"60deg": 2, "80deg": 1,
                     "evanescent": "strictly < 1"}), profiles=["real", "cfs", "alpha0"])


def sweep_geometry(job):
    geometry, ps = job
    wall, layers, refinement = geometry
    base = Cell(wall=wall, layers=layers, dy=.0125 if refinement == "y_half" else .025,
                dz=.0125 if refinement == "z_half" else .025, theta=35.)
    rows = []
    for pol, order, factor in product(("TE", "TM"), (1, 2, 3), (1., .5, .25)):
        real = None
        for preset in ("real", "cfs", "alpha0"):
            cell = replace(base, polarization=pol, order=order, dt=DT*factor, preset=preset)
            values = real if preset == "alpha0" else [spectrum(block(cell, p)) for p in ps]
            if preset == "real":
                real = values
            rows.append(dict(cell=asdict(cell), refinement=refinement, values=values))
    return rows


def fixed_jobs(job):
    fmin, angle, ps = job
    rows = []
    for pol, order, factor in product(("TE", "TM"), (0, 1, 2, 3), (1., .5, .25)):
        cell = Cell(polarization=pol, order=order, dt=DT*factor, preset="cfs", theta=angle, f_min=fmin)
        ky = 2*np.pi*order/cell.period+2*np.pi*fmin/C0*np.sin(np.deg2rad(angle))
        hy = cell.period/cell.ny
        values = [spectrum(block(cell, p, wavenumber=2/hy*np.sin(ky*hy/2))) for p in ps]
        rows.append(dict(cell=asdict(cell), physical_ky=ky, values=values))
    return rows


def run_sweep(ps, workers, quick=False):
    geometry = list(product(("periodic", "pec", "pmc"), (6, 12, 24), ("base", "y_half", "z_half")))
    if quick:
        geometry = [("periodic", 12, "base")]
    cart, fixed = [], []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for g, rows in zip(geometry, pool.map(sweep_geometry, ((g, ps) for g in geometry))):
            cart.extend(rows)
            print(json.dumps(dict(geometry=g, cases=len(cart), seconds=time.monotonic()-start)), flush=True)
        jobs = [(f, a, ps) for f, a in product((.3e9, 2e9), (0., 10., 30., 45., 60., 75.))]
        for rows in pool.map(fixed_jobs, jobs):
            fixed.extend(rows)
    summary = []
    for i, p in enumerate(ps):
        result = dict(p=p)
        for name, rows in (("cartesian", cart), ("fixed_k", fixed)):
            worst = max(rows, key=lambda r: r["values"][i]["rho"])
            result[name] = dict(cases=len(rows), failed=sum(r["values"][i]["rho"] > THRESHOLD for r in rows),
                                worst=worst["values"][i], worst_case=worst["cell"])
        result["passes"] = all(result[n]["failed"] == 0 for n in ("cartesian", "fixed_k"))
        summary.append(result)
    return dict(p_grid=ps, threshold=THRESHOLD, complete=not quick, summary=summary,
                cartesian=cart, fixed_k=fixed, seconds=time.monotonic()-start)


def refine(path, workers):
    """Locate a threshold with failing blocks, then certify a new full grid.

    The inexpensive bisection is only a grid-selection heuristic. Every added
    ratio is subsequently evaluated on all 1,458 + 288 cases.
    """
    original = json.loads(path.read_text())
    assert original["complete"]
    hi = next(s["p"] for s in original["summary"] if s["passes"])
    lo = max(s["p"] for s in original["summary"] if s["p"] < hi and not s["passes"])
    index = original["p_grid"].index(lo)
    unique = {}
    for family in ("cartesian", "fixed_k"):
        for row in original[family]:
            if row["values"][index]["rho"] <= THRESHOLD:
                continue
            cell = dict(row["cell"])
            if cell["preset"] == "alpha0":
                cell["preset"] = "real"
            unique[json.dumps(cell, sort_keys=True)] = (cell, row.get("physical_ky"))
    probes = []
    for data, physical_ky in unique.values():
        cell = Cell(**data)
        kwargs = {}
        if physical_ky is not None:
            hy = cell.period/cell.ny
            kwargs["wavenumber"] = 2/hy*np.sin(physical_ky*hy/2)
        low, high = lo, hi
        while high-low > 1e-8:
            mid = (low+high)/2
            if spectrum(block(cell, mid, **kwargs))["rho"] > THRESHOLD:
                low = mid
            else:
                high = mid
        probes.append(dict(cell=data, failing_p=low, passing_p=high))
    critical = max(r["passing_p"] for r in probes)
    bin_index = int(np.floor(critical/1e-5))
    ps = [round(i*1e-5, 8) for i in range(bin_index-1, bin_index+3)]
    assert not set(ps).intersection(original["p_grid"])
    print(json.dumps(dict(threshold_predictor=critical, refined_grid=ps)), flush=True)
    report = run_sweep(ps, workers)
    report["threshold_predictor"] = dict(source=str(path), bracket=[lo, hi], probes=probes,
        caveat="Heuristic bisection selects grid only; full sweeps supply the reported gates")
    return report


def power(ps, steps):
    rows = []
    start = time.monotonic()
    for p, pol in product(ps, ("TE", "TM")):
        cell = Cell(period=4*1.28, polarization=pol)
        b = block(cell, p)
        orders = np.arange(cell.ny//2+1)
        b.k = 2*np.sin(np.pi*orders/cell.ny)/(cell.period/cell.ny)
        # Keep B0's exact field seed, independent of the added memory count.
        x = np.zeros((b.size, len(orders)))
        original = np.random.default_rng(431).normal(size=(sum(cell.lengths), len(orders)))
        x[:b.offsets[3]] = original[:b.offsets[3]]
        x /= np.linalg.norm(x)
        total, checkpoints = 0., []
        for step in range(1, steps+1):
            x = b.step(x)
            if step % 100 == 0:
                norm = np.linalg.norm(x)
                total += np.log(norm)
                x /= norm
                if step >= steps//2:
                    checkpoints.append((step, total))
            if step % 20000 == 0:
                print(json.dumps(dict(p=p, polarization=pol, step=step,
                                      seconds=time.monotonic()-start)), flush=True)
        fit = np.polyfit(np.array(checkpoints)[:, 0], np.array(checkpoints)[:, 1], 1)
        rows.append(dict(p=p, polarization=pol, steps=steps, period_m=cell.period, ny=cell.ny,
            harmonics=len(orders), dominant_order=int(orders[np.argmax(np.linalg.norm(x, axis=0))]),
            fitted_rho=float(np.exp(fit[0])), log_slope=float(fit[0]), passes=bool(np.exp(fit[0]) <= THRESHOLD),
            seed=431, normalization_interval=100, fit_window_steps=[steps//2, steps],
            fit_checkpoints=checkpoints))
    return dict(rows=rows, seconds=time.monotonic()-start,
                limitation="Matrix-free Fourier stack, projected transverse means, same scope as B0")


def power_job(job):
    p, steps = job
    return power((p,), steps)


def verify(ps):
    """Exact p=0 regression, both realizations, and independent wave elimination."""
    checks = []
    old_sweep = json.loads((ROOT/"results/sweep.jsonl").read_text().splitlines()[0])
    cell = Cell(**old_sweep["cell"])
    assert np.array_equal(block(cell, 0).matrix(), Block(cell).matrix())
    baseline = json.loads((ROOT/"results/validation.json").read_text())["cases"]
    native_rows = []
    for pol, order, stored in (("TM", 1, baseline[0]), ("TE", 1, baseline[1]), ("TE", 0, baseline[2])):
        cell = Cell(polarization=pol, order=order)
        for rounded in (False, True):
            value = oscillatory_spectrum(native(cell, 0, rounded=rounded), cell.dt)
            previous = stored["float32_replay" if rounded else "float64"]
            assert value == previous, (value, previous)
            native_rows.append(dict(polarization=pol, order=order, rounded=rounded, spectrum=value,
                                    exact_B0_match=True))
    for pol, preset, p in product(("TE", "TM"), ("real", "cfs"), (0., .01, .1, .3)):
        cell = Cell(polarization=pol, preset=preset, theta=35.)
        b = block(cell, p, material="vacuum")
        A = b.matrix()
        idx = b.active_indices("current")
        eigenvalues, eigenvectors = linalg.eig(A[np.ix_(idx, idx)], check_finite=False)
        oscillatory = ((abs(np.angle(eigenvalues)) > 2*np.pi*cell.dt*1e6)
                       & (np.linalg.norm(eigenvectors[:b.n-1], axis=0) > .01))
        chosen = np.argmax(np.where(oscillatory, abs(eigenvalues), 0))
        lam = eigenvalues[chosen]
        state = np.zeros(b.size, complex)
        state[idx] = eigenvectors[:, chosen]
        u = state[:b.n-1]
        ge, gh = [transfer(g, lam, cell.dt, "current") for g in b.profile]
        te, th = (np.ones(b.n-1 if pol == "TE" else b.n) for _ in range(2)) if p == 0 else (
            transfer(g, lam, cell.dt, "current") for g in b.tangential)
        K2 = -(lam-1)**2/(b.t*b.t*lam)-b.k*b.k*te*th
        if pol == "TE":
            L = ge[1:-1, None]*(b.B@(gh[:, None]*b.F))+np.diag(K2)
        else:
            L = ge[1:-1, None]*(b.B@((gh/K2)[:, None]*b.F))+np.eye(b.n-1)
        error = float(np.linalg.norm(L@u)/(np.linalg.norm(L)*np.linalg.norm(u)))
        assert error < 1e-11, (pol, preset, p, error)
        checks.append(dict(p=p, polarization=pol, preset=preset, wave_elimination_relative_error=error))
    native_curve = []
    for p, pol in product(ps, ("TE", "TM")):
        cell = Cell(polarization=pol)
        native_curve.append(dict(p=p, polarization=pol, spectrum=oscillatory_spectrum(native(cell, p), cell.dt)))
    return dict(passed=True, p0_native=native_rows, wave_elimination=checks, native_curve=native_curve)


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+"\n")


def merge_sweeps(paths):
    reports = [json.loads(path.read_text()) for path in paths]
    assert all(r["complete"] for r in reports)
    ps = sorted({p for r in reports for p in r["p_grid"]})
    summary = sorted([s for r in reports for s in r["summary"]], key=lambda s: s["p"])
    assert len(summary) == len(ps), "Input sweeps must have disjoint p grids"
    out = dict(p_grid=ps, summary=summary, complete=True, threshold=THRESHOLD)
    for family in ("cartesian", "fixed_k"):
        reference = reports[0][family]
        assert all(len(r[family]) == len(reference) for r in reports)
        merged = []
        for i, row in enumerate(reference):
            values = {}
            for r in reports:
                assert r[family][i]["cell"] == row["cell"]
                values.update(zip(r["p_grid"], r[family][i]["values"]))
            merged.append({**row, "values": [values[p] for p in ps]})
        out[family] = merged
    return out


def compare_b0(sweep, reflections, powers):
    index = sweep["p_grid"].index(0.)
    comparisons = {}
    old_cart = [json.loads(line) for line in (ROOT/"results/sweep.jsonl").read_text().splitlines()]
    old_fixed = json.loads((ROOT/"results/bloch_sweep.json").read_text())["rows"]
    for family, old in (("cartesian", old_cart), ("fixed_k", old_fixed)):
        rows = sweep[family]
        assert len(rows) == len(old)
        worst_error = 0.
        for row, previous in zip(rows, old):
            assert row["cell"] == previous["cell"]
            value = row["values"][index]
            expected = previous["results"]["current"]
            for name in ("rho", "frequency_GHz", "all_rho"):
                worst_error = max(worst_error, abs(value[name]-expected[name]))
                assert value[name] == expected[name], (family, row["cell"], name, value, expected)
        comparisons[family] = dict(cases=len(rows), exact=True, max_absolute_error=worst_error)
    old_reflection = json.loads((ROOT/"results/reflection.json").read_text())["rows"]
    count = 0
    for row in reflections["rows"]:
        if row["p"] != 0:
            continue
        angle = {"normal": 0., "60deg": 60., "80deg": 80., "evanescent": 0.}[row["metric"]]
        previous = next(r for r in old_reflection if r["preset"] == row["preset"] and
            r["evanescent"] == (row["metric"] == "evanescent") and r["theta"] == angle and
            round(r["frequency_GHz"], 8) == round(row["frequency_GHz"], 8))
        assert row["R"] == previous["R"]["current"]
        count += 1
    comparisons["reflection"] = dict(rows=count, exact=True, max_absolute_error=0.)
    old_power = json.loads((ROOT/"results/power.json").read_text())["results"]
    for row in powers:
        if row["p"] == 0:
            previous = next(r for r in old_power if r["method"] == "current" and
                            r["polarization"] == row["polarization"])
            assert row["fitted_rho"] == previous["fitted_rho"], (row, previous)
    comparisons["power"] = dict(polarizations=2, steps=100000, exact=True)
    return comparisons


def summarize(sweep_paths, power_paths):
    sweep = merge_sweeps(sweep_paths)
    ps = sweep["p_grid"]
    assert len(sweep["cartesian"]) == 1458 and len(sweep["fixed_k"]) == 288
    reflections = reflection_curve(ps)
    write(ROOT/"results/mpml_reflection.json", reflections)
    powers = [row for path in power_paths for row in json.loads(path.read_text())["rows"]]
    assert len(powers) == 2*len(ps)
    assert {(r["p"], r["polarization"]) for r in powers} == set(product(ps, ("TE", "TM")))
    assert all(r["steps"] == 100000 and np.isfinite(r["fitted_rho"]) for r in powers)
    verification = json.loads((ROOT/"results/mpml_verify.json").read_text())
    assert verification["passed"] and sorted({r["p"] for r in verification["native_curve"]}) == ps
    baseline = compare_b0(sweep, reflections, powers)
    curve = []
    for s, r in zip(sweep["summary"], reflections["summary"]):
        assert s["p"] == r["p"]
        p = s["p"]
        large = [row for row in powers if row["p"] == p]
        large_pass = (len(large) == 2 and all(row["steps"] == 100000 and row["passes"] for row in large))
        curve.append(dict(p=p, spectral=s, reflection=r, power=large,
                          passivity_passes=bool(s["passes"] and large_pass),
                          accepted=bool(s["passes"] and large_pass and r["passes"])))
    chosen = next((r["p"] for r in curve if r["accepted"]), None)
    closest = next((r["p"] for r in curve if r["passivity_passes"]), None)
    certified_spectrum = [r["p"] for r in curve if r["spectral"]["passes"]]
    threshold = None
    if certified_spectrum:
        hi = min(certified_spectrum)
        lo = max(r["p"] for r in curve if r["p"] < hi and not r["spectral"]["passes"])
        threshold = dict(failing_p=lo, passing_p=hi, width=hi-lo,
                         scope="Adjacent evaluated ratios; no monotonicity proof")
    files = [Path(__file__), ROOT/"model.py", ROOT/"reflection.py",
             ROOT.parents[3]/"witwin/maxwell/fdtd/boundary/cpml.py"]
    files += sweep_paths+power_paths
    files += [ROOT/"results"/f for f in ("mpml_reflection.json", "mpml_verify.json",
        "mpml_native_regression.json", "mpml_operator_regression.json", "sweep.jsonl",
        "bloch_sweep.json", "reflection.json", "power.json", "validation.json")]
    closest_spectral = min(certified_spectrum) if certified_spectrum else None
    return dict(selected_p=chosen, closest_passivity_p=closest, closest_spectral_p=closest_spectral,
        closest_rejected_p=closest if closest is not None else closest_spectral,
        closest_definition="Smallest p passing all passivity checks, else smallest p with zero finite-sweep failures",
        spectral_threshold_bracket=threshold,
        evaluated_ratios=len(ps), total_spectral_cases=len(ps)*1746,
        total_reflection_rows=len(reflections["rows"]), total_large_cell_runs=len(powers),
        p0_regression=baseline, curve=curve, source_sweeps=[str(p) for p in sweep_paths],
        source_power=[str(p) for p in power_paths],
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        interpretation="No tested p passes" if chosen is None else "Smallest accepted evaluated p",
        scope="CPU harmonic model and native-equivalent basis projection; no CUDA qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reflection", "sweep", "refine", "verify", "power", "summarize"))
    parser.add_argument("--p", type=float, nargs="+", default=P_GRID)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sweeps", type=Path, nargs="+", default=[ROOT/"results/mpml_sweep.json"])
    parser.add_argument("--powers", type=Path, nargs="+", default=[ROOT/"results/mpml_power_initial.json"])
    args = parser.parse_args()
    start = time.monotonic()
    if args.mode == "reflection":
        report = reflection_curve(args.p)
    elif args.mode == "sweep":
        report = run_sweep(args.p, args.workers, args.quick)
    elif args.mode == "refine":
        report = refine(args.sweeps[0], args.workers)
    elif args.mode == "verify":
        report = verify(args.p)
    elif args.mode == "power":
        if args.workers == 1 or len(args.p) == 1:
            report = power(args.p, args.steps)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                reports = list(pool.map(power_job, ((p, args.steps) for p in args.p)))
            report = dict(rows=[r for part in reports for r in part["rows"]],
                          limitation=reports[0]["limitation"])
    else:
        report = summarize(args.sweeps, args.powers)
    report["seconds"] = time.monotonic()-start
    write(args.output or ROOT/"results"/f"mpml_{args.mode}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k in ("summary", "passed", "seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
