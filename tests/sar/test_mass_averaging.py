"""cubical-prefix-v1 mass-averaging: golden cases, brute-force parity, convergence.

These tests drive the averaging algorithm directly through
``compute_mass_averaged_sar`` with synthetic per-cell fields (isolating the
integral-image search from the FDTD pipeline) and through the full
``Result.sar(..., averaging=...)`` path for serialization/peak API coverage.
"""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess.sar_averaging import compute_mass_averaged_sar
from witwin.maxwell.scene import prepare_scene


def _uniform_grid_inputs(
    *,
    n=11,
    dx=0.1,
    rho=1000.0,
    q=1.0,
    frequencies=(1.0e9,),
    occupancy=None,
    valid=None,
    device="cpu",
):
    """Synthetic per-cell inputs on a uniform ``n^3`` grid for the averaging kernel."""
    shape = (n, n, n)
    rho_cell = (rho if torch.is_tensor(rho) else torch.full(shape, float(rho))).to(device)
    cell_volume = torch.full(shape, dx ** 3, device=device)
    occ = torch.ones(shape, device=device) if occupancy is None else occupancy.to(device)
    q_field = q if torch.is_tensor(q) else torch.full((len(frequencies), *shape), float(q))
    q_field = q_field.to(device)
    valid_mask = (occ > 1e-6) & (rho_cell > 0) if valid is None else valid.to(device)
    coords = {
        axis: torch.arange(n, device=device, dtype=torch.float64) * dx + 0.5 * dx
        for axis in ("x", "y", "z")
    }
    cell_sizes = tuple(torch.full((n,), dx, device=device) for _ in range(3))
    freqs = torch.tensor(frequencies, dtype=torch.float64, device=device)
    return dict(
        power_total=q_field,
        rho_cell=rho_cell,
        cell_volume=cell_volume,
        occupancy=occ,
        valid=valid_mask,
        coordinates=coords,
        cell_sizes=cell_sizes,
        frequencies=freqs,
    )


def _brute_force_average(inputs, m0, min_fraction):
    """O(N*k^3) reference: grow a symmetric interior cube until mass >= m0."""
    power = inputs["power_total"].detach().cpu().numpy()  # [F,n,n,n]
    rho = inputs["rho_cell"].detach().cpu().numpy()
    vol = inputs["cell_volume"].detach().cpu().numpy()
    occ = inputs["occupancy"].detach().cpu().numpy()
    valid = inputs["valid"].detach().cpu().numpy()
    F, nx, ny, nz = power.shape
    mass_cell = rho * vol
    power_cell = power * vol[None]
    tvol_cell = occ * vol

    avg = np.full((F, nx, ny, nz), np.nan)
    mass_field = np.full((nx, ny, nz), np.nan)
    hw_field = np.full((nx, ny, nz), -1, dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not valid[i, j, k]:
                    continue
                hmax = min(i, nx - 1 - i, j, ny - 1 - j, k, nz - 1 - k)
                chosen = None
                for h in range(hmax + 1):
                    sl = (slice(i - h, i + h + 1), slice(j - h, j + h + 1), slice(k - h, k + h + 1))
                    enclosed_mass = mass_cell[sl].sum()
                    if enclosed_mass >= m0:
                        chosen = h
                        break
                if chosen is None:
                    continue
                sl = (
                    slice(i - chosen, i + chosen + 1),
                    slice(j - chosen, j + chosen + 1),
                    slice(k - chosen, k + chosen + 1),
                )
                enclosed_mass = mass_cell[sl].sum()
                cube_vol = vol[sl].sum()
                tissue_frac = tvol_cell[sl].sum() / cube_vol
                if tissue_frac < min_fraction:
                    continue
                enclosed_power = power_cell[(slice(None), *sl)].sum(axis=(1, 2, 3))
                avg[:, i, j, k] = enclosed_power / enclosed_mass
                mass_field[i, j, k] = enclosed_mass
                hw_field[i, j, k] = chosen
    return avg, mass_field, hw_field


def test_uniform_one_gram_cube_exact_size_and_mass():
    # dx=0.1, rho=1000 -> mass_cell = 1 kg. A 3x3x3 cube is exactly 27 kg.
    inputs = _uniform_grid_inputs(n=11, dx=0.1, rho=1000.0, q=2.0)
    m0 = 27.0
    averaged, peaks = compute_mass_averaged_sar(
        mw.SARAveraging(mass=(m0,)), **inputs
    )
    peak = peaks[m0]
    assert float(peak.mass_kg[0]) == pytest.approx(27.0, rel=1e-6)
    assert int(peak.cube_half_width[0]) == 1
    torch.testing.assert_close(
        peak.cube_size_m, torch.full_like(peak.cube_size_m, 0.3), rtol=1e-5, atol=1e-6
    )
    # Uniform field -> averaged SAR equals point SAR q/rho (float32 prefix sums).
    assert float(peak.sar[0]) == pytest.approx(2.0 / 1000.0, rel=1e-4)
    # Strict-interior: border centers (need h>=1) are invalid.
    valid = averaged[m0]["valid"]
    assert not bool(valid[0, 0, 0])
    assert bool(valid[5, 5, 5])
    assert int(valid.sum()) == 9 * 9 * 9


def test_prefix_matches_bruteforce_random():
    """Load-bearing: the integral-image search equals an O(N*k^3) reference."""
    torch.manual_seed(0)
    n = 9
    rho = torch.rand(n, n, n) * 500.0 + 800.0
    occ = torch.rand(n, n, n)
    occ = torch.where(occ < 0.2, torch.zeros_like(occ), occ)  # some air cells
    q = torch.rand(2, n, n, n) * 3.0
    inputs = _uniform_grid_inputs(n=n, dx=0.1, rho=rho, q=q, occupancy=occ,
                                  frequencies=(1.0e9, 2.4e9))
    for m0, min_frac in ((5.0, 0.1), (12.0, 0.0), (20.0, 0.3)):
        averaged, _ = compute_mass_averaged_sar(
            mw.SARAveraging(mass=(m0,), min_tissue_fraction=min_frac), **inputs
        )
        ref_avg, ref_mass, ref_hw = _brute_force_average(inputs, m0, min_frac)
        got = averaged[m0]["sar"].detach().cpu().numpy()
        got_mass = averaged[m0]["mass_kg"].detach().cpu().numpy()
        got_hw = averaged[m0]["cube_half_width"].detach().cpu().numpy()
        # Validity masks must agree exactly.
        np.testing.assert_array_equal(np.isnan(got[0]), np.isnan(ref_avg[0]))
        finite = ~np.isnan(ref_avg)
        np.testing.assert_allclose(got[finite], ref_avg[finite], rtol=1e-5, atol=1e-9)
        mfinite = ~np.isnan(ref_mass)
        np.testing.assert_allclose(got_mass[mfinite], ref_mass[mfinite], rtol=1e-5, atol=1e-9)
        np.testing.assert_array_equal(got_hw[mfinite], ref_hw[mfinite])


def test_peak_monotonic_in_mass():
    # Spatially varying q so a larger averaging cube smooths the peak down.
    n = 15
    idx = torch.arange(n, dtype=torch.float32)
    gx = torch.exp(-((idx - 7) ** 2) / 8.0)
    q = (gx[:, None, None] * gx[None, :, None] * gx[None, None, :])[None]  # [1,n,n,n]
    inputs = _uniform_grid_inputs(n=n, dx=0.1, rho=1000.0, q=q)
    _, peaks = compute_mass_averaged_sar(
        mw.SARAveraging(mass=(1.0, 10.0)), **inputs
    )
    peak_1 = float(peaks[1.0].sar[0])
    peak_10 = float(peaks[10.0].sar[0])
    assert peak_10 <= peak_1
    assert int(peaks[10.0].cube_half_width[0]) >= int(peaks[1.0].cube_half_width[0])


def test_two_material_halfspace_average_is_mass_weighted():
    # Left half rho=1000 q=1, right half rho=2000 q=4. A cube straddling the
    # interface averages sum(q*V)/sum(rho*V) over its cells.
    n = 9
    rho = torch.empty(n, n, n)
    q = torch.empty(1, n, n, n)
    rho[: n // 2 + 1] = 1000.0
    rho[n // 2 + 1 :] = 2000.0
    q[0, : n // 2 + 1] = 1.0
    q[0, n // 2 + 1 :] = 4.0
    dx = 0.1
    inputs = _uniform_grid_inputs(n=n, dx=dx, rho=rho, q=q)
    # Target mass forces a 3x3x3 cube (h=1) even in the rho=2000 region.
    m0 = 3.0  # small enough that h=1 in either region reaches it
    averaged, _ = compute_mass_averaged_sar(mw.SARAveraging(mass=(m0,)), **inputs)
    center = (4, 4, 4)  # on the interface plane index n//2 = 4
    sl = (slice(3, 6), slice(3, 6), slice(3, 6))
    vol = dx ** 3
    enclosed_power = float((q[0][sl] * vol).sum())
    enclosed_mass = float((rho[sl] * vol).sum())
    expected = enclosed_power / enclosed_mass
    got = float(averaged[m0]["sar"][0][center])
    assert got == pytest.approx(expected, rel=1e-5)


def test_partial_occupancy_uses_effective_mass():
    # Effective density is occupancy-weighted; enclosed mass must use it.
    n = 7
    occ = torch.full((n, n, n), 0.5)
    rho = torch.full((n, n, n), 1000.0)  # effective (already occupancy-weighted)
    inputs = _uniform_grid_inputs(n=n, dx=0.1, rho=rho, q=1.0, occupancy=occ)
    m0 = 1.0
    averaged, peaks = compute_mass_averaged_sar(mw.SARAveraging(mass=(m0,)), **inputs)
    # mass_cell = rho*V = 1000*0.001 = 1 kg. A single cell already reaches 1 kg,
    # but min_tissue_fraction default 0.1 <= 0.5 so it stays valid.
    peak = peaks[m0]
    assert int(peak.cube_half_width[0]) == 0
    assert float(peak.mass_kg[0]) == pytest.approx(1.0, rel=1e-6)


def test_min_tissue_fraction_rejects_air_makeup():
    # A lone tissue cell surrounded by air: reaching mass needs a big cube that is
    # mostly air, so min_tissue_fraction invalidates it (no air-mass makeup).
    n = 9
    occ = torch.zeros(n, n, n)
    rho = torch.zeros(n, n, n)
    occ[4, 4, 4] = 1.0
    rho[4, 4, 4] = 1000.0
    q = torch.zeros(1, n, n, n)
    q[0, 4, 4, 4] = 5.0
    valid = occ > 1e-6
    inputs = _uniform_grid_inputs(n=n, dx=0.1, rho=rho, q=q, occupancy=occ, valid=valid)
    # mass_cell of the tissue cell = 1 kg. Ask for 4 kg -> needs a cube reaching 4
    # kg, impossible (only one tissue cell) -> unreachable -> invalid regardless.
    averaged, _ = compute_mass_averaged_sar(mw.SARAveraging(mass=(4.0,)), **inputs)
    assert bool(torch.all(torch.isnan(averaged[4.0]["sar"])))
    # A reachable mass (1 kg = the single cell, h=0) stays valid (fraction 1.0).
    averaged1, _ = compute_mass_averaged_sar(mw.SARAveraging(mass=(1.0,)), **inputs)
    assert bool(averaged1[1.0]["valid"][4, 4, 4])
    # Now force h=1 by asking 1.0 kg but zero out the center's own mass fraction
    # via a low fill: a cube of mostly-air fails the fraction gate.
    occ2 = torch.full((n, n, n), 0.05)  # 5% fill everywhere < default 0.1
    rho2 = torch.full((n, n, n), 50.0)  # effective density (0.05 * 1000)
    inputs2 = _uniform_grid_inputs(n=n, dx=0.1, rho=rho2, q=1.0, occupancy=occ2)
    averaged2, _ = compute_mass_averaged_sar(mw.SARAveraging(mass=(0.5,)), **inputs2)
    assert bool(torch.all(torch.isnan(averaged2[0.5]["sar"])))


def test_unreachable_mass_marks_invalid():
    inputs = _uniform_grid_inputs(n=7, dx=0.1, rho=1000.0, q=1.0)
    # Whole 7^3 grid holds 343 kg, but strict-interior cube from any center is at
    # most (2*3+1)^3=343 only from the exact center; ask for more than any interior
    # cube can hold -> all invalid.
    huge = 1.0e6
    averaged, peaks = compute_mass_averaged_sar(mw.SARAveraging(mass=(huge,)), **inputs)
    assert bool(torch.all(torch.isnan(averaged[huge]["sar"])))
    assert bool(torch.isnan(peaks[huge].sar[0]))
    assert int(peaks[huge].cube_half_width[0]) == -1


def test_strict_interior_boundary_invalidates_clipped_centers():
    inputs = _uniform_grid_inputs(n=11, dx=0.1, rho=1000.0, q=1.0)
    m0 = 27.0  # needs h=1
    averaged, _ = compute_mass_averaged_sar(mw.SARAveraging(mass=(m0,)), **inputs)
    valid = averaged[m0]["valid"]
    # Every face cell cannot host an h=1 cube.
    assert not bool(valid[0].any())
    assert not bool(valid[-1].any())
    assert not bool(valid[:, 0].any())


def test_grid_convergence_of_peak_averaged_sar():
    """Peak 1 g averaged SAR converges as the fixed-mass cube is refined.

    A continuous Gaussian absorbed-power density sampled on three grids: the peak
    averaged SAR over a fixed physical averaging mass must approach a limit as the
    cube discretization refines (finer grids closer to the finest)."""
    rho = 1000.0
    m0 = 1.0  # 1 g
    length = 1.0  # domain edge (m)

    def peak_at(n):
        dx = length / n
        centers = (torch.arange(n, dtype=torch.float32) + 0.5) * dx
        c = length / 2
        sig = 0.15
        g = torch.exp(-((centers - c) ** 2) / (2 * sig ** 2))
        q = (g[:, None, None] * g[None, :, None] * g[None, None, :])[None]
        inputs = _uniform_grid_inputs(n=n, dx=dx, rho=rho, q=q)
        _, peaks = compute_mass_averaged_sar(mw.SARAveraging(mass=(m0,)), **inputs)
        return float(peaks[m0].sar[0])

    p = [peak_at(n) for n in (16, 24, 40)]
    # Monotone convergence: successive differences shrink and finest bracketed.
    d1 = abs(p[0] - p[2])
    d2 = abs(p[1] - p[2])
    assert d2 < d1
    assert all(math.isfinite(v) and v > 0 for v in p)


def test_averaged_sar_is_differentiable():
    n = 9
    q = torch.rand(1, n, n, n, requires_grad=True)
    inputs = _uniform_grid_inputs(n=n, dx=0.1, rho=1000.0, q=q)
    averaged, _ = compute_mass_averaged_sar(mw.SARAveraging(mass=(3.0,)), **inputs)
    sar = averaged[3.0]["sar"]
    objective = torch.nan_to_num(sar, nan=0.0).sum()
    objective.backward()
    assert q.grad is not None
    assert torch.all(torch.isfinite(q.grad))
    assert float(q.grad.abs().sum()) > 0.0


# --- Full-pipeline coverage: Result.sar averaging + peak API + serialization ---

_USE_RHO = object()


def _uniform_cube_result(*, dx=0.1, sigma=0.5, rho=1000.0, e0=2.0,
                         frequencies=(1.0e9,), device="cpu"):
    from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor

    material = mw.Material(sigma_e=sigma, mass_density=rho, name="tissue")
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 2.0),) * 3),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.none(),
        structures=(
            mw.Structure(
                name="bulk",
                geometry=mw.Box(position=(1.0, 1.0, 1.0), size=(8.0, 8.0, 8.0)),
                material=material,
            ),
        ),
        device=device,
    )
    monitor = mw.PowerLossMonitor(
        "loss", position=(1.0, 1.0, 1.0), size=(1.0, 1.0, 1.0),
        frequencies=frequencies, channels=("conduction",),
    )
    scene.add_monitor(monitor)
    prepared = prepare_scene(scene)
    compiled = compile_power_loss_monitor(prepared, monitor)
    fields = {}
    for component in ("Ex", "Ey", "Ez"):
        shape = compiled.full_component_shapes[component]
        if len(frequencies) > 1:
            shape = (len(frequencies), *shape)
        fields[component.upper()] = torch.full(shape, complex(e0), dtype=torch.complex64, device=device)
    result = mw.Result(method="fdtd", scene=scene, prepared_scene=prepared,
                       frequencies=frequencies, fields=fields)
    return result, dict(sigma=sigma, rho=rho, e0=e0)


def test_result_sar_peak_matches_point_analytic_for_uniform_field():
    result, params = _uniform_cube_result()
    m0 = params["rho"] * (0.3) ** 3  # 3x3x3 cube at dx=0.1
    sar = result.sar("loss", averaging=mw.SARAveraging(mass=(m0,)))
    analytic = 0.5 * params["sigma"] * (3.0 * params["e0"] ** 2) / params["rho"]
    peak = sar.peak(m0)
    assert float(peak.sar[0]) == pytest.approx(analytic, rel=1e-4)
    assert int(peak.cube_half_width[0]) == 1
    assert float(peak.mass_kg[0]) == pytest.approx(m0, rel=1e-4)
    assert peak.position.shape == (1, 3)
    assert peak.profile == mw.AVERAGING_PROFILE


def test_peak_requires_averaging_request():
    result, _ = _uniform_cube_result()
    sar = result.sar("loss")  # no averaging
    with pytest.raises(NotImplementedError, match="No mass-averaged peak"):
        sar.peak(1e-3)
    with pytest.raises(NotImplementedError, match="No mass-averaged SAR field"):
        sar.averaged_sar(1e-3)


def test_peak_unknown_mass_fails_closed():
    result, params = _uniform_cube_result()
    m0 = params["rho"] * (0.3) ** 3
    sar = result.sar("loss", averaging=mw.SARAveraging(mass=(m0,)))
    with pytest.raises(KeyError, match="No mass-averaged SAR for mass"):
        sar.peak(999.0)


def test_averaged_and_peaks_serialization_roundtrip(tmp_path):
    result, params = _uniform_cube_result(frequencies=(1.0e9, 2.0e9))
    m0 = params["rho"] * (0.3) ** 3
    sar = result.sar("loss", averaging=mw.SARAveraging(mass=(m0, 5 * m0)))
    path = tmp_path / "sar_avg.pt"
    sar.save(path)
    loaded = mw.SARResult.load(path)

    assert set(loaded.averaging_masses) == set(sar.averaging_masses)
    a = sar.averaged_sar(m0)
    b = loaded.averaged_sar(m0)
    torch.testing.assert_close(
        torch.nan_to_num(b, nan=0.0), torch.nan_to_num(a.cpu(), nan=0.0)
    )
    pa = sar.peak(m0)
    pb = loaded.peak(m0)
    torch.testing.assert_close(
        torch.nan_to_num(pb.sar, nan=0.0), torch.nan_to_num(pa.sar.cpu(), nan=0.0)
    )
    torch.testing.assert_close(pb.cube_size_m, pa.cube_size_m.cpu())
    assert int(pb.cube_half_width[0]) == int(pa.cube_half_width[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_averaging_stays_on_cuda_device():
    inputs = _uniform_grid_inputs(n=11, dx=0.1, rho=1000.0, q=1.0, device="cuda")
    averaged, peaks = compute_mass_averaged_sar(mw.SARAveraging(mass=(27.0,)), **inputs)
    assert averaged[27.0]["sar"].device.type == "cuda"
    assert peaks[27.0].sar.device.type == "cuda"
