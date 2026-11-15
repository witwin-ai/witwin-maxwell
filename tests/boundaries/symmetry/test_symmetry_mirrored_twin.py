"""A folded scene equals its explicit mirrored twin.

``Scene.symmetry`` stands for the explicit mirrored domain: the soft plane-wave
aperture, the launch reference of beams and every domain-spanning flux plane are
those of that domain. The CUDA tests run each folded scene against the same scene
unfolded without symmetry and compare the kept part of the z-normal plane fields
and the mirrored-domain flux.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.quadrature import finite_plane_image_planes, symmetry_image_planes

FREQ = 1.2e9
HALF = 0.32
DX = 0.02
FOLDS = {
    "pec_x": (("PEC", "low"), None, None),
    "pmc_y": (None, ("PMC", "low"), None),
    "pec_x_high": (("PEC", "high"), None, None),
    "pmc_y_high": (None, ("PMC", "high"), None),
    "pec_x_pmc_y": (("PEC", "low"), ("PMC", "low"), None),
}


def _bounds(symmetry):
    bounds = []
    for entry in symmetry:
        if entry is None:
            bounds.append((-HALF, HALF))
        else:
            bounds.append((0.0, HALF) if entry[1] == "low" else (-HALF, 0.0))
    return tuple(bounds)


# --------------------------------------------------------------------------- #
# CPU: image planes of flux planes
# --------------------------------------------------------------------------- #
def test_image_planes_are_the_folded_axes_tangential_to_the_plane():
    scene = mw.Scene(domain=mw.Domain(bounds=((0.0, 0.3), (-0.4, 0.0), (-0.5, 0.5))), grid=mw.GridSpec.uniform(0.1),
                     symmetry=(("PEC", "low"), ("PMC", "high"), None))
    assert symmetry_image_planes(scene, "z") == {"x": (0.0, "low", 0.3), "y": (0.0, "high", -0.4)}
    assert symmetry_image_planes(scene, "x") == {"y": (0.0, "high", -0.4)}
    assert symmetry_image_planes(mw.Scene(), "z") == {}


def test_finite_plane_image_planes_split_symmetric_kept_and_straddling_rectangles():
    images = {"x": (0.0, "low", 0.3), "y": (0.0, "high", -0.4)}
    symmetric, straddled = finite_plane_image_planes(images, {"x": (-0.1, 0.1), "y": (-0.2, -0.1)})
    assert symmetric == {"x": images["x"]} and straddled == []
    # Clipped to the mirrored domain [-0.3, 0.3] first: an oversized rectangle is symmetric.
    symmetric, straddled = finite_plane_image_planes(images, {"x": (-5.0, 0.4), "y": (-0.2, 0.3)})
    assert symmetric == {"x": images["x"]} and straddled == ["y"]
    symmetric, straddled = finite_plane_image_planes(images, {"x": (0.0, 0.2), "y": (-0.4, 0.0)})
    assert symmetric == {} and straddled == []


# --------------------------------------------------------------------------- #
# CUDA: folded run == explicit mirrored twin
# --------------------------------------------------------------------------- #
def _scene(symmetry, folded, source):
    pulse = mw.GaussianPulse(frequency=FREQ, fwidth=0.4e9)
    structures = ()
    if source == "plane_wave":
        structures = (mw.Structure(geometry=mw.Box(position=(0, 0, 0), size=(4 * HALF, 4 * HALF, 0.08)),
                                   material=mw.Material(eps_r=3.0)),)
        sources = (mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=pulse),)
    else:
        sources = (mw.GaussianBeam(direction=(0, 0, 1), polarization=(1, 0, 0), beam_waist=0.12,
                                   focus=(0, 0, 0), source_time=pulse),)
    monitors = (
        mw.FluxMonitor("flux", axis="z", position=0.16, frequencies=(FREQ,)),
        mw.PlaneMonitor("plane", axis="z", position=0.16, fields=("Ex", "Hy"), frequencies=(FREQ,)),
        mw.monitors.FluxTimeMonitor("flux_t", axis="z", position=0.16),
        mw.monitors.FinitePlaneMonitor(
            "rect", position=(0.0, 0.0, 0.16), size=(0.2, 0.2, 0.0), fields=("Ex", "Ey", "Hx", "Hy"),
            frequencies=(FREQ,), compute_flux=True),
    )
    return mw.Scene(domain=mw.Domain(bounds=_bounds(symmetry) if folded else ((-HALF, HALF),) * 3),
                    grid=mw.GridSpec.uniform(DX), boundary=mw.BoundarySpec.pml(num_layers=8),
                    symmetry=symmetry if folded else None, structures=structures, sources=sources,
                    monitors=monitors)


def _run(scene):
    return mw.Simulation.fdtd(scene, frequencies=[FREQ], config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=500),
        spectral_sampler=mw.fdtd.SpectralSampler(normalize_source=True))).run()


def _plane(result, name):
    component = result.raw_monitor("plane")["components"][name]
    coords = tuple(np.asarray(torch.as_tensor(axis).cpu(), dtype=np.float64) for axis in component["coords"])
    return coords, torch.as_tensor(component["data"]).detach().cpu().squeeze()


def _flux(result, name):
    return torch.as_tensor(result.monitor(name).values).detach().cpu().reshape(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("source", ["plane_wave", "gaussian_beam"])
@pytest.mark.parametrize("fold", list(FOLDS))
def test_folded_run_equals_its_mirrored_twin(fold, source):
    symmetry = FOLDS[fold]
    folded = _run(_scene(symmetry, True, source))
    twin = _run(_scene(symmetry, False, source))
    for name in ("Ex", "Hy"):
        (folded_u, folded_v), folded_values = _plane(folded, name)
        (twin_u, twin_v), twin_values = _plane(twin, name)
        rows = [int(np.argmin(np.abs(twin_u - value))) for value in folded_u]
        columns = [int(np.argmin(np.abs(twin_v - value))) for value in folded_v]
        np.testing.assert_allclose(twin_u[rows], folded_u, atol=1e-9)
        np.testing.assert_allclose(twin_v[columns], folded_v, atol=1e-9)
        kept = twin_values[np.ix_(rows, columns)]
        bound = 1.0e-4 * float(kept.abs().max())
        assert float((folded_values - kept).abs().max()) <= bound, name
    for name in ("flux", "rect", "flux_t"):
        folded_flux, twin_flux = _flux(folded, name), _flux(twin, name)
        assert folded_flux.shape == twin_flux.shape, name
        bound = 1.0e-4 * float(twin_flux.abs().max())
        assert float((folded_flux - twin_flux).abs().max()) <= bound, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_unfolded_scene_flux_payload_records_no_image_planes():
    result = _run(_scene((None, None, None), False, "plane_wave"))
    assert "symmetry_image_planes" not in result.raw_monitor("flux")
    folded = _run(_scene(FOLDS["pec_x_pmc_y"], True, "plane_wave"))
    assert set(folded.raw_monitor("flux")["symmetry_image_planes"]) == {"x", "y"}
