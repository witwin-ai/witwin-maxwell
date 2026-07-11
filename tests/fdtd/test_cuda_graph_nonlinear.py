from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

_FREQ = 2.0e9


def _kerr_chi3():
    # Pure third-order (chi3) Kerr: keeps the static decay tensors and rewrites
    # only the curl coefficient via the curl-only Kerr kernel.
    return mw.Material(eps_r=4.0, kerr_chi3=1.0e-18)


def _chi2():
    # Second-order (chi2) nonlinearity: routes through the general nonlinear
    # coefficient kernel that rewrites both decay and curl every step.
    return mw.Material(eps_r=4.0, nonlinearity=mw.NonlinearSusceptibility(chi2=1.0e-12))


def _tpa():
    # Two-photon absorption: the field-dependent conductivity channel of the
    # general nonlinear kernel (sigma_NL = tpa_sigma * |E|^2).
    return mw.Material(eps_r=4.0, nonlinearity=mw.TwoPhotonAbsorption(beta=1.0e-11, n0=1.5))


def _nonlinear_dipole_scene(material_factory):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))),
        grid=mw.GridSpec.uniform(0.008),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.04)),
            material=material_factory(),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.02, 0.0, -0.02),
            polarization="Ez",
            source_time=mw.GaussianPulse(frequency=_FREQ, fwidth=0.7e9),
        )
    )
    return scene


def _run(material_factory, *, cuda_graph):
    result = mw.Simulation.fdtd(
        _nonlinear_dipole_scene(material_factory),
        frequencies=(_FREQ,),
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=300),
        cuda_graph=cuda_graph,
    ).run()
    view = result.at(frequency=_FREQ)
    fields = {a: np.asarray(getattr(view.E, a).detach().cpu()) for a in ("x", "y", "z")}
    return fields, result.stats()


# Each instantaneous-nonlinear class (curl-only Kerr chi3, general chi2, and
# general two-photon-absorption loss) is now captured into the field-update CUDA
# graph. The graph replay must reproduce the eager host path bit-for-bit: the
# nonlinear coefficient kernels recompute the dynamic curl/decay tensors in place
# from the live field with no per-step host input, and every nonlinear
# polarization vanishes at E = 0 so the zero field stays a fixed point through
# warmup/capture.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
@pytest.mark.parametrize(
    "material_factory",
    [_kerr_chi3, _chi2, _tpa],
    ids=["kerr_chi3", "chi2", "tpa"],
)
def test_cuda_graph_nonlinear_matches_non_graph_bitexact(material_factory):
    ref, ref_stats = _run(material_factory, cuda_graph=False)
    got, got_stats = _run(material_factory, cuda_graph=True)
    assert ref_stats["cuda_graph_active"] is False
    # Real-field nonlinear scene without TFSF / magnetic sources -> captured.
    assert got_stats["cuda_graph_active"] is True
    # The nonlinear update lives entirely in the field-update block; the GPU-driven
    # DFT tail captures exactly as it does for the base real-field path.
    assert got_stats["tail_graph_active"] is True
    for component, field in ref.items():
        assert np.array_equal(field, got[component]), component
