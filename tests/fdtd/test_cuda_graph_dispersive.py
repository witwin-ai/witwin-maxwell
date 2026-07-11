from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

_FREQ = 2.0e9


def _electric_drude():
    return mw.Material.drude(eps_inf=1.0, plasma_frequency=2.0e9, gamma=1.0e8)


def _electric_lorentz():
    return mw.Material.lorentz(
        eps_inf=1.5, delta_eps=1.2, resonance_frequency=1.0e9, gamma=1.5e8
    )


def _electric_debye():
    return mw.Material.debye(eps_inf=2.0, delta_eps=3.0, tau=2.0e-10)


def _magnetic_lorentz():
    return mw.Material(
        mu_lorentz_poles=(
            mw.LorentzPole(delta_eps=4.0, resonance_frequency=2.0e9, gamma=1.0e8),
        ),
    )


def _dispersive_dipole_scene(material_factory):
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
        _dispersive_dipole_scene(material_factory),
        frequencies=(_FREQ,),
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=300),
        cuda_graph=cuda_graph,
    ).run()
    view = result.at(frequency=_FREQ)
    fields = {a: np.asarray(getattr(view.E, a).detach().cpu()) for a in ("x", "y", "z")}
    return fields, result.stats()


# Each linear-dispersion class (electric Debye/Drude/Lorentz ADE and the
# magnetic mu-pole ADE) is now captured into the field-update / tail CUDA graphs.
# The graph replay must reproduce the eager host path bit-for-bit: the ADE advance
# and polarization-current subtraction carry no per-step host input, and the pole
# state stays at its zero fixed point through warmup/capture.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
@pytest.mark.parametrize(
    "material_factory",
    [_electric_drude, _electric_lorentz, _electric_debye, _magnetic_lorentz],
    ids=["electric_drude", "electric_lorentz", "electric_debye", "magnetic_lorentz"],
)
def test_cuda_graph_dispersive_matches_non_graph_bitexact(material_factory):
    ref, ref_stats = _run(material_factory, cuda_graph=False)
    got, got_stats = _run(material_factory, cuda_graph=True)
    assert ref_stats["cuda_graph_active"] is False
    # Real-field dispersive scene without TFSF / magnetic sources -> captured.
    assert got_stats["cuda_graph_active"] is True
    # The electric ADE polarization-current subtraction lives in the post-source
    # tail, so the GPU-driven DFT tail must capture too.
    assert got_stats["tail_graph_active"] is True
    for component, field in ref.items():
        assert np.array_equal(field, got[component]), component
