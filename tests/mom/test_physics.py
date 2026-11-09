"""Physical contract checks independent of the MoM discretization."""

import cmath
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.physics import compile_medium, incident_fields, source_phasor, validate_scene


def test_conduction_loss_is_counted_once_and_branch_is_passive():
    frequency, sigma = 2.4e9, 0.07
    medium = compile_medium(mw.Material(eps_r=3, sigma_e=sigma), frequency)
    assert medium.epsilon_r.imag == pytest.approx(sigma / (2 * math.pi * frequency * EPSILON_0))
    assert medium.k.real > 0 and medium.k.imag > 0
    assert medium.eta == pytest.approx(medium.omega * medium.mu / medium.k)
    assert medium.k**2 == pytest.approx(medium.omega**2 * medium.epsilon * medium.mu)


def test_dispersion_and_conduction_compose_without_frequency_sampler_ambiguity():
    pole = mw.DebyePole(delta_eps=1.5, tau=1e-10)
    material = mw.Material(eps_r=2, sigma_e=0.1, debye_poles=(pole,))
    samples = [compile_medium(material, 1e9 + index * 1e7) for index in range(20)]
    for medium in samples:
        assert medium.epsilon_r == pytest.approx(material.relative_permittivity(medium.frequency))
        assert medium.k.imag >= 0 and medium.eta.real > 0
    assert max(abs(b.k - a.k) / abs(a.k) for a, b in zip(samples, samples[1:])) < 0.02


@pytest.mark.parametrize("material", [
    mw.Material(kerr_chi3=1e-20), mw.Material(pec=True),
])
def test_unsupported_material_is_rejected(material):
    with pytest.raises(ValueError):
        compile_medium(material, 1e9)


@pytest.mark.parametrize("frequency", [0, -1, float("nan"), float("inf")])
def test_invalid_frequency_is_rejected(frequency):
    with pytest.raises(ValueError):
        compile_medium(mw.Material(), frequency)


def test_lossy_background_is_rejected():
    with pytest.raises(ValueError, match="lossless"):
        compile_medium(mw.Material(sigma_e=0.01), 1e9, background=True)


def test_shared_cw_time_phase_has_correct_phasor_sign():
    source = mw.PlaneWave(source_time=mw.CW(frequency=1e9, amplitude=2, phase=math.pi / 2))
    phasor = source_phasor(source, 1e9)
    assert phasor == pytest.approx(-2j)
    for t in (0, 1e-10, 4e-10):
        assert (phasor * cmath.exp(-2j * math.pi * 1e9 * t)).real == pytest.approx(source.source_time.evaluate(t))
    with pytest.raises(ValueError, match="frequencies"):
        source_phasor(source, 2e9)


def test_incident_fields_reject_cpu_fallback():
    with pytest.raises(ValueError, match="CUDA"):
        incident_fields(mw.PlaneWave(), torch.zeros((2, 3)), compile_medium(mw.Material(), 1e9))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA field evaluation required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_plane_wave_complex_amplitude_translation_and_maxwell_relation(dtype):
    medium = compile_medium(mw.Material(), 1e9)
    source = mw.PlaneWave()
    points = torch.tensor([[0.01, 0.02, 0.03], [0.04, -0.02, 0.11]], device="cuda", dtype=dtype)
    e, h = incident_fields(source, points, medium, amplitude=1j)
    assert e.is_cuda and h.is_cuda
    torch.testing.assert_close(h[:, 1], e[:, 0] / medium.eta)
    torch.testing.assert_close(e[:, 1:], torch.zeros_like(e[:, 1:]))
    shift = (0, 0, 0.07)
    translated, _ = incident_fields(source, points, medium, amplitude=1j, phase_origin=shift)
    torch.testing.assert_close(translated, e * cmath.exp(-1j * medium.k * shift[2]))
    zero_e, zero_h = incident_fields(source, points, medium, amplitude=0)
    assert torch.count_nonzero(zero_e) == 0 and torch.count_nonzero(zero_h) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Scene defaults to CUDA")
@pytest.mark.host_validation
def test_scene_scope_allows_pml_but_rejects_periodicity_and_internal_source():
    structure = mw.Structure(mw.Sphere(radius=0.1), mw.Material(eps_r=2))
    scene = mw.Scene(structures=[structure], sources=[mw.PlaneWave()], boundary=mw.BoundarySpec.pml())
    exterior, interior = validate_scene(scene, 1e9)
    assert exterior.epsilon_r == 1 and interior.epsilon_r == 2
    scene.boundary = mw.BoundarySpec.periodic()
    with pytest.raises(ValueError, match="unbounded"):
        validate_scene(scene, 1e9)
    scene.boundary = mw.BoundarySpec.none()
    scene.sources = [mw.PointDipole(position=(0, 0, 0))]
    with pytest.raises(ValueError, match="PlaneWave"):
        validate_scene(scene, 1e9)
