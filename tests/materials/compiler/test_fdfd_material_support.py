import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.fdfd.materials import compile_fdfd_materials
from witwin.core import Box
from witwin.maxwell.scene import prepare_scene


def _build_cuda_scene(*, material):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
            material=material,
        )
    )
    return scene


def test_fdfd_rejects_polarized_tensor_at_preparation():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.15, 0.15),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized_tensor"),
    )
    simulation = mw.Simulation.fdfd(scene, frequency=1.0e9)
    with pytest.raises(ValueError, match="^polarized_tensor subpixel averaging is FDTD-only; use averaging='polarized' for FDFD$"):
        simulation.prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDFD backend")
def test_fdfd_operator_uses_axis_aligned_anisotropic_permittivity_components():
    cp = pytest.importorskip("cupy")
    frequency = 2.0e9
    scene = _build_cuda_scene(
        material=mw.Material(
            eps_r=1.0,
            epsilon_tensor=mw.DiagonalTensor3(2.0, 4.0, 8.0),
        )
    )

    solver = mw.Simulation.fdfd(scene, frequency=frequency).prepare().solver
    compiled_scene = prepare_scene(scene)
    solver._ensure_material_components()
    diagonal = cp.asnumpy(solver._build_matrix_gpu_yee_3d().diagonal())

    n_ex = compiled_scene.N_ex
    n_ey = compiled_scene.N_ey
    ex_mean = diagonal[:n_ex].real.mean()
    ey_mean = diagonal[n_ex:n_ex + n_ey].real.mean()
    ez_mean = diagonal[n_ex + n_ey:].real.mean()

    assert ex_mean < ey_mean < ez_mean


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDFD backend")
def test_fdfd_operator_uses_component_specific_sigma_tensor_conductivity():
    cp = pytest.importorskip("cupy")
    frequency = 2.0e9
    scene = _build_cuda_scene(
        material=mw.Material(
            eps_r=2.0,
            sigma_e_tensor=mw.DiagonalTensor3(0.0, 0.0, 5.0),
        )
    )

    solver = mw.Simulation.fdfd(scene, frequency=frequency).prepare().solver
    compiled_scene = prepare_scene(scene)
    solver._ensure_material_components()
    diagonal = cp.asnumpy(solver._build_matrix_gpu_yee_3d().diagonal())

    n_ex = compiled_scene.N_ex
    n_ey = compiled_scene.N_ey
    ex_imag = np.abs(diagonal[:n_ex].imag).mean()
    ey_imag = np.abs(diagonal[n_ex:n_ex + n_ey].imag).mean()
    ez_imag = np.abs(diagonal[n_ex + n_ey:].imag).mean()

    assert ex_imag < 1.0e-6
    assert ey_imag < 1.0e-6
    assert ez_imag > 1.0e3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDFD backend")
def test_fdfd_supports_static_magnetic_response():
    scene = _build_cuda_scene(material=mw.Material(eps_r=2.0, mu_r=1.5))

    result = mw.Simulation.fdfd(scene, frequency=1.0e9,
        solver=mw.GMRES(solver_type='direct',precision='double')).run()
    assert result.solver_stats['converged']
    assert bool(torch.isfinite(result.tensor('Ez')).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDFD backend")
def test_fdfd_rejects_kerr_media():
    scene = _build_cuda_scene(material=mw.Material(eps_r=2.0, kerr_chi3=1.0e-10))

    with pytest.raises(NotImplementedError, match="Kerr nonlinear media"):
        mw.Simulation.fdfd(scene, frequency=1.0e9).prepare()


@pytest.mark.parametrize('boundary', [mw.BoundarySpec.pml(num_layers=2),
    mw.BoundarySpec.periodic(), mw.BoundarySpec.bloch(wavevector=(0.3, 0.0, 0.0))])
def test_fdfd_yee_samples_preserve_staggering_and_conductivity_sign(boundary):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.1), boundary=boundary, device='cpu',
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging='polarized'))
    scene.add_structure(mw.Structure(mw.Box(position=(0.17, 0., 0.), size=(0.4, 2., 2.)),
        mw.Material(eps_r=4., mu_r=2., sigma_e=0.02, sigma_m=0.03)))
    prepared = prepare_scene(scene)
    model = prepared.compile_materials()
    epsilon, permeability = compile_fdfd_materials(prepared, 1e9, edges=True)
    for axis in 'xyz':
        for values, key, sigma_key, prefix, constant in (
            (epsilon, 'eps', 'sigma_e', 'E', EPSILON_0),
            (permeability, 'mu', 'sigma_m', 'H', MU_0)):
            component = prefix + axis
            expected = model['edge_components'][key][component].to(torch.complex64)
            expected = expected + 1j * model['edge_components'][sigma_key][component] / (2*np.pi*1e9*constant)
            staggered = {axis} if prefix == 'E' else set('xyz') - {axis}
            assert values[axis].shape == tuple(getattr(prepared, f'N{a}') - (a in staggered) for a in 'xyz')
            torch.testing.assert_close(values[axis], expected, rtol=2e-7, atol=1e-7)
