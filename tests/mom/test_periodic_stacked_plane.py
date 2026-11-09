"""3D source-plane remainder under explicit spectral and height limits."""
import pytest
import torch
from test_periodic_dyadic_plane_limit import problem
from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference, _cutoff_check

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_stacked_plane_rejects_other_layers_and_nonprimary_singularities():
    kernel = problem(3)
    sample = kernel.lattice.vectors
    settings = dict(height=.1, shells=1, max_shells=2, max_halvings=3, absolute_tolerance=.02)
    for height in (.45, .9, 1.):
        with pytest.raises(ValueError, match='half the lattice-plane spacing'):
            kernel.estimate_stacked_plane_regular(sample.new_zeros(1, 3), **{**settings, 'height': height})
    with pytest.raises(ValueError, match='source-plane coordinates'):
        kernel.estimate_stacked_plane_regular(sample[2:3], **settings)
    with pytest.raises(ValueError, match='Nonprimary lattice singularities'):
        kernel.estimate_stacked_plane_regular(100*sample[:1], **settings)


def test_actual_stacked_plane_scalar_limit_matches_independent_images(tmp_path):
    kernel = problem(3)
    points = kernel.lattice.vectors.new_tensor([[0., 0., 0.], [.1, .07, 0.]])
    value, receipt = kernel.estimate_stacked_plane_regular(points, height=.12, shells=8,
        max_shells=512, max_halvings=3, absolute_tolerance=.02)
    torch.save(dict(value=value, receipt=receipt, points=points), tmp_path/'stacked-plane.pt')
    expected, reference = LossyDyadicImageReference(kernel.lattice, kernel.medium, 2).estimate_regular(
        points, relative_tolerance=0., absolute_tolerance=1e-6)
    check = _cutoff_check(expected, value, 0., .02)
    assert check['passed'], check
    assert receipt['periodic_dimension'] == 3 and receipt['converged']
    assert reference['image_cutoff_converged']
    assert not receipt['physical_qualification'] and not receipt['certified_error_bound']
