"""Bravais geometry and direct-image controls; no periodic solver qualification."""
import math
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec,LossyImageReference
from witwin.maxwell.mom.kernels import FREE_SPACE

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dimension',[1,2,3])
def test_bravais_reciprocal_bloch_matching_and_owned_state(dimension):
    vectors=torch.tensor([[1.,.2,.1],[.1,1.2,.3],[.2,.1,.9]],device='cuda',dtype=torch.float64)[:dimension]
    beta=torch.arange(1,dimension+1,device='cuda',dtype=vectors.dtype)@vectors*.13
    lattice=LatticeSpec(vectors,beta)
    torch.testing.assert_close(lattice.vectors@lattice.reciprocal.T,
        2*math.pi*torch.eye(dimension,device='cuda',dtype=vectors.dtype),rtol=1e-13,atol=1e-13)
    assert lattice.matches_incident(beta+lattice.reciprocal.sum(0))
    assert not lattice.matches_incident(beta+.1*vectors[0])
    shift=torch.arange(1,dimension+1,device='cuda')[None]
    torch.testing.assert_close(lattice.phase(shift),torch.exp(1j*((shift.to(vectors.dtype)@vectors)@beta)))
    vectors.mul_(2)
    lattice.assert_current()
    lattice.reciprocal.add_(1)
    with pytest.raises(RuntimeError,match='mutated'):
        lattice.receipt()


@pytest.mark.parametrize('dimension',[1,2,3])
def test_lossy_images_bloch_shift_gradient_and_singular_split(dimension):
    vectors=torch.tensor([[1.,.2,.1],[.1,1.2,.3],[.2,.1,.9]],device='cuda',dtype=torch.float64)[:dimension]
    lattice=LatticeSpec(vectors,.17*vectors.sum(0))
    point=torch.tensor([[.13,.19,.23]],device='cuda',dtype=torch.float64,requires_grad=True)
    k=2.+3j
    errors=[]
    for shells in (2,6):
        kernel=LossyImageReference(lattice,shells)
        value,gradient=kernel.evaluate(point,k)
        shifted,shifted_gradient=kernel.evaluate(point+vectors[0],k)
        phase=torch.exp(1j*(vectors[0]@lattice.bloch))
        errors.append(float((shifted-phase*value).detach().abs().max()))
    assert errors[1]<errors[0]*1e-3
    assert errors[1]<1e-7
    assert (shifted_gradient-phase*gradient).abs().max()<1e-6
    derivative,=torch.autograd.grad(value.real.sum(),point,retain_graph=True)
    torch.testing.assert_close(derivative,gradient.real,rtol=1e-12,atol=1e-12)
    remainder=kernel.evaluate(point,k,exclude_zero=True)
    free=FREE_SPACE.evaluate(point,k)
    for full,part,singular in zip((value,gradient),remainder,free):
        torch.testing.assert_close(full,part+singular,rtol=1e-12,atol=1e-12)
    assert not kernel.receipt()['truncation_certified']
    with pytest.raises(ValueError,match='positive absorption'):
        kernel.evaluate(point,2.)


def test_lattice_invalid_geometry_and_image_singular_guard():
    vectors=torch.eye(3,device='cuda',dtype=torch.float64)
    zero=vectors.new_zeros(3)
    for invalid in (vectors*0,vectors[[0,0]],vectors.cpu()):
        with pytest.raises(ValueError):
            LatticeSpec(invalid,zero)
    with pytest.raises(ValueError,match='periodic lattice span'):
        LatticeSpec(vectors[:1],vectors[1])
    lattice=LatticeSpec(vectors,zero)
    reference=LossyImageReference(lattice,2,image_batch=7)
    with pytest.raises(ValueError,match='Coincident'):
        reference.evaluate(zero[None],2+3j)
    for value in reference.evaluate(zero[None],2+3j,exclude_zero=True):
        assert torch.isfinite(value).all()
    point=vectors.new_tensor([[.1,.2,.3]])
    other=LossyImageReference(lattice,2,image_batch=11)
    for first,second in zip(reference.evaluate(point,2+3j),other.evaluate(point,2+3j)):
        torch.testing.assert_close(first,second,rtol=1e-13,atol=1e-13)


def test_ewald_3d_matches_independent_lossy_images_and_native_gradient():
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    vectors=torch.tensor([[1.,.2,.1],[.1,1.2,.3],[.2,.1,.9]],device='cuda',dtype=torch.float64)
    lattice=LatticeSpec(vectors,.17*vectors.sum(0))
    points=vectors.new_tensor([[.13,.19,.23],[-.2,.07,.09]],requires_grad=True)
    k=2+3j
    reference=LossyImageReference(lattice,10).evaluate(points,k)
    for eta in (1.5,2.5):
        kernel=EwaldHelmholtzKernel(lattice,eta,real_shells=4,reciprocal_shells=4,quadrature_order=96)
        values=kernel.evaluate(points,k)
        for value,expected in zip(values,reference):
            torch.testing.assert_close(value,expected,rtol=1e-9,atol=1e-10)
        derivative,=torch.autograd.grad(values[0].real.sum(),points,retain_graph=True)
        torch.testing.assert_close(derivative,values[1].real,rtol=1e-11,atol=1e-11)
        shifted=kernel.evaluate(points+vectors[1],k)
        phase=torch.exp(1j*(vectors[1]@lattice.bloch))
        for value,expected in zip(shifted,values):
            torch.testing.assert_close(value,phase*expected,rtol=1e-12,atol=1e-12)


def test_ewald_3d_resolvent_poles_and_independent_cutoff_receipts():
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel,PeriodicConvergenceError
    vectors=torch.eye(3,device='cuda',dtype=torch.float64)
    lattice=LatticeSpec(vectors,vectors.new_zeros(3))
    kernel=EwaldHelmholtzKernel(lattice,2.,real_shells=3,reciprocal_shells=3,quadrature_order=64)
    points=vectors.new_tensor([[.17,.22,.31]])
    for pole in (0.,2*math.pi):
        with pytest.raises(ValueError,match='spectral pole'):
            kernel.evaluate(points,pole)
    with pytest.raises(ValueError,match='complete spectral pole search'):
        kernel.evaluate(points,100.)
    result,receipt=kernel.estimate(points,2.)
    assert receipt['converged'] and len(receipt['checks'])==3
    assert not receipt['physical_qualification'] and not receipt['certified_error_bound']
    poor=EwaldHelmholtzKernel(lattice,.6,real_shells=1,reciprocal_shells=3,quadrature_order=64)
    with pytest.raises(PeriodicConvergenceError) as failure:
        poor.estimate(points,2.)
    assert failure.value.receipt['failure_reason']=='independent_cutoff_estimate_failed'


def test_ewald_near_singularity_analytic_heat_moments_match_images():
    from dataclasses import replace
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    vectors=torch.eye(3,device='cuda',dtype=torch.float64)
    lattice=LatticeSpec(vectors,vectors.new_tensor([.1,.2,.3]))
    points=vectors.new_tensor([[.001,.002,.001],[.00001,.00002,.00001]])
    kernel=EwaldHelmholtzKernel(lattice,2.,quadrature_order=64)
    actual=kernel.evaluate(points,2+3j)
    refined=replace(kernel,quadrature_order=256).evaluate(points,2+3j)
    reference=LossyImageReference(lattice,12).evaluate(points,2+3j)
    for a,b,c in zip(actual,refined,reference):
        torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-12)
        torch.testing.assert_close(a,c,rtol=2e-14,atol=1e-10)


def test_ewald_reciprocity_helmholtz_equation_and_finite_absorption_limit():
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    vectors=torch.eye(3,device='cuda',dtype=torch.float64)
    beta=vectors.new_tensor([.1,.2,.3])
    kernel=EwaldHelmholtzKernel(LatticeSpec(vectors,beta),2.)
    reciprocal=EwaldHelmholtzKernel(LatticeSpec(vectors,-beta),2.)
    points=vectors.new_tensor([[.17,.22,.31]],requires_grad=True)
    k=2.
    value,gradient=kernel.evaluate(points,k)
    opposite=reciprocal.evaluate(-points,k)
    torch.testing.assert_close(value,opposite[0],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(gradient,-opposite[1],rtol=1e-12,atol=1e-12)
    for part in ('real','imag'):
        laplacian=0
        for axis in range(3):
            derivative,=torch.autograd.grad(getattr(gradient[:,axis],part).sum(),points,retain_graph=True)
            laplacian=laplacian+derivative[:,axis]
        torch.testing.assert_close(laplacian,-k*k*getattr(value,part),rtol=1e-9,atol=1e-10)
    errors=[]
    for loss in (.1,.03,.01,.003):
        lossy,_=kernel.evaluate(points,k+1j*loss)
        errors.append(float((lossy-value).detach().abs().max()))
    assert all(b<.4*a for a,b in zip(errors,errors[1:]))
    assert errors[-1]<.002


@pytest.mark.parametrize('dimension',[1,2])
@pytest.mark.parametrize('k',[2+3j,2+.5j,-2+.5j])
def test_open_axis_ewald_contour_matches_lossy_image_reference(dimension,k):
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    vectors=torch.tensor([[1.,.2,.1],[.1,1.2,.3]],device='cuda',dtype=torch.float64)[:dimension]
    lattice=LatticeSpec(vectors,.17*vectors.sum(0))
    points=vectors.new_tensor([[.13,.19,.23],[-.2,.07,.09]],requires_grad=True)
    reference=LossyImageReference(lattice,60).evaluate(points,k)
    kernel=EwaldHelmholtzKernel(lattice,2.,quadrature_order=96,real_shells=4,reciprocal_shells=4)
    actual=kernel.evaluate(points,k)
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    derivative,=torch.autograd.grad(actual[0].real.sum(),points)
    torch.testing.assert_close(derivative,actual[1].real,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('dimension',[1,2])
def test_open_axis_lossless_ewald_matches_independent_radiation_series(dimension):
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    vectors=torch.tensor([[1.,.2,.1],[.1,1.2,.3]],device='cuda',dtype=torch.float64)[:dimension]
    lattice=LatticeSpec(vectors,.17*vectors.sum(0))
    normal=torch.linalg.cross(vectors[0],vectors.new_tensor([.1,1.,.3]) if dimension==1 else vectors[1])
    normal=normal/normal.norm()
    points=torch.stack((.23*vectors[0]+.7*normal,.13*vectors[0]-.8*normal))
    kernel=EwaldHelmholtzKernel(lattice,2.,quadrature_order=96,real_shells=4,reciprocal_shells=4)
    k=2.
    actual=kernel.evaluate(points,k)
    axis=torch.arange(-20,21,device='cuda')
    indices=torch.stack(torch.meshgrid(*([axis]*dimension),indexing='ij'),-1).reshape(-1,dimension)
    q=indices.to(points.dtype)@lattice.reciprocal+lattice.bloch
    parallel=(points@lattice.reciprocal.T/(2*math.pi))@lattice.vectors
    perpendicular=points-parallel
    rho=perpendicular.norm(dim=-1)
    phase=torch.exp(1j*(points@q.T))
    gamma2=k*k-q.square().sum(-1)
    if dimension==2:
        gamma=gamma2.to(torch.complex128).sqrt()
        term=1j*torch.exp(1j*rho[:,None]*gamma)/(2*lattice.cell_measure*gamma)
        radial=1j*gamma*term
    else:
        propagating=gamma2>0
        # Real Bessel arguments provide an independent, non-heat-integral oracle.
        gamma=gamma2.abs().sqrt()
        x=rho[:,None]*gamma
        h0=torch.special.bessel_j0(x)+1j*torch.special.bessel_y0(x)
        h1=torch.special.bessel_j1(x)+1j*torch.special.bessel_y1(x)
        term=torch.where(propagating,1j*h0/(4*lattice.cell_measure),
            torch.special.modified_bessel_k0(x)/(2*math.pi*lattice.cell_measure))
        radial=torch.where(propagating,-1j*gamma*h1/(4*lattice.cell_measure),
            -gamma*torch.special.modified_bessel_k1(x)/(2*math.pi*lattice.cell_measure))
    expected=(phase*term).sum(-1)
    gradient=(phase[...,None]*(term[...,None]*(1j*q)
        +radial[...,None]*perpendicular[:,None,:]/rho[:,None,None])).sum(-2)
    torch.testing.assert_close(actual[0],expected,rtol=1e-9,atol=1e-10)
    torch.testing.assert_close(actual[1],gradient,rtol=1e-9,atol=1e-10)
    with pytest.raises(ValueError,match='spectral pole'):
        kernel.evaluate(points,float(lattice.bloch.norm()))


@pytest.mark.parametrize('dimension',[1,2,3])
def test_periodic_regular_part_origin_near_gradient_and_primary_image(dimension):
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    vectors=torch.tensor([[1.,.2,.1],[.1,1.2,.3],[.2,.1,.9]],device='cuda',dtype=torch.float64)[:dimension]
    lattice=LatticeSpec(vectors,.17*vectors.sum(0))
    kernel=EwaldHelmholtzKernel(lattice,2.,quadrature_order=96,real_shells=4,reciprocal_shells=4)
    points=vectors.new_tensor([[0.,0.,0.],[1e-10,2e-10,-1e-10],[.13,.19,.23],[.6,.4,.3]],requires_grad=True)
    k=2+3j
    actual=kernel.regular_part(points,k)
    reference=LossyImageReference(lattice,12).evaluate(points,k,exclude_zero=True)
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    derivative,=torch.autograd.grad(actual[0].real.sum(),points)
    torch.testing.assert_close(derivative,actual[1].real,rtol=1e-10,atol=1e-10)
    _,receipt=kernel.estimate(points.detach()[:2],k,regular=True)
    assert receipt['converged']
    with pytest.raises(ValueError,match='Coincident'):
        kernel.regular_part(vectors[:1],k)


@pytest.mark.parametrize('dimension',[1,2,3])
def test_ewald_query_batch_preserves_values_and_gradients(dimension):
    from dataclasses import replace
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    vectors=torch.tensor([[1.,.2,.1],[.1,1.2,.3],[.2,.1,.9]],device='cuda',dtype=torch.float64)[:dimension]
    lattice=LatticeSpec(vectors,.17*vectors.sum(0))
    kernel=EwaldHelmholtzKernel(lattice,2.,real_shells=2,reciprocal_shells=3)
    points=torch.linspace(.001,.19,99,device='cuda',dtype=torch.float64).reshape(33,3).requires_grad_()
    for method in ('evaluate','regular_part'):
        actual=getattr(kernel,method)(points,2+3j)
        reference=getattr(replace(kernel,query_batch=16),method)(points,2+3j)
        for a,b in zip(actual,reference):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        gradient,=torch.autograd.grad(actual[0].real.sum(),points)
        torch.testing.assert_close(gradient,actual[1].real,rtol=1e-10,atol=1e-10)
    with pytest.raises(ValueError,match='query_batch'):
        replace(kernel,query_batch=0)


@pytest.mark.parametrize('dimension',[1,2,3])
def test_regular_part_distant_image_singularity_and_neighborhood(dimension):
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    from witwin.maxwell.mom.kernels import FREE_SPACE
    vectors=torch.tensor([[1.,.2,.1],[.1,1.2,.3],[.2,.1,.9]],device='cuda',dtype=torch.float64)[:dimension]
    lattice=LatticeSpec(vectors,.17*vectors.sum(0))
    kernel=EwaldHelmholtzKernel(lattice,2.,real_shells=2,reciprocal_shells=3)
    image=4*vectors[0]
    with pytest.raises(ValueError,match='Coincident'):
        kernel.regular_part(image[None],2+3j)
    offset=vectors.new_tensor([[.01,.02,.03]])
    points=(image+offset).requires_grad_()
    actual=kernel.regular_part(points,2+3j)
    shifted=kernel.evaluate(offset,2+3j)
    phase=torch.exp(1j*(image@lattice.bloch))
    free=FREE_SPACE.evaluate(points,2+3j)
    for value,wrapped,primary in zip(actual,shifted,free):
        torch.testing.assert_close(value,phase*wrapped-primary,rtol=1e-11,atol=1e-10)
    derivative,=torch.autograd.grad(actual[0].real.sum(),points)
    torch.testing.assert_close(derivative,actual[1].real,rtol=1e-10,atol=1e-10)


def test_half_cell_reduction_preserves_scalar_and_projected_bloch_phase():
    from witwin.maxwell.mom.periodic import EwaldHelmholtzKernel
    from witwin.maxwell.mom.periodic_limits import projected_bloch_green
    vectors = torch.eye(3, device='cuda', dtype=torch.float64)
    lattice = LatticeSpec(vectors, vectors.new_tensor([.2, .1, .05]))
    kernel = EwaldHelmholtzKernel(lattice, 2., real_shells=1, reciprocal_shells=1)
    points = vectors.new_tensor([[-.5, .2, .3], [.5, .2, .3], [1.5, .2, .3]])
    phase = torch.exp(1j*lattice.bloch[0])
    ordinary = kernel.evaluate(points, .8+.3j)
    projected = projected_bloch_green(kernel, points, eigen_order=torch.zeros(3, device='cuda', dtype=torch.int64))
    for value in (*ordinary, projected['green'], projected['gradient'], projected['residue_z']):
        torch.testing.assert_close(value[1:], value[:-1]*phase, rtol=1e-12, atol=1e-13)
