import pytest
import torch
import math
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.modes import RegularModes, build_modal_rhs, modal_response, calibrate_modal_response
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_modal_origin_linearity_and_angular_convergence():
    medium = compile_medium(mw.Material(), 3e8)
    modes = RegularModes(3,medium,angular_order=12)
    fine = RegularModes(3,medium,angular_order=16)
    points = torch.tensor([[0.,0.,0.],[.1,.2,.3],[-.2,.1,.15]],device="cuda",dtype=torch.float64)
    e,h = modes.evaluate(points)
    ef,hf = fine.evaluate(points)
    assert modes.count == 30
    torch.testing.assert_close(e,ef,atol=1e-11,rtol=1e-9)
    torch.testing.assert_close(h,hf,atol=1e-13,rtol=1e-9)
    c = torch.randn(modes.count,device="cuda",dtype=torch.complex128,requires_grad=True)
    electric,magnetic = modes.evaluate(points,c)
    torch.testing.assert_close(electric,torch.einsum("qmc,m->qc",e,c))
    torch.testing.assert_close(magnetic,torch.einsum("qmc,m->qc",h,c))
    assert torch.isfinite(e).all()
    electric.abs().square().sum().backward()
    assert torch.isfinite(c.grad).all()


def test_modal_curl_equations_and_control_energy():
    medium = compile_medium(mw.Material(),3e8)
    modes = RegularModes(2,medium)
    points = torch.tensor([[.12,.04,.08]],device="cuda",dtype=torch.float64)
    step = 1e-5
    de,dh = [],[]
    for axis in range(3):
        delta = torch.zeros_like(points); delta[:,axis] = step
        ep,hp = modes.evaluate(points+delta)
        em,hm = modes.evaluate(points-delta)
        de.append((ep-em)/(2*step)); dh.append((hp-hm)/(2*step))
    ce = torch.stack((de[1][...,2]-de[2][...,1],de[2][...,0]-de[0][...,2],de[0][...,1]-de[1][...,0]),-1)
    ch = torch.stack((dh[1][...,2]-dh[2][...,1],dh[2][...,0]-dh[0][...,2],dh[0][...,1]-dh[1][...,0]),-1)
    e,h = modes.evaluate(points)
    torch.testing.assert_close(ce,1j*medium.omega*medium.mu*h,rtol=2e-8,atol=1e-11)
    torch.testing.assert_close(ch,-1j*medium.omega*medium.epsilon*e,rtol=2e-8,atol=1e-13)
    modes.normalize_control_surface(.3)
    gram = modes.normalize_control_surface(.3)
    torch.testing.assert_close(gram,torch.eye(modes.count,device="cuda",dtype=gram.dtype),rtol=1e-10,atol=1e-10)


def test_modal_rhs_linear_and_cutoff_prefix():
    medium = compile_medium(mw.Material(),3e8)
    basis = RWGBasis(sphere_mesh(radius=.1,subdivisions=0))
    a = RegularModes(1,medium,angular_order=12)
    b = RegularModes(3,medium,angular_order=12)
    rhs = build_modal_rhs(basis,a,order=4)
    bigger = build_modal_rhs(basis,b,order=4)
    torch.testing.assert_close(rhs,bigger[:,:a.count])
    assert rhs.is_cuda and rhs.dtype == torch.complex128
    assert torch.linalg.matrix_rank(rhs) == a.count


def test_regular_vspherical_modes_match_independent_bessel_radial_oracle():
    # Plane-wave expansion: https://dlmf.nist.gov/10.60.E7.
    # This oracle evaluates radial functions independently, not angular sums.
    from scipy.special import spherical_jn
    medium = compile_medium(mw.Material(), 3e8)
    modes = RegularModes(3, medium, angular_order=20)
    radii = [0., .025, .12, .3]
    points = torch.tensor([[r, 0., 0.] for r in radii], device="cuda", dtype=torch.float64)
    e, h = modes.evaluate(points)
    te = modes.ordering.index((1, 0, "TE"))
    tm = modes.ordering.index((1, 0, "TM"))
    prefactor = math.sqrt(3/(8*math.pi))
    expected_e = torch.zeros((4, 3), device="cuda", dtype=torch.complex128)
    expected_tm = torch.zeros_like(expected_e)
    for index, radius in enumerate(radii):
        z = medium.k.real*radius
        expected_e[index, 1] = -prefactor*spherical_jn(1, z)
        derivative_factor = 2/3 if z == 0 else (spherical_jn(1, z)+z*spherical_jn(1, z, derivative=True))/z
        expected_tm[index, 2] = 1j*prefactor*derivative_factor
    torch.testing.assert_close(e[:, te], expected_e, atol=2e-14, rtol=2e-12)
    torch.testing.assert_close(e[:, tm], expected_tm, atol=2e-14, rtol=2e-12)
    torch.testing.assert_close(h[:, te], expected_tm/medium.eta, atol=2e-16, rtol=2e-12)
    torch.testing.assert_close(h[:, tm], -expected_e/medium.eta, atol=2e-16, rtol=2e-12)


@pytest.fixture(scope="module")
def physical_modal_operator():
    from witwin.maxwell.mom.linalg import DenseOperator
    from witwin.maxwell.mom.operator import IntegrationConfig, assemble_weak
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0))
    exterior = compile_medium(mw.Material(), 3e8)
    interior = compile_medium(mw.Material(eps_r=2.5, sigma_e=.003), 3e8)
    matrix, metadata = assemble_weak(basis, exterior, interior, IntegrationConfig(order=4, singular_order=6))
    mass = basis.mass_matrix().to(matrix.dtype)
    handle = DenseOperator(matrix, torch.block_diag(mass, mass), metadata)
    handle.basis, handle.exterior, handle.interior = basis, exterior, interior
    handle.integration = IntegrationConfig(order=6, singular_order=8)
    return handle


def test_complete_actual_pmchwt_modal_response_reuses_one_lu(physical_modal_operator):
    handle = physical_modal_operator
    modes = RegularModes(3, handle.exterior)
    modes.normalize_control_surface(.3)
    response, log = modal_response(handle, modes)
    assert response.shape == (2*handle.basis.n_edges, modes.count)
    assert log["solver"]["converged"] and log["solver"]["rhs_count"] == modes.count
    assert log["complete_column_count"] == 30
    lu = handle._lu
    repeated, next_log = modal_response(handle, modes)
    assert handle._lu is lu and next_log["solver"]["cache_reused"]
    torch.testing.assert_close(response, repeated)
    rhs = build_modal_rhs(handle.basis, modes)
    assert torch.linalg.matrix_norm(handle.matrix@response-rhs)/torch.linalg.matrix_norm(rhs) < 1e-12
    coefficients = torch.randn(modes.count, device="cuda", dtype=torch.complex128)
    torch.testing.assert_close(response@coefficients, torch.linalg.solve(handle.matrix, rhs@coefficients), atol=1e-10, rtol=1e-10)


def test_cutoff_calibration_checks_added_response_not_only_prefix(physical_modal_operator):
    handle = physical_modal_operator
    _, _, coarse_report = calibrate_modal_response(handle, cutoffs=(1, 3, 5), control_radius=.3)
    assert coarse_report['cutoff_tolerance'] == .0005
    assert not coarse_report['cutoff_converged'] and not coarse_report['converged']
    response, modes, report = calibrate_modal_response(handle, cutoffs=(1, 3, 5, 7), control_radius=.3)
    assert response.shape[-1] == modes.count == 126
    assert report["algebra_converged"] and report["angular_converged"]
    rows = report["rows"]
    assert [row["added_column_count"] for row in rows] == [6, 24, 40, 56]
    assert rows[1]["added_response_relative_operator_norm"] > rows[2]["added_response_relative_operator_norm"] > 0
    assert rows[1]["added_response_relative_operator_norm"] > .1
    assert rows[2]["added_response_relative_operator_norm"] < .005
    assert rows[2]["added_response_relative_operator_norm"] > report['cutoff_tolerance']
    assert rows[3]["added_response_relative_operator_norm"] < report['cutoff_tolerance']
    assert rows[2]["angular_relative_error"] < 1e-6
    assert rows[2]["prefix_relative_error"] < 1e-8
    assert not report["uncomputed_tail_bound_certified"]
    assert not report["spatial_reference_validated"]
    assert report["cutoff_converged"] and report["converged"]


def test_regular_source_applicability_guards(physical_modal_operator):
    handle = physical_modal_operator
    modes = RegularModes(1, handle.exterior)
    with pytest.raises(ValueError, match="enclose"):
        modes.validate_enclosing_sphere(handle.basis, .05)
    inside = torch.tensor([[.01, 0., 0.]], device="cuda", dtype=torch.float64)
    outside = torch.tensor([[.4, 0., 0.]], device="cuda", dtype=torch.float64)
    with pytest.raises(ValueError, match="source singularities"):
        modes.validate_enclosing_sphere(handle.basis, .3, source_positions=inside)
    assert modes.validate_enclosing_sphere(handle.basis, .3, source_positions=outside)["singularity_positions_checked"]
    other_medium = compile_medium(mw.Material(eps_r=2), 3e8, background=True)
    with pytest.raises(ValueError, match="operator exterior"):
        modal_response(handle, RegularModes(1, other_medium))


def test_actual_te_modal_scattering_converges_toward_independent_sphere_reference(physical_modal_operator):
    from scipy.special import spherical_jn, spherical_yn
    from witwin.maxwell.mom.fields import far_field
    from witwin.maxwell.mom.linalg import DenseOperator
    from witwin.maxwell.mom.operator import IntegrationConfig, assemble_weak
    coarse = physical_modal_operator
    z0, z1 = coarse.exterior.k*.1, coarse.interior.k*.1
    j0, j1 = spherical_jn(1, z0), spherical_jn(1, z1)
    h0 = j0+1j*spherical_yn(1, z0)
    d0 = j0+z0*spherical_jn(1, z0, derivative=True)
    d1 = j1+z1*spherical_jn(1, z1, derivative=True)
    dh = h0+z0*(spherical_jn(1, z0, derivative=True)+1j*spherical_yn(1, z0, derivative=True))
    # Match j0+a*h0=c*j1 and d0+a*dh=c*d1 for nonmagnetic TE waves.
    scattering = (j0*d1-d0*j1)/(dh*j1-h0*d1)
    expected_far_y = math.sqrt(3/(8*math.pi))*scattering/coarse.exterior.k
    query = torch.tensor([[1., 0., 0.]], device="cuda", dtype=torch.float64)
    errors = []
    for level in (0, 1):
        if level == 0:
            handle = coarse
        else:
            basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=level))
            config = IntegrationConfig(order=4, singular_order=6)
            matrix, metadata = assemble_weak(basis, coarse.exterior, coarse.interior, config)
            mass = basis.mass_matrix().to(matrix.dtype)
            handle = DenseOperator(matrix, torch.block_diag(mass, mass), metadata)
            handle.basis, handle.exterior, handle.interior = basis, coarse.exterior, coarse.interior
            handle.integration = IntegrationConfig(order=6, singular_order=8)
        modes = RegularModes(1, handle.exterior)
        response, log = modal_response(handle, modes)
        assert log["solver"]["converged"]
        fe, _ = far_field(handle, response[:, modes.ordering.index((1, 0, "TE"))], query)
        errors.append(abs(complex(fe[0, 1])/expected_far_y-1))
    # This checks independent phase and mesh convergence, not the G0 accuracy target.
    assert errors[1] < .4*errors[0]
    assert errors[1] < .25
