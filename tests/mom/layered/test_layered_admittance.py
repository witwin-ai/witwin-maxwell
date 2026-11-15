"""conductor_model="surface_admittance": native operators, DC exactness, contracts."""
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from tests.support.layered_dsa_oracle import assemble, divided_difference, helmholtz, rwg_basis
from tests.support.rfic_closed_references import STRIP_DC_OHM, network_metrics, strip_fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

UM = 1e-6
SIGMA = 5.8e7


def _box(n, size):
    """Closed outward box mesh with its RWG basis (test geometry only)."""
    from witwin.maxwell.mom.layered.mesh import mesh_closed_conductors
    planes = torch.tensor((0., size[2]), device="cuda", dtype=torch.float64)
    layout = mesh_closed_conductors([(0., size[0], 0., size[1], 0., size[2])], [], planes, max(size) / n,
                                    cuts=[((), ()), ((), ())], h_edge=None, growth_ratio=1.,
                                    interfaces=(), groups=[1])
    mesh, basis = layout.mesh, layout.basis
    areas = mesh.areas
    return mesh.vertices, mesh.faces, rwg_basis(mesh.faces, basis.edges, basis.plus_faces, basis.minus_faces,
                                                basis.support_slots[:, 0], basis.support_slots[:, 1],
                                                basis.edge_lengths, areas)


@pytest.mark.parametrize("frequency", (1e4, 1e9, 1e10))
def test_native_interior_operators_match_oracle(frequency):
    """Native blocks vs the torch oracle; bound = 10x the oracle's own order change."""
    from witwin.maxwell.mom.layered.admittance import interior_operators
    vertices, faces, basis = _box(4, (8 * UM, 4 * UM, 2 * UM))
    body = SimpleNamespace(vertices=vertices, local_faces=faces, basis=basis, trial=basis)
    omega = 2 * math.pi * frequency
    eps0 = 8.8541878128e-12
    mu0 = 4e-7 * math.pi
    kb = complex(omega * math.sqrt(mu0 * eps0))
    kc = complex(omega * (mu0 * (eps0 + 1j * SIGMA / omega)) ** .5)
    native = interior_operators(body, kc, kb)
    repeat = interior_operators(body, kc, kb)
    for key in native:
        assert torch.equal(native[key], repeat[key])
    references = []
    for orders in ((6, 6, 8, 4), (8, 8, 11, 6)):
        ops = assemble(vertices, faces, basis, basis,
                       [helmholtz(kc), helmholtz(kb), divided_difference(kc, kb)],
                       singular=orders[0], order=orders[1], fan=orders[2], far=orders[3])
        references.append(dict(vc=ops[0][0], vb=ops[1][0], vhat=ops[2][0], pc=ops[0][3], pb=ops[1][3],
                               kb=ops[1][2], khat=ops[2][2]))
    low, high = references
    for key, value in high.items():
        scale = value.norm()
        bound = 10 * (low[key] - value).norm() / scale + 1e-8
        error = (native[key] - value).norm() / scale
        assert error < bound, (key, error.item(), bound.item())


def _strip(model, maximum=20., edge=2., thickness=2.):
    scene, config = strip_fixture(model=model, maximum_um=maximum, edge_um=edge, thickness_um=thickness)
    return scene, config


def _strip_run(frequencies):
    scene, config = _strip("surface_admittance")
    return mw.Simulation.mom(scene, frequencies=frequencies, config=config).run()


def test_strip_dc_resistance_and_metadata():
    """Low regime: uniform-current local impedance gives R = l/(sigma A) (target 1e-6)."""
    result = _strip_run([1.0])
    r = network_metrics(result)["R_ohm"][0]
    assert abs(r / STRIP_DC_OHM - 1) < 1e-6
    metadata = result.network.metadata
    assert metadata["conductor_model"] == "surface_admittance"
    assert metadata["port_bridges"] == "ideal_pec_block"
    assert metadata["conductor_impedance"] == "differential_surface_admittance"
    assert metadata["low_frequency_interior"] == "uniform_current_local_impedance"
    assert [b["owner"] for b in metadata["admittance_bodies"]] == [0]
    assert metadata["admittance_bodies"][0]["thickness"] == pytest.approx(2 * UM, rel=1e-12)


# G3 true-corner reference L (H): lattice FastHenry (scripts/dev/rfic/strip_reference_fasthenry_g3.py, exact ring
# corners, cut-plane delta gap, PEC, direct, 0.383802 nH) plus the copper-run increment L_cu(f) - L_PEC of the A1c
# 25x13 centreline model (corners are PEC in both, so its +1.2% corner over-estimate cancels in the increment).
# The A1c centreline value (0.396576 nH) was the cause of the former -1.06%/-1.25% "offset".
STRIP_REFERENCE_L = {1e6: 0.392624e-9, 1e7: 0.392593e-9, 1e8: 0.392579e-9, 1e9: 0.392065e-9, 1e10: 0.388177e-9}


def test_strip_inductance_true_corner_reference():
    """Low (<= 1e8) and DSA (1e9, 1e10) regimes within 1% of the true-corner reference."""
    m = network_metrics(_strip_run(list(STRIP_REFERENCE_L)))
    for f, l in zip(m["frequency_Hz"], m["L_H"]):
        assert abs(l / STRIP_REFERENCE_L[f] - 1) < 1e-2, (f, l)


def test_internal_inductance_closed_form():
    """dL'_int against independent 2D quadrature of the area and perimeter log integrals."""
    from scipy import integrate
    from witwin.maxwell.mom.layered.admittance import MU0, internal_inductance
    a, b = 10., 2.
    area = 4 * integrate.dblquad(lambda v, u: (a - u) * (b - v) * .5 * math.log(u * u + v * v), 0, a, 0, b,
                                 epsabs=1e-12, epsrel=1e-12)[0] / (a * b) ** 2
    sides = (((0, 0), (a, 0)), ((a, 0), (a, b)), ((a, b), (0, b)), ((0, b), (0, 0)))
    total = 0.
    for p0, p1 in sides:
        for q0, q1 in sides:
            lp, lq = math.dist(p0, p1), math.dist(q0, q1)

            def integrand(t, s):
                x = [p0[k] + s * (p1[k] - p0[k]) - q0[k] - t * (q1[k] - q0[k]) for k in range(2)]
                return .5 * math.log(x[0] * x[0] + x[1] * x[1]) * lp * lq
            # Split at t = s so the self-term log singularity sits on integration bounds.
            total += sum(integrate.dblquad(integrand, 0, 1, lo, hi, epsabs=1e-11, epsrel=1e-11)[0]
                         for lo, hi in ((lambda s: 0., lambda s: s), (lambda s: s, lambda s: 1.)))
    perimeter = total / (2 * (a + b)) ** 2
    expected = MU0 / (2 * math.pi) * (perimeter - area)
    value = internal_inductance(a * UM, b * UM, "cuda")
    assert abs(value / expected - 1) < 1e-8
    # Thin limit -> 0, square above the round-wire mu0/(8 pi).
    assert internal_inductance(10 * UM, 1e-3 * UM, "cuda") < 1e-3 * expected
    assert internal_inductance(4 * UM, 4 * UM, "cuda") > MU0 / (8 * math.pi)


def test_regime_weight_is_c1_and_exact_outside_the_window():
    from witwin.maxwell.mom.layered.admittance import CROSSOVER, MU0, regime_weight
    body = SimpleNamespace(thickness=2 * UM, sigma=SIGMA)

    def frequency(ratio):
        return (ratio / body.thickness) ** 2 / (math.pi * MU0 * SIGMA)
    low, high = CROSSOVER
    assert regime_weight(body, frequency(low) * .999) == 0.
    assert regime_weight(body, frequency(high) * 1.001) == 1.
    for edge in (low, high):
        # Zero slope in log f at both window ends (C1).
        h = 1e-4
        left, right = (regime_weight(body, frequency(edge) * math.exp(d)) for d in (-h, h))
        assert abs(right - left) / (2 * h) < 1e-3


def test_strip_skin_regime_and_repeat():
    first, second = _strip_run([1e9]), _strip_run([1e9])
    assert torch.equal(first.network.s, second.network.s)
    assert torch.equal(mw.mom.surface_currents(first).currents, mw.mom.surface_currents(second).currents)
    assert torch.linalg.matrix_norm(first.network.s, ord=2).max().item() <= 1 + 1e-12
    m = network_metrics(first)
    # Quasi-static reference at 1 GHz: R=0.181666 ohm (25x13 filaments), L from the G3 true-corner table.
    assert abs(m["R_ohm"][0] / 0.181666 - 1) < 2e-2
    assert abs(m["L_H"][0] / STRIP_REFERENCE_L[1e9] - 1) < 2e-2


def test_regime_blocks_dsa_above_window_and_passive_inside():
    """s = 1 folds exactly the A1 DSA block; the window block is passive, the low block symmetric."""
    from witwin.maxwell.mom.layered import admittance as ad
    scene, config = _strip("surface_admittance")
    sa = mw.Simulation.mom(scene, frequencies=[1e9], config=config).prepare().solver.prepared.admittance
    body = sa.bodies[0]
    assert ad.regime_weight(body, 1e9) == 1.
    eps_b = ad.background_permittivity(sa.stack, *body.z_range, 1e9)
    ws, _ = ad.body_impedance(body, 1e9, eps_b)
    rx = body.exterior.to_dense().to(torch.complex128)
    assert torch.equal(sa.body_blocks(1e9)[body.owner][1], rx.mT @ ws @ rx)
    middle = (math.sqrt(ad.CROSSOVER[0] * ad.CROSSOVER[1]) / body.thickness) ** 2 / (math.pi * ad.MU0 * SIGMA)
    assert ad.regime_weight(body, middle) == pytest.approx(.5, abs=1e-12)
    block = sa.body_blocks(middle)[body.owner][1]
    assert torch.equal(block, sa.body_blocks(middle)[body.owner][1])
    # Passive (the DSA block itself is not symmetric: A1 measured a few percent).
    eigenvalues = torch.linalg.eigvalsh((block + block.mH) / 2)
    assert eigenvalues.min() >= -1e-9 * eigenvalues.max()
    low = sa.body_blocks(1e6)[body.owner][1]
    assert (low - low.mT).abs().max() <= 1e-12 * low.abs().max()


def test_surface_admittance_refusals():
    scene, config = _strip("surface_admittance")
    rough = mw.Scene(background=scene.background)
    for structure in scene.structures:
        material = structure.material
        if isinstance(material, mw.media.LossyMetalMedium):
            material = mw.media.LossyMetalMedium(conductivity=SIGMA, roughness=mw.media.HammerstadRoughness(1e-6))
        rough = rough.updated(structures=(*rough.structures, mw.Structure(structure.geometry, material)))
    rough = rough.updated(ports=(*rough.ports, *scene.ports))
    with pytest.raises(NotImplementedError, match="roughness"):
        mw.Simulation.mom(rough, frequencies=[1e9], config=config).prepare()
    trainable = mw.Scene(background=scene.background)
    sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
    for structure in scene.structures:
        material = structure.material
        if isinstance(material, mw.media.LossyMetalMedium):
            material = mw.media.LossyMetalMedium(conductivity=sigma)
        trainable = trainable.updated(structures=(*trainable.structures, mw.Structure(structure.geometry, material)))
    trainable = trainable.updated(ports=(*trainable.ports, *scene.ports))
    with pytest.raises(NotImplementedError, match="does not propagate gradients"):
        mw.Simulation.mom(trainable, frequencies=[1e9], config=config).prepare()


def _strip_scene(stack, return_material):
    """The strip fixture with a chosen stack and left-arm material (refusal controls)."""
    scene = mw.Scene(background=stack)
    copper = mw.media.LossyMetalMedium(conductivity=SIGMA)
    for x0, x1, y0, y1, medium in ((0., 200., -5., 5., copper), (-10., 0., -115., 5., return_material),
                                   (200., 210., -115., 5., mw.Material.pec()), (0., 95., -115., -105., mw.Material.pec()),
                                   (105., 200., -115., -105., mw.Material.pec())):
        vertices = torch.tensor(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=torch.float64) * UM
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(mw.geometry.PlanarPEC(vertices, normal_axis="z", coordinate=100 * UM,
                                      physical_thickness=2 * UM), medium),
        ))
    scene = scene.updated(ports=(
        *scene.ports,
        mw.LumpedPort("p", positive=(105 * UM, -110 * UM, 100 * UM), negative=(95 * UM, -110 * UM, 100 * UM),
                      voltage_path=mw.AxisPath("x"), reference_impedance=50.),
    ))
    config = mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=10 * UM, max_edge_length_m=20 * UM,
                                                          edge_cell_m=2 * UM, conductor_model="surface_admittance"))
    return scene, config


def test_surface_admittance_body_refusals():
    air = mw.mom.LayerStack((mw.mom.Layer("air", 1e-3),), mw.mom.HalfSpace(), mw.mom.HalfSpace())
    scene, config = _strip_scene(air, mw.media.LossyMetalMedium(conductivity=1e7))
    with pytest.raises(NotImplementedError, match="different conductivity"):
        mw.Simulation.mom(scene, frequencies=[1e9], config=config).prepare()
    split = mw.mom.LayerStack((mw.mom.Layer("low", 100 * UM), mw.mom.Layer("high", 1e-3)), mw.mom.HalfSpace(), mw.mom.HalfSpace())
    scene, config = _strip_scene(split, mw.Material.pec())
    with pytest.raises(NotImplementedError, match="one common dielectric layer"):
        mw.Simulation.mom(scene, frequencies=[1e9], config=config).prepare()


def test_surface_admittance_refuses_edge_touching_bodies():
    air = mw.mom.LayerStack((mw.mom.Layer("air", 1e-3),), mw.mom.HalfSpace(), mw.mom.HalfSpace())
    scene, config = _strip_scene(air, mw.Material.pec())
    copper = mw.media.LossyMetalMedium(conductivity=SIGMA)
    for x0, x1, y0, y1 in ((300., 310., 0., 10.), (310., 320., 10., 20.)):
        vertices = torch.tensor(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=torch.float64) * UM
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(mw.geometry.PlanarPEC(vertices, normal_axis="z", coordinate=100 * UM,
                                      physical_thickness=2 * UM), copper),
        ))
    with pytest.raises(NotImplementedError, match="manifold"):
        mw.Simulation.mom(scene, frequencies=[1e9], config=config).prepare()
