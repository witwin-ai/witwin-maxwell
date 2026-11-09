"""Deterministic CUDA Galerkin MPIE for planar and bent layered RWG currents."""

from dataclasses import replace

import torch

from ..quadrature import gauss_rule
from ..topology import INTERIOR
from .build import build_extension


def _rule(order, device):
    u, w = gauss_rule(order, device=device)
    a, b = torch.meshgrid(u, u, indexing="ij")
    return torch.stack((1 - a, a * (1 - b), a * b,
                        w[:, None] * w[None, :] * a), -1).reshape(-1, 4).contiguous()



# Symmetric triangle rules; weights integrate the reference area 1/2.
# Degree 2/4/5/6/7/10 rules use 3/6/7/12/13/25 points respectively.
def _symmetric_rule(nodes, device):
    from itertools import permutations
    if nodes == 1:
        orbits = [(1/3, 1/3, 1/3, .5)]
    elif nodes == 3:
        orbits = [(2/3, 1/6, 1/6, 1/6)]
    elif nodes == 6:
        orbits = [(.108103018168070, .445948490915965, .445948490915965, .111690794839005),
                  (.816847572980459, .091576213509771, .091576213509771, .054975871827661)]
    elif nodes == 7:
        orbits = [(1/3, 1/3, 1/3, .1125),
                  (.059715871789770, .470142064105115, .470142064105115, .066197076394253),
                  (.797426985353087, .101286507323456, .101286507323456, .062969590272414)]
    elif nodes == 12:
        orbits = [(.501426509658179, .249286745170910, .249286745170910, .0583931378631895),
                  (.873821971016996, .063089014491502, .063089014491502, .0254224531851035),
                  (.053145049844816, .310352451033785, .636502499121399, .041425537809187)]
    elif nodes == 13:
        orbits = [(1/3, 1/3, 1/3, -.074785022233841),
                  (.479308067841920, .260345966079040, .260345966079040, .087807628716604),
                  (.869739794195568, .065130102902216, .065130102902216, .026673617804419),
                  (.048690315425316, .312865496004874, .638444188569809, .0385568804451285)]
    elif nodes == 25:
        orbits = [(1/3, 1/3, 1/3, .090817990382754/2),
                  (.028844733232685, .485577633383657, .485577633383657, .036725957756467/2),
                  (.781036849029926, .109481575485037, .109481575485037, .045321059435528/2),
                  (.141707219414880, .307939838764121, .550352941820999, .072757916845420/2),
                  (.025003534762686, .246672560639903, .728323904597411, .028327242531057/2),
                  (.009540815400299, .066803251012200, .923655933587500, .009421666963733/2)]
    return torch.tensor([(*p, w) for a, b, c, w in orbits
                         for p in sorted(set(permutations((a, b, c))))],
                        device=device, dtype=torch.float64)


def _adaptive_rule(device, network_accuracy=False):
    rules = [_rule(8, device)] + [_symmetric_rule(n, device) for n in (25, 13, 12, 7, 6, 3, 1)]
    thresholds = ((0, 0, 3, 4, 8, 24, 1024, 1e8) if network_accuracy else
                  (0, 3, 8, 16, 32, 128, 8192, 1e9))
    # Each rule carries its fp64 squared threshold and point count in the
    # final metadata row. The network budget is 1e-6; standalone matrix
    # queries retain the tighter existing far-entry gate.
    return torch.stack([torch.cat((
        torch.nn.functional.pad(rule, (0, 0, 0, 64 - len(rule))),
        rule.new_tensor([[threshold**2, len(rule), 0, 0]])))
        for rule, threshold in zip(rules, thresholds)])


def _outer_rule(device):
    # Contact/self retains the reference rule; smooth analytic outer integrals
    # use the measured lower orders. Selection is per static image.
    rules = [_rule(n, device) for n in (32, 24, 16, 8)]
    return torch.stack([torch.nn.functional.pad(rule, (0, 0, 0, 1024 - len(rule))) for rule in rules])


def triangle_static_integrals(triangles, observers, free_vertices):
    """CUDA fp64 int 1/R and int (r'-free_vertex)/R, without the 1/(4*pi).

    Inputs are paired [B,3,3], [B,3], [B,3] CUDA float64 tensors. Heights
    may be zero, including observations on edges and vertices.
    """
    build_extension()
    return torch.ops.witwin_maxwell_mom_layered.triangle_static(
        triangles.contiguous(), observers.contiguous(), free_vertices.contiguous())


def triangle_cross_integrals(triangles, observers, beta):
    """Closed-form static radial cross contraction for horizontal/vertical faces.

    Return [batch,3] unnormalized integrals for the three free vertices:
    int (r_h-v_h).(obs_h-r_h)/[R*(R+d)], d=beta*(obs_z-r_z).
    Image triangles have d>=0 and beta is +1 or -1 per query.
    The radial identity f_h.H=(obs_h-v_h).H-1+d/R and integration
    by parts reduce the integral to triangle moments and edge integrals
    of log(R+d) and 1/(R+d). The latter follow from the rational substitution
    t=(s+sqrt(s*s+A))/sqrt(A); no area quadrature is used.
    """
    build_extension()
    return torch.ops.witwin_maxwell_mom_layered.triangle_cross_static(
        triangles.contiguous(), observers.contiguous(), beta.contiguous())


def assemble_layered(layout, table, *, frequency_index=0, near_ratio=2.0,
                     far_order=None, near_order=None, far_dtype=torch.float64, source_tile=64,
                     separate=False, _tiles=False, _network_accuracy=False):
    """Return complex128 Z[N,N], using a bounded [batch,faces,source_tile,10] buffer.

    For exp(-i*w*t), Z=-i*w*mu0*vector+i/(w*eps0)*div^T*scalar*div.
    Thus homogeneous Z is NEGATIVE of single_layer.py's weak electric L.
    K^Phi already includes 1/eps_r; both terms retain vacuum prefactors.
    Near classification always uses fp64 squared distances with strict '<';
    regular far vector moments select float32 or float64 arithmetic. Scalar
    integration remains fp64 to preserve cancellation between static images.
    """
    if far_dtype not in (torch.float32, torch.float64):
        raise ValueError("far_dtype must be torch.float32 or torch.float64")
    mesh, basis = layout.mesh, layout.basis
    triangles = mesh.triangles.contiguous()
    if triangles.dtype != torch.float64:
        raise ValueError("Layered assembly requires fp64 geometry")
    if not (0 <= frequency_index < len(table.frequencies) or separate and frequency_index == -1) or source_tile < 1 or near_ratio <= 0:
        raise ValueError("Require an admitted frequency, source_tile>=1 and near_ratio>0")
    extent = mesh.vertices.amax(0) - mesh.vertices.amin(0)
    if bool(torch.linalg.vector_norm(extent[:2]) > table.rho_max):
        raise ValueError("Layout exceeds Green table rho_max")
    horizontal = layout.triangle_planes >= 0
    if not bool((triangles[horizontal, :, 2] == table.planes[layout.triangle_planes[horizontal], None]).all()):
        raise ValueError("Mesh triangles must lie on their table metal planes")
    walls = not bool(horizontal.all())
    if walls and table.vertical is None:
        raise ValueError("Wall assembly requires declared via spans")
    if walls and not bool(((triangles[:, :, 2, None] >= table.vertical.segments[None, None, :, 0])
                           & (triangles[:, :, 2, None] <= table.vertical.segments[None, None, :, 1])).all(1).any(-1).all()):
        raise ValueError("Every coupled triangle must lie within one declared vertical segment")
    areas2 = (mesh.areas * 2).contiguous()
    edge2 = (triangles - triangles.roll(1, 1)).square().sum(-1).amax(-1).contiguous()
    slopes = (basis.face_signs * basis.edge_lengths[basis.face_edges] / areas2[:, None]).contiguous()
    supports = torch.stack((basis.plus_faces, basis.minus_faces), -1).contiguous()
    slots = (mesh.topology.incidence_slots[mesh.topology.kind == INTERIOR].contiguous()
             if layout.support_slots is None else layout.support_slots)
    geometry = [triangles, layout.triangle_planes, areas2, mesh.centroids.contiguous(), edge2,
                slopes, supports, slots, basis.face_edges.contiguous()]
    green = [table.remainder, table.derivatives, table.rho, table.planes, table.image_offsets,
             table.image_distances, table.image_coefficients, table.media, table.sigma, table.frequencies]
    if walls:
        v = table.vertical
        green += v.assembly_tensors()
    build_extension()
    operation = (torch.ops.witwin_maxwell_mom_layered.assemble_tiles if _tiles else
                 torch.ops.witwin_maxwell_mom_layered.assemble_parts if separate else
                 torch.ops.witwin_maxwell_mom_layered.assemble)
    return operation(
        green, geometry, _adaptive_rule(mesh.device, _network_accuracy) if far_order is None else _rule(far_order, mesh.device),
        _outer_rule(mesh.device) if near_order is None else _rule(near_order, mesh.device),
        frequency_index, source_tile, near_ratio, far_dtype == torch.float64,
        table.exact_mode, table.ground)


def assemble_potentials(layout, table, *, frequency_start=0, frequency_stop=None, **options):
    """Return V[N,N], P[F,F]; frequency_index=-1 batches all frequencies.

    P0_t=1/area_t on face t (unit integral), so D is signed edge length.
    Z=-i*w*mu0*V+i/(w*eps0)*D.T@P@D. This flux convention avoids
    area-dependent amplification of roundoff in numerical D*Lambda checks.
    The original summed-Z gather is unchanged (T0).
    """
    # Frequency slices are views: no all-sweep dense potential allocation.
    table = replace(table, frequencies=table.frequencies[frequency_start:frequency_stop],
                    remainder=table.remainder[frequency_start:frequency_stop],
                    derivatives=table.derivatives[frequency_start:frequency_stop],
                    image_coefficients=table.image_coefficients[frequency_start:frequency_stop])
    if table.vertical is not None:
        v = table.vertical
        table = replace(table, vertical=replace(
            v, remainder=v.remainder[frequency_start:frequency_stop],
            derivatives=v.derivatives[frequency_start:frequency_stop],
            image_coefficients=v.image_coefficients[frequency_start:frequency_stop],
            horizontal_remainder=v.horizontal_remainder[frequency_start:frequency_stop],
            horizontal_derivatives=v.horizontal_derivatives[frequency_start:frequency_stop],
            horizontal_coefficients=v.horizontal_coefficients[frequency_start:frequency_stop]))
        # tensors() must use this chunk's frequency array, not the parent sweep.
        from weakref import proxy
        table.vertical.parent = proxy(table)
    return assemble_layered(layout, table, separate=True, **options)
