"""Test oracle of the A1 interior far kernel (R5-2).

serial_far_interior_operators mirrors admittance.interior_operators but runs the
far pairs through far_serial_kernel (one thread per pair, nodes accumulated in
(i,j,k,l) order), the pre-R5-2 far kernel. The production far_kernel (one warp
per pair, per-lane output slots summed in node order) must be torch.equal to it.
"""

import torch

from witwin.maxwell.mom.layered.admittance import GAUSS_ORDERS, LAUNCH, RULES, _gauss_table


def serial_far_interior_operators(body, kc, kb, rules=RULES, launch=LAUNCH):
    from witwin.maxwell.mom.layered.build import build_extension
    build_extension()
    vertices = (body.vertices - body.vertices.mean(0)).contiguous()
    face_bytes = 47 * 16 * len(body.local_faces)
    tile = max(1, min(len(body.local_faces), (1 << 30) // face_bytes))
    names = ("vc", "vb", "vhat", "kb", "khat", "pc", "pb")
    values = torch.ops.witwin_maxwell_mom_layered.admittance_operators_serial_far(
        vertices, body.local_faces.contiguous(), _gauss_table(vertices.device),
        *(t.contiguous() for t in body.basis), *(t.contiguous() for t in body.trial),
        kc.real, kc.imag, kb.real, kb.imag, [*rules, GAUSS_ORDERS, *launch], tile)
    return dict(zip(names, values))
