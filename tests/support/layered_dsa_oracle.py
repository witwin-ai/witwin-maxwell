"""Torch oracle for the surface-admittance interior operators (test use only).

Face-pair Galerkin integrals of a closed triangle surface for scalar kernels
g(R): V = <f, S f>, P = <div f, S div f>, K = <f, curl S f> and the local
Gram/rotated Gram blocks. Touching pairs use the shared joint Duffy rules,
near pairs observer Gauss nodes times distance-scaled source fans, far pairs
product rules. Accumulation uses index_put (order is not fixed): this module
is an accuracy oracle, never a runtime path.

A basis is (faces[N,2], slots[N,2], coefficients[N,2]); entry (face,slot,c)
means c*(r - v_slot) on that face, face=-1 is padding. RWG: c=+-l/(2A).
"""
import math

import torch

from witwin.maxwell.mom.quadrature import (classify_all_pairs, distance_scaled_duffy_fan, touching_pair_rule,
                                           triangle_rule)


def helmholtz(k):
    """g(R)=exp(ikR)/(4 pi R) and dg/dR."""
    def kernel(r):
        value = torch.exp(1j * k * r) / (4 * math.pi * r)
        return value, value * (1j * k - 1 / r)
    return kernel


def divided_difference(kc, kb):
    """F=(G_c-G_b)/(kc^2-kb^2), dF/dR; series for |kc| R < 0.5."""
    def kernel(r):
        r = r.to(torch.complex128)
        a = torch.exp(1j * kb * r)
        e = torch.expm1(1j * (kc - kb) * r)
        direct = a * e / (4 * math.pi * r * (kc - kb) * (kc + kb))
        derivative_direct = (1j * kc * torch.exp(1j * kc * r) - 1j * kb * a) / (
            4 * math.pi * r * (kc - kb) * (kc + kb)) - direct / r
        # sum_{n>=1} i^n R^(n-1) (kc^n-kb^n)/n! / (4 pi (kc^2-kb^2))
        series = torch.zeros_like(r)
        dseries = torch.zeros_like(r)
        power = torch.ones_like(r)
        factorial = 1.0
        for n in range(1, 30):
            factorial *= n
            coefficient = (1j ** n) * (kc ** n - kb ** n) / factorial / (4 * math.pi * (kc * kc - kb * kb))
            series = series + coefficient * power
            if n >= 2:
                dseries = dseries + coefficient * (n - 1) * power / r
            power = power * r
        small = (abs(kc) * r.real) < .5
        return torch.where(small, series, direct), torch.where(small, dseries, derivative_direct)
    return kernel


def rwg_basis(mesh_faces, edges, plus, minus, plus_slots, minus_slots, lengths, areas):
    faces = torch.stack((plus, minus), -1)
    slots = torch.stack((plus_slots, minus_slots), -1)
    coefficient = torch.stack((lengths / (2 * areas[plus]), -lengths / (2 * areas[minus.clamp_min(0)])), -1)
    coefficient = torch.where(faces >= 0, coefficient, 0.)
    return faces, slots, coefficient


def _incidence(basis, count):
    faces, slots, coefficient = basis
    entries = (faces >= 0).nonzero()
    face = faces[entries[:, 0], entries[:, 1]]
    order = torch.argsort(face, stable=True)
    entries, face = entries[order], face[order]
    counts = torch.bincount(face, minlength=count)
    width = int(counts.max())
    start = torch.cumsum(counts, 0) - counts
    position = torch.arange(len(face), device=face.device) - start[face]
    function = torch.full((count, width), -1, dtype=torch.int64, device=face.device)
    slot = torch.zeros((count, width), dtype=torch.int64, device=face.device)
    value = torch.zeros((count, width), dtype=torch.float64, device=face.device)
    function[face, position] = entries[:, 0]
    slot[face, position] = slots[entries[:, 0], entries[:, 1]]
    value[face, position] = coefficient[entries[:, 0], entries[:, 1]]
    return function, slot, value


def _blocks(obs, src, weight, to, ts, kernels, magnetic):
    """Local [B,3,3] V, [B] P and K blocks from paired nodes; K uses n_s x h."""
    delta = obs - src
    ns = torch.linalg.cross(ts[:, 1] - ts[:, 0], ts[:, 2] - ts[:, 0])
    ns = ns / torch.linalg.vector_norm(ns, dim=-1, keepdim=True)
    r = torch.linalg.vector_norm(delta, dim=-1)
    f = obs[:, :, None] - to[:, None]          # [B,Q,3(slot),3]
    h = src[:, :, None] - ts[:, None]
    dot = torch.einsum("bqic,bqjc->bqij", f, h)
    out = []
    for kernel in kernels:
        g, dg = kernel(r)
        w = weight.to(torch.complex128)
        v = torch.einsum("bq,bqij->bij", g * w, dot.to(torch.complex128))
        p = (g * w).sum(1)
        if magnetic:
            grad = (dg / r)[..., None] * delta.to(torch.complex128)          # grad_r g
            hr = torch.linalg.cross(ns[:, None, None].expand_as(h), h, dim=-1)
            cross = torch.linalg.cross(grad[:, :, None].expand(-1, -1, 3, -1), hr.to(torch.complex128), dim=-1)
            k = torch.einsum("bqic,bqjc,bq->bij", f.to(torch.complex128), cross, w)
        else:
            k = torch.zeros_like(v)
        out.append((v, p, k))
    return out


def assemble(vertices, faces, test, source, kernels, *, order=6, singular=7, far=4, fan=10, batch=1 << 21):
    """Per kernel [V, P, K] for test x source bases plus the face scalar F x F.

    P is the basis-level <div f, S div h>; the last entry is P_face[t,s]=int int g,
    from which exact tree charges form loop-free scalar blocks.
    """
    device = vertices.device
    triangles = vertices[faces]
    count = len(faces)
    ti, si = _incidence(test, count), _incidence(source, count)
    m, n = len(test[0]), len(source[0])
    result = [[torch.zeros((m, n), dtype=torch.complex128, device=device) for _ in range(3)]
              + [torch.zeros((count, count), dtype=torch.complex128, device=device)] for _ in kernels]

    class Mesh:
        pass
    mesh = Mesh()
    mesh.vertices, mesh.faces, mesh.device = vertices, faces, device
    classes = classify_all_pairs(mesh)
    normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True)

    def scatter(rows, cols, values):
        fr, sr, vr = (x[rows] for x in ti)
        fc, sc, vc = (x[cols] for x in si)
        valid = (fr[:, :, None] >= 0) & (fc[:, None, :] >= 0)
        b, a, c = valid.nonzero().unbind(-1)
        mi, nj = fr[b, a], fc[b, c]
        weight = vr[b, a] * vc[b, c]
        for target, (v, p, k) in zip(result, values):
            target[3].index_put_((rows, cols), p, accumulate=True)
            target[0].index_put_((mi, nj), weight * v[b, sr[b, a], sc[b, c]], accumulate=True)
            target[1].index_put_((mi, nj), 4 * weight * p[b], accumulate=True)
            target[2].index_put_((mi, nj), weight * k[b, sr[b, a], sc[b, c]], accumulate=True)

    sizes = (6 * singular ** 4, 6 * singular ** 4, 2 * singular ** 4, order * order * 3 * fan * fan, far ** 4)
    for category in range(5):
        ids = (classes == category).nonzero().flatten()
        for part in ids.split(max(1, batch // sizes[category])):
            rows, cols = part // count, part % count
            if category < 3:
                obs, src, w = touching_pair_rule(triangles[rows], triangles[cols], faces[rows], faces[cols],
                                                 category, singular)
                planar = (normals[rows] * normals[cols]).sum(-1).abs() > 1 - 1e-12
                values = _blocks(obs, src, w, triangles[rows], triangles[cols], kernels, True)
                if category == 0:
                    values = [(v, p, torch.zeros_like(k)) for v, p, k in values]
                else:
                    values = [(v, p, torch.where(planar[:, None, None] & (((obs - src) * normals[rows, None]).sum(-1).abs().amax(1) == 0)[:, None, None], 0, k)) for v, p, k in values]
            elif category == 3:
                points, wo = triangle_rule(triangles[rows], order)
                q = points.shape[1]
                inner, wi = distance_scaled_duffy_fan(points.reshape(-1, 3),
                                                      triangles[cols, None].expand(-1, q, -1, -1).reshape(-1, 3, 3),
                                                      fan)
                inner, wi = inner.reshape(len(rows), q, -1, 3), wi.reshape(len(rows), q, -1)
                s = inner.shape[2]
                obs = points[:, :, None].expand(-1, -1, s, -1).reshape(len(rows), -1, 3)
                w = (wo[:, :, None] * wi).reshape(len(rows), -1)
                values = _blocks(obs, inner.reshape(len(rows), -1, 3), w, triangles[rows], triangles[cols],
                                 kernels, True)
            else:
                po, wo = triangle_rule(triangles[rows], far)
                ps, ws = triangle_rule(triangles[cols], far)
                q, s = po.shape[1], ps.shape[1]
                obs = po[:, :, None].expand(-1, -1, s, -1).reshape(len(rows), -1, 3)
                src = ps[:, None].expand(-1, q, -1, -1).reshape(len(rows), -1, 3)
                w = (wo[:, :, None] * ws[:, None]).reshape(len(rows), -1)
                values = _blocks(obs, src, w, triangles[rows], triangles[cols], kernels, True)
            scatter(rows, cols, values)
    return result


def local_grams(vertices, faces, test, source, *, order=4):
    """Gram <f,h> and rotated Gram <f, n x h> (same-face products)."""
    triangles = vertices[faces]
    count = len(faces)
    ti, si = _incidence(test, count), _incidence(source, count)
    points, w = triangle_rule(triangles, order)
    normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    f = points[:, :, None] - triangles[:, None]
    gram = torch.einsum("tqic,tqjc,tq->tij", f, f, w)
    rotated = torch.einsum("tqic,tqjc,tq->tij", f, torch.linalg.cross(normals[:, None, None].expand_as(f), f, dim=-1), w)
    m, n = len(test[0]), len(source[0])
    out = [torch.zeros((m, n), dtype=torch.float64, device=vertices.device) for _ in range(2)]
    fr, sr, vr = ti
    fc, sc, vc = si
    valid = (fr[:, :, None] >= 0) & (fc[:, None, :] >= 0)
    t, a, c = valid.nonzero().unbind(-1)
    weight = vr[t, a] * vc[t, c]
    for target, block in zip(out, (gram, rotated)):
        target.index_put_((fr[t, a], fc[t, c]), weight * block[t, sr[t, a], sc[t, c]], accumulate=True)
    return out
