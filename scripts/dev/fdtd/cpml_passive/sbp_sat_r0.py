"""Exact CPU R0 check of a split-grid Yee SBP/SAT PML interface.

Central and impedance-upwind fluxes use the same staggered SBP pair. SATs
enter both the field derivatives and their driven memories. A failed R0
terminates this experiment; this script performs no qualification sweeps.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import linalg

from model import Block, Cell

ROOT = Path(__file__).resolve().parent
R = sp.Rational


def operators():
    """Three subdomains: one PML edge, four interior edges, one PML edge."""
    h, t, k = R(1, 2), R(1, 4), R(1, 2)
    lengths = (1, 4, 1)
    starts = (0, 1, 5)
    tags = [(d, j) for d, n in enumerate(lengths) for j in range(n + 1)
            if (d, j) not in ((0, 0), (2, 1))]
    ne, nh = len(tags), sum(lengths)
    F0 = sp.zeros(nh, ne)
    he = []
    for col, (d, j) in enumerate(tags):
        he.append(h / 2 if j in (0, lengths[d]) else h)
        if j:
            F0[starts[d] + j - 1, col] += 1 / h
        if j < lengths[d]:
            F0[starts[d] + j, col] -= 1 / h
    He, Hh = sp.diag(*he), h * sp.eye(nh)
    Je, Jh, S = sp.zeros(2, ne), sp.zeros(2, nh), sp.zeros(nh, 2)
    for face in range(2):
        Je[face, tags.index((face, lengths[face]))] = -1
        Je[face, tags.index((face + 1, 0))] = 1
        left, right = starts[face + 1] - 1, starts[face + 1]
        Jh[face, left], Jh[face, right] = -1, 1
        S[left, face] = S[right, face] = 1
    F = F0 + Hh.inv() * S * Je / 2
    B = -He.inv() * F.T * Hh
    eu = sp.diag(*[4 if (d, j) == (1, 2) else 1 for d, j in tags])
    ev = sp.diag(1, 1, 4, 4, 1, 1)
    pe = [i for i, (d, _) in enumerate(tags) if d != 1]
    ph = [0, nh - 1]
    return dict(h=h, t=t, k=k, tags=tags, ne=ne, nh=nh, He=He, Hh=Hh,
                F=F, B=B, Je=Je, Jh=Jh, eu=eu, ev=ev, pe=pe, ph=ph)


def amplification(o, tau, *, pml=True):
    """Exact H-before-E map, with midpoint SATs inside each local ADE kick."""
    h, t, k = (o[key] for key in ('h', 't', 'k'))
    ne, nh = o['ne'], o['nh']
    He, Hh, F, B = (o[key] for key in ('He', 'Hh', 'F', 'B'))
    eu, ev, pe, ph = (o[key] for key in ('eu', 'ev', 'pe', 'ph'))
    cuts = (0, ne, ne + nh, ne + 2 * nh, ne + 2 * nh + len(pe),
            ne + 2 * nh + len(pe) + len(ph))
    size = cuts[-1]
    U, V, W, P, Q = [sp.eye(size)[cuts[j]:cuts[j + 1], :] for j in range(5)]
    p, q = sp.zeros(ne, size), sp.zeros(nh, size)
    be, bh, ce, ch = sp.eye(ne), sp.eye(nh), sp.zeros(ne), sp.zeros(nh)
    for row, index in enumerate(pe):
        p[index, :] = P[row, :]
        if pml:
            be[index, index], ce[index, index] = R(4, 5), -R(1, 10)
    for row, index in enumerate(ph):
        q[index, :] = Q[row, :]
        if pml:
            bh[index, index], ch[index, index] = R(4, 5), -R(1, 10)
    De, Dh = tau * o['Je'].T * o['Je'], tau * o['Jh'].T * o['Jh']
    Le, Lh = He.inv() * De, Hh.inv() * Dh
    ge, gh = sp.eye(ne) + ce, sp.eye(nh) + ch
    Wn = (sp.eye(nh) + t * gh * Lh / 2).inv() * (
        W + t * (gh * F * U + bh * q / h + k * V) - t * gh * Lh * W / 2)
    dh = F * U - Lh * (Wn + W) / 2
    qn = bh * q + h * ch * dh
    Un = (sp.eye(ne) + t * eu.inv() * ge * Le / 2).inv() * (
        U + t * eu.inv() * (ge * B * Wn + be * p / h)
        - t * eu.inv() * ge * Le * U / 2)
    de = B * Wn - Le * (Un + U) / 2
    pn = be * p + h * ce * de
    Vn = V - t * k * ev.inv() * Wn
    A = Un.col_join(Vn).col_join(Wn).col_join(pn[pe, :]).col_join(qn[ph, :])
    assert not A.atoms(sp.Float)
    # Verify the original coupled equations, independently of elimination.
    assert Wn - W - t * (dh + qn / h + k * V) == sp.zeros(nh, size)
    assert Un - U - t * eu.inv() * (de + pn / h) == sp.zeros(ne, size)
    return A, dict(Ubar=(Un + U) / 2, Wbar=(Wn + W) / 2, pn=pn, qn=qn,
                   De=De, Dh=Dh, dh=dh, de=de)


def yee_storage(o):
    K = o['Hh'] * o['F'].row_join(o['k'] * sp.eye(o['nh']))
    Me = sp.diag(o['eu'] * o['He'], o['ev'] * o['Hh'])
    return Me.row_join(o['t'] * K.T / 2).col_join(
        (o['t'] * K / 2).row_join(o['Hh']))


def exact_work(o, A, stages):
    """Full complex-state work identity as an exact real symmetric matrix."""
    P = yee_storage(o)
    padded = sp.diag(P, sp.zeros(A.rows - P.rows))
    Ubar, Wbar = stages['Ubar'], stages['Wbar']
    loss = o['t'] * (Ubar.T * stages['De'] * Ubar + Wbar.T * stages['Dh'] * Wbar)
    work = o['t'] / o['h'] * (
        Ubar.T * o['He'] * stages['pn'] + Wbar.T * o['Hh'] * stages['qn'])
    residual = A.T * padded * A - padded + 2 * loss - work - work.T
    assert residual == sp.zeros(A.rows)
    return loss


def bare_controls(o, tau):
    """Certify the interface itself and compare conforming central flux to B0."""
    A, stages = amplification(o, tau, pml=False)
    exact_work(o, A, stages)
    P = yee_storage(o)
    m = P.rows
    Af = A[:m, :m]
    _, diagonal = P.LDLdecomposition(hermitian=False)
    positive_pivots = list(diagonal.diagonal())
    assert all(v > 0 for v in positive_pivots)
    loss = o['t'] * (stages['Ubar'][:, :m].T * stages['De'] * stages['Ubar'][:, :m]
                      + stages['Wbar'][:, :m].T * stages['Dh'] * stages['Wbar'][:, :m])
    assert Af.T * P * Af - P + 2 * loss == sp.zeros(m)
    # Duplicated interface E traces are glued to the original B0 grid.
    cell = Cell(dz=.5, dy=.5, zhalf=1., period=2., layers=1, slab=False)
    coefficients = [(np.ones(n, dtype=float), np.ones(n), np.zeros(n)) for n in (7, 6)]
    b0 = Block(cell, material='vacuum', coefficients=coefficients, wavenumber=.5)
    b0.t, b0.h, b0.k = o['t'], o['h'], o['k']
    b0.a = np.array([1, 1, R(1, 4), 1, 1], dtype=object)
    b0.av = np.array([1, 1, R(1, 4), R(1, 4), 1, 1], dtype=object)
    for name in ('ke', 'kh', 'be', 'bh', 'ce', 'ch'):
        setattr(b0, name, np.array([R(v) for v in getattr(b0, name)], dtype=object))
    primary = sp.Matrix(b0.step(np.eye(b0.size, dtype=int).astype(object)).tolist())[:17, :17]
    glue_e = sp.zeros(o['ne'], 5)
    starts = (0, 1, 5)
    for row, (d, j) in enumerate(o['tags']):
        glue_e[row, starts[d] + j - 1] = 1
    glue = sp.diag(glue_e, sp.eye(6), sp.eye(6))
    central, _ = amplification(o, R(0), pml=False)
    assert central[:m, :m] * glue == glue * primary
    return dict(positive_storage_LDL_pivots=[str(v) for v in positive_pivots],
                exact_nonpositive_work_identity=True,
                exact_central_conforming_primary_B0_replay=True,
                field_states=m,
                numerical_rho=float(max(abs(linalg.eigvals(np.array(Af, float))))))


def routh_certificate(polynomial, z):
    """Exact bilinear disk-to-half-plane transform and Routh sign count."""
    s = sp.Symbol('s')
    degree = sp.degree(polynomial, z)
    transformed = sp.Poly(sp.cancel((1 - s) ** degree * polynomial.subs(z, (1 + s) / (1 - s))), s)
    assert transformed.degree() == degree
    _, primitive = transformed.clear_denoms(convert=True)
    _, primitive = primitive.primitive()
    if primitive.LC() < 0:
        primitive = -primitive
    coefficients = primitive.all_coeffs()
    width = (len(coefficients) + 1) // 2
    table = [list(coefficients[::2]) + [sp.S.Zero] * (width - len(coefficients[::2])),
             list(coefficients[1::2]) + [sp.S.Zero] * (width - len(coefficients[1::2]))]
    for _ in range(len(coefficients) - 2):
        upper, lower = table[-2:]
        assert lower[0] != 0
        row = [sp.cancel((lower[0] * upper[j + 1] - upper[0] * lower[j + 1]) / lower[0])
               for j in range(width - 1)] + [sp.S.Zero]
        assert any(v != 0 for v in row)
        table.append(row)
    column = [row[0] for row in table]
    assert all(v != 0 for v in column)
    signs = [int(sp.sign(v)) for v in column]
    changes = sum(a != b for a, b in zip(signs, signs[1:]))
    return dict(degree=int(degree), bilinear_integer_coefficients=[str(v) for v in coefficients],
                routh_first_column=[str(v) for v in column], signs=signs,
                exact_outside_unit_disk_count=changes)


def case(o, name, tau):
    A, stages = amplification(o, tau)
    exact_work(o, A, stages)
    z = sp.Symbol('z')
    factors = sp.factor_list(A.charpoly(z).as_expr())[1]
    certificates = []
    static = 0
    for polynomial, multiplicity in factors:
        if polynomial == z - 1:
            static += multiplicity
            continue
        item = routh_certificate(polynomial, z)
        item.update(characteristic_integer_coefficients=[str(v) for v in sp.Poly(polynomial, z).all_coeffs()],
                    multiplicity=multiplicity)
        certificates.append(item)
    outside = sum(item['exact_outside_unit_disk_count'] * item['multiplicity'] for item in certificates)
    assert outside > 0
    Af = np.array(A, float)
    values, vectors = linalg.eig(Af)
    worst = np.argmax(abs(values))
    v, lam = vectors[:, worst], values[worst]
    return dict(name=name, tau=str(tau), state_count=A.rows, static_root_multiplicity=static,
                matrix=[[str(v) for v in row] for row in A.tolist()],
                characteristic_factors=certificates, exact_outside_unit_disk_count=outside,
                rho=float(abs(lam)), worst_eigenvalue=[float(lam.real), float(lam.imag)],
                eigen_residual=float(np.linalg.norm(Af @ v - lam * v)),
                eigenmode_storage_growth_factor=float(abs(lam) ** 2),
                exact_full_field_memory_work_identity=True,
                bare_interface_control=bare_controls(o, tau),
                forbids_positive_definite_quadratic_storage=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run with CUDA_VISIBLE_DEVICES empty'
    o = operators()
    assert o['He'] * o['B'] + o['F'].T * o['Hh'] == sp.zeros(o['ne'], o['nh'])
    report = dict(date='2026-11-03', decision='NO-GO', stage='R0', cpu_only=True,
                  graded_proof_attempted=False, R1_run=False,
                  parameters=dict(dt='1', c='1/4', h='1/2', k_h='1/2', kappa='1',
                                  b='4/5', c_memory='-1/10', intervals=[1, 4, 1],
                                  electric_node_epsilon=[1, 1, 1, 4, 1, 1, 1],
                                  electric_half_epsilon=[1, 1, 4, 4, 1, 1]),
                  state_order=['u:7', 'v:6', 'w:6', 'p:2', 'q:2'],
                  interface_node_tags=o['tags'],
                  exact_sbp_identity=True,
                  scope='Two specified fluxes for the B0 normal-memory PML; no family-wide impossibility claim',
                  cases=[case(o, 'central', R(0)), case(o, 'impedance_upwind', R(1, 2))])
    report['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ('sbp_sat_r0.py', 'model.py')}
    target = ROOT / 'results' / 'sbp_sat_r0.json'
    encoded = json.dumps(report, indent=2) + '\n'
    assert len(encoded.encode()) < 200_000
    target.write_text(encoded)
    print(json.dumps(dict(decision=report['decision'], R1_run=False, result=str(target),
                          bytes=len(encoded.encode()), cases=[
                              {key: row[key] for key in ('name', 'state_count', 'rho',
                                                        'exact_outside_unit_disk_count', 'eigen_residual')}
                              for row in report['cases']]), indent=2))


if __name__ == '__main__':
    main()
