"""FastHenry inputs for the strip loop of tests/support/rfic_closed_references.strip_fixture.

centerline: the A1c model (one segment per run piece, centerline corners), delta gap at x=100 on the bottom run.
lattice:    true-geometry cell-centred PEEC. The closed ring x[-10,210] y[-115,5] minus the hole x(0,200) y(-105,-5)
            is split into tensor-grid cells graded towards every boundary line; x/y segments join adjacent cell
            centres (width = cell size across, height = thickness with nhinc graded filaments). Corners are exact.
            The delta gap is the plane x=100 on the bottom run: every row is cut, both cut faces tied, as the
            MoM ring gap. Copper (sigma 58 S/um) occupies x[0,200] y[-5,5] unless --pec; PEC is 1e10 S/um.
--length X moves the right leg / gap (loop-length scaling). Units um (or --mm, see the flag).
G3 finding: the centerline model over-estimates the loop L by ~1.2% on the 10x2 um strip (corner overlap / missing
outer quarter, ~4.8 pH localized); the lattice model is the reference. Run FastHenry with -m direct -t 1e-10
(<= ~23k filaments) or -m multi -o 6 -t 1e-8; the default multipole order 2 / tol 1e-3 is not a reference.
"""
import argparse, math

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--mode", choices=("centerline", "lattice"), default="lattice")
p.add_argument("--h", type=float, default=2.)
p.add_argument("--pec", action="store_true")
p.add_argument("--edge", type=float, default=0.2)
p.add_argument("--growth", type=float, default=1.6)
p.add_argument("--hmax", type=float, default=8.)
p.add_argument("--nhinc", type=int, default=5)
p.add_argument("--rh", type=float, default=2.)
p.add_argument("--nwinc", type=int, default=25)
p.add_argument("--freqs", nargs=2, default=("1e9", "1e9"))
p.add_argument("--ndec", default="1")
p.add_argument("--length", type=float, default=200.)
p.add_argument("--mm", action="store_true", help="write .units mm (geometry x1000) and freq/1000: Z(f/1000) equals the physical Z(f)")
a = p.parse_args()
CU, PEC = (1e10 if a.pec else 58.), 1e10
lines = [f"* strip loop {a.mode} h={a.h} pec={a.pec} mm={a.mm}", ".units mm" if a.mm else ".units um"]


def graded(lo, hi):
    """Cell boundaries on [lo, hi] mirrored-geometric from both ends (edge, growth, capped at hmax)."""
    half, steps, s = (hi - lo) / 2, [], a.edge
    while sum(steps) + s <= half:
        steps.append(s)
        s = min(s * a.growth, a.hmax)
    rest = half - sum(steps)
    if steps and rest < .5 * steps[-1]:
        steps[-1] += rest
    else:
        steps.append(rest)
    left = [lo]
    for d in steps:
        left.append(left[-1] + d)
    right = [hi - (v - lo) for v in left]
    return left[:-1] + [ (lo + hi) / 2 ] + right[-2::-1]


if a.mode == "centerline":
    lines.append(f".default nwinc={a.nwinc} nhinc={a.nhinc} rw=2 rh={a.rh} h={a.h} w=10")
    X = a.length
    pts = [(-5, 0, PEC), (0, 0, PEC)] + [(X * k / 10, 0, CU) for k in range(1, 11)] + [(X + 5, 0, PEC)]
    pts += [(X + 5, -110 * k / 6, PEC) for k in range(1, 7)]
    right = [X + 5 - (X / 2) * k / 5 for k in range(1, 6)]
    pts += [(x, -110, PEC) for x in right] + [(X / 2, -110, PEC)]
    left = [X / 2 - 5 - (X / 2) * k / 5 for k in range(0, 6)]
    seq2 = [(X / 2, -110, None)] + [(x, -110, PEC) for x in left]
    seq2 += [(-5, -110 + 110 * k / 6, PEC) for k in range(1, 7)]
    names = []
    for k, (x, y, _) in enumerate(pts + seq2):
        lines.append(f"N{k} x={x} y={y} z=100")
        names.append(f"N{k}")
    elems = []
    n1 = len(pts)
    for i in range(1, n1):
        elems.append((names[i - 1], names[i], pts[i][2]))
    for i in range(1, len(seq2)):
        elems.append((names[n1 + i - 1], names[n1 + i], seq2[i][2]))
    for i, (u, v, s) in enumerate(elems):
        lines.append(f"E{i} {u} {v} sigma={s}")
    lines.append(f".equiv {names[-1]} {names[0]}")
    lines.append(f".external {names[n1 - 1]} {names[n1]} p")
else:
    lines.append(f".default nwinc=1 nhinc={a.nhinc} rh={a.rh} h={a.h}")
    xb = sorted(set(graded(-10, 0) + graded(0, a.length / 2) + graded(a.length / 2, a.length) + graded(a.length, a.length + 10)))
    yb = sorted(set(graded(-115, -105) + graded(-105, -5) + graded(-5, 5)))

    def inside(x, y):
        return -10 < x < a.length + 10 and -115 < y < 5 and not (0 < x < a.length and -105 < y < -5)

    def sigma(x, y):
        return CU if (0 < x < a.length and -5 < y < 5) else PEC

    xc = [(u + v) / 2 for u, v in zip(xb, xb[1:])]
    yc = [(u + v) / 2 for u, v in zip(yb, yb[1:])]
    dx = [v - u for u, v in zip(xb, xb[1:])]
    dy = [v - u for u, v in zip(yb, yb[1:])]
    cell = {}
    for i, x in enumerate(xc):
        for j, y in enumerate(yc):
            if inside(x, y):
                cell[i, j] = f"N{len(cell)}"
                lines.append(f"{cell[i, j]} x={x:.12g} y={y:.12g} z=100")
    gap = xb.index(a.length / 2)
    elems, plus, minus = [], [], []

    def series(s1, l1, s2, l2):
        return (l1 + l2) / (l1 / s1 + l2 / s2)

    for (i, j), n in cell.items():
        if (i + 1, j) in cell:
            if i + 1 == gap and yc[j] < -100:
                # Cut face pair on the plane x=100: left cell -> P_j, M_j -> right cell.
                pn, mn = f"NP{j}", f"NM{j}"
                lines.append(f"{pn} x={a.length / 2:.12g} y={yc[j]:.12g} z=100")
                lines.append(f"{mn} x={a.length / 2:.12g} y={yc[j]:.12g} z=100")
                elems.append((n, pn, dy[j], sigma(xc[i], yc[j])))
                elems.append((mn, cell[i + 1, j], dy[j], sigma(xc[i + 1], yc[j])))
                plus.append(pn)
                minus.append(mn)
            else:
                elems.append((n, cell[i + 1, j], dy[j],
                              series(sigma(xc[i], yc[j]), dx[i], sigma(xc[i + 1], yc[j]), dx[i + 1])))
        if (i, j + 1) in cell:
            elems.append((n, cell[i, j + 1], dx[i],
                          series(sigma(xc[i], yc[j]), dy[j], sigma(xc[i], yc[j + 1]), dy[j + 1])))
    for k, (u, v, w, s) in enumerate(elems):
        lines.append(f"E{k} {u} {v} w={w:.12g} sigma={s:.12g}")
    lines.append(".equiv " + " ".join(plus))
    lines.append(".equiv " + " ".join(minus))
    lines.append(f".external {minus[0]} {plus[0]} p")
    print("cells", len(cell), "segments", len(elems), "filaments", len(elems) * a.nhinc)
scale = 1e-3 if a.mm else 1.
lines.append(f".freq fmin={float(a.freqs[0])*scale:g} fmax={float(a.freqs[1])*scale:g} ndec={a.ndec}")
lines.append(".end")
open(a.out, "w").write("\n".join(lines) + "\n")
