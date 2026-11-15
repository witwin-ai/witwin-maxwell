"""F-F reference: true-geometry sub-strip PEEC lattice of the octagonal differential inductor for FastHenry.

Geometry: tests/support/rfic_a4_fixtures.ff_layout (same parameters). Every conductor piece (octagonal half rings, the
inner ring, the top diagonal, the underpass) is split across its width into n graded sub-strips; sub-strip j of an
octagonal piece follows the regular octagon of its own apothem (exact mitred corners: its vertices lie on the corner
bisectors), sub-strip j of a diagonal with vertical end cuts follows the parallel line at its own vertical offset, so
the pieces meet sub-strip to sub-strip on the cut planes x = +-X. Transverse links join neighbouring sub-strips at
every station (corners, stations <= step apart, cut ends); a link's width is the centreline length it represents.
This is the cell-centred PEEC idea of the G3 lattice (scripts/dev/rfic/strip_reference_fasthenry_g3.py) on the
octagon's own frames. The port is the cut plane x = 0 of the outer ring (all sub-strips cut, both faces tied); vias are
4x4 um bars between the plane nodes of the central sub-strips. PEC (sigma 1e10 per mm), thickness h (zero-thickness
limit by extrapolation in h), one filament per segment. Units: '.units mm' with micrometre numbers and f/1000, so the
imaginary part of Z equals the physical one (avoids short-filament warnings).
Corner error of a sub-strip is O((w/n)^2) per corner -> the lattice converges like 1/n; use Richardson in n.

python octagon_reference_r7.py fh OUT.inp --n 16 --step 4 --h 0.1 [--ratio 1.3]
python octagon_reference_r7.py pair OUT.inp --n 16 --h 0.1 --length 400   (calibration pair, see pair())
python octagon_reference_r7.py parse DIR      (reads DIR/Zc.mat, prints L at the physical frequency)
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

FREQUENCY = 1e8


def offsets(width, n, ratio):
    """Sub-strip (centre, width) across [-width/2, width/2], geometric grading toward both edges."""
    g = [ratio ** k for k in range(n // 2)]
    sizes = g + ([ratio ** (n // 2)] if n % 2 else []) + g[::-1]
    total = sum(sizes)
    out, lo = [], -width / 2
    for size in sizes:
        out.append((lo + width * size / total / 2, width * size / total))
        lo += width * size / total
    return out


def octagon_corners(apothem, start, stop, clockwise):
    """Corner list of the octagon of ``apothem`` passed between the cut points ``start`` and ``stop``."""
    radius = apothem / math.cos(math.pi / 8)
    corners = [(radius * math.cos(math.pi / 8 + k * math.pi / 4), radius * math.sin(math.pi / 8 + k * math.pi / 4))
               for k in range(8)]

    def angle(p):
        return math.atan2(p[1], p[0]) % (2 * math.pi)

    a0, a1 = angle(start), angle(stop)
    sign = -1 if clockwise else 1
    span = ((a1 - a0) * sign) % (2 * math.pi)
    chosen = sorted((k for k in range(8) if ((angle(corners[k]) - a0) * sign) % (2 * math.pi) < span),
                    key=lambda k: ((angle(corners[k]) - a0) * sign) % (2 * math.pi))
    return [start] + [corners[k] for k in chosen] + [stop]


def subdivide(path, counts):
    points = [path[0]]
    for (p, q), pieces in zip(zip(path[:-1], path[1:]), counts):
        for i in range(1, pieces + 1):
            points.append((p[0] + (q[0] - p[0]) * i / pieces, p[1] + (q[1] - p[1]) * i / pieces))
    return points


def generate(n, step, h, ratio):
    from tests.support import rfic_a4_fixtures as fx
    r1, r2 = fx.FF_R
    w, x, zt, zb = fx.FF_W, fx.FF_X, fx.FF_Z[1], fx.FF_Z[0]
    subs = offsets(w, n, ratio)
    lines = [f"* F-F octagonal differential inductor lattice n={n} step={step} h={h} ratio={ratio}", ".units mm",
             f".default nwinc=1 nhinc=1 h={h} sigma=1e10"]
    nodes, elements = {}, []

    def node(p, z):
        name = f"N{len(nodes)}"
        nodes[name] = (p[0], p[1], z)
        return name

    def piece(stations, z):
        """stations: list of per-station lists of n sub-strip points; returns the node grid [station][j]."""
        grid = [[node(p, z) for p in row] for row in stations]
        for k in range(len(stations)):
            for j in range(n):
                if k + 1 < len(stations):
                    elements.append((grid[k][j], grid[k + 1][j], subs[j][1]))
            # Transverse links: width = centreline length represented by the station.
            c0 = [sum(p[i] for p in stations[k]) / n for i in (0, 1)]
            lengths = []
            for other in (k - 1, k + 1):
                if 0 <= other < len(stations):
                    c1 = [sum(p[i] for p in stations[other]) / n for i in (0, 1)]
                    lengths.append(math.dist(c0, c1) / 2)
            for j in range(n - 1):
                elements.append((grid[k][j], grid[k][j + 1], sum(lengths)))
        return grid

    def octagon_piece(apothem, start_x, start_side, stop_x, stop_side, clockwise, z):
        def path(a):
            return octagon_corners(a, (start_x, start_side * a), (stop_x, stop_side * a), clockwise)
        nominal = path(apothem)
        counts = [max(1, math.ceil(math.dist(p, q) / step)) for p, q in zip(nominal[:-1], nominal[1:])]
        rows = [subdivide(path(apothem + centre), counts) for centre, _ in subs]
        return piece([[rows[j][k] for j in range(n)] for k in range(len(rows[0]))], z)

    def straight_piece(points, z):
        """Polyline of (x, y_centre) with vertical cuts: sub-strip j at y + offset_j."""
        stations = []
        for (xa, ya), (xb, yb) in zip(points[:-1], points[1:]):
            pieces = max(1, math.ceil(math.dist((xa, ya), (xb, yb)) / step))
            for i in range(0 if not stations else 1, pieces + 1):
                xs, ys = xa + (xb - xa) * i / pieces, ya + (yb - ya) * i / pieces
                stations.append([(xs, ys + c) for c, _ in subs])
        return piece(stations, z)

    def station_nodes(grid, target):
        """Node row whose centre is closest to target (via landing)."""
        best = min(grid, key=lambda row: math.dist(nodes[row[n // 2]][:2], target))
        return best

    outer_left = octagon_piece(r1, 0., -1, -x, 1, True, zt)
    diagonal = straight_piece([(-x, r1), (x, r2)], zt)
    inner = octagon_piece(r2, x, 1, -x, 1, True, zt)
    outer_right = octagon_piece(r1, x, 1, 0., -1, True, zt)
    under = straight_piece([(-x - 6, r2), (-x, r2), (x, r1), (x + 6, r1)], zb)
    lines += [f"{k} x={v[0]:.12g} y={v[1]:.12g} z={v[2]:.12g}" for k, v in nodes.items()]
    # Joints on the cut planes x = +-X: identical points, tied node to node.
    equivs = []
    for a, b in ((outer_left[-1], diagonal[0]), (diagonal[-1], inner[0])):
        equivs += [f".equiv {p} {q}" for p, q in zip(a, b)]
    # Vias: bars from the central sub-strip node of the nearest station on each plane.
    for target, top_grid, bottom_grid in (((-x - 3, r2), inner, under), ((x + 3, r1), outer_right, under)):
        upper = station_nodes(top_grid, target)[n // 2]
        lower = station_nodes(bottom_grid, target)[n // 2]
        elements.append((upper, lower, fx.FF_VIA))
    for k, (u, v, width) in enumerate(elements):
        zu, zv = nodes[u][2], nodes[v][2]
        extra = f" h={fx.FF_VIA}" if zu != zv else ""
        if zu != zv:
            # A vertical bar: FastHenry needs the width direction; x.
            extra += " wx=1 wy=0 wz=0"
        lines.append(f"E{k} {u} {v} w={width:.12g}{extra}")
    lines += equivs
    plus, minus = outer_left[0], outer_right[-1]
    lines.append(".equiv " + " ".join(plus))
    lines.append(".equiv " + " ".join(minus))
    lines.append(f".external {plus[0]} {minus[0]} diff")
    lines.append(f".freq fmin={FREQUENCY / 1000:g} fmax={FREQUENCY / 1000:g} ndec=1")
    lines.append(".end")
    return "\n".join(lines) + "\n", len(elements)


def pair(n, h, length, ratio):
    """Calibration lattice: the F-F trace pair (w 8, gap 4 um) as two straight strips of n graded sub-strips, one segment
    per sub-strip over ``length``, far ends tied, port across the near ends. Length differencing gives the per-length
    L' whose exact zero-thickness value is the 2D SDA pair (quasistatic_sda_r7.references()["ff_pair"]); the error of
    the same (n, h -> 0) procedure calibrates the lattice discretization of the inductor."""
    from tests.support import rfic_a4_fixtures as fx
    w, gap = fx.FF_W, fx.FF_R[0] - fx.FF_R[1] - fx.FF_W
    subs = offsets(w, n, ratio)
    lines = [f"* F-F pair calibration n={n} h={h} length={length} ratio={ratio}", ".units mm",
             f".default nwinc=1 nhinc=1 h={h} sigma=1e10"]
    near, far, k = {"a": [], "b": []}, [], 0
    for strip, centre in (("a", (w + gap) / 2), ("b", -(w + gap) / 2)):
        for j, (c, width) in enumerate(subs):
            lines.append(f"N{strip}{j}s x=0 y={centre + c:.12g} z={fx.FF_Z[1]}")
            lines.append(f"N{strip}{j}e x={length:.12g} y={centre + c:.12g} z={fx.FF_Z[1]}")
            lines.append(f"E{k} N{strip}{j}s N{strip}{j}e w={width:.12g}")
            k += 1
            near[strip].append(f"N{strip}{j}s")
            far.append(f"N{strip}{j}e")
    lines += [".equiv " + " ".join(near["a"]), ".equiv " + " ".join(near["b"]), ".equiv " + " ".join(far),
              f".external {near['a'][0]} {near['b'][0]} pair", f".freq fmin={FREQUENCY / 1000:g} fmax={FREQUENCY / 1000:g} ndec=1",
              ".end"]
    return "\n".join(lines) + "\n", k


def parse(directory):
    text = (Path(directory) / "Zc.mat").read_text().split("\n")
    for i, line in enumerate(text):
        if line.startswith("Impedance matrix for frequency"):
            frequency = float(line.split("=")[1].split()[0]) * 1000
            value = text[i + 1].split()
            real, imag = float(value[0]), float(value[1].rstrip("j"))
            return dict(frequency_Hz=frequency, L_H=imag / (2 * math.pi * frequency), R_ohm=real)
    raise ValueError("no impedance matrix")


def mom_record(edge, maximum):
    import torch
    from witwin.maxwell.mom.layered import extraction as ex
    from tests.support import rfic_a4_fixtures as fx
    options = fx.ff_options(edge) if maximum is None else fx.ff_options(edge, maximum)
    frequencies = (1e8, 1e9)
    result = fx.simulate(fx.octagonal_inductor(), frequencies, options)
    network = result.network
    repeat = torch.equal(fx.simulate(fx.octagonal_inductor(), frequencies, options).network.s, network.s)
    y = network.to_y()
    return dict(fixture="F-F", edge_m=edge, max_edge_m=options["max_edge_length_m"], frequency_Hz=list(frequencies),
                unknowns=network.metadata["unknowns"], mesher=network.metadata.get("planar_mesher"),
                L_H=ex.inductance(network, "diff", others="open").tolist(),
                R_ohm=ex.resistance(network, "diff", others="open").tolist(),
                max_singular_S=torch.linalg.matrix_norm(network.s, ord=2).max().item(),
                y_real=y.real.flatten().tolist(), y_imag=y.imag.flatten().tolist(), exact_repeat=repeat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("fh", "pair", "parse"))
    parser.add_argument("path")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--step", type=float, default=4.)
    parser.add_argument("--h", type=float, default=0.1)
    parser.add_argument("--ratio", type=float, default=1.3)
    parser.add_argument("--length", type=float, default=400.)
    args = parser.parse_args()
    if args.mode == "fh":
        text, count = generate(args.n, args.step, args.h, args.ratio)
        Path(args.path).write_text(text)
        print(json.dumps(dict(segments=count, n=args.n, step=args.step, h=args.h, ratio=args.ratio)))
    elif args.mode == "pair":
        text, count = pair(args.n, args.h, args.length, args.ratio)
        Path(args.path).write_text(text)
        print(json.dumps(dict(segments=count, n=args.n, h=args.h, length=args.length, ratio=args.ratio)))
    else:
        print(json.dumps(parse(args.path)))
