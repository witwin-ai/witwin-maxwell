"""A4 acceptance rows from the committed evidence (tests/mom/layered/data/fixture_f{d,e,f}_*.json).

Every row: quantity, MoM finest value, Richardson value, reference, u_ref (reference convergence + validity bound),
u_mesh (1.25 |q_ext - q_finest|), relative errors and the two gates:
  (A) |q_ext / ref - 1| <= u_ref + u_mesh  (derived tolerance; solver converges to the independent reference);
  (B) |q_finest / ref - 1| <= target       (plan-level accuracy at the finest locked mesh).
"""
import math

from tests.support.rfic_a4_fixtures import richardson

C0 = 299792458.0


def row(quantity, sequence, ratio, reference, u_ref, target):
    estimate = richardson(sequence, ratio)
    u_mesh = estimate["uncertainty"] / abs(reference)
    error_ext = estimate["value"] / reference - 1
    error_fine = sequence[-1] / reference - 1
    return dict(quantity=quantity, mom=sequence, mom_extrapolated=estimate["value"], order=estimate["order"],
                reference=reference, u_ref=u_ref, u_mesh=u_mesh, tolerance=u_ref + u_mesh, error_extrapolated=error_ext,
                error_finest=error_fine, target=target,
                pass_derived=abs(error_ext) <= u_ref + u_mesh, pass_target=abs(error_fine) <= target,
                **{"pass": abs(error_ext) <= u_ref + u_mesh and abs(error_fine) <= target})


def sequence(data, prefix):
    labels = data["sequences"][prefix]
    return labels, [data["mom"][label] for label in labels]


def quasi_static_eps(record, pair):
    """eps_eff(f) = e0 + a f^2 through the two named frequencies (dispersion is even in f for a bounded line)."""
    frequencies = record["frequency_Hz"][1:]
    (fa, ea), (fb, eb) = [(f, record["eps_eff"][frequencies.index(f)]) for f in pair]
    slope = (eb - ea) / (fb * fb - fa * fa)
    return ea - slope * fa * fa


def composed(quantity, parts, value, reference, u_ref, u_mesh, finest, target):
    """A row whose MoM value is composed from Richardson limits of other rows (u_mesh propagated linearly)."""
    error_ext, error_fine = value / reference - 1, finest / reference - 1
    return dict(quantity=quantity, composed_from=parts, mom_extrapolated=value, mom_finest=finest, reference=reference,
                u_ref=u_ref, u_mesh=u_mesh, tolerance=u_ref + u_mesh, error_extrapolated=error_ext,
                error_finest=error_fine, target=target, pass_derived=abs(error_ext) <= u_ref + u_mesh,
                pass_target=abs(error_fine) <= target, **{"pass": abs(error_ext) <= u_ref + u_mesh and
                                                          abs(error_fine) <= target})


def fe(data):
    """C' and L' = eps_eff / (c^2 C') converge with different orders (charge vs edge current), so each is extrapolated
    on its own; eps_eff = c^2 L' C' and Z0 = sqrt(L'/C') are composed from the two limits."""
    rows = []
    for kind in ("microstrip", "cpw"):
        reference = data["reference"]["values"]["fe_" + kind]
        pairs = data["reference"]["dispersion_pairs"][kind]
        _, records = sequence(data, kind)
        eps0 = [quasi_static_eps(r, pairs[0]) for r in records]
        # Dispersion-model bound: the f^2 limit from the next frequency pair, on the finest mesh.
        u_disp = abs(quasi_static_eps(records[-1], pairs[1]) / eps0[-1] - 1)
        low = data["reference"]["low_frequency_bound"][kind]
        c_mom = [r["C_per_length_F_per_m"] for r in records]
        l_mom = [e / (C0 * C0 * c) for e, c in zip(eps0, c_mom)]
        l_ref = 1 / (C0 * C0 * reference["C_air_F_per_m"])
        c_row = row(f"{kind} C' [F/m]", c_mom, 2., reference["C_F_per_m"], 1e-9 + low, 1e-2)
        l_row = row(f"{kind} L' [H/m]", l_mom, 2., l_ref, 1e-9 + low + u_disp, 1e-2)
        rows += [c_row, l_row]
        mesh = l_row["u_mesh"] + c_row["u_mesh"]
        u_ref = c_row["u_ref"] + l_row["u_ref"]
        rows.append(composed(f"{kind} eps_eff (f->0)", ("L'", "C'"), C0 * C0 * l_row["mom_extrapolated"] *
                             c_row["mom_extrapolated"], reference["eps_eff"], u_ref, mesh, eps0[-1], 5e-3))
        rows.append(composed(f"{kind} Z0 [ohm]", ("L'", "C'"), math.sqrt(l_row["mom_extrapolated"] /
                             c_row["mom_extrapolated"]), reference["Z0_ohm"], u_ref / 2, mesh / 2,
                             math.sqrt(l_mom[-1] / c_mom[-1]), 1e-2))
    return rows


def fd(data):
    rows = []
    reference = data["reference"]["values"]["fd"]
    _, records = sequence(data, "mim")
    for width, values in reference["widths"].items():
        if width not in records[0]["C_per_length_F_per_m"]:
            continue
        for j, frequency in enumerate(records[0]["frequency_Hz"]):
            mom = [r["C_per_length_F_per_m"][width][j] for r in records]
            u_ref = values["order_spread_rel"] + data["reference"]["quasi_static_bound"][j]
            rows.append(row(f"C' {width} at {frequency:.0e} Hz [F/m]", mom, 2., values["C_per_length_F_per_m"], u_ref,
                            5e-3))
    # Square plates: eps A/d + 4 s f(s); the 3D edge/corner field is bounded by the width dependence of the 2D
    # fringe: u = 4 s |f(s) - f(s/2)| / C_ref (the port and corner terms are inside this bound).
    area = reference["area_capacitance_F_per_m2"]
    for case_index, case in enumerate(records[0]["cases"]):
        if case["width_m"] != case["length_m"]:
            continue
        s = case["width_m"]
        f_s = reference["widths"][f"{s * 1e6:g}um"]["fringe_per_edge_F_per_m"]
        f_half = reference["widths"][f"{s * 5e5:g}um"]["fringe_per_edge_F_per_m"]
        ref = area * s * s + 4 * s * f_s
        u_ref = 4 * s * abs(f_s - f_half) / ref
        mom = [r["cases"][case_index]["C_F"][1] for r in records]
        rows.append(row(f"square {s * 1e6:g} um C at 1e9 Hz [F]", mom, 2., ref, u_ref, 1e-2))
    return rows


def ff(data):
    reference = data["reference"]["values"]
    _, records = sequence(data, "inductor")
    mom = [r["L_H"][0] for r in records]
    return [row("differential L at 1e8 Hz [H]", mom, 2 ** .5, reference["L_H"], reference["u_rel"], 1e-2)]


ACCEPTANCE = dict(fd=fd, fe=fe, ff=ff)
