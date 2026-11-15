"""E1 acceptance: PEC microstrip on a wideband-Debye substrate, 0.1-100 GHz.

Two lines (L1, L2) with identical ports; the port error boxes cancel in M2 M1^-1,
whose trace is 2 cosh(gamma dL). ABCD uses the native current-derived admittance.
* 0.1-20 GHz: vertical delta-gap ports from the strip ends to the PEC ground. Their
  vertical z table is refused by admission at 60/100 GHz for this stack and mesh (also
  for constant eps), so they stop at 20 GHz.
* 20-100 GHz: horizontal series delta-gap ports to open end pads (pad-gap-line-gap-pad).
  Pad-to-pad and surface-wave coupling of these ports is not part of the line model and
  shows up as a spurious attenuation; it is measured with the lossless frozen stack
  eps = Re eps(f) (same mesh, same frequency) and subtracted: alpha_d = alpha - alpha_0.
  The vertical-port band reports raw alpha. (Series ports are unusable below ~10 GHz:
  their pad capacitance and nonlocal coupling swamp (gamma dL)^2.)
With PEC metal and a laterally open stack the line attenuation is dielectric loss only.
References:
* eps_eff: Hammerstad-Jensen static value with Kirschning-Jansen dispersion at Re eps(f);
* alpha_d = k0 eps_r (eps_eff - 1) tan_delta / (2 sqrt(eps_eff) (eps_r - 1)) [Np/m],
  with eps_r = Re eps(f), tan_delta = Im eps(f)/Re eps(f) and the extracted eps_eff.

Usage: python scripts/dev/rfic/e1_dispersive_microstrip.py <output.json>
"""

import json
import math
import sys

import torch

import witwin.maxwell as mw
from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered.network import prepare_layered_network, solve_prepared_network
from tests.support.coupled_line_references import hammerstad_jensen_microstrip

C0 = 299792458.0
HEIGHT = WIDTH = 100e-6
LENGTHS = (1.0e-3, 1.5e-3)
LOW_BAND = (1e8, 3e8, 1e9, 3e9, 1e10, 2e10)
HIGH_BAND = (2e10, 3e10, 6e10, 1e11)


def kirschning_jansen(u, eps_r, eps_static, frequency, height):
    """Kirschning-Jansen microstrip dispersion (0.6 % for u 0.1-100, eps_r <= 20, h/lambda0 <= 0.13)."""
    fn = frequency * 1e-9 * height * 1e3
    p1 = 0.27488 + (0.6315 + 0.525 / (1 + 0.0157 * fn) ** 20) * u - 0.065683 * math.exp(-8.7513 * u)
    p2 = 0.33622 * (1 - math.exp(-0.03442 * eps_r))
    p3 = 0.0363 * math.exp(-4.6 * u) * (1 - math.exp(-(fn / 38.7) ** 4.97))
    p4 = 1 + 2.751 * (1 - math.exp(-(eps_r / 15.916) ** 8))
    p = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
    return eps_r - (eps_r - eps_static) / (1 + p)


def abcd(y):
    """ABCD from the port admittance (NetworkData stores S, which loses digits for |Z| >> z0)."""
    y11, y12, y21, y22 = y[..., 0, 0], y[..., 0, 1], y[..., 1, 0], y[..., 1, 1]
    return -torch.stack((torch.stack((y22, torch.ones_like(y11)), -1),
                         torch.stack((y11 * y22 - y12 * y21, y11), -1)), -2) / y21[..., None, None]


def vertical_ports(length):
    rectangles = [(0., length, -WIDTH / 2, WIDTH / 2, HEIGHT)]
    ports = [mw.LumpedPort(name, positive=(x, 0., HEIGHT), negative=(x, 0., 0.), voltage_path=mw.AxisPath("z"),
                           reference_impedance=50.) for name, x in (("p1", 0.), ("p2", length))]
    return rectangles, ports


def series_ports(length):
    gap, pad = WIDTH / 4, 2 * WIDTH
    rectangles = [(-gap - pad, -gap, -WIDTH / 2, WIDTH / 2, HEIGHT), (0., length, -WIDTH / 2, WIDTH / 2, HEIGHT),
                  (length + gap, length + gap + pad, -WIDTH / 2, WIDTH / 2, HEIGHT)]
    ports = [mw.LumpedPort(name, positive=(x, 0., HEIGHT), negative=(x + side * gap, 0., HEIGHT),
                           voltage_path=mw.AxisPath("x"), reference_impedance=50.)
             for name, x, side in (("p1", 0., -1), ("p2", length, 1))]
    return rectangles, ports


def gamma_dl(eps, layout, frequencies):
    """gamma*dL per frequency of the two-line pair on a substrate with permittivity ``eps``."""
    stack = LayerStack((Layer("substrate", HEIGHT, eps),), "pec", HalfSpace())
    matrices = []
    for length in LENGTHS:
        rectangles, ports = layout(length)
        prepared = prepare_layered_network(stack, rectangles, ports, frequencies,
                                           bridge_width=WIDTH, max_edge=HEIGHT / 2, edge_cell=WIDTH / 16)
        network, currents = solve_prepared_network(prepared)
        # Native current-derived admittance: I_p = sum_n s_n l_n x_n (network.py sign chain).
        y = torch.einsum("np,fnq->fpq", prepared.excitation, currents)
        check = (torch.linalg.matrix_norm(y - network.to_y()) / torch.linalg.matrix_norm(y)).max().item()
        print(f"{layout.__name__} L={length:g}: unknowns {prepared.layout.basis.n_edges}, "
              f"|Y_currents - Y_S|/|Y| <= {check:.2e}", flush=True)
        matrices.append(abcd(y))
    difference = matrices[1] @ torch.linalg.inv(matrices[0])
    return torch.arccosh(torch.diagonal(difference, dim1=-2, dim2=-1).sum(-1) / 2).cpu().tolist()


def extract(model, layout, frequencies, *, subtract_lossless):
    gammas = gamma_dl(model, layout, frequencies)
    delta = LENGTHS[1] - LENGTHS[0]
    rows = []
    for f, g in zip(frequencies, gammas):
        eps = model.relative_permittivity(f)
        eps_r, tan_delta = eps.real, eps.imag / eps.real
        k0 = 2 * math.pi * f / C0
        beta, alpha = abs(g.imag) / delta, abs(g.real) / delta
        alpha_0 = abs(gamma_dl(eps_r, layout, (f,))[0].real) / delta if subtract_lossless else 0.0
        eps_eff = (beta / k0) ** 2
        static, _ = hammerstad_jensen_microstrip(WIDTH / HEIGHT, eps_r)
        eps_ref = kirschning_jansen(WIDTH / HEIGHT, eps_r, static, f, HEIGHT)
        alpha_ref = k0 * eps_r * (eps_eff - 1) * tan_delta / (2 * math.sqrt(eps_eff) * (eps_r - 1))
        alpha_d = alpha - alpha_0
        rows.append(dict(frequency_Hz=f, eps_r=eps_r, tan_delta=tan_delta, eps_eff=eps_eff,
                         eps_eff_reference=eps_ref, eps_eff_relative=eps_eff / eps_ref - 1,
                         alpha_Np_per_m=alpha, alpha_lossless_Np_per_m=alpha_0, alpha_d_Np_per_m=alpha_d,
                         alpha_d_reference=alpha_ref, alpha_d_relative=alpha_d / alpha_ref - 1,
                         beta_dL=abs(g.imag)))
        print(f"{layout.__name__:15s} {f:9.3g} eps_r={eps_r:.4f} tand={tan_delta:.5f} eps_eff={eps_eff:.4f} "
              f"ref={eps_ref:.4f} ({rows[-1]['eps_eff_relative']:+.2%}) alpha={alpha:.5g} alpha0={alpha_0:.3g} "
              f"alpha_d={alpha_d:.5g} ref={alpha_ref:.5g} ({rows[-1]['alpha_d_relative']:+.2%}) "
              f"beta*dL={abs(g.imag):.3g}", flush=True)
    return rows


def main(output):
    model = mw.media.WidebandDebye.from_loss_tangent(4.0, 0.02, 1e9)
    results = dict(model=dict(eps_inf=model.eps_inf, delta_eps=model.delta_eps, f1=model.f1, f2=model.f2),
                   height=HEIGHT, width=WIDTH, lengths=LENGTHS,
                   vertical_ports=extract(model, vertical_ports, LOW_BAND, subtract_lossless=False),
                   series_ports=extract(model, series_ports, HIGH_BAND, subtract_lossless=True))
    with open(output, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
