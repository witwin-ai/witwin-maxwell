"""Fixed-geometry thin-PEC transmission lines for propagation qualification.

Homogeneous stripline and CPW have exact TEM phase/loss and conformal impedance
references. Microstrip and coupled microstrip retain an air/dielectric interface.
All conductor widths, gaps and layer heights are SI dimensions independent of dx.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.special import ellipk
import witwin.maxwell as mw

C0 = 299792458.
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
WIDTH = .010
HEIGHT = .005
GAP = .005
PAIR_WIDTH = .005
PAIR_OFFSET = (PAIR_WIDTH+GAP)/2
KINDS = ("microstrip", "stripline", "cpw", "differential_pair")


def analytic_homogeneous_line(kind, frequencies, *, eps_r=2.25, sigma_e=0.):
    """Exact TEM propagation; zero-thickness conformal impedance for wide grounds.

    Stripline: Cohn mapping k=tanh(pi*w/(2*b)), b=2*HEIGHT.
    CPW: k=WIDTH/(WIDTH+2*GAP). scipy.ellipk takes the squared modulus.
    Finite side/outer boundaries require their own padding convergence.
    """
    if kind == "stripline":
        k = math.tanh(math.pi*WIDTH/(4*HEIGHT))
    elif kind == "cpw":
        k = WIDTH/(WIDTH+2*GAP)
    else:
        raise ValueError("Exact homogeneous reference is available for stripline and cpw.")
    ratio = float(ellipk(1-k*k)/ellipk(k*k))/4
    omega = 2*np.pi*np.asarray(frequencies, dtype=np.float64)
    gamma = np.sqrt(1j*omega*MU0*(sigma_e+1j*omega*EPS0*eps_r))
    # Maxwell modal phasors use exp(-i omega t): lossy characteristic Z has
    # negative imaginary part; alpha/beta below remain positive magnitudes.
    z0 = np.conj(ratio*np.sqrt(1j*omega*MU0/(sigma_e+1j*omega*EPS0*eps_r)))
    return {"alpha_np_per_m": gamma.real, "beta_rad_per_m": gamma.imag,
            "characteristic_impedance_ohm": z0}


def _contour(x, center_y, half_y, half_z):
    # Float64 integration geometry preserves its physical half-grid coordinates.
    import torch
    return mw.Box(position=torch.tensor((x, center_y, 0.), dtype=torch.float64),
                  size=torch.tensor((0., 2*half_y, 2*half_z), dtype=torch.float64))


def _port(kind, name, x, direction, dx, half_y, half_z, longitudinal_dl, metal="pec"):
    # V integrates E from the positive conductor toward its negative reference.
    # Contours use the right-hand circulation around forward positive current.
    current_x = x + (.5*longitudinal_dl if direction == "+" else -.5*longitudinal_dl)
    contour_y = (math.ceil(WIDTH/(2*dx))+.5)*dx
    contour_z = 1.5*dx
    model = {"conductor_model": "good_conductor"} if metal == "copper" else {}
    if kind == "cpw":
        path = ((x, 0., 0.), (x, WIDTH/2+GAP, 0.))
        modes = (mw.WaveModeSpec("tem", polarization="Ey", voltage_path=path, **model,
                    current_contour=_contour(current_x, 0., contour_y, contour_z)),)
    elif kind == "differential_pair":
        common_half_y = (math.ceil((PAIR_OFFSET+PAIR_WIDTH/2)/dx)+.5)*dx
        odd_half_y = (math.ceil(PAIR_WIDTH/(2*dx))+.5)*dx
        modes = (
            mw.WaveModeSpec("tem", mode_index=0, name="common", polarization="Ez", **model,
                voltage_path=((x, PAIR_OFFSET, 0.), (x, PAIR_OFFSET, -HEIGHT)),
                current_contour=_contour(current_x, 0., common_half_y, contour_z)),
            mw.WaveModeSpec("tem", mode_index=1, name="differential", polarization="Ey", **model,
                voltage_path=((x, PAIR_OFFSET, 0.), (x, -PAIR_OFFSET, 0.)),
                current_contour=_contour(current_x, PAIR_OFFSET, odd_half_y, contour_z)),
        )
    else:
        modes = (mw.WaveModeSpec("tem", polarization="Ez", **model,
                    voltage_path=((x, 0., 0.), (x, 0., -HEIGHT)),
                    current_contour=_contour(current_x, 0., contour_y, contour_z)),)
    return mw.WavePort(name, position=(x, 0., 0.), size=(0., 2*half_y, 2*half_z),
                       direction=direction, reference_plane=x, modes=modes)


def printed_line_scene(kind, *, dx=.0025, port_separation=.040, eps_r=2.25,
                       sigma_e=0., pml_thickness=.020, transverse_padding=0., device="cuda",
                       longitudinal_dl=None, pair_perturbation=False, axial_half_length=.060, metal="pec", transverse_refinement_radius=None, copper_frequency_range=None):
    """Build a two-port planar line (four modal channels for a differential pair).

    Port reference planes are fixed physical x coordinates. Conductors/material
    span the entire normal-axis PML; its physical thickness stays fixed under
    refinement. Homogeneous CPW/stripline use the same dielectric on both sides.
    """
    if metal not in ("pec", "copper"):
        raise ValueError("metal must be 'pec' or 'copper'.")
    if metal == "copper" and pair_perturbation:
        raise NotImplementedError("The overlapping paddle requires a single nonoverlapping Cu support.")
    if pair_perturbation and kind != "differential_pair":
        raise ValueError("pair_perturbation applies only to differential_pair.")
    if kind not in KINDS:
        raise ValueError(f"Printed line kind must be one of {KINDS}.")
    if dx <= 0 or not 0 < port_separation < 2*axial_half_length or eps_r <= 0 or sigma_e < 0:
        raise ValueError("Printed-line steps, separation and permittivity must be positive; sigma_e is nonnegative.")
    longitudinal_dl = dx if longitudinal_dl is None else float(longitudinal_dl)
    if longitudinal_dl < dx:
        raise ValueError("longitudinal_dl must be at least the transverse dx.")
    half_x, half_y, half_z = float(axial_half_length), .060+transverse_padding, .050+transverse_padding
    if transverse_padding < 0 or pml_thickness <= 0:
        raise ValueError("Printed-line padding is nonnegative and PML thickness must be positive.")
    nodes = [np.arange(-round(half/dx), round(half/dx)+1, dtype=np.float64)*dx
             for half in (half_x, half_y, half_z)]
    if longitudinal_dl != dx:
        nodes[0] = np.arange(-round(half_x/longitudinal_dl), round(half_x/longitudinal_dl)+1, dtype=np.float64)*longitudinal_dl
        # Coarse outer endpoint cells keep every PML at the same physical width.
        # Transverse refinement remains uniform throughout the mode aperture.
        for index, half in ((1, half_y), (2, half_z)):
            interior = nodes[index][(nodes[index] > -half+longitudinal_dl+1e-12)
                                    & (nodes[index] < half-longitudinal_dl-1e-12)]
            nodes[index] = np.r_[-half, -half+longitudinal_dl, interior, half-longitudinal_dl, half]
    if transverse_refinement_radius is not None:
        radius=float(transverse_refinement_radius)
        if not math.isfinite(radius) or radius < .020 or radius >= min(half_y,half_z)-longitudinal_dl:
            raise ValueError("transverse_refinement_radius must cover at least 20 mm and leave coarse outer cells.")
        if any(not math.isclose(radius/step,round(radius/step),rel_tol=0.,abs_tol=1e-9) for step in (dx,longitudinal_dl)):
            raise ValueError("transverse_refinement_radius must align with both transverse and longitudinal steps.")
        fine=np.arange(-round(radius/dx),round(radius/dx)+1,dtype=np.float64)*dx
        for index,half in ((1,half_y),(2,half_z)):
            if not math.isclose(half/longitudinal_dl,round(half/longitudinal_dl),rel_tol=0.,abs_tol=1e-9):
                raise ValueError("Graded transverse domain bounds must align with longitudinal_dl.")
            outer=np.arange(-round(half/longitudinal_dl),round(half/longitudinal_dl)+1,dtype=np.float64)*longitudinal_dl
            nodes[index]=np.r_[outer[outer < -radius-1e-12],fine,outer[outer > radius+1e-12]]
    layers = max(1, math.ceil(pml_thickness/longitudinal_dl-1e-12))
    extended_x, extended_y = half_x+pml_thickness+.040, half_y+pml_thickness+.040
    structures = []
    if kind in ("stripline", "cpw"):
        structures.append(mw.Structure(mw.Box(size=(4*extended_x, 4*extended_y, 4*(half_z+pml_thickness))),
                         mw.Material(eps_r=eps_r, sigma_e=sigma_e), name="uniform_dielectric"))
    else:
        structures.append(mw.Structure(mw.Box(position=(0., 0., -HEIGHT/2),
                         size=(2*extended_x, 2*extended_y, HEIGHT)),
                         mw.Material(eps_r=eps_r, sigma_e=sigma_e), name="substrate"))

    def sheet(name, y0, y1, z):
        if metal == "copper":
            geometry=mw.Box(position=torch.tensor((0., (y0+y1)/2, z),dtype=torch.float64),
                            size=torch.tensor((2*extended_x, y1-y0, 0.),dtype=torch.float64))
            material=mw.CopperFoil(thickness=35e-6,frequency_range=(.8e9,1.2e9) if copper_frequency_range is None else copper_frequency_range)
        else:
            geometry=mw.PlanarPEC.rectangle(((-extended_x, extended_x), (y0, y1)),
                                           coordinate=z, physical_thickness=35e-6)
            material=mw.Material.pec()
        structures.append(mw.Structure(geometry, material, name=name))
    if kind != "cpw":
        sheet("ground", -extended_y, extended_y, -HEIGHT)
    if kind == "stripline":
        sheet("top_ground", -extended_y, extended_y, HEIGHT)
    if kind == "cpw":
        sheet("ground_left", -extended_y, -WIDTH/2-GAP, 0.)
        sheet("ground_right", WIDTH/2+GAP, extended_y, 0.)
    if kind == "differential_pair":
        sheet("signal_positive", PAIR_OFFSET-PAIR_WIDTH/2, PAIR_OFFSET+PAIR_WIDTH/2, 0.)
        sheet("signal_negative", -PAIR_OFFSET-PAIR_WIDTH/2, -PAIR_OFFSET+PAIR_WIDTH/2, 0.)
    else:
        sheet("signal", -WIDTH/2, WIDTH/2, 0.)
    if pair_perturbation:
        # A local asymmetric capacitive paddle, outside both unchanged port
        # cross-sections, provides actual common/differential conversion.
        structures.append(mw.Structure(
            mw.PlanarPEC.rectangle(((-.005, .005), (PAIR_OFFSET-PAIR_WIDTH/2, .010)),
                                   coordinate=0., physical_thickness=35e-6),
            mw.Material.pec(), name="positive_signal_paddle"))
    ports = [_port(kind, "in", -port_separation/2, "+", dx, half_y-.010, half_z-.010, longitudinal_dl, metal),
             _port(kind, "out", port_separation/2, "-", dx, half_y-.010, half_z-.010, longitudinal_dl, metal)]
    return mw.Scene(domain=mw.Domain(bounds=tuple((float(a[0]), float(a[-1])) for a in nodes)),
                    grid=mw.GridSpec.custom(*nodes), boundary=mw.BoundarySpec.pml(num_layers=layers),
                    device=device, structures=structures, ports=ports,
                    metadata={"physical_model": f"printed-{kind}-v1", "dx_m": dx,
                              "port_separation_m": port_separation, "axial_half_length_m": half_x, "longitudinal_dl_m": longitudinal_dl,
                              "pml_physical_thickness_m": layers*longitudinal_dl,
                              "transverse_refinement_radius_m": transverse_refinement_radius,
                              "physical_copper_thickness_m": 35e-6, "metal": metal,
                              "copper_frequency_range_hz": (.8e9,1.2e9) if copper_frequency_range is None else tuple(copper_frequency_range),
                              "launch_conductor_model": "good_conductor" if metal == "copper" else "pec",
                              "pair_perturbation": bool(pair_perturbation), "eps_r": eps_r, "sigma_e": sigma_e})
