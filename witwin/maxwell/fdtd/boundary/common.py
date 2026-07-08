from __future__ import annotations

import torch

BOUNDARY_NONE = 0
BOUNDARY_PML = 1
BOUNDARY_PERIODIC = 2
BOUNDARY_BLOCH = 3
BOUNDARY_PEC = 4
BOUNDARY_PMC = 5

BOUNDARY_KIND_TO_CODE = {
    "none": BOUNDARY_NONE,
    "pml": BOUNDARY_PML,
    "periodic": BOUNDARY_PERIODIC,
    "bloch": BOUNDARY_BLOCH,
    "pec": BOUNDARY_PEC,
    "pmc": BOUNDARY_PMC,
    # Mur ABC is applied in PyTorch after the E-update; the update kernel treats
    # these faces as inert so no dedicated CUDA boundary code is required.
    "mur": BOUNDARY_NONE,
}


def has_complex_fields(solver) -> bool:
    return bool(getattr(solver, "complex_fields_enabled", False))


def combine_complex_spectral_components(real_field_real, real_field_imag, imag_field_real=None, imag_field_imag=None):
    if imag_field_real is None or imag_field_imag is None:
        return real_field_real + 1j * real_field_imag
    return (real_field_real - imag_field_imag) + 1j * (real_field_imag + imag_field_real)


def initialize_complex_fields(solver):
    solver.Ex_imag = torch.zeros_like(solver.Ex)
    solver.Ey_imag = torch.zeros_like(solver.Ey)
    solver.Ez_imag = torch.zeros_like(solver.Ez)
    solver.Hx_imag = torch.zeros_like(solver.Hx)
    solver.Hy_imag = torch.zeros_like(solver.Hy)
    solver.Hz_imag = torch.zeros_like(solver.Hz)
