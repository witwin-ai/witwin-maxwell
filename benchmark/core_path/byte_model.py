"""Analytic FP32 streaming DRAM floors, not measured traffic.

The current six component kernels take two cross fields, decay, curl, inverse
spacing and an in-place field; CPML adds two psi arrays and six 1-D profiles.
Neighbor pairs within one input stream are assumed cached. Cross-component
streams are counted separately. Spacing, profiles and coefficient tables are
assumed cache-resident. Sources, surface clamps, padding and cache-line waste
are excluded. These assumptions must be checked with the emitted profiler run.

O1 shares cross-field streams across components (72 B rather than 96 B), uses
scalar vacuum coefficients and three 32-bit Yee-edge material IDs per cell.
Interface edges retain one curl value (lossless) or decay+curl (lossy). O1 DFT
remains separate; its fusion belongs to O2. Dense psi storage still streams all
12 arrays, whereas slab storage streams only the active per-axis fraction.
"""

from __future__ import annotations

import argparse
import json


MATERIALS = ("vacuum", "lossless dielectric", "lossy dielectric")
BOUNDARIES = ("PEC", "CPML dense", "CPML compressed")


def bytes_per_step(material: str, boundary: str, *, size: int = 128, layers: int = 8,
                   dft: str = "none", frequencies: int = 1, components: int = 3,
                   interface_fraction: float = 0.0) -> dict:
    # 6 stores + 6 self reads + 12 cross-field streams. Without uniform scalar
    # specialization, 12 coefficient streams add 48 B: the July audit's 144 B.
    current_fields = (6 + 6 + 12) * 4
    historical = current_fields + 12 * 4
    fused_fields = (6 + 6 + 6) * 4
    # Standard/dense kernels specialize the pair together; slab specializes
    # decay and curl independently. Magnetic coefficients are uniform (mu=1).
    current_coeff = 0 if material == "vacuum" else (
        12 if material == "lossless dielectric" and boundary == "CPML compressed" else 24
    )
    fused_coeff = 0 if material == "vacuum" else 12 + (
        12 * interface_fraction if material == "lossy dielectric" else 0
    )
    psi_fraction = {"PEC": 0.0, "CPML dense": 1.0,
                    "CPML compressed": 2 * layers / size}[boundary]
    psi = 12 * 2 * 4 * psi_fraction
    # A complex accumulator has two reads and two writes, plus one field read.
    dft_fraction = {"none": 0.0, "plane": 1 / size, "volume": 1.0}[dft]
    spectral = (4 + 2 * 2 * 4) * components * frequencies * dft_fraction
    return {
        "material": material, "boundary": boundary, "dft": dft,
        "current_fields": current_fields, "current_coefficients": current_coeff,
        "o1_fields": fused_fields, "o1_coefficients": fused_coeff,
        "psi": psi, "dft_bytes": spectral,
        "current": current_fields + current_coeff + psi + spectral,
        "o1": fused_fields + fused_coeff + psi + spectral,
        "historical_uncompressed_coefficients_pec": historical,
    }


def table(*, size=128, layers=8, frequencies=1, components=3, interface_fraction=0.0,
          monitored_dft="plane") -> str:
    lines = [
        "| Medium | Boundary | DFT | Current B/cell/step | O1 B/cell/step |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for material in MATERIALS:
        for boundary in BOUNDARIES:
            for dft in ("none", monitored_dft):
                row = bytes_per_step(material, boundary, size=size, layers=layers, dft=dft,
                                     frequencies=frequencies, components=components,
                                     interface_fraction=interface_fraction)
                lines.append(f"| {material} | {boundary} | {dft} | {row['current']:.4f} | {row['o1']:.4f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--frequencies", type=int, default=1)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--interface-fraction", type=float, default=0.0)
    parser.add_argument("--monitored-dft", choices=("plane", "volume"), default="plane")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        from .scenes import build_ladder_case
        print(json.dumps(build_ladder_case(f"vacuum_{args.size}_plane_dft").plan()))
    print(table(size=args.size, layers=args.layers, frequencies=args.frequencies,
                components=args.components, interface_fraction=args.interface_fraction,
                monitored_dft=args.monitored_dft))
    print("Historical audit cross-check: (6 stores + 6 self reads + 12 cross-field streams + 12 coefficient streams) * 4 = 144 B/cell/step.")
    print("Current scalar specializations remove uniform coefficient streams. These are streaming floors; no GPU measurement has been made in O0.")


if __name__ == "__main__":
    main()
