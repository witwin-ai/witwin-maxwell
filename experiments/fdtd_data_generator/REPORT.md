# 256³ FDTD dataset report

> **Historical transient result — not a steady-state dataset.** These files
> used a Gaussian pulse for only 256 steps (2.438 ns, 4.876 carrier cycles).
> The pulse had not reached any sample's nearest object by its peak; the
> resulting field was a localized source wave packet. Keep this report only
> for provenance. New converged CW results are written to
> `dataset_256_steady`.

## Result

- Completed samples: 100 / 100
- Field shape per sample: `(256, 256, 256, 6)`, `float32`
- Permittivity shape per sample: `(256, 256, 256)`, `float32`
- Field channels: Ex/Ey/Ez real and imaginary parts
- Output: uncompressed NPZ plus one JSON sidecar per sample
- Random seed scheme: `20260715 + sample_id`

## Simulation settings

- Solver: WiTwin Maxwell FDTD, full-field single-frequency DFT
- Frequency: 2.0 GHz
- Grid spacing: 0.0093685143125 m (16 points per wavelength)
- Physical domain: 2.398339664 m per side
- External CPML: 10 cells per face
- Time steps: 256
- Random scene: 1–4 boxes, spheres, or cylinders with relative permittivity 2–12
- Source: randomized Gaussian-pulse point dipole
- Output grid: external CPML cropped; Yee fields and permittivity interpolated to cell centers

## Timing

- Mean solve time: 1.703638 s/sample
- Solve range: 1.596418–2.366009 s/sample
- Mean complete sample time (scene + solve + save): 3.643796 s/sample
- Sum of complete sample times: 364.379564 s
- Two-GPU campaign wall time: 254.604915 s (4 min 14.60 s)
- Final resumable completion pass: 96.300185 s
- Stable non-CUDA-Graph GPU peak: about 6.092 GiB per worker

The campaign wall time includes the first real-sample benchmark, an initial
CUDA-Graph batch that exposed graph-memory accumulation after long repeated
scene creation, and the successful resumable non-graph completion pass. The
generator now defaults to the stable non-CUDA-Graph mode.

## Storage

- Data files: 46,976,256,800 bytes
- Sample metadata: 250,795 bytes
- Dataset payload + sample metadata: 46,976,507,595 bytes
- Entire dataset directory: 46,976,508,871 bytes
- Entire directory in binary units: 43.750283 GiB
- Per NPZ sample: 469,762,568 bytes (448.000496 MiB)

## Validation

All 100 sequential NPZ names and JSON sidecars are present. Every NPZ contains
exactly `field.npy` and `permittivity.npy` with the same byte size. Samples
0, 49, and 99 were fully loaded and verified for shape, dtype, and finite
values.
