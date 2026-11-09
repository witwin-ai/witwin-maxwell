# WiTwin Maxwell converged CW-FDTD data generator

This experiment creates random box, sphere, and cylinder scenes driven by a
continuous-wave point dipole, solves the 3-D FDTD transient, and exports the
converged complex 2 GHz phasor field.

Each sample contains:

- `sample_NNNNN.npz`: `field` shaped `(N,N,N,6)` and `permittivity` shaped
  `(N,N,N)`, both `float32`;
- `sample_NNNNN_meta.json`: the exact scene, DFT window, solve attempts,
  convergence certificate, timings, storage, and peak GPU memory;
- `summary.json`: aggregate convergence, runtime, and exact byte counts.

The field channels are `Ex_real`, `Ex_imag`, `Ey_real`, `Ey_imag`, `Ez_real`,
and `Ez_imag`. External CPML cells are cropped and staggered Yee components
are interpolated to physical-domain cell centers.

## Steady-state contract

The default run uses a fixed 80-cycle Hanning DFT window after at least five
domain round trips of transient propagation. A time probe is placed half a
wavelength from the source and at every object center. Two adjacent final
four-cycle windows are fitted to complex Ex/Ey/Ez phasors. Both the vector
phasor change and vector-energy-weighted sinusoid-fit residual must be at most
1%.

If a scene fails, the generator reruns it with a longer pre-DFT transient
while keeping the 80-cycle DFT window fixed. The wait doubles up to 640 cycles.
A sample that still fails is rejected and is never written. This prevents
early transient fields from entering a nominally steady dataset.

```bash
python \
  experiments/fdtd_data_generator/generate.py \
  --num-samples 100 --grid-size 256 --gpus 0,1
```

The default output is `dataset_256_steady`; the historical 256-step
Gaussian-pulse dataset in `dataset_256` is not overwritten. Generation is
resumable only for samples whose data size, configuration, generator contract,
and convergence certificate all match exactly. Existing partial or
incompatible samples cause an error instead of being overwritten.

Uncompressed NPZ is the default for predictable throughput. CUDA Graph is off
by default for stable long batches. `--time-steps` is a fixed-step diagnostic
override; normal dataset generation should use the automatic steady-state
runtime.
