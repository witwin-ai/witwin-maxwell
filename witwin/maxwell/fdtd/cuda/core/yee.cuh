#pragma once

#include <cuda_runtime.h>

// T is the field precision: float, or double for float64 fields.
template <typename T>
struct CoreAxis {
  const T *inv, *kappa, *b, *c;
  int low_mode, high_mode;
};

template <typename T>
struct CorePsi {
  T* data;
  int low, high_start, high;
};

template <typename T>
struct CoreHalf {
  unsigned int nx, ny, nz;
  unsigned int block_strip;
  T* field[3];
  const T* source[3];
  CoreAxis<T> axes[3];
  CorePsi<T> psi[6];
  const void* indices;
  const T* table;
  T coefficients[6];
  // Live and modulated coefficients are float32-only.
  const float* dense[6];
  const float *mod_cos[3], *mod_sin[3], *mod_omega[3];
  const double* modulation_time;
};

void launch_core(CoreHalf<float> half, bool electric, int cpml, bool uniform,
                 bool wide_indices, bool live, bool modulated, cudaStream_t stream);
// float64 fields: static uniform or indexed coefficients only.
void launch_core(CoreHalf<double> half, bool electric, int cpml, bool uniform,
                 bool wide_indices, cudaStream_t stream);
