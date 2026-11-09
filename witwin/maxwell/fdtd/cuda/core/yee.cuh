#pragma once

#include <cuda_runtime.h>

struct CoreAxis {
  const float *inv, *kappa, *b, *c;
  int low_mode, high_mode;
};

struct CorePsi {
  float* data;
  int low, high_start, high;
};

struct CoreHalf {
  unsigned int nx, ny, nz;
  unsigned int block_strip;
  float* field[3];
  const float* source[3];
  CoreAxis axes[3];
  CorePsi psi[6];
  const void* indices;
  const float* table;
  float coefficients[6];
  const float* dense[6];
  const float *mod_cos[3], *mod_sin[3], *mod_omega[3];
  const double* modulation_time;
};

void launch_core(CoreHalf half, bool electric, int cpml, bool uniform,
                 bool wide_indices, bool live, bool modulated, cudaStream_t stream);
