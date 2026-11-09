#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "../source_gather.cuh"

struct CoreWaveform {
  float omega, inv_sigma, pi_frequency, amplitude, phase, delay, pedestal, pedestal_inv_sigma;
};

struct CoreSource {
  float* field;
  const float *patch, *second, *signals, *times, *delays, *activation;
  const int64_t *step, *delay_indices;
  int shape[3], patch_shape[3], offsets[3];
  int64_t count, delay_count;
  int kind, time_kind, wrap;
  bool causal, tabulated;
  CoreWaveform waveform;
  SourceGather gather;
};

struct CoreDft {
  const int64_t* step;
  const float *cosine, *sine, *field[3];
  float *real[3], *imag[3];
  int64_t sizes[3], frequencies;
};

struct CoreAuxiliary {
  const int64_t* step;
  float *electric, *magnetic, *boundary_h, *fields[3];
  const float *electric_decay, *electric_curl, *magnetic_decay, *magnetic_curl;
  const float *signals, *boundary_signals, *coeff, *positions;
  const int* indices;
  int64_t electric_count, magnetic_count;
  int source_index, boundary_width;
  bool reference, boundary;
  float origin, inv_ds;
  SourceGather gather;
};

void auxiliary_boundary_indexed(CoreAuxiliary auxiliary, cudaStream_t stream);
void auxiliary_correction_indexed(CoreAuxiliary auxiliary, bool magnetic, cudaStream_t stream);

void compile_delayed_amplitudes(CoreSource source, float* output, int64_t steps, cudaStream_t stream);
void source_patch_indexed(CoreSource source, cudaStream_t stream);
void auxiliary_electric_indexed(int64_t count, int source_index, float* electric,
    const float* magnetic, const float* decay, const float* curl,
    const float* signals, const int64_t* step, cudaStream_t stream);
void dft_indexed(CoreDft dft, cudaStream_t stream);
void commit_step(int64_t* step, cudaStream_t stream);
