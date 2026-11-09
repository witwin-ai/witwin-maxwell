#pragma once

#include <cstdint>
#include <cuda_runtime.h>

constexpr int CORE_OBSERVER_GROUPS = 16;

template <typename AccumT>
struct CoreObserverGroup {
  const float* field;
  int64_t field_size;
  const int64_t* offsets;
  const int64_t* frequency_map;
  AccumT* real;
  AccumT* imag;
  int64_t count, points, frequencies;
  int retard;
};

template <typename AccumT>
struct CoreObserverBatch {
  CoreObserverGroup<AccumT> groups[CORE_OBSERVER_GROUPS];
  int64_t prefix[CORE_OBSERVER_GROUPS + 1];
  int group_count;
  const AccumT* weights;
  int64_t weight_count;
  const int* active;
};

void launch_observer_gather(const CoreObserverBatch<float>& batch, cudaStream_t stream);
void launch_observer_gather(const CoreObserverBatch<double>& batch, cudaStream_t stream);
