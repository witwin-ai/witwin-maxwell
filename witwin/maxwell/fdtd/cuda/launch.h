#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

inline dim3 linear_grid(long long elements, int block_size = 256) {
  const auto blocks = (elements + block_size - 1) / block_size;
  return dim3(static_cast<unsigned int>(blocks), 1, 1);
}

inline cudaStream_t current_cuda_stream() {
  return at::cuda::getCurrentCUDAStream();
}

#define WITWIN_CUDA_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()

