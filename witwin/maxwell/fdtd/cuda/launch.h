#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>

#include <cuda_runtime.h>

inline dim3 linear_grid(long long elements, int block_size = 256) {
  const auto blocks = (elements + block_size - 1) / block_size;
  return dim3(static_cast<unsigned int>(blocks), 1, 1);
}

inline cudaStream_t current_cuda_stream() {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      torch::stable::accelerator::getCurrentDeviceIndex(),
      &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

#define WITWIN_CUDA_CHECK() STD_CUDA_KERNEL_LAUNCH_CHECK()

