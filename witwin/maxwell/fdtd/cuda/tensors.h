#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

inline void check_cuda_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

inline void check_same_cuda_device(const at::Tensor& reference, const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same CUDA device as the reference tensor");
}

inline void check_float32_tensor(const at::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
}

inline void check_contiguous_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

