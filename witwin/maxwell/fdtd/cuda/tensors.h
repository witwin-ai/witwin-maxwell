#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <optional>

inline void check_cuda_tensor(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

inline void check_same_cuda_device(
    const torch::stable::Tensor& reference,
    const torch::stable::Tensor& tensor,
    const char* name) {
  STD_TORCH_CHECK(
      tensor.device() == reference.device(),
      name,
      " must be on the same CUDA device as the reference tensor");
}

inline void check_float32_tensor(const torch::stable::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Float,
      name,
      " must be float32");
}

inline void check_contiguous_tensor(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

