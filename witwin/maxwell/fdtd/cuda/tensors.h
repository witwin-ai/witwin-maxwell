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

inline void check_bounded_x_launch(
    const torch::stable::Tensor& field,
    int64_t local_x_begin,
    int64_t local_x_end,
    int64_t global_x_offset,
    int64_t global_x_extent,
    const char* name) {
  STD_TORCH_CHECK(
      local_x_begin >= 0 && local_x_begin <= local_x_end &&
          local_x_end <= field.size(0),
      name,
      " local x launch interval must satisfy 0 <= begin <= end <= field.size(0)");
  STD_TORCH_CHECK(global_x_offset >= 0, name, " global x offset must be nonnegative");
  STD_TORCH_CHECK(global_x_extent > 0, name, " global x extent must be positive");
  STD_TORCH_CHECK(
      global_x_offset <= global_x_extent &&
          field.size(0) <= global_x_extent - global_x_offset,
      name,
      " padded local x allocation must fit within the component global x extent");
}
