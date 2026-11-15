#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include "../../../fdtd/cuda/launch.h"
#include <string>
#include <vector>

inline void check_tensor(const torch::stable::Tensor& tensor, const torch::stable::Tensor& first,
                         torch::headeronly::ScalarType dtype, const char* op, const std::string& name) {
  STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.device() == first.device(),
              op, ": ", name, " must be a contiguous CUDA tensor on the same device");
  STD_TORCH_CHECK(tensor.scalar_type() == dtype, op, ": ", name, " must have dtype ", dtype);
}

inline void check_tensors(const std::vector<torch::stable::Tensor>& tensors, const torch::stable::Tensor& first,
                          torch::headeronly::ScalarType dtype, const char* op, const char* name) {
  for (size_t i = 0; i < tensors.size(); ++i)
    check_tensor(tensors[i], first, dtype, op, std::string(name) + "[" + std::to_string(i) + "]");
}
