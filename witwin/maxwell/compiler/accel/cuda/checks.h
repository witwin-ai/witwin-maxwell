#pragma once

#include <ATen/ATen.h>
#include <string>
#include <vector>

inline void check_tensor(const at::Tensor& tensor, const at::Tensor& first,
                         at::ScalarType dtype, const char* op, const std::string& name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.device() == first.device(),
              op, ": ", name, " must be a contiguous CUDA tensor on the same device");
  TORCH_CHECK(tensor.scalar_type() == dtype, op, ": ", name, " must have dtype ", dtype);
}

inline void check_tensors(const std::vector<at::Tensor>& tensors, const at::Tensor& first,
                          at::ScalarType dtype, const char* op, const char* name) {
  for (size_t i = 0; i < tensors.size(); ++i)
    check_tensor(tensors[i], first, dtype, op, std::string(name) + "[" + std::to_string(i) + "]");
}
