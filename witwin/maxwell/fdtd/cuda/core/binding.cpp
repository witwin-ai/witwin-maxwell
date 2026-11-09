#include <torch/extension.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "yee.cuh"
#include "step.cuh"
#include "observers.cuh"

class CoreObserverGather {
 public:
  CoreObserverGather(std::vector<at::Tensor> fields, std::vector<at::Tensor> offsets,
                     std::vector<at::Tensor> frequency_maps, std::vector<at::Tensor> real,
                     std::vector<at::Tensor> imag, std::vector<bool> magnetic,
                     at::Tensor weights, at::Tensor active)
      : weights_(std::move(weights)), active_(std::move(active)) {
    const auto count = fields.size();
    TORCH_CHECK(offsets.size() == count && frequency_maps.size() == count &&
        real.size() == count && imag.size() == count && magnetic.size() == count,
        "observer group lists must have matching lengths");
    const auto dtype = weights_.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
        "observer weights must be float32 or float64");
    check(weights_, dtype, "weights");
    check(active_, at::kInt, "active");
    TORCH_CHECK(weights_.dim() == 3 && weights_.size(0) > 0 && weights_.size(2) == 4 &&
        active_.numel() == weights_.size(0), "invalid weights or active mask");
    std::vector<at::Tensor> inputs{weights_, active_}, outputs;
    for (size_t i = 0; i < count; ++i) {
      check(fields[i], at::kFloat, "field");
      check(offsets[i], at::kLong, "offsets");
      check(frequency_maps[i], at::kLong, "frequency_map");
      check(real[i], dtype, "real");
      check(imag[i], dtype, "imag");
      TORCH_CHECK(real[i].dim() == 3 && real[i].sizes().equals(imag[i].sizes()),
          "accumulators must match [B,F,P]");
      const auto b = real[i].size(0), f = real[i].size(1), p = real[i].size(2);
      TORCH_CHECK(b > 0 && f > 0 && p > 0 && fields[i].dim() == 4 &&
          fields[i].size(0) == b, "invalid batch field or accumulator dimensions");
      TORCH_CHECK(offsets[i].dim() == 2 && offsets[i].size(0) == b && offsets[i].size(1) == p &&
          frequency_maps[i].dim() == 2 && frequency_maps[i].size(0) == b &&
          frequency_maps[i].size(1) == f && weights_.size(0) == b,
          "invalid offsets, frequency map, weights or active mask");
      inputs.insert(inputs.end(), {fields[i], offsets[i], frequency_maps[i]});
      outputs.insert(outputs.end(), {real[i], imag[i]});
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      for (size_t j = 0; j < i; ++j) disjoint(outputs[i], outputs[j]);
      for (const auto& input : inputs) disjoint(outputs[i], input);
    }
    tensors_ = std::move(inputs);
    tensors_.insert(tensors_.end(), outputs.begin(), outputs.end());
    if (dtype == at::kDouble)
      prepare(fields, offsets, frequency_maps, real, imag, magnetic, double_batches_);
    else
      prepare(fields, offsets, frequency_maps, real, imag, magnetic, float_batches_);
  }

  void launch() {
    c10::cuda::CUDAGuard guard(weights_.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(weights_.get_device());
    for (const auto& batch : float_batches_) launch_observer_gather(batch, stream);
    for (const auto& batch : double_batches_) launch_observer_gather(batch, stream);
  }

 private:
  void check(const at::Tensor& tensor, at::ScalarType dtype, const char* name) const {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.device() == weights_.device(),
        name, " must be contiguous CUDA storage on the weights device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has incorrect dtype");
  }

  static void disjoint(const at::Tensor& a, const at::Tensor& b) {
    if (a.numel() == 0 || b.numel() == 0) return;
    const auto x = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto y = reinterpret_cast<uintptr_t>(b.data_ptr());
    TORCH_CHECK(x + a.numel() * a.element_size() <= y || y + b.numel() * b.element_size() <= x,
        "core observer writable storage must not overlap another input or accumulator");
  }

  template <typename AccumT>
  void prepare(const std::vector<at::Tensor>& fields, const std::vector<at::Tensor>& offsets,
               const std::vector<at::Tensor>& frequency_maps, const std::vector<at::Tensor>& real,
               const std::vector<at::Tensor>& imag, const std::vector<bool>& magnetic,
               std::vector<CoreObserverBatch<AccumT>>& batches) {
    for (size_t i = 0; i < fields.size(); ++i) {
      if (i % CORE_OBSERVER_GROUPS == 0) {
        batches.emplace_back(CoreObserverBatch<AccumT>{});
        batches.back().weights = weights_.data_ptr<AccumT>();
        batches.back().weight_count = weights_.numel() / 4;
        batches.back().active = active_.data_ptr<int>();
      }
      auto& batch = batches.back();
      const int index = batch.group_count++;
      batch.groups[index] = {fields[i].data_ptr<float>(), fields[i].numel(),
          offsets[i].data_ptr<int64_t>(), frequency_maps[i].data_ptr<int64_t>(),
          real[i].data_ptr<AccumT>(), imag[i].data_ptr<AccumT>(), real[i].numel(),
          real[i].size(2), real[i].size(1), magnetic[i] ? 2 : 0};
      batch.prefix[index + 1] = batch.prefix[index] + real[i].numel();
    }
  }

  at::Tensor weights_, active_;
  std::vector<at::Tensor> tensors_;
  std::vector<CoreObserverBatch<float>> float_batches_;
  std::vector<CoreObserverBatch<double>> double_batches_;
};

// Tensor ownership and pointer assembly happen once, before graph capture.
class CoreLaunch {
 public:
  CoreLaunch(std::vector<at::Tensor> tensors, std::vector<int64_t> integers,
             std::vector<double> coefficients, bool electric, int cpml,
             bool uniform, bool wide_indices, bool live, bool modulated)
      : tensors_(std::move(tensors)), electric_(electric), cpml_(cpml),
        uniform_(uniform), wide_indices_(wide_indices), live_(live), modulated_(modulated) {
    TORCH_CHECK(integers.size() == (cpml_ ? 28 : 10), "invalid core launch integers");
    const auto strip = integers.back();
    TORCH_CHECK(strip >= 0 && uint64_t(strip) <= std::numeric_limits<unsigned>::max(),
        "core Yee block strip must be a nonnegative 32-bit unsigned integer");
    for (int a = 0; a < 3; ++a)
      TORCH_CHECK(integers[a] > 0 && uint64_t(integers[a]) <= std::numeric_limits<unsigned>::max(),
          "core Yee dimensions must be positive 32-bit unsigned integers");
    half_.nx = integers[0]; half_.ny = integers[1]; half_.nz = integers[2];
    half_.block_strip = strip;
    const uint64_t grid_xy = ((uint64_t(half_.nz) + 127) / 128) * ((uint64_t(half_.ny) + 1) / 2);
    constexpr uint64_t block_limit = uint64_t(1) << 32;
    TORCH_CHECK(grid_xy < block_limit && uint64_t(half_.nx) <= (block_limit - 1) / grid_xy,
        "core Yee launch requires fewer than 2^32 blocks");
    for (int c = 0; c < 3; ++c) {
      half_.field[c] = tensors_[c].data_ptr<float>();
      half_.source[c] = tensors_[3 + c].data_ptr<float>();
      half_.axes[c] = {tensors_[6 + c].data_ptr<float>(), nullptr, nullptr, nullptr,
                       int(integers[3 + 2*c]), int(integers[4 + 2*c])};
      if (cpml_) {
        half_.axes[c].kappa = tensors_[11 + 3*c].data_ptr<float>();
        half_.axes[c].b = tensors_[12 + 3*c].data_ptr<float>();
        half_.axes[c].c = tensors_[13 + 3*c].data_ptr<float>();
      }
    }
    half_.indices = tensors_[9].data_ptr();
    half_.table = tensors_[10].data_ptr<float>();
    for (int p = 0; p < 6; ++p) {
      half_.coefficients[p] = coefficients[p];
      if (cpml_) half_.psi[p] = {tensors_[20 + p].data_ptr<float>(),
          int(integers[9 + 3*p]), int(integers[10 + 3*p]), int(integers[11 + 3*p])};
    }
    int next = cpml_ ? 26 : 11;
    if (live_) {
      for (int p = 0; p < 6; ++p) half_.dense[p] = tensors_[next++].data_ptr<float>();
    }
    if (modulated_) {
      for (int c = 0; c < 3; ++c) {
        half_.mod_cos[c] = tensors_[next++].data_ptr<float>();
        half_.mod_sin[c] = tensors_[next++].data_ptr<float>();
        half_.mod_omega[c] = tensors_[next++].data_ptr<float>();
      }
      half_.modulation_time = tensors_[next].data_ptr<double>();
    }
  }

  void launch() {
    c10::cuda::CUDAGuard guard(tensors_[0].device());
    launch_core(half_, electric_, cpml_, uniform_, wide_indices_, live_, modulated_,
                c10::cuda::getCurrentCUDAStream(tensors_[0].get_device()));
  }

 private:
  std::vector<at::Tensor> tensors_;
  CoreHalf half_{};
  bool electric_;
  int cpml_;
  bool uniform_, wide_indices_, live_, modulated_;
};

class CoreStepLaunch {
 public:
  CoreStepLaunch(int operation, std::vector<at::Tensor> tensors,
                 std::vector<int64_t> integers, std::vector<double> scalars)
      : operation_(operation), tensors_(std::move(tensors)) {
    step_ = tensors_[0].data_ptr<int64_t>();
    if (operation_ == 0) {
      source_.step = step_;
      source_.field = tensors_[1].data_ptr<float>();
      source_.patch = tensors_[2].data_ptr<float>();
      source_.second = tensors_[3].data_ptr<float>();
      source_.signals = tensors_[4].data_ptr<float>();
      source_.times = tensors_[5].data_ptr<float>();
      source_.delays = tensors_[6].data_ptr<float>();
      source_.activation = tensors_[7].data_ptr<float>();
      source_.delay_indices = tensors_[8].data_ptr<int64_t>();
      for (int a = 0; a < 3; ++a) {
        source_.shape[a] = tensors_[1].size(a);
        source_.patch_shape[a] = tensors_[2].size(a);
        source_.offsets[a] = integers[a];
      }
      source_.count = tensors_[2].numel();
      source_.kind = integers[3]; source_.time_kind = integers[4];
      source_.wrap = integers[5]; source_.causal = integers[6];
      source_.tabulated = integers[7];
      if (source_.wrap) {
        TORCH_CHECK(tensors_.size() == 12, "periodic core sources require a prepared gather");
        for (int index = 9; index < 12; ++index) {
          TORCH_CHECK(tensors_[index].is_cuda() && tensors_[index].is_contiguous() &&
              tensors_[index].device() == tensors_[1].device() &&
              tensors_[index].scalar_type() == at::kLong, "invalid core source gather");
        }
        TORCH_CHECK(tensors_[9].dim() == 1 && tensors_[10].dim() == 2 && tensors_[10].size(1) == 2 &&
            tensors_[11].dim() == 2 && tensors_[11].size(1) == 2 &&
            tensors_[9].numel() == tensors_[10].size(0)+1, "invalid core source gather shape");
        source_.gather = {tensors_[9].data_ptr<int64_t>(), tensors_[10].data_ptr<int64_t>(),
            tensors_[11].data_ptr<int64_t>(), tensors_[10].size(0)};
      }
      source_.delay_count = source_.tabulated ? tensors_[4].size(1) : 0;
      constexpr float two_pi = 6.283185307179586f, pi = 3.141592653589793f;
      source_.waveform = {two_pi * float(scalars[0]),
          std::fmax(two_pi * float(scalars[1]), 1.0e-30f), pi * float(scalars[0]),
          float(scalars[2]), float(scalars[3]), float(scalars[4]), float(scalars[5]),
          two_pi * float(scalars[6])};
    } else if (operation_ == 1) {
      source_index_ = integers[0];
    } else if (operation_ >= 4 && operation_ <= 6) {
      auxiliary_.step = step_;
      auxiliary_.electric = tensors_[1].data_ptr<float>();
      auxiliary_.magnetic = tensors_[2].data_ptr<float>();
      auxiliary_.electric_decay = tensors_[3].data_ptr<float>();
      auxiliary_.electric_curl = tensors_[4].data_ptr<float>();
      auxiliary_.magnetic_decay = tensors_[5].data_ptr<float>();
      auxiliary_.magnetic_curl = tensors_[6].data_ptr<float>();
      auxiliary_.signals = tensors_[7].data_ptr<float>();
      auxiliary_.boundary_signals = tensors_[8].data_ptr<float>();
      auxiliary_.boundary_h = tensors_[9].data_ptr<float>();
      auxiliary_.electric_count = tensors_[1].numel();
      auxiliary_.magnetic_count = tensors_[2].numel();
      auxiliary_.source_index = integers[0];
      auxiliary_.boundary_width = integers[0];
      TORCH_CHECK(auxiliary_.electric_count == auxiliary_.magnetic_count + 1,
          "invalid auxiliary Yee line lengths");
      TORCH_CHECK(tensors_[3].numel() == auxiliary_.electric_count &&
          tensors_[4].numel() == auxiliary_.electric_count &&
          tensors_[5].numel() == auxiliary_.magnetic_count &&
          tensors_[6].numel() == auxiliary_.magnetic_count,
          "auxiliary loss arrays must match the full line");
      if (operation_ == 4) {
        TORCH_CHECK(auxiliary_.source_index == 2 && tensors_[8].dim() == 2 &&
            tensors_[8].size(1) == 2 && tensors_[9].numel() == 2,
            "boundary continuation requires the two-cell incident stencil");
      } else {
        auxiliary_.reference = integers[1];
        auxiliary_.boundary = integers[2];
        for (int a = 0; a < 3; ++a) auxiliary_.fields[a] = tensors_[10+a].data_ptr<float>();
        auxiliary_.coeff = tensors_[13].data_ptr<float>();
        if (auxiliary_.reference) auxiliary_.indices = tensors_[14].data_ptr<int>();
        else auxiliary_.positions = tensors_[14].data_ptr<float>();
        auxiliary_.gather = {tensors_[15].data_ptr<int64_t>(), tensors_[16].data_ptr<int64_t>(),
            tensors_[17].data_ptr<int64_t>(), tensors_[16].size(0)};
        auxiliary_.origin = float(scalars[0]);
        auxiliary_.inv_ds = float(scalars[1]);
      }
    } else if (operation_ == 2) {
      dft_.step = step_;
      dft_.cosine = tensors_[1].data_ptr<float>();
      dft_.sine = tensors_[2].data_ptr<float>();
      dft_.frequencies = tensors_[1].size(1);
      for (int a = 0; a < 3; ++a) {
        dft_.field[a] = tensors_[3+3*a].data_ptr<float>();
        dft_.real[a] = tensors_[4+3*a].data_ptr<float>();
        dft_.imag[a] = tensors_[5+3*a].data_ptr<float>();
        dft_.sizes[a] = tensors_[3+3*a].numel();
      }
    }
  }

  void compile_delayed(at::Tensor delays, at::Tensor output) {
    c10::cuda::CUDAGuard guard(tensors_[0].device());
    auto source = source_;
    source.delays = delays.data_ptr<float>();
    source.delay_count = delays.numel();
    compile_delayed_amplitudes(source, output.data_ptr<float>(), output.size(0),
        c10::cuda::getCurrentCUDAStream(tensors_[0].get_device()));
  }

  void launch() {
    c10::cuda::CUDAGuard guard(tensors_[0].device());
    const auto stream = c10::cuda::getCurrentCUDAStream(tensors_[0].get_device());
    switch (operation_) {
      case 0: source_patch_indexed(source_, stream); break;
      case 1: auxiliary_electric_indexed(tensors_[1].numel(), source_index_,
          tensors_[1].data_ptr<float>(), tensors_[2].data_ptr<float>(),
          tensors_[3].data_ptr<float>(), tensors_[4].data_ptr<float>(),
          tensors_[5].data_ptr<float>(), step_, stream); break;
      case 2: dft_indexed(dft_, stream); break;
      case 3: commit_step(step_, stream); break;
      case 4: auxiliary_boundary_indexed(auxiliary_, stream); break;
      case 5: auxiliary_correction_indexed(auxiliary_, true, stream); break;
      case 6: auxiliary_correction_indexed(auxiliary_, false, stream); break;
    }
  }

 private:
  int operation_, source_index_ = 0;
  std::vector<at::Tensor> tensors_;
  int64_t* step_;
  CoreSource source_{};
  CoreDft dft_{};
  CoreAuxiliary auxiliary_{};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.attr("abi") = "core_abi_v11";
  pybind11::class_<CoreObserverGather>(module, "CoreObserverGather")
      .def(pybind11::init<std::vector<at::Tensor>, std::vector<at::Tensor>,
           std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>,
           std::vector<bool>, at::Tensor, at::Tensor>())
      .def("launch", &CoreObserverGather::launch);
  pybind11::class_<CoreLaunch>(module, "CoreLaunch")
      .def(pybind11::init<std::vector<at::Tensor>, std::vector<int64_t>,
           std::vector<double>, bool, int, bool, bool, bool, bool>())
      .def("launch", &CoreLaunch::launch);
  pybind11::class_<CoreStepLaunch>(module, "CoreStepLaunch")
      .def(pybind11::init<int, std::vector<at::Tensor>, std::vector<int64_t>, std::vector<double>>())
      .def("compile_delayed", &CoreStepLaunch::compile_delayed)
      .def("launch", &CoreStepLaunch::launch);
}
