#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "../launch.h"
#include "../tensors.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include "yee.cuh"
#include "step.cuh"
#include "observers.cuh"

class CoreHandle {
 public:
  virtual ~CoreHandle() = default;
  virtual void launch() = 0;
};

class CoreObserverGather : public CoreHandle {
 public:
  CoreObserverGather(std::vector<torch::stable::Tensor> fields, std::vector<torch::stable::Tensor> offsets,
                     std::vector<torch::stable::Tensor> frequency_maps, std::vector<torch::stable::Tensor> real,
                     std::vector<torch::stable::Tensor> imag, std::vector<bool> magnetic,
                     torch::stable::Tensor weights, torch::stable::Tensor active)
      : weights_(std::move(weights)), active_(std::move(active)) {
    const auto count = fields.size();
    STD_TORCH_CHECK(offsets.size() == count && frequency_maps.size() == count &&
        real.size() == count && imag.size() == count && magnetic.size() == count,
        "observer group lists must have matching lengths");
    const auto dtype = weights_.scalar_type();
    STD_TORCH_CHECK(dtype == torch::headeronly::ScalarType::Float || dtype == torch::headeronly::ScalarType::Double,
        "observer weights must be float32 or float64");
    check(weights_, dtype, "weights");
    check(active_, torch::headeronly::ScalarType::Int, "active");
    // float32 fields accumulate into float32 or float64 spectra; float64 fields only into float64.
    const auto field_dtype = count ? fields[0].scalar_type() : torch::headeronly::ScalarType::Float;
    STD_TORCH_CHECK(field_dtype == torch::headeronly::ScalarType::Float ||
        (field_dtype == torch::headeronly::ScalarType::Double && dtype == field_dtype),
        "observer fields must be float32, or float64 with float64 weights");
    STD_TORCH_CHECK(weights_.dim() == 3 && weights_.size(0) > 0 && weights_.size(2) == 4 &&
        active_.numel() == weights_.size(0), "invalid weights or active mask");
    std::vector<torch::stable::Tensor> inputs{weights_, active_}, outputs;
    for (size_t i = 0; i < count; ++i) {
      check(fields[i], field_dtype, "field");
      check(offsets[i], torch::headeronly::ScalarType::Long, "offsets");
      check(frequency_maps[i], torch::headeronly::ScalarType::Long, "frequency_map");
      check(real[i], dtype, "real");
      check(imag[i], dtype, "imag");
      STD_TORCH_CHECK(real[i].dim() == 3 && real[i].sizes().equals(imag[i].sizes()),
          "accumulators must match [B,F,P]");
      const auto b = real[i].size(0), f = real[i].size(1), p = real[i].size(2);
      STD_TORCH_CHECK(b > 0 && f > 0 && p > 0 && fields[i].dim() == 4 &&
          fields[i].size(0) == b, "invalid batch field or accumulator dimensions");
      STD_TORCH_CHECK(offsets[i].dim() == 2 && offsets[i].size(0) == b && offsets[i].size(1) == p &&
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
    if (field_dtype == torch::headeronly::ScalarType::Double)
      prepare(fields, offsets, frequency_maps, real, imag, magnetic, field64_batches_);
    else if (dtype == torch::headeronly::ScalarType::Double)
      prepare(fields, offsets, frequency_maps, real, imag, magnetic, double_batches_);
    else
      prepare(fields, offsets, frequency_maps, real, imag, magnetic, float_batches_);
  }

  void launch() override {
    torch::stable::accelerator::DeviceGuard guard(weights_.get_device_index());
    const auto stream = current_cuda_stream();
    for (const auto& batch : float_batches_) launch_observer_gather(batch, stream);
    for (const auto& batch : double_batches_) launch_observer_gather(batch, stream);
    for (const auto& batch : field64_batches_) launch_observer_gather(batch, stream);
  }

 private:
  void check(const torch::stable::Tensor& tensor, torch::headeronly::ScalarType dtype, const char* name) const {
    STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.device() == weights_.device(),
        name, " must be contiguous CUDA storage on the weights device");
    STD_TORCH_CHECK(tensor.scalar_type() == dtype, name, " has incorrect dtype");
  }

  static void disjoint(const torch::stable::Tensor& a, const torch::stable::Tensor& b) {
    if (a.numel() == 0 || b.numel() == 0) return;
    const auto x = reinterpret_cast<uintptr_t>(a.const_data_ptr());
    const auto y = reinterpret_cast<uintptr_t>(b.const_data_ptr());
    STD_TORCH_CHECK(x + a.numel() * a.element_size() <= y || y + b.numel() * b.element_size() <= x,
        "core observer writable storage must not overlap another input or accumulator");
  }

  template <typename FieldT, typename AccumT>
  void prepare(const std::vector<torch::stable::Tensor>& fields, const std::vector<torch::stable::Tensor>& offsets,
               const std::vector<torch::stable::Tensor>& frequency_maps, const std::vector<torch::stable::Tensor>& real,
               const std::vector<torch::stable::Tensor>& imag, const std::vector<bool>& magnetic,
               std::vector<CoreObserverBatch<FieldT, AccumT>>& batches) {
    for (size_t i = 0; i < fields.size(); ++i) {
      if (i % CORE_OBSERVER_GROUPS == 0) {
        batches.emplace_back(CoreObserverBatch<FieldT, AccumT>{});
        batches.back().weights = weights_.const_data_ptr<AccumT>();
        batches.back().weight_count = weights_.numel() / 4;
        batches.back().active = active_.const_data_ptr<int>();
      }
      auto& batch = batches.back();
      const int index = batch.group_count++;
      batch.groups[index] = {fields[i].const_data_ptr<FieldT>(), fields[i].numel(),
          offsets[i].const_data_ptr<int64_t>(), frequency_maps[i].const_data_ptr<int64_t>(),
          real[i].mutable_data_ptr<AccumT>(), imag[i].mutable_data_ptr<AccumT>(), real[i].numel(),
          real[i].size(2), real[i].size(1), magnetic[i] ? 2 : 0};
      batch.prefix[index + 1] = batch.prefix[index] + real[i].numel();
    }
  }

  torch::stable::Tensor weights_, active_;
  std::vector<torch::stable::Tensor> tensors_;
  std::vector<CoreObserverBatch<float, float>> float_batches_;
  std::vector<CoreObserverBatch<float, double>> double_batches_;
  std::vector<CoreObserverBatch<double, double>> field64_batches_;
};

// Tensor ownership and pointer assembly happen once, before graph capture.
// Fields, coefficients and CPML state share one precision: float32, or float64
// with static (uniform or indexed) coefficients.
class CoreLaunch : public CoreHandle {
 public:
  CoreLaunch(std::vector<torch::stable::Tensor> tensors, std::vector<int64_t> integers,
             std::vector<double> coefficients, bool electric, int cpml,
             bool uniform, bool wide_indices, bool live, bool modulated)
      : tensors_(std::move(tensors)), electric_(electric), cpml_(cpml),
        uniform_(uniform), wide_indices_(wide_indices), live_(live), modulated_(modulated) {
    STD_TORCH_CHECK(integers.size() == (cpml_ ? 28 : 10), "invalid core launch integers");
    const auto strip = integers.back();
    STD_TORCH_CHECK(strip >= 0 && uint64_t(strip) <= std::numeric_limits<unsigned>::max(),
        "core Yee block strip must be a nonnegative 32-bit unsigned integer");
    for (int a = 0; a < 3; ++a)
      STD_TORCH_CHECK(integers[a] > 0 && uint64_t(integers[a]) <= std::numeric_limits<unsigned>::max(),
          "core Yee dimensions must be positive 32-bit unsigned integers");
    const uint64_t grid_xy = ((uint64_t(integers[2]) + 127) / 128) * ((uint64_t(integers[1]) + 1) / 2);
    constexpr uint64_t block_limit = uint64_t(1) << 32;
    STD_TORCH_CHECK(grid_xy < block_limit && uint64_t(integers[0]) <= (block_limit - 1) / grid_xy,
        "core Yee launch requires fewer than 2^32 blocks");
    // The integer guards above precede any tensor access.
    double_ = tensors_[0].scalar_type() == torch::headeronly::ScalarType::Double;
    if (double_) {
      STD_TORCH_CHECK(!live_ && !modulated_, "float64 core Yee launches require static coefficients");
      prepare(half64_, integers, coefficients);
    } else {
      prepare(half_, integers, coefficients);
    }
  }

  void launch() override {
    torch::stable::accelerator::DeviceGuard guard(tensors_[0].get_device_index());
    if (double_)
      launch_core(half64_, electric_, cpml_, uniform_, wide_indices_, current_cuda_stream());
    else
      launch_core(half_, electric_, cpml_, uniform_, wide_indices_, live_, modulated_,
                  current_cuda_stream());
  }

 private:
  template <typename T>
  void prepare(CoreHalf<T>& half, const std::vector<int64_t>& integers, const std::vector<double>& coefficients) {
    // Every floating tensor except the float32 live/modulated tables carries the field precision.
    const auto dtype = tensors_[0].scalar_type();
    const int fixed = cpml_ ? 26 : 11;
    for (int index = 0; index < fixed; ++index)
      STD_TORCH_CHECK(index == 9 || tensors_[index].scalar_type() == dtype, "core launch tensor dtypes must match the fields");
    half.nx = integers[0]; half.ny = integers[1]; half.nz = integers[2];
    half.block_strip = integers.back();
    for (int c = 0; c < 3; ++c) {
      half.field[c] = tensors_[c].mutable_data_ptr<T>();
      half.source[c] = tensors_[3 + c].const_data_ptr<T>();
      half.axes[c] = {tensors_[6 + c].const_data_ptr<T>(), nullptr, nullptr, nullptr,
                      int(integers[3 + 2*c]), int(integers[4 + 2*c])};
      if (cpml_) {
        half.axes[c].kappa = tensors_[11 + 3*c].const_data_ptr<T>();
        half.axes[c].b = tensors_[12 + 3*c].const_data_ptr<T>();
        half.axes[c].c = tensors_[13 + 3*c].const_data_ptr<T>();
      }
    }
    half.indices = tensors_[9].const_data_ptr();
    half.table = tensors_[10].const_data_ptr<T>();
    for (int p = 0; p < 6; ++p) {
      half.coefficients[p] = coefficients[p];
      if (cpml_) half.psi[p] = {tensors_[20 + p].mutable_data_ptr<T>(),
          int(integers[9 + 3*p]), int(integers[10 + 3*p]), int(integers[11 + 3*p])};
    }
    int next = fixed;
    if (live_) {
      for (int p = 0; p < 6; ++p) half.dense[p] = tensors_[next++].const_data_ptr<float>();
    }
    if (modulated_) {
      for (int c = 0; c < 3; ++c) {
        half.mod_cos[c] = tensors_[next++].const_data_ptr<float>();
        half.mod_sin[c] = tensors_[next++].const_data_ptr<float>();
        half.mod_omega[c] = tensors_[next++].const_data_ptr<float>();
      }
      half.modulation_time = tensors_[next].const_data_ptr<double>();
    }
  }

  std::vector<torch::stable::Tensor> tensors_;
  CoreHalf<float> half_{};
  CoreHalf<double> half64_{};
  bool electric_;
  int cpml_;
  bool uniform_, wide_indices_, live_, modulated_, double_ = false;
};

class CoreStepLaunch : public CoreHandle {
 public:
  CoreStepLaunch(int operation, std::vector<torch::stable::Tensor> tensors,
                 std::vector<int64_t> integers, std::vector<double> scalars)
      : operation_(operation), tensors_(std::move(tensors)) {
    step_ = tensors_[0].mutable_data_ptr<int64_t>();
    if (operation_ == 0) {
      source_.step = step_;
      source_.field = tensors_[1].mutable_data_ptr<float>();
      source_.patch = tensors_[2].const_data_ptr<float>();
      source_.second = tensors_[3].const_data_ptr<float>();
      source_.signals = tensors_[4].const_data_ptr<float>();
      source_.times = tensors_[5].const_data_ptr<float>();
      source_.delays = tensors_[6].const_data_ptr<float>();
      source_.activation = tensors_[7].const_data_ptr<float>();
      source_.delay_indices = tensors_[8].const_data_ptr<int64_t>();
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
        STD_TORCH_CHECK(tensors_.size() == 12, "periodic core sources require a prepared gather");
        for (int index = 9; index < 12; ++index) {
          STD_TORCH_CHECK(tensors_[index].is_cuda() && tensors_[index].is_contiguous() &&
              tensors_[index].device() == tensors_[1].device() &&
              tensors_[index].scalar_type() == torch::headeronly::ScalarType::Long, "invalid core source gather");
        }
        STD_TORCH_CHECK(tensors_[9].dim() == 1 && tensors_[10].dim() == 2 && tensors_[10].size(1) == 2 &&
            tensors_[11].dim() == 2 && tensors_[11].size(1) == 2 &&
            tensors_[9].numel() == tensors_[10].size(0)+1, "invalid core source gather shape");
        source_.gather = {tensors_[9].const_data_ptr<int64_t>(), tensors_[10].const_data_ptr<int64_t>(),
            tensors_[11].const_data_ptr<int64_t>(), tensors_[10].size(0)};
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
      auxiliary_.electric = tensors_[1].mutable_data_ptr<float>();
      auxiliary_.magnetic = tensors_[2].mutable_data_ptr<float>();
      auxiliary_.electric_decay = tensors_[3].const_data_ptr<float>();
      auxiliary_.electric_curl = tensors_[4].const_data_ptr<float>();
      auxiliary_.magnetic_decay = tensors_[5].const_data_ptr<float>();
      auxiliary_.magnetic_curl = tensors_[6].const_data_ptr<float>();
      auxiliary_.signals = tensors_[7].const_data_ptr<float>();
      auxiliary_.boundary_signals = tensors_[8].const_data_ptr<float>();
      auxiliary_.boundary_h = tensors_[9].mutable_data_ptr<float>();
      auxiliary_.electric_count = tensors_[1].numel();
      auxiliary_.magnetic_count = tensors_[2].numel();
      auxiliary_.source_index = integers[0];
      auxiliary_.boundary_width = integers[0];
      STD_TORCH_CHECK(auxiliary_.electric_count == auxiliary_.magnetic_count + 1,
          "invalid auxiliary Yee line lengths");
      STD_TORCH_CHECK(tensors_[3].numel() == auxiliary_.electric_count &&
          tensors_[4].numel() == auxiliary_.electric_count &&
          tensors_[5].numel() == auxiliary_.magnetic_count &&
          tensors_[6].numel() == auxiliary_.magnetic_count,
          "auxiliary loss arrays must match the full line");
      if (operation_ == 4) {
        STD_TORCH_CHECK(auxiliary_.source_index == 2 && tensors_[8].dim() == 2 &&
            tensors_[8].size(1) == 2 && tensors_[9].numel() == 2,
            "boundary continuation requires the two-cell incident stencil");
      } else {
        auxiliary_.reference = integers[1];
        auxiliary_.boundary = integers[2];
        for (int a = 0; a < 3; ++a) auxiliary_.fields[a] = tensors_[10+a].mutable_data_ptr<float>();
        auxiliary_.coeff = tensors_[13].const_data_ptr<float>();
        if (auxiliary_.reference) auxiliary_.indices = tensors_[14].const_data_ptr<int>();
        else auxiliary_.positions = tensors_[14].const_data_ptr<float>();
        auxiliary_.gather = {tensors_[15].const_data_ptr<int64_t>(), tensors_[16].const_data_ptr<int64_t>(),
            tensors_[17].const_data_ptr<int64_t>(), tensors_[16].size(0)};
        auxiliary_.origin = float(scalars[0]);
        auxiliary_.inv_ds = float(scalars[1]);
      }
    } else if (operation_ == 7) {
      STD_TORCH_CHECK(tensors_.size() == 12 && integers.size() == 8 && scalars.size() == 12,
          "invalid core Bloch source");
      for (int index = 6; index < 9; ++index) {
        STD_TORCH_CHECK(tensors_[index].is_cuda() && tensors_[index].is_contiguous() &&
            tensors_[index].device() == tensors_[1].device() &&
            tensors_[index].scalar_type() == torch::headeronly::ScalarType::Long, "invalid core Bloch gather");
      }
      STD_TORCH_CHECK(tensors_[6].dim() == 1 && tensors_[7].dim() == 2 && tensors_[7].size(1) == 2 &&
          tensors_[8].dim() == 2 && tensors_[8].size(1) == 2 && tensors_[6].numel() == tensors_[7].size(0)+1,
          "invalid core Bloch gather shape");
      bloch_.kind = integers[5];
      STD_TORCH_CHECK(bloch_.kind == 2 || (tensors_[5].dim() == 2 && tensors_[5].size(1) == 2),
          "Bloch signals must be [steps, 2]");
      STD_TORCH_CHECK(bloch_.kind != 2 || (tensors_[9].numel() == tensors_[3].numel() &&
          (!integers[7] || tensors_[10].numel() == tensors_[3].numel())), "invalid Bloch delay patches");
      bloch_.step = step_;
      bloch_.real = tensors_[1].mutable_data_ptr<float>();
      bloch_.imag = tensors_[2].mutable_data_ptr<float>();
      bloch_.patch = tensors_[3].const_data_ptr<float>();
      bloch_.second = tensors_[4].const_data_ptr<float>();
      bloch_.signals = tensors_[5].const_data_ptr<float>();
      for (int a = 0; a < 3; ++a) {
        bloch_.patch_shape[a] = tensors_[3].size(a);
        bloch_.offsets[a] = integers[a];
      }
      bloch_.axis_a = integers[3]; bloch_.axis_b = integers[4];
      bloch_.waveform.kind = integers[6]; bloch_.causal = integers[7];
      bloch_.ca = float(scalars[0]); bloch_.sa = float(scalars[1]);
      bloch_.cb = float(scalars[2]); bloch_.sb = float(scalars[3]);
      // Each scalar is the float32 cast torch applies to its Python double operand.
      bloch_.waveform.amplitude = float(scalars[4]); bloch_.waveform.two_pi_frequency = float(scalars[5]);
      bloch_.waveform.phase = float(scalars[6]); bloch_.waveform.delay = float(scalars[7]);
      bloch_.waveform.inv_sigma = float(scalars[8]); bloch_.waveform.pedestal = float(scalars[9]);
      bloch_.waveform.pedestal_inv_sigma = float(scalars[10]); bloch_.waveform.pi_frequency = float(scalars[11]);
      bloch_.times = tensors_[11].const_data_ptr<float>();
      bloch_.delays = tensors_[9].const_data_ptr<float>();
      bloch_.activation = tensors_[10].const_data_ptr<float>();
      bloch_.gather = {tensors_[6].const_data_ptr<int64_t>(), tensors_[7].const_data_ptr<int64_t>(),
          tensors_[8].const_data_ptr<int64_t>(), tensors_[7].size(0)};
    } else if (operation_ == 2) {
      dft_.step = step_;
      dft_.cosine = tensors_[1].const_data_ptr<float>();
      dft_.sine = tensors_[2].const_data_ptr<float>();
      dft_.frequencies = tensors_[1].size(1);
      for (int a = 0; a < 3; ++a) {
        dft_.field[a] = tensors_[3+3*a].const_data_ptr<float>();
        dft_.real[a] = tensors_[4+3*a].mutable_data_ptr<float>();
        dft_.imag[a] = tensors_[5+3*a].mutable_data_ptr<float>();
        dft_.sizes[a] = tensors_[3+3*a].numel();
      }
    }
  }

  void compile_delayed(torch::stable::Tensor delays, torch::stable::Tensor output) {
    torch::stable::accelerator::DeviceGuard guard(tensors_[0].get_device_index());
    auto source = source_;
    source.delays = delays.const_data_ptr<float>();
    source.delay_count = delays.numel();
    compile_delayed_amplitudes(source, output.mutable_data_ptr<float>(), output.size(0),
        current_cuda_stream());
  }

  void launch() override {
    torch::stable::accelerator::DeviceGuard guard(tensors_[0].get_device_index());
    const auto stream = current_cuda_stream();
    switch (operation_) {
      case 0: source_patch_indexed(source_, stream); break;
      case 1: auxiliary_electric_indexed(tensors_[1].numel(), source_index_,
          tensors_[1].mutable_data_ptr<float>(), tensors_[2].mutable_data_ptr<float>(),
          tensors_[3].mutable_data_ptr<float>(), tensors_[4].mutable_data_ptr<float>(),
          tensors_[5].mutable_data_ptr<float>(), step_, stream); break;
      case 2: dft_indexed(dft_, stream); break;
      case 3: commit_step(step_, stream); break;
      case 4: auxiliary_boundary_indexed(auxiliary_, stream); break;
      case 5: auxiliary_correction_indexed(auxiliary_, true, stream); break;
      case 6: auxiliary_correction_indexed(auxiliary_, false, stream); break;
      case 7: bloch_source_indexed(bloch_, stream); break;
    }
  }

 private:
  int operation_, source_index_ = 0;
  std::vector<torch::stable::Tensor> tensors_;
  int64_t* step_;
  CoreSource source_{};
  CoreDft dft_{};
  CoreAuxiliary auxiliary_{};
  CoreBlochSource bloch_{};
};

int64_t core_launch(std::vector<torch::stable::Tensor> tensors, std::vector<int64_t> integers,
                    std::vector<double> coefficients, bool electric, int64_t cpml,
                    bool uniform, bool wide_indices, bool live, bool modulated) {
  return static_cast<int64_t>(reinterpret_cast<intptr_t>(new CoreLaunch(
      std::move(tensors), std::move(integers), std::move(coefficients),
      electric, cpml, uniform, wide_indices, live, modulated)));
}

int64_t core_step_launch(int64_t operation, std::vector<torch::stable::Tensor> tensors,
                        std::vector<int64_t> integers, std::vector<double> scalars) {
  return static_cast<int64_t>(reinterpret_cast<intptr_t>(new CoreStepLaunch(
      operation, std::move(tensors), std::move(integers), std::move(scalars))));
}

int64_t core_observer_gather(std::vector<torch::stable::Tensor> fields,
                            std::vector<torch::stable::Tensor> offsets,
                            std::vector<torch::stable::Tensor> frequency_maps,
                            std::vector<torch::stable::Tensor> real,
                            std::vector<torch::stable::Tensor> imag,
                            std::vector<bool> magnetic,
                            torch::stable::Tensor weights, torch::stable::Tensor active) {
  return static_cast<int64_t>(reinterpret_cast<intptr_t>(new CoreObserverGather(
      std::move(fields), std::move(offsets), std::move(frequency_maps),
      std::move(real), std::move(imag), std::move(magnetic),
      std::move(weights), std::move(active))));
}

void launch(int64_t handle) {
  reinterpret_cast<CoreHandle*>(static_cast<intptr_t>(handle))->launch();
}

void compile_delayed(int64_t handle, torch::stable::Tensor delays, torch::stable::Tensor output) {
  static_cast<CoreStepLaunch*>(reinterpret_cast<CoreHandle*>(static_cast<intptr_t>(handle)))
      ->compile_delayed(std::move(delays), std::move(output));
}

void release(int64_t handle) {
  delete reinterpret_cast<CoreHandle*>(static_cast<intptr_t>(handle));
}

STABLE_TORCH_LIBRARY(witwin_maxwell_fdtd_core, m) {
  m.def("core_launch(Tensor[] tensors, int[] integers, float[] coefficients, bool electric, int cpml, bool uniform, bool wide_indices, bool live, bool modulated) -> int");
  m.def("core_step_launch(int operation, Tensor[] tensors, int[] integers, float[] scalars) -> int");
  m.def("core_observer_gather(Tensor[] fields, Tensor[] offsets, Tensor[] frequency_maps, Tensor[] real, Tensor[] imag, bool[] magnetic, Tensor weights, Tensor active) -> int");
  m.def("launch(int handle) -> ()");
  m.def("compile_delayed(int handle, Tensor delays, Tensor(a!) output) -> ()");
  m.def("release(int handle) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(witwin_maxwell_fdtd_core, CompositeExplicitAutograd, m) {
  m.impl("core_launch", TORCH_BOX(&core_launch));
  m.impl("core_step_launch", TORCH_BOX(&core_step_launch));
  m.impl("core_observer_gather", TORCH_BOX(&core_observer_gather));
  m.impl("launch", TORCH_BOX(&launch));
  m.impl("compile_delayed", TORCH_BOX(&compile_delayed));
  m.impl("release", TORCH_BOX(&release));
}
