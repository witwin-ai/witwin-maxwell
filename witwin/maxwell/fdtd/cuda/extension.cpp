#include <torch/extension.h>
#include <torch/cuda.h>

#include <vector>

bool is_available() {
  return torch::cuda::is_available();
}

void synchronize_noop_cuda();
std::vector<at::Tensor> debug_linear_indices_cuda(std::vector<int64_t> shape);
void update_magnetic_hx_standard_cuda(
    at::Tensor hx,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dy,
    double inv_dz);
void update_magnetic_hy_standard_cuda(
    at::Tensor hy,
    const at::Tensor& ex,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dx,
    double inv_dz);
void update_magnetic_hz_standard_cuda(
    at::Tensor hz,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dx,
    double inv_dy);
void update_magnetic_hx_cpml_cuda(
    at::Tensor hx,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_y,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dy,
    double inv_dz);
void update_magnetic_hy_cpml_cuda(
    at::Tensor hy,
    const at::Tensor& ex,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dx,
    double inv_dz);
void update_magnetic_hz_cpml_cuda(
    at::Tensor hz,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_y,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    double inv_dx,
    double inv_dy);
void update_magnetic_hx_cpml_compressed_cuda(
    at::Tensor hx,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_y,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dy,
    double inv_dz,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    c10::optional<double> uniform_decay,
    c10::optional<double> uniform_curl);
void update_magnetic_hy_cpml_compressed_cuda(
    at::Tensor hy,
    const at::Tensor& ex,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dx,
    double inv_dz,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    c10::optional<double> uniform_decay,
    c10::optional<double> uniform_curl);
void update_magnetic_hz_cpml_compressed_cuda(
    at::Tensor hz,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_y,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    double inv_dx,
    double inv_dy,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    c10::optional<double> uniform_decay,
    c10::optional<double> uniform_curl);
void update_electric_ex_standard_cuda(
    at::Tensor ex,
    const at::Tensor& hy,
    const at::Tensor& hz,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dy,
    double inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ey_standard_cuda(
    at::Tensor ey,
    const at::Tensor& hx,
    const at::Tensor& hz,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dx,
    double inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ez_standard_cuda(
    at::Tensor ez,
    const at::Tensor& hx,
    const at::Tensor& hy,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dx,
    double inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode);
void update_electric_ex_bloch_cuda(
    at::Tensor ex_real,
    at::Tensor ex_imag,
    const at::Tensor& hy_real,
    const at::Tensor& hy_imag,
    const at::Tensor& hz_real,
    const at::Tensor& hz_imag,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dy,
    double inv_dz);
void update_electric_ey_bloch_cuda(
    at::Tensor ey_real,
    at::Tensor ey_imag,
    const at::Tensor& hx_real,
    const at::Tensor& hx_imag,
    const at::Tensor& hz_real,
    const at::Tensor& hz_imag,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dx,
    double inv_dz);
void update_electric_ez_bloch_cuda(
    at::Tensor ez_real,
    at::Tensor ez_imag,
    const at::Tensor& hx_real,
    const at::Tensor& hx_imag,
    const at::Tensor& hy_real,
    const at::Tensor& hy_imag,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    double inv_dx,
    double inv_dy);
void update_electric_ex_cpml_cuda(
    at::Tensor ex,
    const at::Tensor& hy,
    const at::Tensor& hz,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_y,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dy,
    double inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ey_cpml_cuda(
    at::Tensor ey,
    const at::Tensor& hx,
    const at::Tensor& hz,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dx,
    double inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ez_cpml_cuda(
    at::Tensor ez,
    const at::Tensor& hx,
    const at::Tensor& hy,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_y,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    double inv_dx,
    double inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode);
void update_electric_ex_cpml_compressed_cuda(
    at::Tensor ex,
    const at::Tensor& hy,
    const at::Tensor& hz,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_y,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dy,
    double inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    c10::optional<double> uniform_decay,
    c10::optional<double> uniform_curl);
void update_electric_ey_cpml_compressed_cuda(
    at::Tensor ey,
    const at::Tensor& hx,
    const at::Tensor& hz,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dx,
    double inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    c10::optional<double> uniform_decay,
    c10::optional<double> uniform_curl);
void update_electric_ez_cpml_compressed_cuda(
    at::Tensor ez,
    const at::Tensor& hx,
    const at::Tensor& hy,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_y,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    double inv_dx,
    double inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    c10::optional<double> uniform_decay,
    c10::optional<double> uniform_curl);
void accumulate_dft_batched_cuda(
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& ez,
    at::Tensor ex_real,
    at::Tensor ex_imag,
    at::Tensor ey_real,
    at::Tensor ey_imag,
    at::Tensor ez_real,
    at::Tensor ez_imag,
    const at::Tensor& weighted_cos,
    const at::Tensor& weighted_sin);
void accumulate_point_observers_cuda(
    const at::Tensor& field,
    const at::Tensor& point_i,
    const at::Tensor& point_j,
    const at::Tensor& point_k,
    at::Tensor real_accum,
    at::Tensor imag_accum,
    double weighted_cos,
    double weighted_sin);
void accumulate_plane_observer_cuda(
    const at::Tensor& field,
    at::Tensor real_accum,
    at::Tensor imag_accum,
    int64_t axis,
    int64_t plane_index,
    double weighted_cos,
    double weighted_sin);
void update_debye_current_cuda(
    const at::Tensor& electric,
    at::Tensor polarization,
    at::Tensor current,
    const at::Tensor& drive,
    double decay,
    double dt);
void update_drude_current_cuda(
    const at::Tensor& electric,
    at::Tensor current,
    const at::Tensor& drive,
    double decay);
void update_lorentz_current_cuda(
    const at::Tensor& electric,
    at::Tensor polarization,
    at::Tensor current,
    const at::Tensor& drive,
    double decay,
    double restoring,
    double dt);
void apply_polarization_current_cuda(
    at::Tensor electric,
    const at::Tensor& current,
    const at::Tensor& inv_permittivity,
    double dt);
void update_kerr_ex_curl_cuda(
    at::Tensor dynamic_curl,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& linear_permittivity,
    const at::Tensor& ex_decay,
    const at::Tensor& chi3,
    double dt,
    double eps0);
void update_kerr_ey_curl_cuda(
    at::Tensor dynamic_curl,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& linear_permittivity,
    const at::Tensor& ey_decay,
    const at::Tensor& chi3,
    double dt,
    double eps0);
void update_kerr_ez_curl_cuda(
    at::Tensor dynamic_curl,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& linear_permittivity,
    const at::Tensor& ez_decay,
    const at::Tensor& chi3,
    double dt,
    double eps0);
void add_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal);
void add_cw_phased_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch_cos,
    const at::Tensor& patch_sin,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal_cos,
    double signal_sin);
void add_time_shifted_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    const at::Tensor& delay_patch,
    const at::Tensor& activation_delay_patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t time_kind,
    double time,
    double frequency,
    double fwidth,
    double amplitude,
    double phase,
    double delay,
    int64_t causal_gate);
void add_source_patch_periodic_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal,
    int64_t axis_a,
    int64_t axis_b,
    int64_t wrap_a,
    int64_t wrap_b);
void add_source_patch_bloch_cuda(
    at::Tensor ex_real,
    at::Tensor ex_imag,
    at::Tensor ey_real,
    at::Tensor ey_imag,
    at::Tensor ez_real,
    at::Tensor ez_imag,
    const at::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal_real,
    double signal_imag,
    int64_t axis_code,
    double phase_cos_a,
    double phase_sin_a,
    double phase_cos_b,
    double phase_sin_b,
    int64_t wrap_axis_a,
    int64_t wrap_axis_b);
void add_scaled_slice_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    const at::Tensor& incident,
    int64_t sample_index,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale);
void add_scaled_line_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    const at::Tensor& incident,
    const at::Tensor& sample_indices,
    int64_t sample_axis,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale);
void add_interpolated_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    const at::Tensor& incident,
    const at::Tensor& sample_positions,
    double origin,
    double ds,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale);
void add_batched_reference_source_patches_cuda(
    at::Tensor field_x,
    at::Tensor field_y,
    at::Tensor field_z,
    const at::Tensor& coeff_data,
    const at::Tensor& incident,
    const at::Tensor& field_codes_per_coeff,
    const at::Tensor& field_offsets,
    const at::Tensor& sample_indices_per_coeff);
void add_batched_interpolated_source_patches_cuda(
    at::Tensor field_x,
    at::Tensor field_y,
    at::Tensor field_z,
    const at::Tensor& coeff_data,
    const at::Tensor& incident,
    const at::Tensor& sample_positions,
    const at::Tensor& field_codes_per_coeff,
    const at::Tensor& field_offsets,
    double origin,
    double ds);
void update_auxiliary_magnetic_cuda(
    at::Tensor magnetic,
    const at::Tensor& electric,
    const at::Tensor& decay,
    const at::Tensor& curl);
void update_auxiliary_electric_cuda(
    at::Tensor electric,
    const at::Tensor& magnetic,
    const at::Tensor& decay,
    const at::Tensor& curl,
    int64_t source_index,
    double source_value);
void reverse_magnetic_adjoint_decay_cuda(
    at::Tensor adj_prev,
    const at::Tensor& adj_mid,
    const at::Tensor& decay);
void reverse_electric_adjoint_to_hx_standard_cuda(
    at::Tensor adj_hx_mid,
    const at::Tensor& adj_hx_post,
    const at::Tensor& adj_ey_post,
    const at::Tensor& adj_ez_post,
    const at::Tensor& ey_curl,
    const at::Tensor& ez_curl,
    double inv_dy,
    double inv_dz);
void reverse_electric_adjoint_to_hy_standard_cuda(
    at::Tensor adj_hy_mid,
    const at::Tensor& adj_hy_post,
    const at::Tensor& adj_ex_post,
    const at::Tensor& adj_ez_post,
    const at::Tensor& ex_curl,
    const at::Tensor& ez_curl,
    double inv_dx,
    double inv_dz);
void reverse_electric_adjoint_to_hz_standard_cuda(
    at::Tensor adj_hz_mid,
    const at::Tensor& adj_hz_post,
    const at::Tensor& adj_ex_post,
    const at::Tensor& adj_ey_post,
    const at::Tensor& ex_curl,
    const at::Tensor& ey_curl,
    double inv_dx,
    double inv_dy);
void reverse_magnetic_adjoint_to_ex_standard_cuda(
    at::Tensor adj_ex_prev,
    at::Tensor grad_eps_ex,
    const at::Tensor& adj_ex_post,
    const at::Tensor& adj_hy_mid,
    const at::Tensor& adj_hz_mid,
    const at::Tensor& ex_decay,
    const at::Tensor& ex_curl,
    const at::Tensor& eps_ex,
    const at::Tensor& hy_mid,
    const at::Tensor& hz_mid,
    const at::Tensor& hy_curl,
    const at::Tensor& hz_curl,
    double inv_dy,
    double inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void reverse_magnetic_adjoint_to_ey_standard_cuda(
    at::Tensor adj_ey_prev,
    at::Tensor grad_eps_ey,
    const at::Tensor& adj_ey_post,
    const at::Tensor& adj_hx_mid,
    const at::Tensor& adj_hz_mid,
    const at::Tensor& ey_decay,
    const at::Tensor& ey_curl,
    const at::Tensor& eps_ey,
    const at::Tensor& hx_mid,
    const at::Tensor& hz_mid,
    const at::Tensor& hx_curl,
    const at::Tensor& hz_curl,
    double inv_dx,
    double inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void reverse_magnetic_adjoint_to_ez_standard_cuda(
    at::Tensor adj_ez_prev,
    at::Tensor grad_eps_ez,
    const at::Tensor& adj_ez_post,
    const at::Tensor& adj_hx_mid,
    const at::Tensor& adj_hy_mid,
    const at::Tensor& ez_decay,
    const at::Tensor& ez_curl,
    const at::Tensor& eps_ez,
    const at::Tensor& hx_mid,
    const at::Tensor& hy_mid,
    const at::Tensor& hx_curl,
    const at::Tensor& hy_curl,
    double inv_dx,
    double inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode);
void reverse_electric_adjoint_to_hx_bloch_cuda(
    at::Tensor adj_hx_mid_real,
    at::Tensor adj_hx_mid_imag,
    const at::Tensor& adj_hx_post_real,
    const at::Tensor& adj_hx_post_imag,
    const at::Tensor& adj_ey_post_real,
    const at::Tensor& adj_ey_post_imag,
    const at::Tensor& adj_ez_post_real,
    const at::Tensor& adj_ez_post_imag,
    const at::Tensor& ey_curl,
    const at::Tensor& ez_curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dy,
    double inv_dz);
void reverse_electric_adjoint_to_hy_bloch_cuda(
    at::Tensor adj_hy_mid_real,
    at::Tensor adj_hy_mid_imag,
    const at::Tensor& adj_hy_post_real,
    const at::Tensor& adj_hy_post_imag,
    const at::Tensor& adj_ex_post_real,
    const at::Tensor& adj_ex_post_imag,
    const at::Tensor& adj_ez_post_real,
    const at::Tensor& adj_ez_post_imag,
    const at::Tensor& ex_curl,
    const at::Tensor& ez_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dx,
    double inv_dz);
void reverse_electric_adjoint_to_hz_bloch_cuda(
    at::Tensor adj_hz_mid_real,
    at::Tensor adj_hz_mid_imag,
    const at::Tensor& adj_hz_post_real,
    const at::Tensor& adj_hz_post_imag,
    const at::Tensor& adj_ex_post_real,
    const at::Tensor& adj_ex_post_imag,
    const at::Tensor& adj_ey_post_real,
    const at::Tensor& adj_ey_post_imag,
    const at::Tensor& ex_curl,
    const at::Tensor& ey_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    double inv_dx,
    double inv_dy);
void reverse_magnetic_adjoint_to_ex_bloch_cuda(
    at::Tensor adj_ex_prev_real,
    at::Tensor adj_ex_prev_imag,
    at::Tensor grad_eps_ex,
    const at::Tensor& adj_ex_post_real,
    const at::Tensor& adj_ex_post_imag,
    const at::Tensor& adj_hy_mid_real,
    const at::Tensor& adj_hy_mid_imag,
    const at::Tensor& adj_hz_mid_real,
    const at::Tensor& adj_hz_mid_imag,
    const at::Tensor& ex_decay,
    const at::Tensor& ex_curl,
    const at::Tensor& eps_ex,
    const at::Tensor& hy_mid_real,
    const at::Tensor& hy_mid_imag,
    const at::Tensor& hz_mid_real,
    const at::Tensor& hz_mid_imag,
    const at::Tensor& hy_curl,
    const at::Tensor& hz_curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dy,
    double inv_dz);
void reverse_magnetic_adjoint_to_ey_bloch_cuda(
    at::Tensor adj_ey_prev_real,
    at::Tensor adj_ey_prev_imag,
    at::Tensor grad_eps_ey,
    const at::Tensor& adj_ey_post_real,
    const at::Tensor& adj_ey_post_imag,
    const at::Tensor& adj_hx_mid_real,
    const at::Tensor& adj_hx_mid_imag,
    const at::Tensor& adj_hz_mid_real,
    const at::Tensor& adj_hz_mid_imag,
    const at::Tensor& ey_decay,
    const at::Tensor& ey_curl,
    const at::Tensor& eps_ey,
    const at::Tensor& hx_mid_real,
    const at::Tensor& hx_mid_imag,
    const at::Tensor& hz_mid_real,
    const at::Tensor& hz_mid_imag,
    const at::Tensor& hx_curl,
    const at::Tensor& hz_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dx,
    double inv_dz);
void reverse_magnetic_adjoint_to_ez_bloch_cuda(
    at::Tensor adj_ez_prev_real,
    at::Tensor adj_ez_prev_imag,
    at::Tensor grad_eps_ez,
    const at::Tensor& adj_ez_post_real,
    const at::Tensor& adj_ez_post_imag,
    const at::Tensor& adj_hx_mid_real,
    const at::Tensor& adj_hx_mid_imag,
    const at::Tensor& adj_hy_mid_real,
    const at::Tensor& adj_hy_mid_imag,
    const at::Tensor& ez_decay,
    const at::Tensor& ez_curl,
    const at::Tensor& eps_ez,
    const at::Tensor& hx_mid_real,
    const at::Tensor& hx_mid_imag,
    const at::Tensor& hy_mid_real,
    const at::Tensor& hy_mid_imag,
    const at::Tensor& hx_curl,
    const at::Tensor& hy_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    double inv_dx,
    double inv_dy);
void accumulate_forward_diff_adjoint_cuda(
    at::Tensor field_grad,
    const at::Tensor& diff_grad,
    int64_t axis,
    double inv_delta);
void accumulate_backward_diff_adjoint_cuda(
    at::Tensor field_grad,
    const at::Tensor& diff_grad,
    int64_t axis,
    double inv_delta);
void reverse_electric_component_ex_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor grad_eps,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& eps,
    const at::Tensor& psi_pos,
    const at::Tensor& psi_neg,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg,
    const at::Tensor& hy_mid,
    const at::Tensor& hz_mid,
    double inv_dy,
    double inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void reverse_electric_component_ey_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor grad_eps,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& eps,
    const at::Tensor& psi_pos,
    const at::Tensor& psi_neg,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg,
    const at::Tensor& hx_mid,
    const at::Tensor& hz_mid,
    double inv_dx,
    double inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void reverse_electric_component_ez_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor grad_eps,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& eps,
    const at::Tensor& psi_pos,
    const at::Tensor& psi_neg,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg,
    const at::Tensor& hx_mid,
    const at::Tensor& hy_mid,
    double inv_dx,
    double inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode);
void reverse_magnetic_component_hx_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg);
void reverse_magnetic_component_hy_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg);
void reverse_magnetic_component_hz_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg);
void reverse_debye_current_cuda(
    at::Tensor adj_electric_prev,
    at::Tensor adj_polarization_prev,
    const at::Tensor& adj_polarization_post,
    const at::Tensor& adj_current_post,
    const at::Tensor& drive,
    double decay,
    double dt);
void reverse_drude_current_cuda(
    at::Tensor adj_electric_prev,
    at::Tensor adj_current_prev,
    const at::Tensor& adj_current_post,
    const at::Tensor& drive,
    double decay);
void reverse_lorentz_current_cuda(
    at::Tensor adj_electric_prev,
    at::Tensor adj_polarization_prev,
    at::Tensor adj_current_prev,
    const at::Tensor& adj_polarization_post,
    const at::Tensor& adj_current_post,
    const at::Tensor& drive,
    double decay,
    double restoring,
    double dt);
void accumulate_tfsf_scalar_sample_adjoint_cuda(
    at::Tensor adj_aux_field,
    const at::Tensor& adj_field_patch,
    const at::Tensor& coeff_patch,
    int64_t sample_index,
    double component_scale);
void accumulate_tfsf_line_sample_adjoint_cuda(
    at::Tensor adj_aux_field,
    const at::Tensor& adj_field_patch,
    const at::Tensor& coeff_patch,
    const at::Tensor& sample_indices,
    int64_t sample_axis_code,
    double component_scale);
void accumulate_tfsf_interpolated_sample_adjoint_cuda(
    at::Tensor adj_aux_field,
    const at::Tensor& adj_field_patch,
    const at::Tensor& coeff_patch,
    const at::Tensor& sample_positions,
    double origin,
    double ds,
    double component_scale);
void reverse_tfsf_auxiliary_electric_cuda(
    at::Tensor adj_electric_prev,
    at::Tensor adj_magnetic_after,
    const at::Tensor& adj_electric_post,
    const at::Tensor& electric_decay,
    const at::Tensor& electric_curl,
    int64_t source_index);
void reverse_tfsf_auxiliary_magnetic_cuda(
    at::Tensor adj_electric_prev,
    at::Tensor adj_magnetic_prev,
    const at::Tensor& adj_magnetic_after,
    const at::Tensor& magnetic_decay,
    const at::Tensor& magnetic_curl);
void clamp_field_face_cuda(at::Tensor field, int64_t axis, int64_t side);
void clamp_pec_boundary_cuda(at::Tensor field, int64_t axis_a, int64_t axis_b);
void project_periodic_boundary_cuda(at::Tensor field, int64_t axis);
void project_bloch_boundary_cuda(
    at::Tensor field_real,
    at::Tensor field_imag,
    int64_t axis,
    double phase_cos,
    double phase_sin);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_available", &is_available, "Return whether CUDA is available to PyTorch.");
  m.def("synchronize_noop", &synchronize_noop_cuda, "Launch a no-op kernel on the current CUDA stream.");
  m.def("debug_linear_indices", &debug_linear_indices_cuda, "Return row-major linear/i/j/k helper tensors.");
  m.def("update_magnetic_hx_standard", &update_magnetic_hx_standard_cuda, "Update standard Hx field.");
  m.def("update_magnetic_hy_standard", &update_magnetic_hy_standard_cuda, "Update standard Hy field.");
  m.def("update_magnetic_hz_standard", &update_magnetic_hz_standard_cuda, "Update standard Hz field.");
  m.def("update_magnetic_hx_cpml", &update_magnetic_hx_cpml_cuda, "Update dense CPML Hx field.");
  m.def("update_magnetic_hy_cpml", &update_magnetic_hy_cpml_cuda, "Update dense CPML Hy field.");
  m.def("update_magnetic_hz_cpml", &update_magnetic_hz_cpml_cuda, "Update dense CPML Hz field.");
  m.def("update_magnetic_hx_cpml_compressed", &update_magnetic_hx_cpml_compressed_cuda, "Update compressed CPML Hx field.");
  m.def("update_magnetic_hy_cpml_compressed", &update_magnetic_hy_cpml_compressed_cuda, "Update compressed CPML Hy field.");
  m.def("update_magnetic_hz_cpml_compressed", &update_magnetic_hz_cpml_compressed_cuda, "Update compressed CPML Hz field.");
  m.def("update_electric_ex_standard", &update_electric_ex_standard_cuda, "Update standard Ex field.");
  m.def("update_electric_ey_standard", &update_electric_ey_standard_cuda, "Update standard Ey field.");
  m.def("update_electric_ez_standard", &update_electric_ez_standard_cuda, "Update standard Ez field.");
  m.def("update_electric_ex_bloch", &update_electric_ex_bloch_cuda, "Update Bloch Ex field.");
  m.def("update_electric_ey_bloch", &update_electric_ey_bloch_cuda, "Update Bloch Ey field.");
  m.def("update_electric_ez_bloch", &update_electric_ez_bloch_cuda, "Update Bloch Ez field.");
  m.def("update_electric_ex_cpml", &update_electric_ex_cpml_cuda, "Update dense CPML Ex field.");
  m.def("update_electric_ey_cpml", &update_electric_ey_cpml_cuda, "Update dense CPML Ey field.");
  m.def("update_electric_ez_cpml", &update_electric_ez_cpml_cuda, "Update dense CPML Ez field.");
  m.def("update_electric_ex_cpml_compressed", &update_electric_ex_cpml_compressed_cuda, "Update compressed CPML Ex field.");
  m.def("update_electric_ey_cpml_compressed", &update_electric_ey_cpml_compressed_cuda, "Update compressed CPML Ey field.");
  m.def("update_electric_ez_cpml_compressed", &update_electric_ez_cpml_compressed_cuda, "Update compressed CPML Ez field.");
  m.def("accumulate_dft_batched", &accumulate_dft_batched_cuda, "Accumulate batched Yee-grid DFT fields.");
  m.def("accumulate_point_observers", &accumulate_point_observers_cuda, "Accumulate point observer samples.");
  m.def("accumulate_plane_observer", &accumulate_plane_observer_cuda, "Accumulate plane observer samples.");
  m.def("update_debye_current", &update_debye_current_cuda, "Update Debye polarization state.");
  m.def("update_drude_current", &update_drude_current_cuda, "Update Drude current state.");
  m.def("update_lorentz_current", &update_lorentz_current_cuda, "Update Lorentz polarization state.");
  m.def("apply_polarization_current", &apply_polarization_current_cuda, "Apply polarization current correction.");
  m.def("update_kerr_ex_curl", &update_kerr_ex_curl_cuda, "Update Kerr-adjusted Ex curl coefficient.");
  m.def("update_kerr_ey_curl", &update_kerr_ey_curl_cuda, "Update Kerr-adjusted Ey curl coefficient.");
  m.def("update_kerr_ez_curl", &update_kerr_ez_curl_cuda, "Update Kerr-adjusted Ez curl coefficient.");
  m.def("add_source_patch", &add_source_patch_cuda, "Add a uniform source patch.");
  m.def("add_cw_phased_source_patch", &add_cw_phased_source_patch_cuda, "Add a CW phased source patch.");
  m.def("add_time_shifted_source_patch", &add_time_shifted_source_patch_cuda, "Add a time-shifted source patch.");
  m.def(
      "add_source_patch_ex_periodic",
      [](at::Tensor field,
         const at::Tensor& patch,
         int64_t offset_i,
         int64_t offset_j,
         int64_t offset_k,
         double signal,
         int64_t wrap_a,
         int64_t wrap_b) {
        add_source_patch_periodic_cuda(field, patch, offset_i, offset_j, offset_k, signal, 1, 2, wrap_a, wrap_b);
      },
      "Add periodic Ex source patch.");
  m.def(
      "add_source_patch_ey_periodic",
      [](at::Tensor field,
         const at::Tensor& patch,
         int64_t offset_i,
         int64_t offset_j,
         int64_t offset_k,
         double signal,
         int64_t wrap_a,
         int64_t wrap_b) {
        add_source_patch_periodic_cuda(field, patch, offset_i, offset_j, offset_k, signal, 0, 2, wrap_a, wrap_b);
      },
      "Add periodic Ey source patch.");
  m.def(
      "add_source_patch_ez_periodic",
      [](at::Tensor field,
         const at::Tensor& patch,
         int64_t offset_i,
         int64_t offset_j,
         int64_t offset_k,
         double signal,
         int64_t wrap_a,
         int64_t wrap_b) {
        add_source_patch_periodic_cuda(field, patch, offset_i, offset_j, offset_k, signal, 0, 1, wrap_a, wrap_b);
      },
      "Add periodic Ez source patch.");
  m.def("add_source_patch_bloch", &add_source_patch_bloch_cuda, "Add a Bloch-periodic source patch.");
  m.def("add_scaled_slice_source_patch", &add_scaled_slice_source_patch_cuda, "Add a scaled TFSF slice source patch.");
  m.def("add_scaled_line_source_patch", &add_scaled_line_source_patch_cuda, "Add a scaled TFSF line source patch.");
  m.def("add_interpolated_source_patch", &add_interpolated_source_patch_cuda, "Add an interpolated TFSF source patch.");
  m.def("add_batched_reference_source_patches", &add_batched_reference_source_patches_cuda, "Add batched reference TFSF patches.");
  m.def("add_batched_interpolated_source_patches", &add_batched_interpolated_source_patches_cuda, "Add batched interpolated TFSF patches.");
  m.def("update_auxiliary_magnetic", &update_auxiliary_magnetic_cuda, "Update auxiliary 1D magnetic field.");
  m.def("update_auxiliary_electric", &update_auxiliary_electric_cuda, "Update auxiliary 1D electric field.");
  m.def("reverse_magnetic_adjoint_decay", &reverse_magnetic_adjoint_decay_cuda, "Reverse magnetic adjoint decay.");
  m.def("reverse_electric_adjoint_to_hx_standard", &reverse_electric_adjoint_to_hx_standard_cuda, "Reverse standard electric update to Hx adjoint.");
  m.def("reverse_electric_adjoint_to_hy_standard", &reverse_electric_adjoint_to_hy_standard_cuda, "Reverse standard electric update to Hy adjoint.");
  m.def("reverse_electric_adjoint_to_hz_standard", &reverse_electric_adjoint_to_hz_standard_cuda, "Reverse standard electric update to Hz adjoint.");
  m.def("reverse_magnetic_adjoint_to_ex_standard", &reverse_magnetic_adjoint_to_ex_standard_cuda, "Reverse standard magnetic update to Ex adjoint and epsilon gradient.");
  m.def("reverse_magnetic_adjoint_to_ey_standard", &reverse_magnetic_adjoint_to_ey_standard_cuda, "Reverse standard magnetic update to Ey adjoint and epsilon gradient.");
  m.def("reverse_magnetic_adjoint_to_ez_standard", &reverse_magnetic_adjoint_to_ez_standard_cuda, "Reverse standard magnetic update to Ez adjoint and epsilon gradient.");
  m.def("reverse_electric_adjoint_to_hx_bloch", &reverse_electric_adjoint_to_hx_bloch_cuda, "Reverse Bloch electric update to Hx adjoint.");
  m.def("reverse_electric_adjoint_to_hy_bloch", &reverse_electric_adjoint_to_hy_bloch_cuda, "Reverse Bloch electric update to Hy adjoint.");
  m.def("reverse_electric_adjoint_to_hz_bloch", &reverse_electric_adjoint_to_hz_bloch_cuda, "Reverse Bloch electric update to Hz adjoint.");
  m.def("reverse_magnetic_adjoint_to_ex_bloch", &reverse_magnetic_adjoint_to_ex_bloch_cuda, "Reverse Bloch magnetic update to Ex adjoint and epsilon gradient.");
  m.def("reverse_magnetic_adjoint_to_ey_bloch", &reverse_magnetic_adjoint_to_ey_bloch_cuda, "Reverse Bloch magnetic update to Ey adjoint and epsilon gradient.");
  m.def("reverse_magnetic_adjoint_to_ez_bloch", &reverse_magnetic_adjoint_to_ez_bloch_cuda, "Reverse Bloch magnetic update to Ez adjoint and epsilon gradient.");
  m.def("accumulate_forward_diff_adjoint", &accumulate_forward_diff_adjoint_cuda, "Accumulate adjoint of a forward finite difference.");
  m.def("accumulate_backward_diff_adjoint", &accumulate_backward_diff_adjoint_cuda, "Accumulate adjoint of a backward finite difference.");
  m.def("reverse_electric_component_ex_cpml", &reverse_electric_component_ex_cpml_cuda, "Reverse CPML Ex electric component.");
  m.def("reverse_electric_component_ey_cpml", &reverse_electric_component_ey_cpml_cuda, "Reverse CPML Ey electric component.");
  m.def("reverse_electric_component_ez_cpml", &reverse_electric_component_ez_cpml_cuda, "Reverse CPML Ez electric component.");
  m.def("reverse_magnetic_component_hx_cpml", &reverse_magnetic_component_hx_cpml_cuda, "Reverse CPML Hx magnetic component.");
  m.def("reverse_magnetic_component_hy_cpml", &reverse_magnetic_component_hy_cpml_cuda, "Reverse CPML Hy magnetic component.");
  m.def("reverse_magnetic_component_hz_cpml", &reverse_magnetic_component_hz_cpml_cuda, "Reverse CPML Hz magnetic component.");
  m.def("reverse_debye_current", &reverse_debye_current_cuda, "Reverse Debye current update.");
  m.def("reverse_drude_current", &reverse_drude_current_cuda, "Reverse Drude current update.");
  m.def("reverse_lorentz_current", &reverse_lorentz_current_cuda, "Reverse Lorentz current update.");
  m.def("accumulate_tfsf_scalar_sample_adjoint", &accumulate_tfsf_scalar_sample_adjoint_cuda, "Accumulate scalar TFSF sample adjoint.");
  m.def("accumulate_tfsf_line_sample_adjoint", &accumulate_tfsf_line_sample_adjoint_cuda, "Accumulate line TFSF sample adjoint.");
  m.def("accumulate_tfsf_interpolated_sample_adjoint", &accumulate_tfsf_interpolated_sample_adjoint_cuda, "Accumulate interpolated TFSF sample adjoint.");
  m.def("reverse_tfsf_auxiliary_electric", &reverse_tfsf_auxiliary_electric_cuda, "Reverse TFSF auxiliary electric update.");
  m.def("reverse_tfsf_auxiliary_magnetic", &reverse_tfsf_auxiliary_magnetic_cuda, "Reverse TFSF auxiliary magnetic update.");
  m.def("clamp_field_face", &clamp_field_face_cuda, "Clamp one boundary face.");
  m.def("clamp_pec_boundary", &clamp_pec_boundary_cuda, "Clamp PEC boundary faces.");
  m.def("project_periodic_boundary", &project_periodic_boundary_cuda, "Project periodic boundary faces.");
  m.def("project_bloch_boundary", &project_bloch_boundary_cuda, "Project Bloch boundary faces.");
}
