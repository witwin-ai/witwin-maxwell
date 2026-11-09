#pragma once

// Keep the large phase product and its range reduction out of FP32. Both the
// electric update and ADE correction must evaluate exactly the same m_next.
__device__ __forceinline__ void modulation_sincos(
    double omega, double time, float& sine, float& cosine) {
  constexpr double two_pi = 6.283185307179586476925286766559;
  sincosf(static_cast<float>(remainder(omega * time, two_pi)), &sine, &cosine);
}
