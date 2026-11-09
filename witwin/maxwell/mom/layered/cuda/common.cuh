#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <cuda_runtime.h>

namespace layered {
using C = c10::complex<double>;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double eps0 = 8.8541878128e-12;
constexpr double c0 = 299792458.0;
__device__ inline double abs(C z) { return hypot(z.real(),z.imag()); }
__device__ inline C ex(C z) {
  double a = exp(z.real());
  return C(a*cos(z.imag()), a*sin(z.imag()));
}
__device__ inline C root(C z) {
  double r = hypot(z.real(), z.imag());
  double x = sqrt(fmax(0.0, (r+z.real())*0.5));
  double y = sqrt(fmax(0.0, (r-z.real())*0.5));
  // Stable near the real axis; select Im(kz)>=0, including the lossless limit.
  if (z.real() >= 0 && x != 0) y = fabs(z.imag())/(2*x);
  if (z.real() < 0 && y != 0) x = fabs(z.imag())/(2*y);
  return z.imag() < 0 ? C(-x,y) : C(x,y);
}
__device__ inline C eps(const C* media, const double* sigma, int l, double w) {
  return media[l]+C(0,sigma[l]/(w*eps0));
}
struct Stack {
  const C* media;
  const double *sigma, *edges;
  int n, pec_bottom, pec_top;
};
__device__ inline C kz(Stack s, int l, C q, double w) {
  return root((w/c0)*(w/c0)*eps(s.media,s.sigma,l,w)-q*q);
}
__device__ inline int region(Stack s, double z) {
  int l=0;
  while(l<s.n-1 && z>=s.edges[l]) ++l;
  return l;
}
// j -> -i from the exp(j*w*t) TL formulas:
// G_A=V_h/(i*w*mu0), K_Phi=k0^2/q^2*(G_A-V_e/(i*w*mu0)).
// Stored voltages are V/(i*w*mu0); Z_h=1/kz, Z_e=kz/(k0^2*eps).
// Carry the scaled TE-TM difference d=(h-e)*k0^2/q^2 algebraically. Computing
// that difference after two independent TL solves loses digits near q=0.
struct Modes {
  C h,e,d;
  __device__ Modes(C value=C(0)):h(value),e(value),d(0) {}
  __device__ Modes(C hv,C ev,C dv):h(hv),e(ev),d(dv) {}
};
__device__ inline Modes operator+(Modes a,Modes b) { return {a.h+b.h,a.e+b.e,a.d+b.d}; }
__device__ inline Modes operator-(Modes a,Modes b) { return {a.h-b.h,a.e-b.e,a.d-b.d}; }
__device__ inline Modes operator*(Modes a,Modes b) {
  return {a.h*b.h,a.e*b.e,a.d*b.h+a.e*b.d};
}
__device__ inline Modes operator/(Modes a,Modes b) {
  return {a.h/b.h,a.e/b.e,(a.d*b.h-a.h*b.d)/(b.h*b.e)};
}
}
