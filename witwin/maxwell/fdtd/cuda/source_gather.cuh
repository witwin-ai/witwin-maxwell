#pragma once
#include <cstdint>

// CSR rows own disjoint targets. Entries retain patch, entry, image order.
struct SourceGather {
  const int64_t *rows, *targets, *entries;
  int64_t count;
};

// Global float atomic addition flushed subnormal inputs and results. Retain
// that arithmetic at former atomic sites without retaining competing writers.
#ifdef __CUDACC__
__device__ __forceinline__ float source_atomic_sum(float field, float delta) {
  float result;
  asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(field), "f"(delta));
  return result;
}
#endif
