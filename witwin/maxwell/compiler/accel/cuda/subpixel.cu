#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Reduce.cuh: contiguous sample means split across x, with four per-thread
// accumulators; strided [window,sample,axis] moments keep four serial chains.
__global__ void window_kernel(const float* sample, const float* offsets, float* out,
                              int rows, int samples, int axis) {
  int row=blockIdx.x*blockDim.y+threadIdx.y, lane=threadIdx.x;
  __shared__ float shared[4][64][2];
  __shared__ float means[4][2], moments[4][3];
  float a[4]={0,0,0,0}, h[4]={0,0,0,0};
  if(row<rows) {
    for(int k=0;k<4;++k) {
      int s=lane+k*blockDim.x;
      if(s<samples) { float v=sample[row*samples+s]; a[k]+=v; h[k]+=1.f/v; }
    }
  }
  float av=((a[0]+a[1])+a[2])+a[3], hv=((h[0]+h[1])+h[2])+h[3];
  if(blockDim.x==64) {
    shared[threadIdx.y][lane][0]=av; shared[threadIdx.y][lane][1]=hv;
    __syncthreads();
    if(lane<32) { av+=shared[threadIdx.y][lane+32][0]; hv+=shared[threadIdx.y][lane+32][1]; }
  }
  for(int d=min(int(blockDim.x),32)/2;d;d/=2) {
    av+=__shfl_down_sync(0xffffffff,av,d);
    hv+=__shfl_down_sync(0xffffffff,hv,d);
  }
  if(lane==0) { means[threadIdx.y][0]=av*(1.f/samples); means[threadIdx.y][1]=hv*(1.f/samples); }
  __syncthreads();
  if(row<rows && lane<3) {
    float v[4]={0,0,0,0};
    for(int s=0;s<samples;++s) {
      float centered=sample[row*samples+s]-means[threadIdx.y][0];
      v[s%4]+=centered*offsets[(row*samples+s)*3+lane];
    }
    moments[threadIdx.y][lane]=((v[0]+v[1])+v[2])+v[3];
  }
  __syncthreads();
  if(row<rows && lane==0) {
    float* m=moments[threadIdx.y];
    float norm=sqrtf((m[0]*m[0]+m[2]*m[2])+m[1]*m[1]);
    float denominator=norm>0.f ? fmaxf(norm,1e-30f) : 1.f;
    denominator*=denominator;
    float w=norm>0.f ? (m[axis]*m[axis])/denominator : 0.f;
    out[row]=w*(1.f/means[threadIdx.y][1])+(1.f-w)*means[threadIdx.y][0];
  }
}

at::Tensor window_setup(at::Tensor sample, at::Tensor offsets, int64_t axis) {
  check_tensor(sample, sample, at::kFloat, "window", "sample");
  check_tensor(offsets, sample, at::kFloat, "window", "offsets");
  c10::cuda::CUDAGuard guard(sample.device());
  auto out=at::empty({sample.size(0)},sample.options());
  int n=sample.size(1), rows=sample.size(0), width=1, height=1;
  while(width*2<=n && width<512) width*=2;
  while(height*2<=rows && height<512) height*=2;
  int bx=std::min(width,32), by=std::min(height,512/bx);
  bx=std::min(width,512/by);
  window_kernel<<<(rows+3)/4,dim3(bx,4),0,c10::cuda::getCurrentCUDAStream()>>>(
      sample.data_ptr<float>(),offsets.data_ptr<float>(),out.data_ptr<float>(),rows,n,axis);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__global__ void reconstruct_kernel(const float* a, const float* inv, const float* gx,
                                   const float* gy, const float* gz, float* out,
                                   int count, int axis, bool node) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  float g[3]={gx[i]*gx[i],gy[i]*gy[i],gz[i]*gz[i]};
  if(node) for(int j=0;j<3;++j) { float t=0.f+g[j]; t+=g[j]; t+=g[j]; g[j]=t; }
  float norm=((0.f+g[0])+g[1])+g[2]+1e-24f;
  float w=g[axis]/norm, contrast=a[i]*inv[i]-1.f;
  if(contrast<1e-5f) contrast=0.f;
  out[i]=a[i]/(1.f+w*contrast);
}

at::Tensor reconstruct_setup(at::Tensor arithmetic, at::Tensor inverse,
                             std::vector<at::Tensor> gradients, int64_t axis, bool node) {
  check_tensor(arithmetic, arithmetic, at::kFloat, "reconstruct", "arithmetic");
  check_tensor(inverse, arithmetic, at::kFloat, "reconstruct", "inverse");
  check_tensors(gradients, arithmetic, at::kFloat, "reconstruct", "gradients");
  c10::cuda::CUDAGuard guard(arithmetic.device());
  auto out=at::empty_like(arithmetic);
  reconstruct_kernel<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      arithmetic.data_ptr<float>(),inverse.data_ptr<float>(),gradients[0].data_ptr<float>(),
      gradients[1].data_ptr<float>(),gradients[2].data_ptr<float>(),out.data_ptr<float>(),out.numel(),axis,node);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
