#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

struct Keys {
  const int* data[6];
  int size[6][3], n[3];
};
__device__ void key_at(Keys p, int i, int* key) {
  int z=i%p.n[2], y=(i/p.n[2])%p.n[1], x=i/(p.n[1]*p.n[2]);
  for(int c=0;c<6;++c)
    key[c]=(x<p.size[c][0] && y<p.size[c][1] && z<p.size[c][2])
      ? p.data[c][(x*p.size[c][1]+y)*p.size[c][2]+z] : 0;
}
__global__ void hash_kernel(Keys p, int* slots, int* inverse, int* overflow, int count, int mask) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  int key[6]; key_at(p,i,key);
  unsigned hash=2166136261u;
  for(int c=0;c<6;++c) { hash=(hash^unsigned(key[c]))*16777619u; hash^=hash>>16; }
  int slot=hash&mask;
  for(int probe=0;probe<128;++probe,slot=(slot+1)&mask) {
    int owner=atomicCAS(slots+slot,-1,i);
    if(owner==-1 || owner==i) { inverse[i]=slot; return; }
    int other[6]; key_at(p,owner,other);
    bool equal=true;
    for(int c=0;c<6;++c) equal&=key[c]==other[c];
    if(equal) { inverse[i]=slot; return; }
  }
  atomicExch(overflow,1);
}
__global__ void gather_kernel(Keys p, const int* owners, int* rows, int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) key_at(p,owners[i],rows+i*6);
}
Keys keys(std::vector<at::Tensor> input, std::vector<int64_t> shape) {
  Keys p{};
  for(int a=0;a<3;++a) p.n[a]=shape[a];
  for(int c=0;c<6;++c) {
    p.data[c]=reinterpret_cast<const int*>(input[c].data_ptr<float>());
    for(int a=0;a<3;++a) p.size[c][a]=input[c].size(a);
  }
  return p;
}
std::vector<at::Tensor> hash_setup(std::vector<at::Tensor> input, std::vector<int64_t> shape) {
  check_tensors(input, input[0], at::kFloat, "hash", "input");
  c10::cuda::CUDAGuard guard(input[0].device());
  auto options=input[0].options().dtype(at::kInt);
  auto slots=at::full({1<<20},-1,options);
  int count=shape[0]*shape[1]*shape[2];
  auto inverse=at::empty({count},options), overflow=at::zeros({1},options);
  hash_kernel<<<(count+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      keys(input,shape),slots.data_ptr<int>(),inverse.data_ptr<int>(),overflow.data_ptr<int>(),count,slots.numel()-1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {slots,inverse,overflow};
}
at::Tensor gather_setup(std::vector<at::Tensor> input, std::vector<int64_t> shape, at::Tensor owners) {
  check_tensors(input, input[0], at::kFloat, "gather", "input");
  check_tensor(owners, input[0], at::kInt, "gather", "owners");
  c10::cuda::CUDAGuard guard(input[0].device());
  auto rows=at::empty({owners.numel(),6},owners.options());
  gather_kernel<<<(owners.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      keys(input,shape),owners.data_ptr<int>(),rows.data_ptr<int>(),owners.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rows;
}

template<class Index>
__global__ void remap_kernel(const int* slots, const int* remap, Index* out, int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) out[i]=remap[slots[i]];
}
at::Tensor remap_setup(at::Tensor slots, at::Tensor remap, bool wide) {
  check_tensor(slots, slots, at::kInt, "remap", "slots");
  check_tensor(remap, slots, at::kInt, "remap", "remap");
  c10::cuda::CUDAGuard guard(slots.device());
  auto out=at::empty(slots.sizes(),slots.options().dtype(wide ? at::kUInt32 : at::kUInt16));
  if(wide)
    remap_kernel<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        slots.data_ptr<int>(),remap.data_ptr<int>(),out.data_ptr<uint32_t>(),out.numel());
  else
    remap_kernel<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        slots.data_ptr<int>(),remap.data_ptr<int>(),out.data_ptr<uint16_t>(),out.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
