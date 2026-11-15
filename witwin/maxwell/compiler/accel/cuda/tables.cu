#include "checks.h"
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
Keys keys(std::vector<torch::stable::Tensor> input, std::vector<int64_t> shape) {
  Keys p{};
  for(int a=0;a<3;++a) p.n[a]=shape[a];
  for(int c=0;c<6;++c) {
    p.data[c]=reinterpret_cast<const int*>(input[c].const_data_ptr<float>());
    for(int a=0;a<3;++a) p.size[c][a]=input[c].size(a);
  }
  return p;
}
std::vector<torch::stable::Tensor> hash_setup(std::vector<torch::stable::Tensor> input, std::vector<int64_t> shape) {
  check_tensors(input, input[0], torch::headeronly::ScalarType::Float, "hash", "input");
  torch::stable::accelerator::DeviceGuard guard(input[0].get_device_index());
  auto slots=torch::stable::full({1<<20}, -1, torch::headeronly::ScalarType::Int, std::nullopt, input[0].device());
  int count=shape[0]*shape[1]*shape[2];
  auto inverse=torch::stable::new_empty(input[0], {count}, torch::headeronly::ScalarType::Int), overflow=torch::stable::new_zeros(input[0], {1}, torch::headeronly::ScalarType::Int);
  hash_kernel<<<(count+255)/256,256,0,current_cuda_stream()>>>(
      keys(input,shape),slots.mutable_data_ptr<int>(),inverse.mutable_data_ptr<int>(),overflow.mutable_data_ptr<int>(),count,slots.numel()-1);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return {slots,inverse,overflow};
}
torch::stable::Tensor gather_setup(std::vector<torch::stable::Tensor> input, std::vector<int64_t> shape, torch::stable::Tensor owners) {
  check_tensors(input, input[0], torch::headeronly::ScalarType::Float, "gather", "input");
  check_tensor(owners, input[0], torch::headeronly::ScalarType::Int, "gather", "owners");
  torch::stable::accelerator::DeviceGuard guard(input[0].get_device_index());
  auto rows=torch::stable::new_empty(owners, {owners.numel(),6});
  gather_kernel<<<(owners.numel()+255)/256,256,0,current_cuda_stream()>>>(
      keys(input,shape),owners.const_data_ptr<int>(),rows.mutable_data_ptr<int>(),owners.numel());
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return rows;
}

template<class Index>
__global__ void remap_kernel(const int* slots, const int* remap, Index* out, int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) out[i]=remap[slots[i]];
}
torch::stable::Tensor remap_setup(torch::stable::Tensor slots, torch::stable::Tensor remap, bool wide) {
  check_tensor(slots, slots, torch::headeronly::ScalarType::Int, "remap", "slots");
  check_tensor(remap, slots, torch::headeronly::ScalarType::Int, "remap", "remap");
  torch::stable::accelerator::DeviceGuard guard(slots.get_device_index());
  auto out=torch::stable::new_empty(slots, slots.sizes(), wide ? torch::headeronly::ScalarType::UInt32 : torch::headeronly::ScalarType::UInt16);
  if(wide)
    remap_kernel<<<(out.numel()+255)/256,256,0,current_cuda_stream()>>>(
        slots.const_data_ptr<int>(),remap.const_data_ptr<int>(),out.mutable_data_ptr<uint32_t>(),out.numel());
  else
    remap_kernel<<<(out.numel()+255)/256,256,0,current_cuda_stream()>>>(
        slots.const_data_ptr<int>(),remap.const_data_ptr<int>(),out.mutable_data_ptr<uint16_t>(),out.numel());
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
