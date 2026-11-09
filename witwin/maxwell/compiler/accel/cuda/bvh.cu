#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {
__device__ unsigned spread(unsigned x) {
  x=(x|(x<<16))&0x030000FFu; x=(x|(x<<8))&0x0300F00Fu;
  x=(x|(x<<4))&0x030C30C3u; return (x|(x<<2))&0x09249249u;
}
__global__ void keys_kernel(const float* t,const float* extent,int64_t* keys,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  unsigned cell[3];
  for(int k=0;k<3;++k) {
    float lo=fminf(t[i*9+k],fminf(t[i*9+3+k],t[i*9+6+k]));
    float hi=fmaxf(t[i*9+k],fmaxf(t[i*9+3+k],t[i*9+6+k]));
    float width=extent[3+k]-extent[k];
    float q=width>0.f?(.5f*(lo+hi)-extent[k])/width:0.f;
    cell[k]=(unsigned)fminf(fmaxf(q*1024.f,0.f),1023.f);
  }
  unsigned morton=(spread(cell[0])<<2)|(spread(cell[1])<<1)|spread(cell[2]);
  keys[i]=(int64_t(morton)<<32)|unsigned(i);
}
__device__ int prefix(const int64_t* keys,int a,int b,int n) {
  return b<0||b>=n?-1:__clzll((unsigned long long)(keys[a]^keys[b]));
}
__global__ void topology_kernel(const int64_t* keys,int* topology,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x,nodes=2*n-1;
  if(i>=n) return;
  int leaf=n-1+i;
  topology[2*nodes+leaf]=i; topology[3*nodes+leaf]=1;
  if(i==n-1) return;
  // Radix topology over unique (Morton, original triangle ID) keys.
  int direction=prefix(keys,i,i+1,n)>prefix(keys,i,i-1,n)?1:-1;
  int minimum=prefix(keys,i,i-direction,n),limit=2;
  while(prefix(keys,i,i+limit*direction,n)>minimum) limit*=2;
  int length=0;
  for(int step=limit/2;step;step/=2)
    if(prefix(keys,i,i+(length+step)*direction,n)>minimum) length+=step;
  int j=i+length*direction,first=min(i,j),last=max(i,j);
  int common=prefix(keys,first,last,n),split=first,step=last-first;
  do {
    step=(step+1)/2;
    int candidate=split+step;
    if(candidate<last && prefix(keys,first,candidate,n)>common) split=candidate;
  } while(step>1);
  topology[i]=split==first?n-1+split:split;
  topology[nodes+i]=split+1==last?n-1+split+1:split+1;
}
__global__ void parents_kernel(const int* topology,int* parents,int nodes) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<nodes && topology[3*nodes+i]==0) {
    parents[topology[i]]=i; parents[topology[nodes+i]]=i;
  }
}
__device__ void store(volatile float* b,int nodes,int node,int k,float lo,float hi) {
  b[k*nodes+node]=lo; b[(18+k)*nodes+node]=hi;
}
__global__ void bounds_kernel(const float* triangles,const int* topology,const int* ids,
    const int* parents,int* arrivals,volatile float* bounds,int nodes) {
  int node=blockIdx.x*blockDim.x+threadIdx.x;
  if(node>=nodes || !topology[3*nodes+node]) return;
  float lo[18],hi[18];
  for(int k=0;k<18;++k) { lo[k]=INFINITY; hi[k]=-INFINITY; }
  int start=topology[2*nodes+node],count=topology[3*nodes+node];
  for(int j=0;j<count;++j) {
    const float* t=triangles+9*ids[start+j];
    for(int k=0;k<3;++k) {
      float a=t[k],b=t[3+k],c=t[6+k],ab=b-a,ac=c-a,bc=c-b;
      float values[5]={a,b,c,ab,ac};
      for(int v=0;v<5;++v) {
        int index=3+3*v+k;
        lo[index]=fminf(lo[index],values[v]); hi[index]=fmaxf(hi[index],values[v]);
        // Nonfinite geometry must never acquire a finite pruning certificate.
        if(!isfinite(values[v])) { lo[index]=-INFINITY; hi[index]=INFINITY; }
      }
      float lower=fminf(fminf(a,b),c),upper=fmaxf(fmaxf(a,b),c);
      lower=fminf(lower,__fadd_rd(a,fminf(0.f,ab)));
      lower=fminf(lower,__fadd_rd(a,fminf(0.f,ac)));
      lower=fminf(lower,__fadd_rd(b,fminf(0.f,bc)));
      upper=fmaxf(upper,__fadd_ru(a,fmaxf(0.f,ab)));
      upper=fmaxf(upper,__fadd_ru(a,fmaxf(0.f,ac)));
      upper=fmaxf(upper,__fadd_ru(b,fmaxf(0.f,bc)));
      lo[k]=fminf(lo[k],nextafterf(lower,-INFINITY));
      hi[k]=fmaxf(hi[k],nextafterf(upper,INFINITY));
    }
  }
  for(int k=0;k<18;++k) store(bounds,nodes,node,k,lo[k],hi[k]);
  __threadfence();
  int parent=parents[node];
  while(parent>=0) {
    if(atomicAdd(arrivals+parent,1)==0) return;
    int left=topology[parent],right=topology[nodes+parent];
    for(int k=0;k<18;++k) store(bounds,nodes,parent,k,
        fminf(bounds[k*nodes+left],bounds[k*nodes+right]),
        fmaxf(bounds[(18+k)*nodes+left],bounds[(18+k)*nodes+right]));
    __threadfence();
    parent=parents[parent];
  }
}
}

at::Tensor mesh_keys(at::Tensor triangles,at::Tensor extent) {
  check_tensor(triangles, triangles, at::kFloat, "mesh_keys", "triangles");
  check_tensor(extent, triangles, at::kFloat, "mesh_keys", "extent");
  c10::cuda::CUDAGuard guard(triangles.device());
  auto keys=at::empty({triangles.size(0)},triangles.options().dtype(at::kLong));
  keys_kernel<<<(keys.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<float>(),extent.data_ptr<float>(),keys.data_ptr<int64_t>(),keys.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return keys;
}
at::Tensor mesh_topology(at::Tensor keys) {
  check_tensor(keys, keys, at::kLong, "mesh_topology", "keys");
  c10::cuda::CUDAGuard guard(keys.device());
  int n=keys.numel();
  auto topology=at::zeros({4,2*n-1},keys.options().dtype(at::kInt));
  topology_kernel<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      keys.data_ptr<int64_t>(),topology.data_ptr<int>(),n);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return topology;
}
at::Tensor mesh_bounds(at::Tensor triangles,at::Tensor topology,at::Tensor ids) {
  check_tensor(triangles, triangles, at::kFloat, "mesh_bounds", "triangles");
  check_tensor(topology, triangles, at::kInt, "mesh_bounds", "topology");
  check_tensor(ids, triangles, at::kInt, "mesh_bounds", "ids");
  c10::cuda::CUDAGuard guard(triangles.device());
  int nodes=topology.size(1);
  auto parents=at::full({nodes},-1,ids.options());
  auto arrivals=at::zeros({nodes},ids.options());
  auto bounds=at::empty({2,18,nodes},triangles.options());
  parents_kernel<<<(nodes+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      topology.data_ptr<int>(),parents.data_ptr<int>(),nodes);
  bounds_kernel<<<(nodes+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<float>(),topology.data_ptr<int>(),ids.data_ptr<int>(),parents.data_ptr<int>(),
      arrivals.data_ptr<int>(),bounds.data_ptr<float>(),nodes);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return bounds;
}
