#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <climits>
#include <vector>

namespace {
struct V { float x,y,z; };
__device__ V load(const float* p) { return {p[0],p[1],p[2]}; }
__device__ V sub(V a,V b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__device__ V project(V a,V b,float t) { return {a.x+b.x*t,a.y+b.y*t,a.z+b.z*t}; }
__device__ float dot(V a,V b) {
  // Reduce.cuh: three contiguous inputs use two lanes, x/z then y;
  // each lane's accumulator starts at positive zero.
  return __fadd_rn(__fadd_rn(__fadd_rn(0.f,a.x*b.x),__fadd_rn(0.f,a.z*b.z)),
                   __fadd_rn(0.f,a.y*b.y));
}
__device__ float norm2(V p,V q) { V d=sub(p,q); return dot(d,d); }
__device__ float clamp_min(float x,float lo) { return x<lo?lo:x; }
__device__ float clamp(float x) { return x<0.f?0.f:(x>1.f?1.f:x); }
__device__ float signed_denom(float x) {
  return x>=0.f?clamp_min(x,FLT_EPSILON):(x>-FLT_EPSILON?-FLT_EPSILON:x);
}
__device__ float triangle_distance(V p,const float* t) {
  V a=load(t),b=load(t+3),c=load(t+6);
  V ab=sub(b,a),ac=sub(c,a),bc=sub(c,b),ap=sub(p,a),bp=sub(p,b),cp=sub(p,c);
  float d1=dot(ab,ap),d2=dot(ac,ap),d3=dot(ab,bp),d4=dot(ac,bp),d5=dot(ab,cp),d6=dot(ac,cp);
  float va=d3*d6-d5*d4,vb=d5*d2-d1*d6,vc=d1*d4-d3*d2;
  float denom=clamp_min((va+vb)+vc,FLT_EPSILON),v=vb/denom,w=vc/denom;
  V face=project(project(a,ab,v),ac,w);
  float result=norm2(p,face);
  // Preserve the chronological torch.where overrides, including overlapping
  // predicates on degenerate triangles. Do not turn this into early returns.
  if(d1<=0.f && d2<=0.f) result=dot(ap,ap);
  if(d3>=0.f && d4<=d3) result=dot(bp,bp);
  if(d6>=0.f && d5<=d6) result=dot(cp,cp);
  if(vc<=0.f && d1>=0.f && d3<=0.f)
    result=norm2(p,project(a,ab,clamp(d1/signed_denom(d1-d3))));
  if(vb<=0.f && d2>=0.f && d6<=0.f)
    result=norm2(p,project(a,ac,clamp(d2/signed_denom(d2-d6))));
  if(va<=0.f && d4-d3>=0.f && d5-d6>=0.f)
    result=norm2(p,project(b,bc,clamp((d4-d3)/clamp_min((d4-d3)+(d5-d6),FLT_EPSILON))));
  return result;
}

// A geometric box alone is insufficient for the current clamped face formula.
// Prune only when outward interval arithmetic proves a vertex/edge predicate
// for every triangle in the node. Otherwise visit it, even for distant boxes.
struct I { float lo,hi; };
__device__ I add(I a,I b) { return {__fadd_rd(a.lo,b.lo),__fadd_ru(a.hi,b.hi)}; }
__device__ I neg(I a) { return {-a.hi,-a.lo}; }
__device__ I minus(I a,I b) { return add(a,neg(b)); }
__device__ I mul(I a,I b) {
  float l[4]={__fmul_rd(a.lo,b.lo),__fmul_rd(a.lo,b.hi),__fmul_rd(a.hi,b.lo),__fmul_rd(a.hi,b.hi)};
  float h[4]={__fmul_ru(a.lo,b.lo),__fmul_ru(a.lo,b.hi),__fmul_ru(a.hi,b.lo),__fmul_ru(a.hi,b.hi)};
  return {fminf(fminf(l[0],l[1]),fminf(l[2],l[3])),fmaxf(fmaxf(h[0],h[1]),fmaxf(h[2],h[3]))};
}
__device__ bool finite(I a) { return isfinite(a.lo)&&isfinite(a.hi); }
__device__ bool prunable(const float* bounds,int nodes,int node,V p,float best) {
  I r[18];
  for(int k=0;k<18;++k) {
    r[k]={bounds[k*nodes+node],bounds[(18+k)*nodes+node]};
    if(!finite(r[k])) return false;
  }
  float q[3]={p.x,p.y,p.z},squared[3];
  for(int k=0;k<3;++k) {
    if(!isfinite(q[k])) return false;
    float delta=fmaxf(0.f,fmaxf(__fsub_rd(r[k].lo,q[k]),__fsub_rd(q[k],r[k].hi)));
    squared[k]=__fmul_rd(delta,delta);
  }
  float lower=__fadd_rd(__fadd_rd(squared[0],squared[2]),squared[1]);
  if(!(lower>best)) return false;  // Visit equality ties, including +/-zero.
  I d[6];
  for(int vertex=0;vertex<3;++vertex) for(int edge=0;edge<2;++edge) {
    I terms[3];
    for(int k=0;k<3;++k)
      terms[k]=mul(r[12+edge*3+k],minus({q[k],q[k]},r[3+vertex*3+k]));
    d[vertex*2+edge]=add(add(terms[0],terms[2]),terms[1]);
    if(!finite(d[vertex*2+edge])) return false;
  }
  I va=minus(mul(d[2],d[5]),mul(d[4],d[3]));
  I vb=minus(mul(d[4],d[1]),mul(d[0],d[5]));
  I vc=minus(mul(d[0],d[3]),mul(d[2],d[1]));
  I u=minus(d[3],d[2]),v=minus(d[4],d[5]);
  if(!finite(va)||!finite(vb)||!finite(vc)||!finite(u)||!finite(v)
      ||!finite(add(u,v))||!finite(minus(d[0],d[2]))||!finite(minus(d[1],d[5]))) return false;
  return (d[0].hi<=0.f && d[1].hi<=0.f)
      || (d[2].lo>=0.f && d[3].hi<=d[2].lo)
      || (d[5].lo>=0.f && d[4].hi<=d[5].lo)
      || (vc.hi<=0.f && d[0].lo>=0.f && d[2].hi<=0.f)
      || (vb.hi<=0.f && d[1].lo>=0.f && d[5].hi<=0.f)
      || (va.hi<=0.f && u.lo>=0.f && v.lo>=0.f);
}
__device__ void consider(V p,const float* triangles,int id,float& best,int& owner) {
  float value=triangle_distance(p,triangles+id*9);
  if((isnan(value)&&(!isnan(best)||id<owner))
     || (!isnan(best)&&(value<best||(value==best&&id<owner)))) { best=value; owner=id; }
}
__global__ void distance_kernel(const float* points,const float* triangles,float* output,
    int count,int triangles_count,const int* topology,const int* ids,const float* bounds,int nodes) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  V p=load(points+3*i);
  float best=INFINITY; int owner=INT_MAX;
  if(!nodes) {
    for(int id=0;id<triangles_count;++id) consider(p,triangles,id,best,owner);
  } else {
    int stack[65],top=0; stack[top++]=0;
    while(top) {
      int node=stack[--top];
      if(prunable(bounds,nodes,node,p,best)) continue;
      int n=topology[3*nodes+node];
      if(n) {
        int start=topology[2*nodes+node];
        for(int j=0;j<n;++j) consider(p,triangles,ids[start+j],best,owner);
      } else {
        stack[top++]=topology[nodes+node]; stack[top++]=topology[node];
      }
    }
  }
  output[i]=sqrtf(fmaxf(best,0.f)+1.e-12f)-1.e-6f;
  if(isnan(best)) output[i]=best;
}
}

at::Tensor mesh_distance(at::Tensor points,at::Tensor triangles,std::vector<at::Tensor> hierarchy) {
  check_tensor(points, points, at::kFloat, "mesh_distance", "points");
  check_tensor(triangles, points, at::kFloat, "mesh_distance", "triangles");
  if (!hierarchy.empty()) check_tensor(hierarchy[0], points, at::kInt, "mesh_distance", "hierarchy[0] (topology)");
  if (!hierarchy.empty()) check_tensor(hierarchy[1], points, at::kInt, "mesh_distance", "hierarchy[1] (ids)");
  if (!hierarchy.empty()) check_tensor(hierarchy[2], points, at::kFloat, "mesh_distance", "hierarchy[2] (bounds)");
  c10::cuda::CUDAGuard guard(points.device());
  auto out=at::empty({points.size(0)},points.options());
  if(points.size(0)) distance_kernel<<<(points.size(0)+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      points.data_ptr<float>(),triangles.data_ptr<float>(),out.data_ptr<float>(),points.size(0),triangles.size(0),
      hierarchy.empty()?nullptr:hierarchy[0].data_ptr<int>(),hierarchy.empty()?nullptr:hierarchy[1].data_ptr<int>(),
      hierarchy.empty()?nullptr:hierarchy[2].data_ptr<float>(),hierarchy.empty()?0:hierarchy[0].size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
