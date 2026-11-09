#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

#include "distance.cuh"

struct PoleEdges {
  const float *geometry, *boxes, *query[3], *width[3], *values, *seed, *old;
  const double *geometry64, *planes[3], *centre[3], *query64[3];
  const bool* carriers;
  const int64_t* slots;
  int n[3], nodes[3], samples[3], plane_count[3];
  int count, structures, seed_count, slot_count, axis;
  bool magnetic, node_background, fill;
  float background, delta, norm_eps, lap_scale;
};

// Each intermediate matches an ATen operation; contraction is disabled at build.
// Positions and the rotation matrix are fp32 scalars promoted by fp64 queries;
// dimensions were converted to fp64 before halving by the descriptor packer.
__device__ double pole_distance64(const double* g, double x, double y, double z) {
  double dx=x-g[4], dy=y-g[5], dz=z-g[6];
  if(g[0]==1.) return sqrt((dx*dx+dy*dy)+dz*dz)-g[16];
  x=(g[7]*dx+g[8]*dy)+g[9]*dz;
  y=(g[10]*dx+g[11]*dy)+g[12]*dz;
  z=(g[13]*dx+g[14]*dy)+g[15]*dz;
  if(g[0]==2.) {
    double qx=fabs(x)-g[16], qy=fabs(y)-g[17], qz=fabs(z)-g[18];
    double ax=fmax(qx,0.), ay=fmax(qy,0.), az=fmax(qz,0.);
    return sqrt((ax*ax+az*az)+ay*ay)+fmin(fmax(fmax(qx,qy),qz),0.);
  }
  int axis=int(g[18]);
  double axial=axis==0?x:axis==1?y:z;
  double a=axis==0?y:x, b=axis==2?y:z;
  double radial=sqrt(a*a+b*b)-g[16], cap=fabs(axial)-g[17];
  double r=fmax(radial,0.), h=fmax(cap,0.);
  return sqrt(r*r+h*h)+fmin(fmax(radial,cap),0.);
}

__device__ double ulp32(float value) {
  // Subtractions round in fp32 BEFORE conversion, as torch.nextafter does.
  float up=nextafterf(value,INFINITY)-value;
  float down=value-nextafterf(value,-INFINITY);
  return double(fmaxf(up,down));
}

__global__ void pole_box_bounds(PoleEdges p, double* bounds) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=p.structures*6) return;
  int j=i/6, side=(i%6)/3, axis=i%3;
  const float* b=p.boxes+j*7;
  if(b[0]==0.f) return;
  float position=b[1+axis], size=b[4+axis], half=size/2.f;
  float declared=side?position+half:position-half;
  double value=double(declared);
  double division_error=fabs(double(half)-double(size)/2.);
  double roundoff=((ulp32(position)/2.+ulp32(size)/4.)+division_error)+ulp32(declared)/2.;
  double best=p.planes[axis][0], error=fabs(best-value);
  for(int k=1;k<p.plane_count[axis];++k) {
    double node=p.planes[axis][k], next=fabs(node-value);
    // Strict less-than preserves argmin's first index on ties.
    if(next<error) {best=node; error=next;}
  }
  bounds[i]=error<=roundoff?best:value;
}

__device__ bool pole_inside(const PoleEdges& p, const double* bounds, int j,
                           double x, double y, double z) {
  if(p.boxes[j*7]!=0.f) {
    const double* b=bounds+j*6;
    return x>=b[0] && x<=b[3] && y>=b[1] && y<=b[4] && z>=b[2] && z<=b[5];
  }
  return pole_distance64(p.geometry64+j*16*24,x,y,z)<=0.;
}

__device__ float pole_average(const PoleEdges& p, const float* field, int x, int y, int z) {
  int start=(x*p.nodes[1]+y)*p.nodes[2]+z;
  if(!p.magnetic) {
    int stride=p.axis==0?p.nodes[1]*p.nodes[2]:p.axis==1?p.nodes[2]:1;
    return .5f*(field[start]+field[start+stride]);
  }
  // Magnetic faces sum low/low, low/high, high/low, high/high, left to right.
  int first=p.axis==0?p.nodes[2]:p.nodes[1]*p.nodes[2];
  int second=p.axis==2?p.nodes[2]:1;
  return .25f*(((field[start]+field[start+second])+field[start+first])+field[start+first+second]);
}

__global__ void pole_owner(PoleEdges p, const double* bounds, int64_t* owner,
                           float* value, bool* touch) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=p.count) return;
  int z=i%p.n[2], y=(i/p.n[2])%p.n[1], x=i/(p.n[1]*p.n[2]);
  bool touched=false;
  int node_count=p.nodes[0]*p.nodes[1]*p.nodes[2];
  for(int slot=0;slot<p.seed_count;++slot)
    touched|=pole_average(p,p.seed+slot*node_count,x,y,z)>0.f;
  int64_t selected=-1;
  float material=p.background;
  for(int j=0;j<p.structures;++j) {
    bool inside=pole_inside(p,bounds,j,p.centre[0][x],p.centre[1][y],p.centre[2][z]);
    if(inside) {selected=j; material=p.values[j];}
    if(p.carriers[j]) touched|=inside;
  }
  owner[i]=selected; value[i]=material; touch[i]=touched;
}

__global__ void pole_support(PoleEdges p, const double* bounds, bool* touch) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=p.count || touch[i]) return;
  int z=i%p.n[2], y=(i/p.n[2])%p.n[1], x=i/(p.n[1]*p.n[2]);
  for(int j=0;j<p.structures;++j) if(p.carriers[j]) {
    for(int sx=0;sx<p.samples[0];++sx)
      for(int sy=0;sy<p.samples[1];++sy)
        for(int sz=0;sz<p.samples[2];++sz) {
          bool occupied;
          if(p.fill) {
            // This is the fill ramp, NOT the binary Kottke window quadrature.
            // Shared coverage() preserves its fp32 stencil, sum, division and
            // reciprocal-multiply order, including the final 1e-3 snap.
            float qx=p.query[0][sx*p.n[0]+x], qy=p.query[1][sy*p.n[1]+y], qz=p.query[2][sz*p.n[2]+z];
            occupied=coverage(p,p.geometry+j*16*24,qx,qy,qz,p.width[0][x],p.width[1][y],p.width[2][z])>0.f;
          } else {
            occupied=pole_inside(p,bounds,j,p.query64[0][sx*p.n[0]+x],
                p.query64[1][sy*p.n[1]+y],p.query64[2][sz*p.n[2]+z]);
          }
          if(occupied) {touch[i]=true; return;}
        }
  }
}

__global__ void pole_scatter(PoleEdges p, const int64_t* owner, const float* value,
                             const bool* touch, float* weights, float* selected) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=p.count) return;
  for(int slot=0;slot<p.slot_count;++slot)
    weights[slot*p.count+i]=owner[i]==p.slots[slot]?1.f:0.f;
  int z=i%p.n[2], y=(i/p.n[2])%p.n[1], x=i/(p.n[1]*p.n[2]);
  // Select/copy only. The node fallback uses the exact reference average order.
  selected[i]=touch[i]?value[i]:(p.node_background?pole_average(p,p.old,x,y,z):p.old[i]);
}

std::vector<at::Tensor> poles_setup(at::Tensor geometry, at::Tensor geometry64,
    at::Tensor boxes, std::vector<at::Tensor> planes, std::vector<at::Tensor> centres,
    std::vector<at::Tensor> queries, std::vector<at::Tensor> widths,
    at::Tensor values, at::Tensor carriers, at::Tensor seed, at::Tensor old,
    at::Tensor slots, std::vector<int64_t> sizes, std::vector<double> constants) {
  TORCH_CHECK(planes.size()==3 && centres.size()==3 && queries.size()==3 && widths.size()==3,
              "poles: planes, centres, queries and widths must have three axes");
  check_tensor(geometry, geometry, at::kFloat, "poles", "geometry");
  check_tensor(geometry64, geometry, at::kDouble, "poles", "geometry64");
  check_tensor(boxes, geometry, at::kFloat, "poles", "boxes");
  check_tensors(planes, geometry, at::kDouble, "poles", "planes");
  check_tensors(centres, geometry, at::kDouble, "poles", "centres");
  TORCH_CHECK(queries[0].scalar_type()==at::kFloat || queries[0].scalar_type()==at::kDouble,
              "poles: queries must be float32 or float64");
  check_tensors(queries, geometry, queries[0].scalar_type(), "poles", "queries");
  check_tensors(widths, geometry, at::kFloat, "poles", "widths");
  check_tensor(values, geometry, at::kFloat, "poles", "values");
  check_tensor(carriers, geometry, at::kBool, "poles", "carriers");
  check_tensor(seed, geometry, at::kFloat, "poles", "seed");
  check_tensor(old, geometry, at::kFloat, "poles", "old");
  check_tensor(slots, geometry, at::kLong, "poles", "slots");
  c10::cuda::CUDAGuard guard(geometry.device());
  PoleEdges p{};
  p.geometry=geometry.data_ptr<float>(); p.geometry64=geometry64.data_ptr<double>();
  p.boxes=boxes.data_ptr<float>(); p.values=values.data_ptr<float>();
  p.carriers=carriers.data_ptr<bool>(); p.seed=seed.data_ptr<float>();
  p.old=old.data_ptr<float>(); p.slots=slots.data_ptr<int64_t>();
  p.count=sizes[0]*sizes[1]*sizes[2]; p.structures=geometry.size(0);
  p.slot_count=slots.numel(); p.seed_count=seed.numel()?seed.size(0):0;
  p.axis=sizes[9]; p.magnetic=sizes[10]; p.node_background=sizes[11];
  p.fill=queries[0].scalar_type()==at::kFloat;
  p.background=constants[0]; p.delta=constants[1]; p.norm_eps=constants[2]; p.lap_scale=constants[3];
  for(int a=0;a<3;++a) {
    p.n[a]=sizes[a]; p.nodes[a]=sizes[a+3]; p.samples[a]=sizes[a+6];
    p.planes[a]=planes[a].data_ptr<double>(); p.plane_count[a]=planes[a].numel();
    p.centre[a]=centres[a].data_ptr<double>(); p.width[a]=widths[a].data_ptr<float>();
    if(p.fill) p.query[a]=queries[a].data_ptr<float>();
    else p.query64[a]=queries[a].data_ptr<double>();
  }
  auto bounds=at::empty({p.structures,6},geometry64.options());
  auto owner=at::empty({p.count},geometry.options().dtype(at::kLong));
  auto value=at::empty({p.count},geometry.options());
  auto touch=at::empty({p.count},geometry.options().dtype(at::kBool));
  auto weights=at::empty({p.slot_count,p.count},geometry.options());
  auto selected=at::empty({p.count},geometry.options());
  auto stream=c10::cuda::getCurrentCUDAStream();
  if(p.structures) {
    pole_box_bounds<<<(p.structures*6+255)/256,256,0,stream>>>(p,bounds.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if(p.count) {
    pole_owner<<<(p.count+255)/256,256,0,stream>>>(p,bounds.data_ptr<double>(),
        owner.data_ptr<int64_t>(),value.data_ptr<float>(),touch.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    pole_support<<<(p.count+255)/256,256,0,stream>>>(p,bounds.data_ptr<double>(),touch.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    pole_scatter<<<(p.count+255)/256,256,0,stream>>>(p,owner.data_ptr<int64_t>(),
        value.data_ptr<float>(),touch.data_ptr<bool>(),weights.data_ptr<float>(),selected.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {weights,selected};
}
