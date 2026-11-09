#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>

struct Setup {
  float* out[6];
  const float *axis[3], *width[3], *geometry, *values, *fills;
  const int *offsets, *ids;
  int n[3], samples[3], begin, count, structures, bins_y, bins_z;
  bool edge;
  float delta, norm_eps, lap_scale, scale, background[6];
};

#include "distance.cuh"

__global__ void coverage_queries(Setup p, const float* points, float* out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=p.count) return;
  int iz=i%p.n[2], iy=(i/p.n[2])%p.n[1], ix=i/(p.n[1]*p.n[2]);
  out[i]=coverage(p,p.geometry,points[3*i],points[3*i+1],points[3*i+2],
                   p.width[0][ix],p.width[1][iy],p.width[2][iz]);
}

at::Tensor coverage_setup(at::Tensor geometry, at::Tensor points,
    std::vector<at::Tensor> widths, std::vector<int64_t> shape, std::vector<double> constants) {
  check_tensor(geometry, geometry, at::kFloat, "coverage", "geometry");
  check_tensor(points, geometry, at::kFloat, "coverage", "points");
  check_tensors(widths, geometry, at::kFloat, "coverage", "widths");
  c10::cuda::CUDAGuard guard(points.device());
  auto out=at::empty({points.size(0)},points.options());
  Setup p{}; p.count=points.size(0); p.geometry=geometry.data_ptr<float>();
  for(int a=0;a<3;++a) {p.n[a]=shape[a]; p.width[a]=widths[a].data_ptr<float>();}
  p.delta=constants[0]; p.norm_eps=constants[1]; p.lap_scale=constants[2];
  if(p.count) {
    coverage_queries<<<(p.count+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        p,points.data_ptr<float>(),out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

__global__ void sample_kernel(Setup p) {
  int local=blockIdx.x*blockDim.x+threadIdx.x;
  if(local>=p.count) return;
  int i=p.begin+local, iz=i%p.n[2], iy=(i/p.n[2])%p.n[1], ix=i/(p.n[1]*p.n[2]);
  int bin=((ix/8)*p.bins_y+iy/8)*p.bins_z+iz/8;
  float sums[6]={0,0,0,0,0,0};
  int sample=0;
  for(int sx=0;sx<p.samples[0];++sx)
    for(int sy=0;sy<p.samples[1];++sy)
      for(int sz=0;sz<p.samples[2];++sz,++sample) {
        float x=p.axis[0][sx*p.n[0]+ix], y=p.axis[1][sy*p.n[1]+iy], z=p.axis[2][sz*p.n[2]+iz];
        float v[6];
        for(int c=0;c<6;++c) v[c]=p.background[c];
        for(int entry=p.offsets[bin];entry<p.offsets[bin+1];++entry) {
          int j=p.ids[entry];
          const float* g=p.geometry+j*16*24;
          float f = g[0]!=0.f ? coverage(p,g,x,y,z,p.width[0][ix],p.width[1][iy],p.width[2][iz])
                              : p.fills[(sample*p.structures+j)*p.count+local];
          for(int c=0;c<6;++c) v[c]=(1.f-f)*v[c]+f*p.values[j*6+c];
        }
        for(int c=0;c<6;++c) sums[c]=(p.edge && sample==0) ? v[c] : sums[c]+v[c];
      }
  for(int c=0;c<6;++c) p.out[c][i]=sums[c]*p.scale;
}

// Node pole weights have a different recurrence from material blends. Keep the
// clear, own-slot add, chronological sample sum and final scale as separate
// fp32 operations, including an optional per-node custom amplitude.
__global__ void sample_pole_kernel(Setup p, float* output, const float* amplitude, int owner) {
  int local=blockIdx.x*blockDim.x+threadIdx.x;
  if(local>=p.count) return;
  int i=p.begin+local, iz=i%p.n[2], iy=(i/p.n[2])%p.n[1], ix=i/(p.n[1]*p.n[2]);
  int bin=((ix/8)*p.bins_y+iy/8)*p.bins_z+iz/8;
  float sum=0.f;
  int sample=0;
  for(int sx=0;sx<p.samples[0];++sx)
    for(int sy=0;sy<p.samples[1];++sy)
      for(int sz=0;sz<p.samples[2];++sz,++sample) {
        float x=p.axis[0][sx*p.n[0]+ix], y=p.axis[1][sy*p.n[1]+iy], z=p.axis[2][sz*p.n[2]+iz];
        float weight=0.f;
        for(int entry=p.offsets[bin];entry<p.offsets[bin+1];++entry) {
          int j=p.ids[entry];
          const float* g=p.geometry+j*16*24;
          float f=g[0]!=0.f?coverage(p,g,x,y,z,p.width[0][ix],p.width[1][iy],p.width[2][iz])
                            :p.fills[(sample*p.structures+j)*p.count+local];
          weight=weight*(1.f-f);
          if(j==owner) weight=weight+(amplitude?f*amplitude[i]:f);
        }
        sum=sum+weight;
      }
  output[i]=sum*p.scale;
}

void sample_setup(std::vector<at::Tensor> outputs, std::vector<at::Tensor> axes,
                  at::Tensor geometry, at::Tensor values, at::Tensor fills,
                  at::Tensor offsets, at::Tensor ids,
                  std::vector<int64_t> sizes, std::vector<double> constants,
                  std::vector<at::Tensor> pole_outputs, std::vector<at::Tensor> amplitudes,
                  std::vector<int64_t> owners) {
  check_tensors(outputs, outputs[0], at::kFloat, "sample", "outputs");
  check_tensors(axes, outputs[0], at::kFloat, "sample", "axes");
  check_tensor(geometry, outputs[0], at::kFloat, "sample", "geometry");
  check_tensor(values, outputs[0], at::kFloat, "sample", "values");
  check_tensor(fills, outputs[0], at::kFloat, "sample", "fills");
  check_tensor(offsets, outputs[0], at::kInt, "sample", "offsets");
  check_tensor(ids, outputs[0], at::kInt, "sample", "ids");
  check_tensors(pole_outputs, outputs[0], at::kFloat, "sample", "pole_outputs");
  check_tensors(amplitudes, outputs[0], at::kFloat, "sample", "amplitudes");
  TORCH_CHECK(amplitudes.size()==pole_outputs.size() && owners.size()==pole_outputs.size(),
              "sample: pole_outputs, amplitudes and owners must have equal length");
  c10::cuda::CUDAGuard guard(outputs[0].device());
  Setup p{};
  for(int c=0;c<6;++c) { p.out[c]=outputs[c].data_ptr<float>(); p.background[c]=constants[4+c]; }
  for(int a=0;a<3;++a) {
    p.axis[a]=axes[a].data_ptr<float>(); p.width[a]=axes[a+3].data_ptr<float>();
    p.n[a]=sizes[a]; p.samples[a]=sizes[a+3];
  }
  p.geometry=geometry.data_ptr<float>(); p.values=values.data_ptr<float>(); p.fills=fills.data_ptr<float>();
  p.offsets=offsets.data_ptr<int>(); p.ids=ids.data_ptr<int>();
  p.begin=sizes[6]; p.count=sizes[7]; p.structures=geometry.size(0); p.edge=sizes[8];
  p.bins_y=(p.n[1]+7)/8; p.bins_z=(p.n[2]+7)/8;
  p.delta=constants[0]; p.norm_eps=constants[1]; p.lap_scale=constants[2]; p.scale=constants[3];
  sample_kernel<<<(p.count+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for(size_t slot=0;slot<pole_outputs.size();++slot) {
    sample_pole_kernel<<<(p.count+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        p,pole_outputs[slot].data_ptr<float>(),
        amplitudes[slot].numel()?amplitudes[slot].data_ptr<float>():nullptr,owners[slot]);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

__global__ void distance_kernel(const float* geometry, const float* points, float* out,
                                int count, int total) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=total) return;
  int point=i%count, structure=i/count;
  out[i]=distance(geometry+structure*16*24,points[point*3],points[point*3+1],points[point*3+2]);
}

at::Tensor distance_setup(at::Tensor geometry, at::Tensor points) {
  check_tensor(geometry, geometry, at::kFloat, "distance", "geometry");
  check_tensor(points, geometry, at::kFloat, "distance", "points");
  c10::cuda::CUDAGuard guard(points.device());
  auto out=at::empty({geometry.size(0),points.size(0)},points.options());
  if(out.numel()) {
    distance_kernel<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      geometry.data_ptr<float>(),points.data_ptr<float>(),out.data_ptr<float>(),points.size(0),out.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

__global__ void bulk_kernel(const float* distances, const float* values, const float* band,
                            float* material, bool* near, int count, int structures,
                            float background, bool smooth) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  float result=background;
  bool interface=false;
  for(int j=0;j<structures;++j) {
    float d=distances[j*count+i];
    if(smooth) {
      float f=fminf(fmaxf(.5f-d/(2.f*band[i]),0.f),1.f);
      result=f*values[j]+(1.f-f)*result;
    } else if(d<0.f) result=values[j];
    interface|=fabsf(d)<band[i];
  }
  material[i]=result; near[i]=interface;
}

std::vector<at::Tensor> bulk_setup(at::Tensor distances, at::Tensor values,
                                 at::Tensor band, double background, bool smooth) {
  check_tensor(distances, distances, at::kFloat, "bulk", "distances");
  check_tensor(values, distances, at::kFloat, "bulk", "values");
  check_tensor(band, distances, at::kFloat, "bulk", "band");
  c10::cuda::CUDAGuard guard(band.device());
  auto material=at::empty_like(band);
  auto near=at::empty(band.sizes(),band.options().dtype(at::kBool));
  bulk_kernel<<<(band.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      distances.data_ptr<float>(),values.data_ptr<float>(),band.data_ptr<float>(),material.data_ptr<float>(),
      near.data_ptr<bool>(),band.numel(),distances.size(0),float(background),smooth);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {material,near};
}

__global__ void polygon_distance_kernel(const float* points, const float* starts,
    const float* ends, float* out, int count, int edges) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  float x=points[2*i], y=points[2*i+1], best=INFINITY;
  int crossings=0;
  for(int j=0;j<edges;++j) {
    float u=starts[2*j], v=starts[2*j+1];
    float eu=ends[2*j]-u, ev=ends[2*j+1]-v;
    float ru=x-u, rv=y-v;
    float norm=fmaxf(eu*eu+ev*ev,FLT_EPSILON);
    // Each lane of ATen's two-element sum starts at positive zero.
    float dot=__fadd_rn(__fadd_rn(0.f,ru*eu),__fadd_rn(0.f,rv*ev));
    float t=fminf(fmaxf(dot/norm,0.f),1.f);
    float du=x-(u+t*eu), dv=y-(v+t*ev);
    best=fminf(best,fmaxf(du*du+dv*dv,0.f));
    float denom=ev>=0.f?fmaxf(ev,FLT_EPSILON):fminf(ev,-FLT_EPSILON);
    float crossing=u+((y-v)/denom)*eu;
    crossings+=((v<=y)!=(ends[2*j+1]<=y)) && x<crossing;
  }
  float distance=sqrtf(best);
  out[i]=(crossings%2==1)?-distance:distance;
}

at::Tensor polygon_distance(at::Tensor points, at::Tensor starts, at::Tensor ends) {
  check_tensor(points, points, at::kFloat, "polygon_distance", "points");
  check_tensor(starts, points, at::kFloat, "polygon_distance", "starts");
  check_tensor(ends, points, at::kFloat, "polygon_distance", "ends");
  c10::cuda::CUDAGuard guard(points.device());
  auto out=at::empty({points.size(0)},points.options());
  if(out.numel()) {
    polygon_distance_kernel<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(),starts.data_ptr<float>(),ends.data_ptr<float>(),
        out.data_ptr<float>(),points.size(0),starts.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

__global__ void conductor_edge_kernel(const float* distance, float* out,
    int nx, int ny, int nz, int axis, int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  int sy=ny-(axis==1), sz=nz-(axis==2);
  int z=i%sz, y=(i/sz)%sy, x=i/(sy*sz);
  int start=(x*ny+y)*nz+z, stride=axis==0?ny*nz:axis==1?nz:1;
  float low=distance[start], high=distance[start+stride];
  float lower=fminf(low,high), upper=fmaxf(low,high), span=upper-lower;
  bool cut=span>0.f;
  float fraction=fminf(fmaxf(-lower/(cut?span:1.f),0.f),1.f);
  out[i]=cut?fraction:float(upper<=0.f);
}

std::vector<at::Tensor> conductor_edges(at::Tensor distance) {
  check_tensor(distance, distance, at::kFloat, "conductor_edges", "distance");
  c10::cuda::CUDAGuard guard(distance.device());
  std::vector<at::Tensor> out;
  for(int axis=0;axis<3;++axis) {
    auto shape=distance.sizes().vec(); --shape[axis];
    out.push_back(at::empty(shape,distance.options()));
    conductor_edge_kernel<<<(out.back().numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        distance.data_ptr<float>(),out.back().data_ptr<float>(),distance.size(0),
        distance.size(1),distance.size(2),axis,out.back().numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
