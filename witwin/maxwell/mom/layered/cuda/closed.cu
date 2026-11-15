#include "common.cuh"
#include <vector>
using namespace layered;
using at::Tensor;

// Each compressed Cartesian cell owns its boundary faces. This constructs a
// surface mesh; no volume-current unknowns are introduced.
__device__ int owner_at(const double* boxes,int B,double x,double y,double z) {
  for(int b=0;b<B;++b) {
    const double* r=boxes+6*b;
    if(x>r[0] && x<r[1] && y>r[2] && y<r[3] && z>r[4] && z<r[5]) return b;
  }
  return -1;
}
// Logical traffic/cell: 24 B axes + up to 48*B B box metadata + 8 B
// output; six comparisons/box. Small box tables are cache-resident.
__global__ void occupancy(const double* x,const double* y,const double* z,
    const double* boxes,int64_t* owners,int nx,int ny,int nz,int B) {
  int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nx*ny*nz) return;
  int k=c%nz,j=(c/nz)%ny,i=c/(ny*nz);
  owners[c]=owner_at(boxes,B,(x[i]+x[i+1])/2,(y[j]+y[j+1])/2,(z[k]+z[k+1])/2);
}
// At most 16 B owner reads + 8 B output per candidate face; integer-only.
// Neighbour owner across local face q, or -1 outside the grid.
__device__ inline int64_t neighbour(const int64_t* owners,int q,int nx,int ny,int nz) {
  int c=q/6,a=(q%6)/2,side=q%2;
  int coord[3]={c/(ny*nz),(c/nz)%ny,c%nz},dims[3]={nx,ny,nz},stride[3]={ny*nz,nz,1};
  int next=coord[a]+(side?1:-1);
  return (next<0 || next>=dims[a]) ? -1 : owners[c+(side?1:-1)*stride[a]];
}
// With material groups, a face between two groups is also kept once, from
// the side with the larger group id (the lossy body for a lossy/PEC contact),
// so it is oriented outward from that body.
__global__ void exposed(const int64_t* owners,int64_t* keep,int nx,int ny,int nz,const int64_t* groups) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=nx*ny*nz*6) return;
  int64_t self=owners[q/6],other=neighbour(owners,q,nx,ny,nz);
  keep[q]=self>=0 && (other<0 || (groups && groups[self]>groups[other]));
}
__device__ void quad(const double* z,int i,int j,int k,int axis,int side,
    int64_t* faces,int64_t* ids,int64_t* kinds,int64_t* owners,
    int out,int owner,const double* planes,int P,int ny,int nz) {
  int u=(axis+1)%3,v=(axis+2)%3,c[3]={i,j,k};c[axis]+=side;
  int vertices[4];
  for(int q=0;q<4;++q) {
    int d[3]={c[0],c[1],c[2]};
    d[u]+=(q==1 || q==2);d[v]+=(q>=2);
    vertices[q]=(d[0]*(ny+1)+d[1])*(nz+1)+d[2];
  }
  int local[6]={0,1,2,0,2,3};
  if((i+j+k)%2) {local[2]=3;local[3]=1;}
  for(int t=0;t<2;++t) for(int n=0;n<3;++n)
    faces[(out+t)*3+n]=vertices[local[t*3+(side?n:2-n)]];
  int plane=-1;
  if(axis==2) for(int p=0;p<P;++p) if(planes[p]==z[c[2]]) plane=p;
  for(int t=0;t<2;++t) {ids[out+t]=plane;kinds[out+t]=(axis==2?side:2);owners[out+t]=owner;}
}
// Each retained quad writes 96 B topology/ownership, plus prefix/owner
// reads and the small plane lookup. No floating-point reductions.
__global__ void emit_body(const double* z,const int64_t* cells,const int64_t* keep,
    const int64_t* prefix,int64_t* faces,int64_t* ids,int64_t* kinds,int64_t* owners,
    const double* planes,int P,int nx,int ny,int nz) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=nx*ny*nz*6 || !keep[q]) return;
  int c=q/6,k=c%nz,j=(c/nz)%ny,i=c/(ny*nz),out=2*(prefix[q]-1);
  quad(z,i,j,k,(q%6)/2,q%2,faces,ids,kinds,owners,out,cells[c],planes,P,ny,nz);
  // Kind 3 marks a material contact face (only emitted with groups).
  if(neighbour(cells,q,nx,ny,nz)>=0) {kinds[out]=3;kinds[out+1]=3;}
}
__global__ void bridge_keep(const double* x,const double* y,const double* z,
    const double* bridges,const double* boxes,int64_t* keep,int nx,int ny,int nz,int H,int B) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=H*nx*ny*(nz+1)) return;
  int k=q%(nz+1),j=(q/(nz+1))%ny,i=(q/((nz+1)*ny))%nx,b=q/((nz+1)*ny*nx);
  const double* r=bridges+5*b;double cx=(x[i]+x[i+1])/2,cy=(y[j]+y[j+1])/2;
  bool physical=false;
  for(int a=0;a<B;++a) {
    const double* s=boxes+6*a;
    physical|=cx>s[0] && cx<s[1] && cy>s[2] && cy<s[3] && z[k]>=s[4] && z[k]<=s[5];
  }
  keep[q]=z[k]==r[4] && cx>r[0] && cx<r[1] && cy>r[2] && cy<r[3] && !physical;
}
__global__ void emit_bridge(const double* z,const int64_t* keep,const int64_t* prefix,
    int64_t* faces,int64_t* ids,int64_t* kinds,int64_t* owners,const double* planes,
    int P,int nx,int ny,int nz,int H,int offset) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=H*nx*ny*(nz+1) || !keep[q]) return;
  int k=q%(nz+1),j=(q/(nz+1))%ny,i=(q/((nz+1)*ny))%nx;
  quad(z,i,j,k-1,2,1,faces,ids,kinds,owners,offset+2*(prefix[q]-1),-1,planes,P,ny,nz);
}
std::vector<Tensor> closed_mesh_impl(Tensor x,Tensor y,Tensor z,Tensor boxes,Tensor bridges,Tensor planes,
    const int64_t* groups) {
  c10::cuda::CUDAGuard guard(x.device());auto stream=c10::cuda::getCurrentCUDAStream();
  int nx=x.numel()-1,ny=y.numel()-1,nz=z.numel()-1,C=nx*ny*nz;
  auto opt=x.options().dtype(at::kLong);
  auto cells=at::empty({C},opt),keep=at::empty({6*C},opt);
  occupancy<<<(C+127)/128,128,0,stream>>>(x.data_ptr<double>(),y.data_ptr<double>(),z.data_ptr<double>(),
      boxes.data_ptr<double>(),cells.data_ptr<int64_t>(),nx,ny,nz,boxes.size(0));
  exposed<<<(6*C+127)/128,128,0,stream>>>(cells.data_ptr<int64_t>(),keep.data_ptr<int64_t>(),nx,ny,nz,groups);
  auto prefix=at::cumsum(keep,0);int body=2*prefix.select(0,6*C-1).item<int64_t>();
  int H=bridges.size(0),Q=H*nx*ny*(nz+1);auto bk=at::empty({Q},opt);
  if(Q) bridge_keep<<<(Q+127)/128,128,0,stream>>>(x.data_ptr<double>(),y.data_ptr<double>(),
      z.data_ptr<double>(),bridges.data_ptr<double>(),boxes.data_ptr<double>(),bk.data_ptr<int64_t>(),
      nx,ny,nz,H,boxes.size(0));
  auto bp=at::cumsum(bk,0);int bridge=Q ? 2*bp.select(0,Q-1).item<int64_t>() : 0;
  auto faces=at::empty({body+bridge,3},opt),ids=at::empty({body+bridge},opt);
  auto kinds=at::empty_like(ids),owners=at::empty_like(ids);
  emit_body<<<(6*C+127)/128,128,0,stream>>>(z.data_ptr<double>(),cells.data_ptr<int64_t>(),
      keep.data_ptr<int64_t>(),prefix.data_ptr<int64_t>(),faces.data_ptr<int64_t>(),ids.data_ptr<int64_t>(),
      kinds.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),planes.data_ptr<double>(),planes.numel(),nx,ny,nz);
  if(Q) emit_bridge<<<(Q+127)/128,128,0,stream>>>(z.data_ptr<double>(),bk.data_ptr<int64_t>(),
      bp.data_ptr<int64_t>(),faces.data_ptr<int64_t>(),ids.data_ptr<int64_t>(),kinds.data_ptr<int64_t>(),
      owners.data_ptr<int64_t>(),planes.data_ptr<double>(),planes.numel(),nx,ny,nz,H,body);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {faces,ids,kinds,owners};
}
std::vector<Tensor> layered_closed_mesh(Tensor x,Tensor y,Tensor z,Tensor boxes,Tensor bridges,Tensor planes) {
  return closed_mesh_impl(x,y,z,boxes,bridges,planes,nullptr);
}
// Surface-admittance mesh: exposed union faces plus one face per contact
// between different material groups (groups[box], larger id emits).
std::vector<Tensor> layered_contact_mesh(Tensor x,Tensor y,Tensor z,Tensor boxes,Tensor bridges,Tensor planes,
    Tensor groups) {
  return closed_mesh_impl(x,y,z,boxes,bridges,planes,groups.data_ptr<int64_t>());
}

// k faces at an edge have k-1 independent sum-zero flux pairs. Pair the
// first incident triangle with every other triangle, in ascending face order.
// 48 B topology output/RWG plus fixed incidence metadata reads. A face's
// padded adjacency is packed by its own thread, so integer writes are unique.
__global__ void basis_pairs(const int64_t* edges,const int64_t* counts,const int64_t* offsets,
    const int64_t* order,const int64_t* bases,int64_t* be,int64_t* supports,int64_t* slots,int E) {
  int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=E) return;
  int first=order[offsets[e]];
  for(int j=1;j<counts[e];++j) {
    int n=bases[e]+j-1,other=order[offsets[e]+j];
    be[2*n]=edges[2*e];be[2*n+1]=edges[2*e+1];
    supports[2*n]=first/3;supports[2*n+1]=other/3;
    slots[2*n]=first%3;slots[2*n+1]=other%3;
  }
}
__global__ void basis_faces(const int64_t* face_edges,const int64_t* counts,const int64_t* offsets,
    const int64_t* order,const int64_t* bases,int64_t* adj,int64_t* signs,int64_t* local,
    int64_t* positions,int F,int W) {
  int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=F) return;int at=0;
  for(int l=0;l<3;++l) {
    int e=face_edges[3*f+l],first=order[offsets[e]];
    for(int j=1;j<counts[e];++j) {
      int other=order[offsets[e]+j];
      if(first/3!=f && other/3!=f) continue;
      int n=bases[e]+j-1,side=first/3==f ? 0 : 1;
      adj[f*W+at]=n;signs[f*W+at]=side ? -1 : 1;local[f*W+at]=l;
      positions[2*n+side]=at++;
    }
  }
}
std::vector<Tensor> layered_junction_basis(Tensor edges,Tensor face_edges,Tensor counts) {
  c10::cuda::CUDAGuard guard(edges.device());auto stream=c10::cuda::getCurrentCUDAStream();
  auto flat=face_edges.reshape({-1}),order=at::argsort(flat,true,0,false);
  auto offsets=at::cumsum(counts,0)-counts;
  auto number=(counts-1).clamp_min(0),bases=at::cumsum(number,0)-number;
  int E=counts.numel(),F=face_edges.size(0),N=number.sum().item<int64_t>();
  int W=3*std::max(int64_t(1),number.max().item<int64_t>());
  auto be=at::empty({N,2},edges.options()),supports=at::empty_like(be),slots=at::empty_like(be);
  auto adj=at::zeros({F,W},edges.options()),signs=at::zeros_like(adj),local=at::zeros_like(adj);
  auto positions=at::empty_like(be);
  basis_pairs<<<(E+127)/128,128,0,stream>>>(edges.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),order.data_ptr<int64_t>(),bases.data_ptr<int64_t>(),be.data_ptr<int64_t>(),
      supports.data_ptr<int64_t>(),slots.data_ptr<int64_t>(),E);
  basis_faces<<<(F+127)/128,128,0,stream>>>(face_edges.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),order.data_ptr<int64_t>(),bases.data_ptr<int64_t>(),adj.data_ptr<int64_t>(),
      signs.data_ptr<int64_t>(),local.data_ptr<int64_t>(),positions.data_ptr<int64_t>(),F,W);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {be,supports,slots,adj,signs,local,positions};
}

// Exact degree-two triangle moment; translate to the first vertex to avoid
// cancellation at RFIC global coordinates. One owner per face/local pair.
// Per entry: <=72 B triangle + 24 B metric + 48 B index reads, 8 B
// output, approximately 70 real FLOP. Adjacent local pairs reuse cache lines.
__global__ void gram(const double* triangles,const double* areas,const double* lengths,
    const int64_t* edges,const int64_t* signs,const int64_t* local,double* out,int F,int W) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=F*W*W) return;
  int j=q%W,i=(q/W)%W,f=q/(W*W);double value=0;
  for(int d=0;d<3;++d) {
    double origin=triangles[9*f+d],mean=0,square=0;
    for(int a=0;a<3;++a) {double v=triangles[9*f+3*a+d]-origin;mean+=v/3;square+=v*v/12;}
    double second=square+(3*mean)*(3*mean)/12;
    double a=triangles[9*f+3*local[f*W+i]+d]-origin,b=triangles[9*f+3*local[f*W+j]+d]-origin;
    value+=second-(a+b)*mean+a*b;
  }
  out[q]=value*double(signs[f*W+i]*signs[f*W+j])*
      lengths[edges[f*W+i]]*lengths[edges[f*W+j]]/(4*areas[f]);
}
Tensor layered_junction_gram(Tensor triangles,Tensor areas,Tensor lengths,Tensor edges,Tensor signs,Tensor local) {
  c10::cuda::CUDAGuard guard(triangles.device());int F=edges.size(0),W=edges.size(1);
  auto out=at::empty({F,W,W},triangles.options());
  gram<<<(out.numel()+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(triangles.data_ptr<double>(),
      areas.data_ptr<double>(),lengths.data_ptr<double>(),edges.data_ptr<int64_t>(),signs.data_ptr<int64_t>(),
      local.data_ptr<int64_t>(),out.data_ptr<double>(),F,W);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// One owner per frequency/face. Reuse the public roughness descriptor.
// Logical traffic: 8 B f + 8 B sigma + 16 B factor + 8 B owner + 16 B
// output; about 10 real FLOP and one sqrt for a lossy face.
__global__ void surface_loss(const double* frequencies,const double* sigma,const C* factor,
    const int64_t* owners,C* impedance,int K,int F,int M) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=K*F) return;
  int f=q/F,t=q%F,m=owners[t];double rs=0;
  if(m>=0 && sigma[m]>0) rs=sqrt(pi*frequencies[f]*(4e-7*pi)/sigma[m]);
  impedance[q]=m>=0 ? C(rs,-rs)*factor[f*M+m] : C(0);
}
Tensor layered_closed_impedance(Tensor frequencies,Tensor sigma,Tensor factors,Tensor owners) {
  c10::cuda::CUDAGuard guard(frequencies.device());int K=frequencies.numel(),F=owners.numel();
  auto out=at::empty({K,F},frequencies.options().dtype(at::kComplexDouble));
  surface_loss<<<(K*F+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(frequencies.data_ptr<double>(),
      sigma.data_ptr<double>(),factors.data_ptr<C>(),owners.data_ptr<int64_t>(),out.data_ptr<C>(),K,F,sigma.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Preserve fixed plane coordinates exactly, including segment endpoints.
// 72 B input + 24 B output/triangle, 15 real FLOP, no reduction/atomics.
__global__ void centroids_kernel(const double* triangles,double* out,int F) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=3*F) return;
  int f=q/3,d=q%3;double origin=triangles[9*f+d];
  out[q]=origin+((triangles[9*f+3+d]-origin)+(triangles[9*f+6+d]-origin))/3;
}
Tensor layered_closed_centroids(Tensor triangles) {
  c10::cuda::CUDAGuard guard(triangles.device());int F=triangles.size(0);
  auto out=at::empty({F,3},triangles.options());
  centroids_kernel<<<(3*F+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<double>(),out.data_ptr<double>(),F);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
