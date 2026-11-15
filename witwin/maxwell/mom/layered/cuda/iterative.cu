#include "common.cuh"
#include <vector>
using namespace layered;
using at::Tensor;

// Matrix-free layered backend: RWG <-> face-vertex maps of the far action,
// near potential values and deterministic sparse kernels. Every output has
// one owner and sums its terms in a fixed order; no floating-point atomics.

struct SupportView {
  const int64_t *supports,*slots,*face_edges;
  const double *slopes,*support_slopes;
  int width;
};
SupportView support_view(Tensor supports,Tensor slots,Tensor slopes,Tensor face_edges,Tensor support_slopes) {
  return {supports.data_ptr<int64_t>(),slots.data_ptr<int64_t>(),face_edges.data_ptr<int64_t>(),
          slopes.data_ptr<double>(),support_slopes.numel() ? support_slopes.data_ptr<double>() : nullptr,
          int(face_edges.size(1))};
}
// The assembly gather's RWG slope of edge n on support side b (vertex slot j).
__device__ inline double support_slope(SupportView v,int n,int b,int face,int j) {
  return v.support_slopes ? v.support_slopes[2*n+b] : v.slopes[3*face+j];
}

// c[s,j,p] = sum over RWGs n supported on face s with free vertex j of
// slope*u[n,p], in face slot order. One thread per (face, column).
__global__ void face_coefficients_kernel(SupportView v,const C* u,C* out,int F,int P) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(F)*P) return;
  int s=i/P,p=i%P;
  C sums[3]{};
  for(int slot=0;slot<v.width;++slot) {
    if(v.slopes[v.width*s+slot]==0) continue;
    int n=v.face_edges[v.width*s+slot];
    for(int b=0;b<2;++b) if(v.supports[2*n+b]==s) {
      int j=v.slots[2*n+b];
      sums[j]+=support_slope(v,n,b,s,j)*u[int64_t(n)*P+p];
    }
  }
  for(int j=0;j<3;++j) out[(int64_t(s)*4+j)*P+p]=sums[j];
  out[(int64_t(s)*4+3)*P+p]=C(0);
}
Tensor layered_face_coefficients(Tensor u,Tensor supports,Tensor slots,Tensor slopes,Tensor face_edges,
    Tensor support_slopes) {
  c10::cuda::CUDAGuard guard(u.device());int F=face_edges.size(0),P=u.size(1);
  auto out=at::empty({F,4,P},u.options());
  face_coefficients_kernel<<<(int64_t(F)*P+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    support_view(supports,slots,slopes,face_edges,support_slopes),u.data_ptr<C>(),out.data_ptr<C>(),F,P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// (V u)[m,p] = sum over the two supports a of slope(m,a) * moments[t_a, j_a, p].
__global__ void edge_gather_kernel(SupportView v,const C* moments,C* out,int N,int P) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(N)*P) return;
  int m=i/P,p=i%P;C sum(0);
  for(int a=0;a<2;++a) {
    int t=v.supports[2*m+a];if(t<0) continue;
    int j=v.slots[2*m+a];
    sum+=support_slope(v,m,a,t,j)*moments[(int64_t(t)*4+j)*P+p];
  }
  out[i]=sum;
}
Tensor layered_edge_gather(Tensor moments,Tensor supports,Tensor slots,Tensor slopes,Tensor face_edges,
    Tensor support_slopes) {
  c10::cuda::CUDAGuard guard(moments.device());int N=supports.size(0),P=moments.size(2);
  auto out=at::empty({N,P},moments.options());
  edge_gather_kernel<<<(int64_t(N)*P+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    support_view(supports,slots,slopes,face_edges,support_slopes),moments.data_ptr<C>(),out.data_ptr<C>(),N,P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

__device__ inline int64_t find_sorted(const int64_t* values,int64_t first,int64_t last,int64_t key) {
  while(first<last) {
    int64_t middle=(first+last)/2;
    if(values[middle]<key) first=middle+1; else last=middle;
  }
  return first;
}

// Near magnetic potential V[m,n] on the RWG pattern: the gather_system sum
// over support pairs (a,b) in the same order, restricted to near face pairs
// (looked up in the row-sorted near list). One thread per pattern entry.
__global__ void near_potential_kernel(SupportView v,const int64_t* rows,const int64_t* cols,
    const int64_t* near_offsets,const int64_t* near_cols,const C* tiles,C* out,int64_t nnz) {
  int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=nnz) return;
  int m=rows[k],n=cols[k];C value(0);
  for(int a=0;a<2;++a) for(int b=0;b<2;++b) {
    int t=v.supports[2*m+a],s=v.supports[2*n+b];
    if(t<0 || s<0) continue;
    int64_t at=find_sorted(near_cols,near_offsets[t],near_offsets[t+1],s);
    if(at==near_offsets[t+1] || near_cols[at]!=s) continue;
    int i=v.slots[2*m+a],j=v.slots[2*n+b];
    value+=(v.support_slopes ? v.support_slopes[2*m+a]*v.support_slopes[2*n+b] :
            v.slopes[3*t+i]*v.slopes[3*s+j])*tiles[10*at+3*i+j];
  }
  out[k]=value;
}
Tensor layered_near_potential(Tensor rows,Tensor cols,Tensor near_offsets,Tensor near_cols,Tensor tiles,
    Tensor supports,Tensor slots,Tensor slopes,Tensor face_edges,Tensor support_slopes) {
  c10::cuda::CUDAGuard guard(rows.device());int64_t nnz=rows.numel();
  auto out=at::empty({nnz},tiles.options());
  near_potential_kernel<<<(nnz+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    support_view(supports,slots,slopes,face_edges,support_slopes),rows.data_ptr<int64_t>(),
    cols.data_ptr<int64_t>(),near_offsets.data_ptr<int64_t>(),near_cols.data_ptr<int64_t>(),
    tiles.data_ptr<C>(),out.data_ptr<C>(),nnz);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Local loss fold V' = V + R/(-i mu0 w) on the CSR pattern: loss_fold's
// first-occurrence column order and (side,slot) sums, one thread per row.
__global__ void near_loss_kernel(const double* blocks,const int64_t* columns,const int64_t* supports,
    const C* impedance,const int64_t* offsets,const int64_t* cols,C* values,C scale,int N,int W) {
  int m=blockIdx.x*blockDim.x+threadIdx.x;if(m>=N) return;
  for(int a=0;a<2*W;++a) {
    if(supports[2*m+a/W]<0) continue;
    int n=columns[2*W*m+a];
    bool first=true;
    for(int b=0;b<a;++b) first&=!(supports[2*m+b/W]>=0 && columns[2*W*m+b]==n);
    if(!first) continue;
    C value(0);
    for(int b=a;b<2*W;++b)
      if(supports[2*m+b/W]>=0 && columns[2*W*m+b]==n)
        value+=blocks[2*W*m+b]*impedance[supports[2*m+b/W]];
    // Zero Gram slots (padding, non-RWG face edges) may name edges outside
    // the pattern; every nonzero block couples RWGs of one face, a near pair.
    if(value==C(0)) continue;
    values[find_sorted(cols,offsets[m],offsets[m+1],n)]+=value*scale;
  }
}
void layered_near_loss(Tensor values,Tensor offsets,Tensor cols,Tensor blocks,Tensor columns,Tensor supports,
    Tensor impedance,double frequency) {
  c10::cuda::CUDAGuard guard(values.device());int N=offsets.numel()-1;
  constexpr double mu0=1/(eps0*c0*c0);
  near_loss_kernel<<<(N+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    blocks.data_ptr<double>(),columns.data_ptr<int64_t>(),supports.data_ptr<int64_t>(),impedance.data_ptr<C>(),
    offsets.data_ptr<int64_t>(),cols.data_ptr<int64_t>(),values.data_ptr<C>(),C(0,1/(mu0*2*pi*frequency)),
    N,blocks.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Segment sums of key-sorted products, ascending within each segment.
__global__ void segment_sum_kernel(const C* values,const int64_t* offsets,C* out,int64_t segments) {
  int64_t s=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(s>=segments) return;
  C sum(0);
  for(int64_t k=offsets[s];k<offsets[s+1];++k) sum+=values[k];
  out[s]=sum;
}
Tensor layered_segment_sum(Tensor values,Tensor offsets) {
  c10::cuda::CUDAGuard guard(values.device());int64_t segments=offsets.numel()-1;
  auto out=at::empty({segments},values.options());
  segment_sum_kernel<<<(segments+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    values.data_ptr<C>(),offsets.data_ptr<int64_t>(),out.data_ptr<C>(),segments);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// y = A x for CSR A [M,N] and row-major x [N,P]: one thread per (row, column),
// ascending column order. ~(24+16) B and 8 real FLOP per term.
__global__ void csr_matmul_kernel(const int64_t* offsets,const int64_t* cols,const C* values,const C* x,
    C* y,int M,int P) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(M)*P) return;
  int r=i/P,p=i%P;C sum(0);
  for(int64_t k=offsets[r];k<offsets[r+1];++k) sum+=values[k]*x[cols[k]*P+p];
  y[i]=sum;
}
Tensor layered_csr_matmul(Tensor offsets,Tensor cols,Tensor values,Tensor x) {
  c10::cuda::CUDAGuard guard(x.device());int M=offsets.numel()-1,P=x.size(1);
  auto y=at::empty({M,P},x.options());
  csr_matmul_kernel<<<(int64_t(M)*P+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    offsets.data_ptr<int64_t>(),cols.data_ptr<int64_t>(),values.data_ptr<C>(),x.data_ptr<C>(),
    y.data_ptr<C>(),M,P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

// Sparse fundamental cycles without the dense [N,N] setup matrix: the walk of
// cycles_kernel, emitted as (column, edge, value) triples. Pass 1 counts the
// entries of each edge's column, pass 2 writes them at the exclusive prefix.
// Integer work plus one +-1/length per entry; O(N*tree_depth), no atomics.
template<bool Fill> __global__ void cycle_entries_kernel(const int64_t* plus,const int64_t* minus,
    const double* length,const int64_t* order,const int64_t* parent,int64_t* counts,
    const int64_t* offsets,int64_t* columns,int64_t* rows,double* values,int N,int loops) {
  int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=N) return;
  int col=order[e];int64_t at=Fill ? offsets[e] : 0;
  auto emit=[&](int row,double value) {
    if(Fill) {columns[at]=col;rows[at]=row;values[at]=value;}
    ++at;
  };
  emit(e,1/length[e]);
  if(col<loops) {
    int a=plus[e],b=minus[e];
    while(a!=b) {
      if(parent[3*a+2]>=parent[3*b+2]) {
        int t=parent[3*a+1];emit(t,(plus[t]==a ? -1.0 : 1.0)/length[t]);a=parent[3*a];
      } else {
        int t=parent[3*b+1];emit(t,(plus[t]==b ? 1.0 : -1.0)/length[t]);b=parent[3*b];
      }
    }
  }
  if(!Fill) counts[e+1]=at;
}
std::vector<Tensor> layered_cycle_entries(Tensor plus,Tensor minus,Tensor lengths,Tensor order,Tensor parents,
    int64_t loops) {
  c10::cuda::CUDAGuard guard(plus.device());int N=plus.numel();
  auto stream=c10::cuda::getCurrentCUDAStream();
  auto counts=at::zeros({N+1},plus.options());
  cycle_entries_kernel<false><<<(N+127)/128,128,0,stream>>>(plus.data_ptr<int64_t>(),minus.data_ptr<int64_t>(),
    lengths.data_ptr<double>(),order.data_ptr<int64_t>(),parents.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),
    nullptr,nullptr,nullptr,nullptr,N,loops);
  auto offsets=at::cumsum(counts,0);
  // Setup-only size transfer.
  int64_t total=offsets.select(0,N).item<int64_t>();
  auto columns=at::empty({total},plus.options()),rows=at::empty({total},plus.options());
  auto values=at::empty({total},lengths.options());
  cycle_entries_kernel<true><<<(N+127)/128,128,0,stream>>>(plus.data_ptr<int64_t>(),minus.data_ptr<int64_t>(),
    lengths.data_ptr<double>(),order.data_ptr<int64_t>(),parents.data_ptr<int64_t>(),nullptr,
    offsets.data_ptr<int64_t>(),columns.data_ptr<int64_t>(),rows.data_ptr<int64_t>(),values.data_ptr<double>(),N,loops);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {columns,rows,values};
}
