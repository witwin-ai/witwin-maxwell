// H-matrix far action of the layered face-pair operator (operator_backend="hmatrix").
//
// The far operator acts on face coefficients x [F,4,P] (three vertex channels
// and one charge channel) like layered_far_apply. Admissible blocks hold
// low-rank factors U [w*m,k] and V [k,w*n] per channel (w=3 vector, w=1
// scalar) for the unordered cluster pair (T,S); the transposed block (S,T) is
// V^T U^T, so the compressed operator stays exactly complex symmetric.
// Inadmissible leaf blocks hold their far face-pair tiles once per unordered
// pair. Every output has one owning thread, sums run in a fixed order and
// there are no atomics: the action is bitwise reproducible.
#include "common.cuh"
#include <vector>

using namespace layered;
using at::Tensor;

namespace {
constexpr int MAX_COLUMNS=16;

// proj[g,side,p] for global rank row g of block b (rank_block[g]):
// side 0: sum_i V_b[r,i] x[S face i/w, comp0+i%w, p]   (V_b x_S)
// side 1: sum_i U_b[i,r] x[T face i/w, comp0+i%w, p]   (U_b^T x_T)
// One warp per (g,side); lanes stride the inner index; fixed shuffle tree.
__global__ void project_kernel(const C* factors,const int64_t* blocks,const int64_t* layout,
    const int64_t* rank_offsets,const int64_t* rank_block,const int64_t* faces,const C* x,C* proj,
    int64_t ranks,int w,int comp0,int P,int p0,int pc) {
  int64_t warp=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
  if(warp>=2*ranks) return;
  int lane=threadIdx.x%32;
  int64_t g=warp/2;int side=warp%2;
  int64_t b=rank_block[g];
  int64_t r=g-rank_offsets[b];
  int64_t rank=rank_offsets[b+1]-rank_offsets[b];
  int64_t row_start=blocks[4*b],m=blocks[4*b+1],col_start=blocks[4*b+2],n=blocks[4*b+3];
  const C* u=factors+layout[2*b];const C* v=factors+layout[2*b+1];
  int64_t count=side==0 ? w*n : w*m;
  int64_t start=side==0 ? col_start : row_start;
  C sums[MAX_COLUMNS];
  for(int q=0;q<pc;++q) sums[q]=C(0);
  for(int64_t i=lane;i<count;i+=32) {
    C factor=side==0 ? v[r*(w*n)+i] : u[i*rank+r];
    const C* xi=x+(faces[start+i/w]*4+comp0+i%w)*P+p0;
    for(int q=0;q<pc;++q) sums[q]+=factor*xi[q];
  }
  for(int q=0;q<pc;++q) {
    double re=sums[q].real(),im=sums[q].imag();
    for(int delta=16;delta>0;delta/=2) {
      re+=__shfl_down_sync(0xffffffff,re,delta);
      im+=__shfl_down_sync(0xffffffff,im,delta);
    }
    if(lane==0) proj[(g*2+side)*P+p0+q]=C(re,im);
  }
}

// out[t,comp0+a,p] += sum over the entries (b,l,side) of face t in CSR order:
// side 0: sum_r U_b[l*w+a,r] proj[b,r,0,p]; side 1: sum_r V_b[r,l*w+a] proj[b,r,1,p].
__global__ void expand_kernel(const C* factors,const int64_t* blocks,const int64_t* layout,
    const int64_t* rank_offsets,const int64_t* entry_offsets,const int64_t* entries,const C* proj,C* out,
    int F,int w,int comp0,int P) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(F)*w*P) return;
  int p=i%P,a=(i/P)%w,t=i/(int64_t(w)*P);
  C sum(0);
  for(int64_t e=entry_offsets[t];e<entry_offsets[t+1];++e) {
    int64_t b=entries[3*e],l=entries[3*e+1],side=entries[3*e+2];
    int64_t first=rank_offsets[b],rank=rank_offsets[b+1]-first;
    if(side==0) {
      const C* u=factors+layout[2*b]+(l*w+a)*rank;
      for(int64_t r=0;r<rank;++r) sum+=u[r]*proj[((first+r)*2)*P+p];
    } else {
      int64_t n=blocks[4*b+3];
      const C* v=factors+layout[2*b+1]+l*w+a;
      for(int64_t r=0;r<rank;++r) sum+=v[r*w*n]*proj[((first+r)*2+1)*P+p];
    }
  }
  out[(int64_t(t)*4+comp0+a)*P+p]+=sum;
}

// Leaf far pairs (t_e,s_e) with tiles in (t_e,s_e) orientation, stored once.
// out[t,k,p] = sum_{e in forward(t)} tile_e x[s_e] + sum_{e in reverse(t)} tile_e^T x[t_e]:
// k<3 contracts the vector tile (transposed on reverse entries), k=3 the scalar tile.
__global__ void leaf_kernel(const C* tiles,const int64_t* pairs,const int64_t* forward_offsets,
    const int64_t* forward,const int64_t* reverse_offsets,const int64_t* reverse,const C* x,C* out,int F,int P) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(F)*4*P) return;
  int p=i%P,k=(i/P)%4,t=i/(4*P);
  C sum(0);
  for(int64_t j=forward_offsets[t];j<forward_offsets[t+1];++j) {
    int64_t e=forward[j];const C* tile=tiles+10*e;const C* c=x+pairs[2*e+1]*4*P+p;
    if(k==3) sum+=tile[9]*c[3*P];
    else for(int q=0;q<3;++q) sum+=tile[3*k+q]*c[q*P];
  }
  for(int64_t j=reverse_offsets[t];j<reverse_offsets[t+1];++j) {
    int64_t e=reverse[j];const C* tile=tiles+10*e;const C* c=x+pairs[2*e]*4*P+p;
    if(k==3) sum+=tile[9]*c[3*P];
    else for(int q=0;q<3;++q) sum+=tile[3*q+k]*c[q*P];
  }
  out[i]=sum;
}
}  // namespace

// Leaf part of the far action: a fresh [F,4,P] output.
Tensor layered_hmatrix_leaf(Tensor tiles,Tensor pairs,Tensor forward_offsets,Tensor forward,
    Tensor reverse_offsets,Tensor reverse,Tensor x) {
  c10::cuda::CUDAGuard guard(x.device());
  int F=x.size(0),P=x.size(2);
  auto out=at::empty_like(x);
  leaf_kernel<<<(int64_t(F)*4*P+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      tiles.data_ptr<C>(),pairs.data_ptr<int64_t>(),forward_offsets.data_ptr<int64_t>(),
      forward.data_ptr<int64_t>(),reverse_offsets.data_ptr<int64_t>(),reverse.data_ptr<int64_t>(),
      x.data_ptr<C>(),out.data_ptr<C>(),F,P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Adds one channel's low-rank part to out [F,4,P] in place.
// blocks [B,4] = (row start, m, column start, n) in cluster positions,
// layout [B,2] = (U offset, V offset) into factors, rank_offsets [B+1],
// rank_block [ranks], faces [F] cluster order, entries [E,3] = (block, local, side).
void layered_hmatrix_lowrank(Tensor factors,Tensor blocks,Tensor layout,Tensor rank_offsets,Tensor rank_block,
    Tensor faces,Tensor entry_offsets,Tensor entries,Tensor x,Tensor out,int64_t w,int64_t comp0) {
  c10::cuda::CUDAGuard guard(x.device());
  auto stream=c10::cuda::getCurrentCUDAStream();
  int F=x.size(0),P=x.size(2);
  int64_t ranks=rank_block.numel();
  if(ranks==0) return;
  auto proj=at::empty({ranks,2,P},x.options());
  for(int p0=0;p0<P;p0+=MAX_COLUMNS) {
    int pc=std::min(MAX_COLUMNS,P-p0);
    project_kernel<<<(2*ranks*32+127)/128,128,0,stream>>>(factors.data_ptr<C>(),blocks.data_ptr<int64_t>(),
        layout.data_ptr<int64_t>(),rank_offsets.data_ptr<int64_t>(),rank_block.data_ptr<int64_t>(),
        faces.data_ptr<int64_t>(),x.data_ptr<C>(),proj.data_ptr<C>(),ranks,w,comp0,P,p0,pc);
  }
  expand_kernel<<<(int64_t(F)*w*P+127)/128,128,0,stream>>>(factors.data_ptr<C>(),blocks.data_ptr<int64_t>(),
      layout.data_ptr<int64_t>(),rank_offsets.data_ptr<int64_t>(),entry_offsets.data_ptr<int64_t>(),
      entries.data_ptr<int64_t>(),proj.data_ptr<C>(),out.data_ptr<C>(),F,w,comp0,P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
