#include "common.cuh"
#include <torch/library.h>
using namespace layered;
using at::Tensor;

// Native network-admittance VJP for parameters of the local loss operator.
//
// Y = V^T (Z+R)^-1 V with R = sum_f Zs_f G_f gives, for the forward currents
// X = (Z+R)^-1 V (all ports), dY_pq/dZs_f = -sum_{mn in f} X_mp G_f,mn X_nq.
// For an upstream gradient gY (conjugate Wirtinger convention of autograd):
//   grad Zs_f = -conj(sum_{mn in f} G_f,mn sum_q L_mq X_nq),  L = X conj(gY).
// The contraction walks the same local Gram rows (blocks, columns, supports)
// as loss_fold and never forms dZ. Faces are grouped by the parameter group
// that owns their impedance; the sum over a group's faces is a fixed-order
// per-thread stride followed by a fixed shared-memory tree, so results are
// bitwise reproducible. No floating-point atomics anywhere.

// Stage 1: one owner per L[k,m,q], fixed p order.
// Traffic 16*(2P+1) B/output, 8P real FLOP/output.
__global__ void adjoint_left_kernel(const C* x,const C* g,C* left,int64_t K,int64_t N,int P) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=K*N*P) return;
  int q=t%P;int64_t km=t/P,k=km/N;C sum(0);
  for(int p=0;p<P;++p) sum+=x[km*P+p]*std::conj(g[(k*P+p)*P+q]);
  left[t]=sum;
}

// Stage 2: one block per (frequency, group). Each lossy face has <= W rows
// (row = 2*edge+side) and each row W local Gram columns: per slot 16 B of
// block/column metadata plus 32P B of L/X rows (cache resident), 8P+4 real
// FLOP. Total O(K*F*W^2*P), linear in the mesh like the loss operator.
constexpr int THREADS=128;
__global__ void loss_adjoint_kernel(const C* left,const C* x,const double* blocks,const int64_t* columns,
    const int64_t* rows,const int64_t* row_offsets,const int64_t* faces,const int64_t* face_offsets,
    C* out,int64_t N,int P,int W,int G) {
  __shared__ C partial[THREADS];
  int k=blockIdx.x/G,group=blockIdx.x%G;C sum(0);
  for(int64_t e=face_offsets[group]+threadIdx.x;e<face_offsets[group+1];e+=THREADS) {
    int64_t face=faces[e];
    for(int64_t r=row_offsets[face];r<row_offsets[face+1];++r) {
      int64_t row=rows[r];
      const C* l=left+(int64_t(k)*N+row/2)*P;
      for(int slot=0;slot<W;++slot) {
        double block=blocks[row*W+slot];
        if(block==0) continue;
        const C* xn=x+(int64_t(k)*N+columns[row*W+slot])*P;
        C dot(0);
        for(int q=0;q<P;++q) dot+=l[q]*xn[q];
        sum+=block*dot;
      }
    }
  }
  partial[threadIdx.x]=sum;
  __syncthreads();
  for(int half=THREADS/2;half>0;half/=2) {
    if(threadIdx.x<half) partial[threadIdx.x]+=partial[threadIdx.x+half];
    __syncthreads();
  }
  if(!threadIdx.x) out[blockIdx.x]=-std::conj(partial[0]);
}

Tensor layered_loss_adjoint(Tensor currents,Tensor grad,Tensor blocks,Tensor columns,Tensor rows,
    Tensor row_offsets,Tensor faces,Tensor face_offsets) {
  c10::cuda::CUDAGuard guard(currents.device());auto stream=c10::cuda::getCurrentCUDAStream();
  int64_t K=currents.size(0),N=currents.size(1);int P=currents.size(2),W=blocks.size(-1);
  int G=face_offsets.numel()-1;
  auto left=at::empty_like(currents),out=at::empty({K,G},currents.options());
  adjoint_left_kernel<<<(K*N*P+127)/128,128,0,stream>>>(currents.data_ptr<C>(),grad.data_ptr<C>(),
      left.data_ptr<C>(),K,N,P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(K*G) loss_adjoint_kernel<<<K*G,THREADS,0,stream>>>(left.data_ptr<C>(),currents.data_ptr<C>(),
      blocks.data_ptr<double>(),columns.data_ptr<int64_t>(),rows.data_ptr<int64_t>(),
      row_offsets.data_ptr<int64_t>(),faces.data_ptr<int64_t>(),face_offsets.data_ptr<int64_t>(),
      out.data_ptr<C>(),N,P,W,G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

TORCH_LIBRARY_FRAGMENT(witwin_maxwell_mom_layered,m) {
  m.def("loss_adjoint(Tensor currents, Tensor grad, Tensor blocks, Tensor columns, Tensor rows, "
        "Tensor row_offsets, Tensor faces, Tensor face_offsets) -> Tensor");
  m.impl("loss_adjoint",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_loss_adjoint)));
}
