#include "common.cuh"
using namespace layered;

namespace {
struct DD { double hi,lo; };
// Fixed series coefficients: hi=round(1/k^2), lo=round(1/k^2-hi).
// Keeping their fp106 sum in constant memory removes two fp64 divides per
// DD iteration; a_k=-(2*k-1)^2/(8*k) is likewise frequency independent.
__device__ __constant__ DD reciprocals[128]={
  {1.0,0.0},{0.25,0.0},{0.1111111111111111,6.1679056923619804e-18},{0.0625,0.0},
  {0.04,-8.326672684688674e-19},{0.027777777777777776,1.5419764230904951e-18},{0.02040816326530612,1.6285159162231251e-18},{0.015625,0.0},
  {0.012345679012345678,6.853228547068867e-19},{0.01,-2.0816681711721684e-19},{0.008264462809917356,-2.2938492244321417e-19},{0.006944444444444444,3.854941057726238e-19},
  {0.005917159763313609,1.3857258535909407e-19},{0.00510204081632653,4.0712897905578128e-19},{0.0044444444444444444,3.854941057726238e-21},{0.00390625,0.0},
  {0.0034602076124567475,-1.050438090989416e-19},{0.0030864197530864196,1.7133071367672168e-19},{0.002770083102493075,-6.006660235376756e-21},{0.0025,-5.204170427930421e-20},
  {0.0022675736961451248,-5.998760319420931e-20},{0.002066115702479339,-5.734623061080354e-20},{0.001890359168241966,-5.123828792464577e-20},{0.001736111111111111,9.637352644315594e-20},
  {0.0016,-7.667477763817488e-20},{0.0014792899408284023,3.4643146339773517e-20},{0.0013717421124828531,1.0024036546710974e-19},{0.0012755102040816326,1.0178224476394532e-19},
  {0.0011890606420927466,1.062286076252147e-19},{0.0011111111111111111,9.637352644315595e-22},{0.001040582726326743,-9.702537651795358e-20},{0.0009765625,0.0},
  {0.0009182736455463728,1.0652858811381907e-20},{0.0008650519031141869,-2.62609522747354e-20},{0.0008163265306122449,-3.026915452979939e-20},{0.0007716049382716049,4.283267841918042e-20},
  {0.0007304601899196494,-1.4176200794368538e-20},{0.0006925207756232687,-1.501665058844189e-21},{0.0006574621959237344,-3.278980929278975e-20},{0.000625,-1.3010426069826053e-20},
  {0.000594883997620464,-2.857237135104572e-20},{0.0005668934240362812,-1.499690079855233e-20},{0.0005408328826392645,-4.040104363669619e-20},{0.0005165289256198347,-1.4336557652700885e-20},
  {0.0004938271604938272,-1.1618364021202689e-20},{0.0004725897920604915,-1.2809571981161442e-20},{0.0004526935264825713,4.073733830524983e-21},{0.00043402777777777775,2.4093381610788986e-20},
  {0.00041649312786339027,-1.6549774936107345e-20},{0.0004,-1.916869440954372e-20},{0.00038446751249519417,-1.1671534344326845e-20},{0.0003698224852071006,8.660786584943379e-21},
  {0.000355998576005696,-1.914433170355323e-20},{0.0003429355281207133,2.5060091366777436e-20},{0.00033057851239669424,-2.435422731252563e-20},{0.00031887755102040814,2.544556119098633e-20},
  {0.0003077870113881194,1.1379284112574854e-20},{0.00029726516052318666,2.6557151906303675e-20},{0.0002872737719046251,-1.6460811495506725e-20},{0.0002777777777777778,2.4093381610788987e-22},
  {0.0002687449610319806,2.422773734000798e-20},{0.00026014568158168577,-2.4256344129488394e-20},{0.0002519526329050139,-2.4735325451892775e-20},{0.000244140625,0.0},
  {0.00023668639053254438,2.290296896907249e-21},{0.0002295684113865932,2.663214702845477e-21},{0.00022276676319893073,-1.1526742856286633e-20},{0.00021626297577854672,-6.56523806868385e-21},
  {0.00021003990758244065,1.2376893105353322e-20},{0.00020408163265306123,-7.567288632449848e-21},{0.00019837333862328903,8.71061059029219e-22},{0.00019290123456790122,1.0708169604795105e-20},
  {0.00018765246762994934,-8.519603297585006e-21},{0.00018261504747991235,-3.5440501985921345e-21},{0.00017777777777777779,-8.519419737574986e-21},{0.00017313019390581717,-3.7541626471104723e-22},
  {0.00016866250632484398,4.7224694053699026e-21},{0.0001643655489809336,-8.197452323197437e-21},{0.0001602307322544464,4.560215835241867e-21},{0.00015625,-3.252606517456513e-21},
  {0.00015241579027587258,-3.920545121508702e-21},{0.000148720999405116,-7.14309283776143e-21},{0.00014515894904920887,1.2366262984910511e-20},{0.0001417233560090703,-3.749225199638082e-21},
  {0.0001384083044982699,-1.1791167571356196e-20},{0.00013520822065981613,-1.0100260909174048e-20},{0.0001321178491214163,-6.2668575830678845e-21},{0.00012913223140495868,-3.584139413175221e-21},
  {0.00012624668602449185,7.463214676779716e-21},{0.0001234567901234568,-2.9045910053006722e-21},{0.00012075836251660427,3.3811763198210545e-21},{0.00011814744801512288,-3.2023929952903606e-21},
  {0.00011562030292519366,5.7836024665336985e-21},{0.00011317338162064282,1.0184334576312457e-21},{0.00011080332409972299,4.6386433667696994e-21},{0.00010850694444444444,6.0233454026972465e-21},
  {0.00010628122010840685,-6.075519135327688e-21},{0.00010412328196584757,-4.137443734026836e-21},{0.00010203040506070809,1.1836509790424342e-21},{0.0001,-4.79217360238593e-21},
  {9.802960494069208e-05,6.726442994919749e-21},{9.611687812379854e-05,-2.917883586081711e-21},{9.425959091337543e-05,1.333663714858689e-21},{9.245562130177515e-05,2.1651966462358448e-21},
  {9.070294784580499e-05,-1.8574030415256206e-21},{8.8999644001424e-05,-4.7860829258883075e-21},{8.734387282732117e-05,-4.884070839893431e-21},{8.573388203017832e-05,6.265022841694359e-21},
  {8.416799932665601e-05,-2.807235866600903e-21},{8.264462809917356e-05,-6.088556828131407e-21},{8.116224332440549e-05,1.4365392797521192e-21},{7.971938775510203e-05,6.3613902977465824e-21},
  {7.831466833737959e-05,2.0463052985277356e-21},{7.694675284702985e-05,2.8448210281437135e-21},{7.561436672967863e-05,6.624085862898205e-21},{7.431629013079666e-05,6.639287976575919e-21},
  {7.305135510263716e-05,-3.643312143643305e-21},{7.181844297615628e-05,-4.115202873876681e-21},{7.06164818868724e-05,8.986528493431684e-22},{6.944444444444444e-05,6.023345402697247e-23},
  {6.830134553650706e-05,6.168570518956527e-21},{6.718624025799516e-05,6.056934335001995e-21},{6.609822195782933e-05,4.3544738254775904e-21},{6.503642039542144e-05,-6.0640860323720985e-21},
  {6.4e-05,2.8961208431432794e-21},{6.298815822625347e-05,-6.183831362973194e-21},{6.2000124000248e-05,-5.8624826069745216e-21},{6.103515625e-05,0.0}
};
__device__ __constant__ double asymptotic[96]={
  -0.125,-0.5625,-1.0416666666666667,-1.53125,-2.025,-2.5208333333333335,-3.017857142857143,-3.515625,
  -4.013888888888889,-4.5125,-5.011363636363637,-5.510416666666667,-6.009615384615385,-6.508928571428571,-7.008333333333334,-7.5078125,
  -8.007352941176471,-8.506944444444445,-9.006578947368421,-9.50625,-10.005952380952381,-10.505681818181818,-11.005434782608695,-11.505208333333334,
  -12.005,-12.504807692307692,-13.00462962962963,-13.504464285714286,-14.004310344827585,-14.504166666666666,-15.004032258064516,-15.50390625,
  -16.00378787878788,-16.503676470588236,-17.00357142857143,-17.50347222222222,-18.00337837837838,-18.50328947368421,-19.003205128205128,-19.503125,
  -20.003048780487806,-20.50297619047619,-21.002906976744185,-21.50284090909091,-22.002777777777776,-22.502717391304348,-23.002659574468087,-23.502604166666668,
  -24.002551020408163,-24.5025,-25.002450980392158,-25.502403846153847,-26.00235849056604,-26.502314814814813,-27.00227272727273,-27.502232142857142,
  -28.00219298245614,-28.502155172413794,-29.002118644067796,-29.502083333333335,-30.00204918032787,-30.50201612903226,-31.001984126984127,-31.501953125,
  -32.00192307692308,-32.50189393939394,-33.00186567164179,-33.501838235294116,-34.0018115942029,-34.50178571428572,-35.00176056338028,-35.501736111111114,
  -36.00171232876713,-36.501689189189186,-37.001666666666665,-37.5016447368421,-38.00162337662338,-38.50160256410256,-39.00158227848101,-39.5015625,
  -40.001543209876544,-40.5015243902439,-41.001506024096386,-41.501488095238095,-42.00147058823529,-42.50145348837209,-43.00143678160919,-43.50142045454545,
  -44.00140449438202,-44.50138888888889,-45.00137362637363,-45.50135869565217,-46.001344086021504,-46.50132978723404,-47.001315789473686,-47.501302083333336
};

__device__ DD add(DD a,DD b) {
  double s=a.hi+b.hi,v=s-a.hi;
  double e=(a.hi-(s-v))+(b.hi-v)+a.lo+b.lo;
  double h=s+e;
  return {h,e-(h-s)};
}
__device__ DD mul(DD a,DD b) {
  double p=a.hi*b.hi;
  double e=__fma_rn(a.hi,b.hi,-p)+a.hi*b.lo+a.lo*b.hi+a.lo*b.lo;
  double h=p+e;
  return {h,e-(h-p)};
}
struct DC { DD re,im; };
__device__ DC add(DC a,DC b) { return {add(a.re,b.re),add(a.im,b.im)}; }
__device__ DC mul(DC a,DC b) {
  DD v=mul(a.im,b.im); v.hi=-v.hi;v.lo=-v.lo;
  return {add(mul(a.re,b.re),v),add(mul(a.re,b.im),mul(a.im,b.re))};
}
__device__ DC scale(DC a,double x) { return {mul(a.re,{x,0}),mul(a.im,{x,0})}; }
__device__ C jzero(C z) {
  // Real-axis nodes use the fp64 libdevice Bessel implementation.
  if(z.imag()==0 && fabs(z.real())<=16) return C(::j0(z.real()));
  double radius=abs(z);
  if(radius<=32) {
    // Double-double series prevents cancellation near the crossover at |z|=32.
    // Needed domain: Re(z)>=0, |Im(z)|<=8. The first omitted term is <1e-32
    // of the absolute series sum; rounding is dominated by final fp64 storage.
    DC zz{{z.real(),0},{z.imag(),0}};
    DC x=scale(mul(zz,zz),-0.25), term{{1,0},{0,0}}, sum=term;
    for(int k=1;k<=128;++k) {
      // Division by k^2 needs a compensated reciprocal in the DD recurrence.
      DD reciprocal=reciprocals[k-1];
      term=mul(term,x);
      term={mul(term.re,reciprocal),mul(term.im,reciprocal)};
      sum=add(sum,term);
      if(k>radius && hypot(term.re.hi,term.im.hi)<1e-34) break;
    }
    return C(sum.re.hi+sum.re.lo,sum.im.hi+sum.im.lo);
  }
  // DLMF 10.17.3: optimally truncate the Hankel P/Q expansions. At |z|=32
  // the least term is O(exp(-64)); complex cosine/sine grow by at most exp(8).
  C p(1),q(0),term(1),inverse=C(1)/z; double previous=1;
  for(int k=1;k<=96;++k) {
    term*=asymptotic[k-1]*inverse;
    double magnitude=abs(term);
    if(magnitude>previous) break;
    int index=k/2;
    C signed_term=(index%2) ? -term : term;
    if(k%2) q+=signed_term; else p+=signed_term;
    previous=magnitude;
    if(magnitude<1e-18) break;
  }
  // Rotate sin(z),cos(z) algebraically; subtracting pi/4 from a large real
  // argument would needlessly lose phase accuracy near Bessel zeros.
  C e=ex(C(0,1)*z),em=ex(C(0,-1)*z);
  C cz=(e+em)*0.5,sz=(e-em)*C(0,-0.5);
  C cosine=(cz+sz)*0.707106781186547524400844362104849039;
  C sine=(sz-cz)*0.707106781186547524400844362104849039;
  C sq=root(0.636619772367581343075535053490057448/z); if(sq.real()<0) sq=-sq;
  return sq*(cosine*p-sine*q);
}
__device__ C hankel(C z,int kind) {
  // DLMF 10.17.5-6: sqrt(2/(pi*z))*exp(s*i*(z-pi/4))*sum a_k*(s*i/z)^k.
  // |z|>=32 on the vertical tails. Evaluate only the decaying exponential;
  // constructing J0 or the other Hankel branch here would overflow.
  if(kind*z.imag()>745) return C(0);
  C term(1),sum(1),inverse=C(1)/z;double previous=1;
  for(int k=1;k<=96;++k) {
    term*=C(0,kind)*asymptotic[k-1]*inverse;
    double magnitude=abs(term);
    if(magnitude>previous) break;
    sum+=term;previous=magnitude;
    if(magnitude<1e-18) break;
  }
  C sq=root(0.636619772367581343075535053490057448/z);if(sq.real()<0) sq=-sq;
  C rotation(0.7071067811865475244,-kind*0.7071067811865475244);
  return sq*ex(C(0,kind)*z)*rotation*sum;
}
__device__ C jone(C z) {
  // Real-axis nodes use the fp64 libdevice Bessel implementation.
  if(z.imag()==0 && fabs(z.real())<=16) return C(::j1(z.real()));
  double radius=abs(z);
  if(radius<=32) {
    // Double-double series prevents cancellation near the crossover at |z|=32.
    // Needed domain: Re(z)>=0, |Im(z)|<=8. The first omitted term is <1e-32
    // of the absolute series sum; rounding is dominated by final fp64 storage.
    DC zz{{z.real(),0},{z.imag(),0}};
    DC x=scale(mul(zz,zz),-0.25), term=scale(zz,0.5), sum=term;
    for(int k=1;k<=128;++k) {
      // Division by k^2 needs a compensated reciprocal in the DD recurrence.
      double divisor=double(k)*(k+1), inverse=1/divisor;
      DD reciprocal{inverse,__fma_rn(-inverse,divisor,1.0)/divisor};
      term=mul(term,x);
      term={mul(term.re,reciprocal),mul(term.im,reciprocal)};
      sum=add(sum,term);
      if(k>radius && hypot(term.re.hi,term.im.hi)<1e-34) break;
    }
    return C(sum.re.hi+sum.re.lo,sum.im.hi+sum.im.lo);
  }
  // DLMF 10.17.3: optimally truncate the Hankel P/Q expansions. At |z|=32
  // the least term is O(exp(-64)); complex cosine/sine grow by at most exp(8).
  C p(1),q(0),term(1),inverse=C(1)/z; double previous=1;
  for(int k=1;k<=96;++k) {
    term*=((4.0-(2.0*k-1)*(2.0*k-1))/(8.0*k))*inverse;
    double magnitude=abs(term);
    if(magnitude>previous) break;
    int index=k/2;
    C signed_term=(index%2) ? -term : term;
    if(k%2) q+=signed_term; else p+=signed_term;
    previous=magnitude;
    if(magnitude<1e-18) break;
  }
  // Rotate sin(z),cos(z) algebraically; subtracting pi/4 from a large real
  // argument would needlessly lose phase accuracy near Bessel zeros.
  C e=ex(C(0,1)*z),em=ex(C(0,-1)*z);
  C cz=(e+em)*0.5,sz=(e-em)*C(0,-0.5);
  C cosine=(sz-cz)*0.707106781186547524400844362104849039;
  C sine=-(sz+cz)*0.707106781186547524400844362104849039;
  C sq=root(0.636619772367581343075535053490057448/z); if(sq.real()<0) sq=-sq;
  return sq*(cosine*p-sine*q);
}
__device__ C hankel_one(C z,int kind) {
  // DLMF 10.17.5-6: sqrt(2/(pi*z))*exp(s*i*(z-pi/4))*sum a_k*(s*i/z)^k.
  // |z|>=32 on the vertical tails. Evaluate only the decaying exponential;
  // constructing J0 or the other Hankel branch here would overflow.
  if(kind*z.imag()>745) return C(0);
  C term(1),sum(1),inverse=C(1)/z;double previous=1;
  for(int k=1;k<=96;++k) {
    term*=C(0,kind)*((4.0-(2.0*k-1)*(2.0*k-1))/(8.0*k))*inverse;
    double magnitude=abs(term);
    if(magnitude>previous) break;
    sum+=term;previous=magnitude;
    if(magnitude<1e-18) break;
  }
  C sq=root(0.636619772367581343075535053490057448/z);if(sq.real()<0) sq=-sq;
  C rotation(-0.7071067811865475244,-kind*0.7071067811865475244);
  return sq*ex(C(0,kind)*z)*rotation*sum;
}
}

// K2: each output writes 16 B, reads q,w (32 B), kind (4 B) and rho (8 B).
// Complex small arguments: ~130 fp64 FLOP per DD iteration, no runtime divides
// for series coefficients. Real |z|<=16 uses libdevice J0. Large arguments:
// ~20*40 FLOP typical, plus hypot per iteration, 1/2 complex exp and one root;
// two complex divides per large argument, hoisted out of the recurrence.
__global__ void bessel_kernel(const C* q,const C* w,const int* kinds,const double* rho,C* out,
                              int F,int Q,int R,int begin,bool midpoint) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t>=int64_t(F)*R*Q) return;
  int j=t%Q,r=(t/Q)%R,f=t/(int64_t(Q)*R);
  // A frequency can have a zero-width log/linear section in the batched rule.
  if(w[f*Q+j]==C(0)) {out[t]=C(0);return;}
  double distance=midpoint ? (rho[begin+r]+rho[begin+r+1])*0.5 : rho[begin+r];
  C z=q[f*Q+j]*distance;
  int kind=kinds[f*Q+j];
  C value=kind ? hankel(z,kind) : jzero(z);
  out[t]=w[f*Q+j]*value*q[f*Q+j]/(2*pi);
}
at::Tensor layered_bessel(at::Tensor q,at::Tensor w,at::Tensor kinds,at::Tensor rho,
                          int64_t begin,int64_t end,bool midpoint) {
  c10::cuda::CUDAGuard guard(q.device());
  auto out=at::empty({q.size(0),end-begin,q.size(1)},q.options());
  int64_t count=out.numel();
  bessel_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<C>(),w.data_ptr<C>(),kinds.data_ptr<int>(),rho.data_ptr<double>(),out.data_ptr<C>(),
      q.size(0),q.size(1),end-begin,begin,midpoint);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
__global__ void j0_kernel(const C* z,C* out,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) out[i]=jzero(z[i]);
}
at::Tensor layered_j0(at::Tensor z) {
  c10::cuda::CUDAGuard guard(z.device());
  auto out=at::empty_like(z);int n=z.numel();
  j0_kernel<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(z.data_ptr<C>(),out.data_ptr<C>(),n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

__global__ void bessel_one_kernel(const C* q,const C* w,const int* kinds,const double* rho,C* out,
                              int F,int Q,int R,int begin,bool midpoint) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t>=int64_t(F)*R*Q) return;
  int j=t%Q,r=(t/Q)%R,f=t/(int64_t(Q)*R);
  // A frequency can have a zero-width log/linear section in the batched rule.
  if(w[f*Q+j]==C(0)) {out[t]=C(0);return;}
  double distance=midpoint ? (rho[begin+r]+rho[begin+r+1])*0.5 : rho[begin+r];
  C z=q[f*Q+j]*distance;
  int kind=kinds[f*Q+j];
  C value=kind ? hankel_one(z,kind) : jone(z);
  out[t]=w[f*Q+j]*value*q[f*Q+j]/(2*pi);
}
at::Tensor layered_bessel_one(at::Tensor q,at::Tensor w,at::Tensor kinds,at::Tensor rho,
                          int64_t begin,int64_t end,bool midpoint) {
  c10::cuda::CUDAGuard guard(q.device());
  auto out=at::empty({q.size(0),end-begin,q.size(1)},q.options());
  int64_t count=out.numel();
  bessel_one_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<C>(),w.data_ptr<C>(),kinds.data_ptr<int>(),rho.data_ptr<double>(),out.data_ptr<C>(),
      q.size(0),q.size(1),end-begin,begin,midpoint);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
