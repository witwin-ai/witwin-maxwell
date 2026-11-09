# PMCHWT FFT channel composition and adjoint

Internal wrapper only, not public backend integration or physics qualification.

Original scalar_moment_action gives A=[C,e B;m B,C] on [J,M],
e=i omega epsilon, m=-i omega mu,
B=diag(G,G,G,-G/k^2),
C(v)=[D_y v_z-D_z v_y,D_z v_x-D_x v_z,D_x v_y-D_y v_x,0].

Forward wraps independent scalar Green/gradient FFT translations of all vector
and divergence input channels and then assembles exactly these material/curl
coefficients. Fourier outputs are Green, Dx, Dy, Dz.

Hermitian adjoint is A^H=[C^H,conj(m)B^H;conj(e)B^H,C^H],
B^H charge factor -1/conj(k)^2; vector factor1.
C^H input channel x receives Dz^H dual_y-Dy^H dual_z;
input y receives Dx^H dual_z-Dz^H dual_x;
input z receives Dy^H dual_x-Dx^H dual_y; charge receives0.
Populate one four-field scalar-adjoint input per J/M vector/divergence channel,
then apply the scalar FFT adjoint. No hidden transposition/conjugation of medium
for forward, and no floating certification is implied.

Validate against independently existing dense scalar_moment_action including
complex electric/magnetic losses and non-unit mu; adjoint oracle must also come
from the existing dense translation or assembled all-column weak matrix.
Validate actual RWG upward/downward projection and original selected weak
quadrature at the original1e-8 balanced norm/1e-11dot/currentgrad gates.
No all-system/global near/resource/scaling/highfreq/periodic/public claim.
