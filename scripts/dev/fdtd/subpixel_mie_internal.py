"""Analytic Mie internal-field enhancement (1/V) int |Ex|^2 dV over r < rmax for an x-polarized unit plane wave (autogrid thesis reference)."""
import math, numpy as np
from scipy.special import spherical_jn, spherical_yn

def internal_enhancement(radius, wavelength, eps, nr=48, nt=48, nphi=32, rmax=None):
    rmax = radius if rmax is None else rmax
    k = 2*math.pi/wavelength; x = k*radius; m = np.sqrt(complex(eps))
    nmax = int(abs(m*x) + 4*abs(m*x)**(1/3) + 15)
    n = np.arange(1, nmax+1)
    jx, djx = spherical_jn(n, x), spherical_jn(n, x, derivative=True)
    hx = jx + 1j*spherical_yn(n, x); dhx = djx + 1j*spherical_yn(n, x, derivative=True)
    xj_d = jx + x*djx; xh_d = hx + x*dhx
    jm, djm = spherical_jn(n, m*x), spherical_jn(n, m*x, derivative=True)
    mxj_d = jm + m*x*djm
    c = (jx*xh_d - hx*xj_d) / (jm*xh_d - hx*mxj_d)
    d = (m*jx*xh_d - m*hx*xj_d) / (m*m*jm*xh_d - hx*mxj_d)
    En = (1j**n)*(2*n+1)/(n*(n+1))
    ur, wr = np.polynomial.legendre.leggauss(nr); r = 0.5*rmax*(ur+1); wr = 0.5*rmax*wr
    ut, wt = np.polynomial.legendre.leggauss(nt)  # mu = cos(theta)
    phi = np.arange(nphi)*2*math.pi/nphi; wp = 2*math.pi/nphi
    R, MU, PH = np.meshgrid(r, ut, phi, indexing="ij")
    W = wr[:,None,None]*r[:,None,None]**2 * wt[None,:,None] * wp
    sin_t = np.sqrt(1-MU**2); cp, sp = np.cos(PH), np.sin(PH)
    rho = m*k*R
    Er = np.zeros(R.shape, complex); Et = np.zeros_like(Er); Ep = np.zeros_like(Er)
    pi_prev, pi_cur = np.zeros_like(MU), np.ones_like(MU)
    for i, nn in enumerate(n):
        tau = nn*MU*pi_cur - (nn+1)*pi_prev
        z = spherical_jn(nn, rho); dz = spherical_jn(nn, rho, derivative=True)
        zr = z/rho; dr = zr + dz  # [rho z]'/rho
        # M_o1n and N_e1n
        Mt = cp*pi_cur*z; Mp = -sp*tau*z
        Nr = cp*nn*(nn+1)*sin_t*pi_cur*zr; Nt = cp*tau*dr; Np = -sp*pi_cur*dr
        Er += En[i]*(-1j*d[i]*Nr); Et += En[i]*(c[i]*Mt - 1j*d[i]*Nt); Ep += En[i]*(c[i]*Mp - 1j*d[i]*Np)
        pi_prev, pi_cur = pi_cur, ((2*nn+1)*MU*pi_cur - (nn+1)*pi_prev)/nn
    Ex = Er*sin_t*cp + Et*MU*cp - Ep*sp
    V = 4/3*math.pi*rmax**3
    return float(np.sum(np.abs(Ex)**2*W)/V)

if __name__ == "__main__":
    print("check m=1:", internal_enhancement(0.2, 0.6, 1.0+1e-12))
    eps0 = 8.8541878128e-12; f = 299792458.0/0.6
    eps = 12.0 + 1j*0.1/(2*math.pi*f*eps0)
    for q in (32, 48, 64):
        print("eps", eps, "quad", q, internal_enhancement(0.2, 0.6, eps, nr=q, nt=q, nphi=q))
