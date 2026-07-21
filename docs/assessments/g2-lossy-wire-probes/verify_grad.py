import numpy as np
from scipy import special

def Zp(a, sigma, mu, f):
    omega = 2*np.pi*f
    m = (1j*omega*mu*sigma)**0.5
    ma = m*a
    R = special.ive(0,ma)/special.ive(1,ma)
    return m/(2*np.pi*a*sigma)*R

def dZp_dsigma_closed(a, sigma, mu, f):
    omega = 2*np.pi*f
    m = (1j*omega*mu*sigma)**0.5
    ma = m*a
    R = special.ive(0,ma)/special.ive(1,ma)
    return (1j*omega*mu)/(4*np.pi*sigma)*(1.0 - R*R)

mu = 4*np.pi*1e-7
for a in (1e-3, 2e-3):
    for sigma in (5.8e7, 1e6, 3.5e7):
        for f in (1e3, 1e6, 1e8, 5e9, 2e10):
            h = sigma*1e-7
            fd = (Zp(a,sigma+h,mu,f)-Zp(a,sigma-h,mu,f))/(2*h)
            cl = dZp_dsigma_closed(a,sigma,mu,f)
            rel = abs(fd-cl)/abs(cl)
            assert rel < 1e-6, (a,sigma,f,rel,fd,cl)
    # DC limit of real part vs dRdc/dsigma
    sigma=5.8e7
    Rdc_grad = -1.0/(np.pi*a*a*sigma*sigma)
    cl_lowf = dZp_dsigma_closed(a,sigma,mu,1.0).real
    print(f"a={a} DC: dRdc/dsig={Rdc_grad:.6e} closed.real(1Hz)={cl_lowf:.6e} rel={abs(cl_lowf-Rdc_grad)/abs(Rdc_grad):.2e}")
print("ALL CLOSED-FORM GRAD CHECKS PASSED")
