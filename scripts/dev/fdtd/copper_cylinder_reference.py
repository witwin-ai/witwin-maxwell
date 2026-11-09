"""Independent full-Maxwell annular copper two-port in cylindrical harmonics.

Validation only. Unlike a regular-core scattering problem, the two vacuum ports
have independently incident/outgoing Hankel waves. The origin is outside the
annular computational domain, so no singular polar-axis approximation is made.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import hankel1,hankel2,h1vp,h2vp,hankel1e,hankel2e

EPS_0=8.8541878128e-12
MU_0=1.25663706212e-6
C_0=299792458.
ETA_0=376.730313668


def _scaled_cylinder(order,z,anchor,outgoing):
    function=hankel1e if outgoing else hankel2e
    sign=1 if outgoing else -1
    scale=np.exp(sign*1j*(z-anchor))
    return function(order,z)*scale,(function(order-1,z)-function(order+1,z))*scale/2


def shell_scattering(radius,thickness,frequency,conductivity,*,order=0,polarization="tm"):
    """Unit-flux-equivalent vacuum Hankel port S matrix at both physical radii.

    TM means scalar Ez; TE means scalar Hz. Port order is inner then outer.
    Positive propagation at the inner port is increasing physical radius.
    """
    if not radius>0 or not 0<thickness<2*radius or frequency<=0 or conductivity<0 or order<0 or int(order)!=order or polarization not in ("te","tm"):
        raise ValueError("Invalid finite cylindrical-shell reference parameters.")
    omega=2*math.pi*frequency
    k=omega/C_0
    epsilon=1+1j*conductivity/(omega*EPS_0)
    kc=k*np.sqrt(epsilon)
    ri,ro=radius-thickness/2,radius+thickness/2
    hi1,hi2,ho1,ho2=hankel1(order,k*ri),hankel2(order,k*ri),hankel1(order,k*ro),hankel2(order,k*ro)
    di1,di2,do1,do2=h1vp(order,k*ri),h2vp(order,k*ri),h1vp(order,k*ro),h2vp(order,k*ro)
    ci1,qi1=_scaled_cylinder(order,kc*ri,kc*ri,True)
    ci2,qi2=_scaled_cylinder(order,kc*ri,kc*ro,False)
    co1,qo1=_scaled_cylinder(order,kc*ro,kc*ri,True)
    co2,qo2=_scaled_cylinder(order,kc*ro,kc*ro,False)
    flux=kc/k/(epsilon if polarization=="te" else 1)
    matrix=np.array(((hi2,0,-ci1,-ci2),(di2,0,-flux*qi1,-flux*qi2),
                     (0,ho1,-co1,-co2),(0,do1,-flux*qo1,-flux*qo2)),dtype=complex)
    rhs=-np.array(((hi1,0),(di1,0),(0,ho2),(0,do2)),dtype=complex)
    column=abs(matrix).max(0)
    balanced=matrix/column
    row=abs(balanced).max(1)
    equilibrated=balanced/row[:,None]
    solution=np.linalg.solve(equilibrated,rhs/row[:,None])/column[:,None]
    for _ in range(2):
        residual=rhs.astype(np.clongdouble)-matrix.astype(np.clongdouble)@solution.astype(np.clongdouble)
        solution+=np.linalg.solve(equilibrated,np.asarray(residual/row[:,None],dtype=complex))/column[:,None]
    return solution[:2]


def collapsed_shell_scattering(radius,thickness,frequency,conductivity,*,order=0,polarization="tm"):
    """Independent physical cylindrical trace solve using exact slab two-port."""
    omega=2*math.pi*frequency
    k=omega/C_0
    gamma=np.sqrt(-1j*omega*MU_0*conductivity)
    eta=np.sqrt(-1j*omega*MU_0/conductivity)
    argument=gamma*thickness
    admittance=np.array(((1/np.tanh(argument),-1/np.sinh(argument)),
                         (-1/np.sinh(argument),1/np.tanh(argument))),dtype=complex)/eta
    f1,f2=hankel1(order,k*radius),hankel2(order,k*radius)
    d1,d2=h1vp(order,k*radius),h2vp(order,k*radius)
    if polarization=="tm":
        unknown_e=np.diag((f2,f1));unknown_i=np.diag((-1j*d2/ETA_0,1j*d1/ETA_0))
        incident_e=np.diag((f1,f2));incident_i=np.diag((-1j*d1/ETA_0,1j*d2/ETA_0))
    elif polarization=="te":
        unknown_e=np.diag((-1j*ETA_0*d2,-1j*ETA_0*d1));unknown_i=np.diag((f2,-f1))
        incident_e=np.diag((-1j*ETA_0*d1,-1j*ETA_0*d2));incident_i=np.diag((f1,-f2))
    else:raise ValueError("Cylinder polarization must be te or tm.")
    return np.linalg.solve(unknown_i-admittance@unknown_e,admittance@incident_e-incident_i)
