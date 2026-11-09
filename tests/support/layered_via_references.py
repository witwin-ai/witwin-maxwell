"""Independent magnetostatic partial-inductance reference for a vertical loop.

A strip of width w has logarithmic capacity w/4 (the Joukowski map of a
circle onto an interval). For a square of side a, the exterior
Schwarz-Christoffel map has derivative C(1-z**-4)**(1/2). Integrating one
quarter of its boundary gives a/C = 4*pi**(3/2)/Gamma(1/4)**2, hence the
equivalent round-wire radius is C = 0.5901702995*a. This is a surface-current
radius, appropriate to a PEC four-wall tube; no internal inductance is added.

Integrating the Neumann kernel along two parallel segments of length l and
separation d gives mu0/(2*pi) * [l*asinh(l/d)-sqrt(l*l+d*d)+d]. Substitute
the capacity radius for each self term, subtract the opposite-side mutual
terms, and note that perpendicular straight segments have zero dot product.
This is the rectangular-loop partial-inductance (Grover) approximation;
finite bends and pad spreading are omitted and controlled by thin aspect ratios.
"""

import math

MU0 = 4e-7 * math.pi


def square_via_radius(side):
    return side * math.gamma(.25)**2 / (4 * math.pi**1.5)


def rectangular_loop_inductance(length, height, strip_width, via_side):
    def partial(l, d):
        return MU0 / (2*math.pi) * (l*math.asinh(l/d)-math.hypot(l,d)+d)
    return 2*(partial(length, strip_width/4) + partial(height, square_via_radius(via_side))
              - partial(length, height) - partial(height, length))


def static_cross_triangle(triangle, observer, beta):
    """Independent signed-fan Duffy integration, without boundary formulas.

    Split about the observer projection and integrate each radial ray. Coplanar
    rays are polynomial; horizontal off-plane radial moments are elementary.
    Vertical off-plane rays use 128-point Gauss integration after the substitution
    t=(h/l)*sinh(v*asinh(l/h)), which resolves the near-axis scale h/l.
    Only the remaining angular coordinate uses adaptive quadrature.
    """
    import numpy as np
    from numpy.polynomial.legendre import leggauss
    from scipy.integrate import quad_vec
    triangle,observer=np.asarray(triangle),np.asarray(observer)
    normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
    normal/=np.linalg.norm(normal)
    signed_height=np.dot(observer-triangle[0],normal)
    projection=observer-signed_height*normal
    height=abs(signed_height)
    horizontal=abs(normal[2])==1
    nodes,weights=leggauss(128)
    nodes,weights=(nodes+1)/2,weights/2
    out=np.zeros(3)
    for i in range(3):
        a,b=triangle[i]-projection,triangle[(i+1)%3]-projection
        jacobian=np.dot(np.cross(a,b),normal)
        if jacobian==0:
            continue
        def angular(u):
            ray=(1-u)*a+u*b
            length=np.linalg.norm(ray)
            C=((projection[None,:2]-triangle[:,:2])*ray[None,:2]).sum(-1)
            W=np.dot(ray[:2],ray[:2])
            if height==0:
                return jacobian*(-C-.5*W)/(length*(length-beta*ray[2]))
            if horizontal:
                x=length/height
                second=((1/6-3*x*x/40+5*x**4/112-35*x**6/1152)/height**2 if x<.01 else
                        (1-np.arcsinh(x)/x)/length**2)
                third=.5/(np.hypot(height,length)+height)**2
            else:
                U=np.arcsinh(length/height)
                t=height/length*np.sinh(U*nodes)
                R=height*np.cosh(U*nodes)
                denominator=R-beta*t*ray[2]
                second=np.sum(weights*t*t/denominator)*U/length
                third=np.sum(weights*t*t*t/denominator)*U/length
            return jacobian*(-C*second-W*third)
        out+=quad_vec(angular,0.,1.,epsabs=2e-12,epsrel=2e-12)[0]
    return out
