"""Analytic covariance and independent discrete Green/CN energy identities.

All states are prescribed algebraic arrays, not CPU FDTD propagation.
"""
import copy
import numpy as np
import pytest
import torch
from scripts.dev.fdtd.printed_line_product_flux import RULES,COORDS,adjacent,native_product_flux


def payload(axis,pair,direction,kind):
    nodes=[np.array([-.7,-.2,.3,1.]),np.array([-1.,-.4,.1,.6,1.4])]
    centers=[(a[1:]+a[:-1])/2 for a in nodes]
    raw=dict(axis=axis,position=0.,components={},**dict(zip(COORDS[axis],centers)))
    selected=dict(axis=axis,position=0.,normal_direction=direction,frequency=1.,frequencies=(1.,),
        cell_widths=dict(zip(COORDS[axis],map(np.diff,nodes))),**dict(zip(COORDS[axis],centers)))
    for number,(e,h,averaging,sign) in enumerate(RULES[axis]):
        coordinates=centers.copy();coordinates[averaging]=nodes[averaging]
        t=np.broadcast_to(coordinates[averaging].reshape((-1,1) if averaging==0 else (1,-1)),tuple(map(len,coordinates)))
        a,b=(2+1j,0j if kind=='constant' else 3-1j)
        c,d=(-1+2j,2+.5j if kind=='correlated' else 0j)
        E=a+b*t;H=c+d*t
        if number!=pair:E=E*0;H=H*0
        for name,value in ((e,E),(h,H)):
            raw['components'][name]=dict(data=torch.tensor(value),coords=coordinates,collocated=False,plane_indices=(0,),plane_weights=(1.,))
            selected[name]=adjacent(torch.tensor(value),averaging)
    return raw,selected,nodes


@pytest.mark.parametrize('axis',list(RULES))
@pytest.mark.parametrize('pair',[0,1])
@pytest.mark.parametrize('direction',['+','-'])
@pytest.mark.parametrize('kind',['constant','one_affine','correlated'])
def test_product_first_has_exact_covariance_and_declared_affine_quadrature_error(axis,pair,direction,kind):
    raw,selected,nodes=payload(axis,pair,direction,kind);actual=native_product_flux(raw,selected)
    _,_,averaging,sign=RULES[axis][pair];normal=1 if direction=='+' else -1
    a,b=2+1j,0j if kind=='constant' else 3-1j;c,d=-1+2j,2+.5j if kind=='correlated' else 0j
    t=nodes[averaging];length=nodes[1-averaging][-1]-nodes[1-averaging][0]
    covariance=.5*normal*sign*np.real(b*np.conj(d))*length*np.sum(np.diff(t)**3)/4
    assert actual['product_first']-actual['old']==pytest.approx(covariance,rel=1e-13,abs=3e-15)
    primitive=lambda x:a*np.conj(c)*x+(a*np.conj(d)+b*np.conj(c))*x*x/2+b*np.conj(d)*x**3/3
    exact=.5*normal*sign*length*np.real(primitive(t[-1])-primitive(t[0]))
    trap_error=.5*normal*sign*np.real(b*np.conj(d))*length*np.sum(np.diff(t)**3)/6
    assert actual['product_first']==pytest.approx(exact+trap_error,rel=1e-13,abs=3e-15)
    # Both affine fields produce a quadratic product; trapezoidal integration
    # has its analytic error. One affine field times a constant is exact.
    if kind=='correlated':assert abs(actual['product_first']-exact)>1e-3


def test_collocated_fallback_preserves_existing_functional_and_missing_native_rejects():
    raw,selected,_=payload('x',0,'+','one_affine')
    for entry in raw['components'].values():entry['collocated']=True
    actual=native_product_flux(raw,selected);assert actual['old']==actual['product_first'] and actual['fallback']=='already_collocated'
    raw,selected,_=payload('x',0,'+','one_affine');del raw['components']['Hz']
    with pytest.raises(ValueError,match='native tangential'):native_product_flux(raw,selected)


def test_piecewise_copper_Et_uses_each_actual_side_and_rejects_unknown_H_continuity():
    y=np.array([0.,1.]);z=np.array([-1.,0.,1.]);shape=(1,2)
    raw=dict(axis='x',position=0.,y=np.array([.5]),z=np.array([-.5,.5]),components={})
    for name,values,coords in [('Ey',[[1.,1.,3.]],[np.array([.5]),z]),('Hz',[[2.,2.,2.]],[np.array([.5]),z]),
                              ('Ez',np.zeros((2,2)),[y,np.array([-.5,.5])]),('Hy',np.zeros((2,2)),[y,np.array([-.5,.5])])]:
        raw['components'][name]=dict(data=torch.tensor(values,dtype=torch.complex128),coords=coords,collocated=False,plane_indices=(0,),plane_weights=(1.,))
    selected=dict(axis='x',normal_direction='+',y=raw['y'],z=raw['z'],cell_widths={'y':np.array([1.]),'z':np.array([1.,1.])})
    trace=dict(component='Ey',normal_axis='z',normal_coordinate=0.,indices=torch.tensor([[0,0,1]]),electric_fields=torch.tensor([[1.],[3.]],dtype=torch.complex128))
    axes={'Ey':(np.array([0.]),np.array([.5]),z)}
    with pytest.raises(ValueError,match='normal-H'):native_product_flux(raw,selected,side_traces=[trace],native_axes=axes,require_sided=True)
    actual=native_product_flux(raw,selected,side_traces=[trace],native_axes=axes,require_sided=True,single_normal_H_DOF_contract=True)
    assert actual['old']==3. and actual['product_first']==4. and actual['side_trace_correction']==1.
    assert actual['side_trace_edge_count']==1


@pytest.mark.parametrize('graded',[False,True])
def test_two_dimensional_Yee_Green_and_CN_modified_energy_balance(graded):
    rng=np.random.default_rng(637)
    x=np.r_[0.,np.cumsum([.2,.3,.17,.4,.23,.31])] if graded else np.arange(7)*.25
    z=np.r_[0.,np.cumsum([.3,.2,.31,.13,.4,.27,.19])] if graded else np.arange(8)*.2
    dx,dz=np.diff(x),np.diff(z);dual=lambda d:np.r_[d[0]/2,(d[1:]+d[:-1])/2,d[-1]/2]
    wx,wz=dual(dx),dual(dz);L,R,K,T=1,5,1,6
    ex=wx[L:R+1].copy();ex[0]=dx[L]/2;ex[-1]=dx[R-1]/2
    ez=wz[K:T+1].copy();ez[0]=dz[K]/2;ez[-1]=dz[T-1]/2
    WE=ex[:,None]*ez[None,:];WHZ=dx[L:R,None]*ez[None,:];WHX=ex[:,None]*dz[None,K:T]
    E0=rng.normal(size=(len(x),len(z)));E1=rng.normal(size=E0.shape);bar=(E0+E1)/2
    HZ=rng.normal(size=(len(x)-1,len(z)));HX=rng.normal(size=(len(x),len(z)-1))
    curlE=lambda E:(-np.diff(E,axis=1)/dz[None,:],np.diff(E,axis=0)/dx[:,None])
    div=(HZ[L:R+1,K:T+1]-HZ[L-1:R,K:T+1])/wx[L:R+1,None]-(HX[L:R+1,K:T+1]-HX[L:R+1,K-1:T])/wz[None,K:T+1]
    def interp(H,dimension,index,primal):
        if dimension==0:return (primal[index]*H[index-1]+primal[index-1]*H[index])/(primal[index]+primal[index-1])
        return (primal[index]*H[:,index-1]+primal[index-1]*H[:,index])/(primal[index]+primal[index-1])
    def product_integral(a,b,width):return np.sum(.5*(a[:-1]*b[:-1]+a[1:]*b[1:])*width)
    flux=product_integral(bar[R,K:T+1],interp(HZ,0,R,dx)[K:T+1],dz[K:T])-product_integral(bar[L,K:T+1],interp(HZ,0,L,dx)[K:T+1],dz[K:T])
    flux-=product_integral(bar[L:R+1,T],interp(HX,1,T,dz)[L:R+1],dx[L:R])-product_integral(bar[L:R+1,K],interp(HX,1,K,dz)[L:R+1],dx[L:R])
    CX,CZ=curlE(bar)
    spatial=np.sum(WE*bar[L:R+1,K:T+1]*div)+np.sum(WHX*CX[L:R+1,K:T]*HX[L:R+1,K:T])+np.sum(WHZ*CZ[L:R,K:T+1]*HZ[L:R,K:T+1])
    assert spatial==pytest.approx(flux,rel=5e-14,abs=2e-14)
    epsilon,mu,sigma,dt=2.,3.,.4,.001
    CX0,CZ0=curlE(E0);CX1,CZ1=curlE(E1)
    prevX,prevZ=HX+dt/mu*CX0,HZ+dt/mu*CZ0
    afterX,afterZ=HX-dt/mu*CX1,HZ-dt/mu*CZ1
    energy=lambda E,ax,az,bx,bz:.5*epsilon*np.sum(WE*E[L:R+1,K:T+1]**2)+.5*mu*(np.sum(WHX*ax[L:R+1,K:T]*bx[L:R+1,K:T])+np.sum(WHZ*az[L:R,K:T+1]*bz[L:R,K:T+1]))
    delta=energy(E1,HX,HZ,afterX,afterZ)-energy(E0,prevX,prevZ,HX,HZ)
    J=epsilon*(E1[L:R+1,K:T+1]-E0[L:R+1,K:T+1])/dt+div+sigma*bar[L:R+1,K:T+1]
    heat=dt*sigma*np.sum(WE*bar[L:R+1,K:T+1]**2);work=dt*np.sum(WE*bar[L:R+1,K:T+1]*J)
    assert delta+heat+dt*flux-work==pytest.approx(0.,abs=3e-14*(abs(delta)+abs(work)+heat+abs(dt*flux)))


def test_CN_harmonic_midpoint_heat_has_explicit_cosine_factor():
    n=np.arange(128);angle=2*np.pi/128;amplitude=2+1j;sigma=.7
    E=np.real(amplitude*np.exp(-1j*angle*n));nextE=np.real(amplitude*np.exp(-1j*angle*(n+1)))
    midpoint=(E+nextE)/2
    assert sigma*np.mean(midpoint**2)==pytest.approx(.5*sigma*abs(amplitude)**2*np.cos(angle/2)**2,rel=3e-15)
    assert sigma*np.mean(midpoint[:-1]**2)!=pytest.approx(.5*sigma*abs(amplitude)**2*np.cos(angle/2)**2,rel=1e-5)
