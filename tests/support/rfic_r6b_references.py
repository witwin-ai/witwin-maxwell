"""Isolated oxide-coupled silicon admittance and an independent electrostatic oracle."""
import numpy as np
import torch
import witwin.maxwell as mw

UM=1e-6
FREQUENCIES=(1e9,3.4e9)

def plate_fixture(maximum=20., edge=5., device="cuda"):
    stack=mw.LayerStack((mw.Layer("silicon",300*UM,11.9,10.),
                         mw.Layer("oxide",6*UM,4.1)),mw.HalfSpace(11.9,10.),mw.HalfSpace())
    scene=mw.Scene(background=stack,device=device)
    outer=torch.tensor(((-70,-70),(70,-70),(70,70),(-70,70)),dtype=torch.float64)*UM
    inner=torch.tensor(((-60,-60),(60,-60),(60,60),(-60,60)),dtype=torch.float64)*UM
    square=torch.tensor(((-50,-50),(50,-50),(50,50),(-50,50)),dtype=torch.float64)*UM
    for vertices,holes in ((square,[]),(outer,[inner])):
        scene.add_structure(mw.Structure(mw.PlanarPEC(vertices,holes=holes,normal_axis="z",
                                coordinate=305*UM,physical_thickness=2*UM),mw.Material.pec()))
    scene.add_port(mw.LumpedPort("plate",positive=(50*UM,0.,305*UM),negative=(60*UM,0.,305*UM),
                    voltage_path=mw.AxisPath("x"),current_surface=mw.Box(
                        position=(55*UM,0.,305*UM),size=(10*UM,10*UM,2*UM)),reference_impedance=50.))
    config=mw.MoMConfig(layered=mw.LayeredSurfaceConfig(bridge_width_m=10*UM,
              max_edge_length_m=maximum*UM,edge_cell_m=edge*UM,growth_ratio=1.5))
    return scene,config

def pulse_geometry(pitch, gap):
    # A rectangular pulse grid represents the metal union, including both port arms.
    edges=np.arange(-70,70+pitch/2,pitch)
    cells,owners=[],[]
    for x0,x1 in zip(edges[:-1],edges[1:]):
        for y0,y1 in zip(edges[:-1],edges[1:]):
            x,y=(x0+x1)/2,(y0+y1)/2
            signal=abs(x)<50 and abs(y)<50
            ring=max(abs(x),abs(y))>60
            arm=50<x<60 and abs(y)<5
            if signal or ring or (arm and abs(x-55)>gap/2):
                cells.append((x0,x1,y0,y1))
                owners.append(0 if signal or (arm and x<55) else 1)
    return np.asarray(cells)*UM,np.asarray(owners)

def rectangle_potential(cells, observers, distance):
    # Closed-form source integration, collocated at the observer-cell centres.
    dx=cells[None,:,0:2]-observers[:,None,0,None]
    dy=cells[None,:,2:4]-observers[:,None,1,None]
    result=np.zeros((len(observers),len(cells)))
    for i in (0,1):
        for j in (0,1):
            x,y=dx[:,:,i],dy[:,:,j]
            r=np.sqrt(x*x+y*y+distance*distance)
            primitive=x*np.log(y+r)+y*np.log(x+r)
            if distance:
                primitive-=distance*np.arctan(x*y/(distance*r))
            result+=(-1 if i!=j else 1)*primitive
    return result/((cells[:,1]-cells[:,0])*(cells[:,3]-cells[:,2]))[None,:]

def electrostatic(pitch, gap=0.):
    cells,owners=pulse_geometry(pitch,gap)
    centres=np.stack(((cells[:,0]+cells[:,1])/2,(cells[:,2]+cells[:,3])/2),-1)
    basis=np.eye(2)[owners]
    frequencies=np.asarray(FREQUENCIES)
    omega=2*np.pi*frequencies
    eps=4.1
    silicon=11.9+1j*10/(omega*8.8541878128e-12)
    rb=(eps-silicon)/(eps+silicon)
    rt=(eps-1)/(eps+1)
    product=rb*rt
    matrices=np.broadcast_to(rectangle_potential(cells,centres,0),(2,len(cells),len(cells))).astype(complex).copy()
    for n in range(45):
        for distance,coefficient in ((10*UM+12*UM*n,rb*product**n),
                                      (2*UM+12*UM*n,rt*product**n)):
            matrices+=coefficient[:,None,None]*rectangle_potential(cells,centres,distance)
        if n:
            matrices+=2*product[:,None,None]**n*rectangle_potential(cells,centres,12*UM*n)
    matrices/=4*np.pi*8.8541878128e-12*eps
    values=[]
    for w,matrix in zip(omega,matrices):
        responses=np.linalg.solve(matrix,basis)
        capacitance=basis.T@responses
        common=-capacitance[:,0].sum()/capacitance.sum()
        charge=capacitance@np.array((1+common,common))
        values.append(-1j*w*charge[0])
    return dict(pitch_um=pitch,gap_um=gap,cells=len(cells),frequency_Hz=frequencies.tolist(),
                y_real=np.real(values).tolist(),y_imag=np.imag(values).tolist())

def lumped_estimate():
    # Equal-area disk spreading is approximate; the finite ring is not an ohmic contact.
    areas=np.array((100*100,140*140-120*120))*UM**2
    cox=8.8541878128e-12*4.1*areas/(5*UM)
    rsub=1/(4*10*np.sqrt(areas/np.pi))
    csub=8.8541878128e-12*11.9/(10*rsub)
    w=2*np.pi*np.asarray(FREQUENCIES)[:,None]
    arm=1/(1/(-1j*w*cox)+1/(1/rsub-1j*w*csub))
    y=1/np.sum(1/arm,axis=1)
    return dict(Cox_F=cox.tolist(),Csub_F=csub.tolist(),Rsub_ohm=rsub.tolist(),
                y_real=y.real.tolist(),y_imag=y.imag.tolist())
