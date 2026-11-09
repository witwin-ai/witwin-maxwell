"""Independent MP100 full-symbol and leading-static plane references."""
import mpmath as mp
from witwin.maxwell.constants import EPSILON_0,MU_0

def mp_plane_remainder_reference(q,eps,mu,frequency,distance, *, return_mp=False):
    """Resolve actual binary constitutive data at100digits, not a proxy medium."""
    with mp.workdps(100):
        def matrix(values):return mp.matrix([[mp.mpc(v.real,v.imag) for v in row] for row in values])
        electric,magnetic=matrix(eps),matrix(mu)
        x,y=map(mp.mpf,q)
        curl=mp.matrix([[0,0,y],[0,0,-x],[-y,x,0]])
        symbol=mp.matrix(6)
        for i in range(3):
            for j in range(3):
                symbol[i,j]=electric[i,j];symbol[i+3,j+3]=-magnetic[i,j]
                symbol[i,j+3]=symbol[i+3,j]=curl[i,j]
        tangential=[0,1,3,4];longitudinal=[2,5]
        def block(rows,cols):return mp.matrix([[symbol[i,j] for j in cols] for i in rows])
        tt,tl,lt,ll=block(tangential,tangential),block(tangential,longitudinal),block(longitudinal,tangential),block(longitudinal,longitudinal)
        reduced=tt-tl*(ll**-1)*lt
        derivative=mp.matrix([[0,0,0,-1],[0,0,1,0],[0,-1,0,0],[1,0,0,0]])
        generator=-(derivative**-1)*reduced
        roots,vectors=mp.eig(generator,left=False,right=True)
        identity=mp.eye(4);embedding=mp.matrix(6,4);forcing=mp.matrix(4,6)
        bottom=-(ll**-1)*lt;right=-tl*(ll**-1)
        for i in range(4):
            for j in range(4):embedding[i,j]=forcing[i,j]=identity[i,j]
        for i in range(2):
            for j in range(4):embedding[i+4,j]=bottom[i,j];forcing[j,i+4]=right[j,i]
        coefficients=(vectors**-1)*(derivative**-1)*forcing
        eta=mp.sqrt(mp.mpf(MU_0)/mp.mpf(EPSILON_0))
        k0=2*mp.pi*mp.mpf(frequency)*mp.sqrt(mp.mpf(EPSILON_0)*mp.mpf(MU_0))
        output=mp.matrix(6);outgoing=[]
        for column,root in enumerate(roots):
            flux=mp.re(vectors[0,column]*mp.conj(vectors[3,column])-vectors[1,column]*mp.conj(vectors[2,column]))
            if mp.im(root)>mp.mpf('1e-70') or (abs(mp.im(root))<=mp.mpf('1e-70') and flux>0):
                outgoing.append(column)
                for i in range(6):
                    for j in range(6):
                        output[i,j]+=mp.j*k0*(embedding*vectors)[i,column]*coefficients[column,j]*mp.exp(mp.j*k0*root*mp.mpf(distance))
        assert len(outgoing)==2
        reorder=[0,1,4,2,3,5]
        source=[-mp.j*eta/k0]*3+[mp.j/k0]*3;field=[1]*3+[1/eta]*3
        physical=mp.matrix([[field[i]*output[reorder[i],reorder[j]]*source[j] for j in range(6)] for i in range(6)])
        static=mp.matrix(6);omega=2*mp.pi*mp.mpf(frequency)
        for offset,tensor,vacuum in ((0,electric,EPSILON_0),(3,magnetic,MU_0)):
            symmetric=(tensor+tensor.T)/2
            a=symmetric[2,2];b=symmetric[2,0]*x+symmetric[2,1]*y
            c=symmetric[0,0]*x*x+(symmetric[0,1]+symmetric[1,0])*x*y+symmetric[1,1]*y*y
            root=(-b+mp.sqrt(b*b-a*c))/a
            assert mp.im(root)>0
            wave=[k0*x,k0*y,k0*root]
            coefficient=mp.exp(mp.j*k0*root*mp.mpf(distance))/(2*k0*(a*root+b)*omega*mp.mpf(vacuum))
            for i in range(3):
                for j in range(3):static[offset+i,offset+j]=coefficient*wave[i]*wave[j]
        # Both spectra remain MP100 until after their difference is formed.
        remainder=physical-static
        if return_mp:
            return {'complete':physical,'static':static,'remainder':remainder}
        return {name:[[complex(value[i,j]) for j in range(6)] for i in range(6)] for name,value in
            (('complete',physical),('static',static),('remainder',remainder))}
