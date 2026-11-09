"""Frozen F-A square spiral through the public Scene -> Simulation -> Result API.

Layout numbers are micrometres. Fourteen sides define 3.5 turns; outer/inner
diameters 200/105 um, width/spacing 10/5 um. Closed ring: inner side 290 um,
45 um clearance from turns. Silicon half-space avoids a backside magnetic image.
"""
from dataclasses import dataclass
import torch
import witwin.maxwell as mw

SIGMA = 5.8e7
UM = 1e-6
SWEEP = tuple(1e8 * 200**(i / 24) for i in range(25))
MESHES = {"test": (20., 5.), "full": (10., 4.9), "shield": (80., 20.),
          "shield_fine": (20., 5.)}
SPIRAL_XY = ((95,95),(-95,95),(-95,-95),(95,-95),(95,80),
             (-80,80),(-80,-80),(80,-80),(80,65),(-65,65),
             (-65,-65),(65,-65),(65,50),(-50,50),(-50,-50))

@dataclass(frozen=True)
class FixtureFA:
    scene: mw.Scene
    config: mw.MoMConfig
    segments: tuple
    port_segment: tuple | None
    mesh: str

    def simulation(self, frequencies=SWEEP):
        return mw.Simulation.mom(self.scene, frequencies=frequencies, config=self.config)


def fixture_fa(*, lossy=True, substrate=True, substrate_sigma=10., ring=True,
               mesh="test", top_thickness_um=2., device="cuda", shield="none",
               shield_grounded=True):
    """Two x-directed ring ports, or one y gap between side-by-side leads.

    Underpass: (-50,-50) to (125,-50), z=302 um. Two 4x4 um Box vias
    reach z=305 um. Return lead ends at (125,80), beside outer lead (125,95).
    Ring ports join x=130 to x=145. One-port variant bridges y=85 to y=90.
    Ring clearance refers to turns; leads approach it for measurement.
    Shields retain their declared poly material when winding metal is set to PEC.
    """
    assert shield in ("none", "pgs", "solid")
    assert shield == "none" or ring
    eps_si,eps_ox = (11.9,4.1) if substrate else (1.,1.)
    sigma = substrate_sigma if substrate else 0.
    stack = mw.LayerStack((mw.Layer("silicon",300*UM,eps_si,sigma),
                           mw.Layer("oxide",6*UM,eps_ox)),
                          mw.HalfSpace(eps_si,sigma),mw.HalfSpace())
    scene = mw.Scene(background=stack,device=device)
    material = mw.LossyMetalMedium(conductivity=SIGMA) if lossy else mw.Material.pec()
    za,zb = 302,305
    segments = []

    def rectangle(x0,x1,y0,y1,z,thickness,sheet_material=material):
        points = torch.tensor(((x0,y0),(x1,y0),(x1,y1),(x0,y1)),dtype=torch.float64)*UM
        scene.add_structure(mw.Structure(mw.PlanarPEC(points,normal_axis="z",coordinate=z*UM,
                                                     physical_thickness=thickness*UM),sheet_material))

    def strip(a,b,z,thickness=top_thickness_um):
        x0,x1 = min(a[0],b[0]),max(a[0],b[0])
        y0,y1 = min(a[1],b[1]),max(a[1],b[1])
        # Square caps include the corner pads without internal centreline
        # breakpoints that would extend across every metal plane.
        rectangle(x0-5,x1+5,y0-5,y1+5,z,thickness)
        segments.append((tuple(v*UM for v in (*a,z)),tuple(v*UM for v in (*b,z)),10*UM,thickness*UM))

    for a,b in zip(SPIRAL_XY[:-1],SPIRAL_XY[1:]):
        strip(a,b,zb)
    strip((95,95),(125,95),zb)
    segments[-1] = (segments[-1][1],segments[-1][0],*segments[-1][2:])
    strip((-50,-50),(125,-50),za,thickness=1.)
    strip((125,-50),(125,80),zb)
    for x,y in ((-50,-50),(125,-50)):
        scene.add_structure(mw.Structure(mw.Box(
            position=torch.tensor((x,y,(za+zb)/2),dtype=torch.float64)*UM,
            size=torch.tensor((4,4,zb-za),dtype=torch.float64)*UM),material))
    segments.insert(14,((-50*UM,-50*UM,zb*UM),(-50*UM,-50*UM,za*UM),4*UM,0.))
    segments.append(((125*UM,-50*UM,za*UM),(125*UM,-50*UM,zb*UM),4*UM,0.))

    def port(name,positive,negative,axis):
        centre = tuple((a+b)*UM/2 for a,b in zip(positive,negative))
        return mw.LumpedPort(name,positive=tuple(v*UM for v in positive),
                             negative=tuple(v*UM for v in negative),voltage_path=mw.AxisPath(axis),
                             current_surface=mw.Box(position=centre,size=(10*UM,15*UM,10*UM)),
                             reference_impedance=50.)
    port_segment = None
    if ring:
        outer = torch.tensor(((-155,-155),(155,-155),(155,155),(-155,155)),dtype=torch.float64)*UM
        inner = torch.tensor(((-145,-145),(145,-145),(145,145),(-145,145)),dtype=torch.float64)*UM
        scene.add_structure(mw.Structure(mw.PlanarPEC(outer,holes=[inner],normal_axis="z",coordinate=zb*UM,
                                                     physical_thickness=top_thickness_um*UM),material))
        for name,y in (("p1",95),("p2",80)):
            scene.add_port(port(name,(130,y,zb),(145,y,zb),"x"))
    else:
        scene.add_port(port("loop",(125,90,zb),(125,85,zb),"y"))
        port_segment = ((125*UM,80*UM,zb*UM),(125*UM,95*UM,zb*UM),10*UM,top_thickness_um*UM)
    if shield != "none":
        # Doped poly: 0.2 um at 5e5 S/m gives 10 ohm/square at DC.
        # The four combs stop at diagonal slots; only the perimeter joins them.
        shield_material = mw.LossyMetalMedium(conductivity=5e5)
        zs = 300.5
        # Active footprint is 220x220 um, with a perimeter outside the coil.
        # Two lateral straps connect it to the ring below the existing ground.
        for box in ((-155,155,-155,-145),(-155,155,145,155),
                    (-155,-145,-145,145),(145,155,-145,145),
                    (-145,-100,-5,5),(100,145,-5,5)):
            rectangle(*box,zs,.2,shield_material)
        if shield == "solid":
            rectangle(-110,110,-110,110,zs,.2,shield_material)
        else:
            for box in ((-110,110,-110,-100),(-110,110,100,110),
                        (-110,-100,-100,100),(100,110,-100,100)):
                rectangle(*box,zs,.2,shield_material)
            for centre in (-80,-50,-20,20,50,80):
                start = abs(centre)+10
                for box in ((centre-5,centre+5,start,100),
                            (centre-5,centre+5,-100,-start),
                            (start,100,centre-5,centre+5),
                            (-100,-start,centre-5,centre+5)):
                    rectangle(*box,zs,.2,shield_material)
        if shield_grounded:
            # Short intermediate ring sections connect staggered via arrays.
            # Keeping the rest open avoids an additional closed eddy-current loop.
            for box in ((-155,-135,-155,-145),(135,155,145,155)):
                rectangle(*box,za,1.)
            # Coincident lower/upper tube rims would create a three-face
            # junction. Each path instead traverses a 10-um landing section.
            for low,high,locations in ((zs,za,((-150,-150),(150,150))),
                                       (za,zb,((-140,-150),(140,150)))):
                for x,y in locations:
                    scene.add_structure(mw.Structure(mw.Box(
                        position=torch.tensor((x,y,(low+high)/2),dtype=torch.float64)*UM,
                        size=torch.tensor((4,4,high-low),dtype=torch.float64)*UM),material))
    maximum,edge = MESHES[mesh]
    config = mw.MoMConfig(layered=mw.LayeredSurfaceConfig(bridge_width_m=10*UM,
                          max_edge_length_m=maximum*UM,edge_cell_m=edge*UM,growth_ratio=1.5))
    return FixtureFA(scene,config,tuple(segments),port_segment,mesh)
