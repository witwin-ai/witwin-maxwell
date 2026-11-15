"""Straight copper strip with an explicit ideal return for inductance measurement."""
import torch
import witwin.maxwell as mw

SIGMA = 5.8e7
UM = 1e-6
STRIP_DC_OHM = 200e-6 / (SIGMA * 10e-6 * 2e-6)


def strip_fixture(*, model="closed", maximum_um=20., edge_um=2., thickness_um=2.,
                  lossy=True):
    """200x10 um copper bar, PEC return and one ideal top-face delta-gap bridge.

    A return is explicitly present because a strip's terminal inductance depends
    on the measurement circuit. Copper occupies x=0..200, y=-5..5 um.
    End contacts cover its two complete x-normal faces; the 10-um PEC return
    follows x=-10..210, y=-115..-105 um and contains a 10-um source gap.
    """
    stack = mw.mom.LayerStack((mw.mom.Layer("air",1e-3),),mw.mom.HalfSpace(),mw.mom.HalfSpace())
    scene = mw.Scene(background=stack)
    material = mw.media.LossyMetalMedium(conductivity=SIGMA) if lossy else mw.Material.pec()
    rectangles = [(0.,200.,-5.,5.,material),(-10.,0.,-115.,5.,mw.Material.pec()),
                  (200.,210.,-115.,5.,mw.Material.pec()),
                  (0.,95.,-115.,-105.,mw.Material.pec()),
                  (105.,200.,-115.,-105.,mw.Material.pec())]
    for x0,x1,y0,y1,medium in rectangles:
        vertices=torch.tensor(((x0,y0),(x1,y0),(x1,y1),(x0,y1)),dtype=torch.float64)*UM
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(mw.geometry.PlanarPEC(vertices,normal_axis="z",coordinate=100*UM,
                physical_thickness=thickness_um*UM),medium),
        ))
    scene = scene.updated(ports=(
        *scene.ports,
        mw.LumpedPort("p",positive=(105*UM,-110*UM,100*UM),
            negative=(95*UM,-110*UM,100*UM),voltage_path=mw.AxisPath("x"),reference_impedance=50.),
    ))
    config=mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=10*UM,
        max_edge_length_m=maximum_um*UM,edge_cell_m=edge_um*UM,conductor_model=model))
    return scene,config


def network_metrics(result):
    network=result.network
    f=network.frequencies.detach().cpu()
    y=network.to_y()
    series=(1/y[:,0,0]) if y.shape[-1]==1 else (-1/y[:,1,0])
    zin=1/y[:,0,0]
    return dict(frequency_Hz=f.tolist(),L_H=(-series.imag.cpu()/(2*torch.pi*f)).tolist(),
                R_ohm=series.real.cpu().tolist(),Q=(-zin.imag/zin.real).cpu().tolist(),
                unknowns=network.metadata["unknowns"])
