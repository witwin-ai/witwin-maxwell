"""Fixed printed dipole with a finite-width, coplanar named-terminal feed."""
from __future__ import annotations

import numpy as np
import torch
import witwin.maxwell as mw
from .printed_patch import _axis_nodes

ARM_END=.023
FEED_GAP=.002
STRIP_WIDTH=.002
SUBSTRATE_HEIGHT=.001
SUBSTRATE_EPS=2.2
COUPON_HALF_X=.030
COUPON_HALF_Y=.015
BOUNDS=((- .040,.040),(-.028,.028),(-.025,.025))


def _box(position,size):
    return mw.Box(position=torch.tensor(position,dtype=torch.float64),
                  size=torch.tensor(size,dtype=torch.float64))


def printed_dipole_scene(*,lateral_dl=.001,substrate_cells=4,air_dl=.002,
        edge_dl=None,metal="pec",dielectric_sigma=0.,frequencies=(2.5e9,3e9,3.5e9),
        pml_layers=12,air_padding=0.,radiation_monitor=True,device="cuda",boundary_cell_widths=None,
        vacuum_reference_y_nodes=None,coupon_y_dl=None,vacuum_reference_z_nodes=None):
    """46 mm tip-to-tip arms on a fixed 1 mm, epsilon=2.2 dielectric coupon.

    The named feed spans the physical 2 mm gap and the full 2 mm strip width.
    All physical dimensions, source aperture, current loop and observation
    surface stay fixed when grid steps change. Copper is a real finite-film
    model; the PEC version retains 35 um only as physical provenance.
    """
    if metal not in ("pec","copper") or dielectric_sigma<0:
        raise ValueError("metal is pec/copper and dielectric_sigma is nonnegative.")
    if lateral_dl<=0 or air_dl<=0 or substrate_cells<1 or air_padding<0:
        raise ValueError("Mesh steps/cell count must be positive; padding is nonnegative.")
    if edge_dl is not None and edge_dl<=0:raise ValueError("edge_dl must be positive.")
    bounds=tuple((lo-air_padding,hi+air_padding) for lo,hi in BOUNDS)
    x_edges=[] if edge_dl is None else [(value-.001,value+.001,edge_dl)
        for value in (-ARM_END,ARM_END,-FEED_GAP/2,FEED_GAP/2)]
    y_edges=[] if edge_dl is None else [(value-.001,value+.001,edge_dl)
        for value in (-STRIP_WIDTH/2,STRIP_WIDTH/2)]
    x=_axis_nodes(bounds[0],(-ARM_END,ARM_END,-FEED_GAP/2,FEED_GAP/2,0.),
        (-COUPON_HALF_X,COUPON_HALF_X),lateral_dl,air_dl,(.0005,),x_edges)
    y=_axis_nodes(bounds[1],(-STRIP_WIDTH/2,STRIP_WIDTH/2,0.),
        (-.005,.005),lateral_dl,air_dl,(-.0035,.0035),y_edges)
    z=_axis_nodes(bounds[2],(-SUBSTRATE_HEIGHT,0.),(-.002,.002),
        SUBSTRATE_HEIGHT/int(substrate_cells),air_dl,(-.0025,.0025))
    if vacuum_reference_y_nodes is not None:
        reference=np.asarray(vacuum_reference_y_nodes,dtype=np.float64)
        if reference.ndim!=1 or reference.size<3 or not np.isfinite(reference).all() or not (np.diff(reference)>0).all():
            raise ValueError("vacuum_reference_y_nodes must be finite and strictly increasing.")
        if not np.allclose(reference[[0,-1]],bounds[1],rtol=0,atol=1e-14):
            raise ValueError("vacuum_reference_y_nodes must retain both physical domain endpoints.")
        lower=int(np.searchsorted(reference,-COUPON_HALF_Y,side='right')-1)
        upper=int(np.searchsorted(reference,COUPON_HALF_Y,side='left'))
        fixed=reference[lower:upper+1]
        # Hold all coupon cells, including both nodes of each interface cell.
        # Only cells wholly in exterior vacuum may be subdivided.
        sides=[]
        for old in (reference[:lower+1],reference[upper:]):
            if np.diff(old).max(initial=0.)<=air_dl*(1+1e-10):sides.append(old)
            else:sides.append(np.linspace(old[0],old[-1],max(1,int(np.ceil((old[-1]-old[0])/air_dl-1e-12)))+1))
        y=np.r_[sides[0][:-1],fixed,sides[1][1:]]
    if vacuum_reference_z_nodes is not None:
        reference=np.asarray(vacuum_reference_z_nodes,dtype=np.float64)
        if reference.ndim!=1 or reference.size<3 or not np.isfinite(reference).all() or not (np.diff(reference)>0).all():
            raise ValueError("vacuum_reference_z_nodes must be finite and strictly increasing.")
        if not np.array_equal(reference[[0,-1]],np.asarray(bounds[2],dtype=np.float64)):
            raise ValueError("vacuum_reference_z_nodes must retain both physical domain endpoints exactly.")
        core_bounds=np.asarray((-.002,.002),dtype=np.float64)
        if not all(np.any(reference==point) for point in core_bounds):
            raise ValueError("vacuum_reference_z_nodes must contain both fixed nearfield boundaries at +/-2 mm.")
        core=z[(z>=core_bounds[0])&(z<=core_bounds[1])]
        if not np.array_equal(core[[0,-1]],core_bounds):
            raise ValueError("The refined normal grid must retain both fixed nearfield boundaries.")
        # Keep all exterior air nodes, including the +/-2.5 mm current-contour
        # half-cell brackets. Substrate refinement changes only the inner band.
        z=np.r_[reference[reference<core_bounds[0]],core,reference[reference>core_bounds[1]]]
    if coupon_y_dl is not None:
        if not np.isfinite(coupon_y_dl) or coupon_y_dl<=0:
            raise ValueError("coupon_y_dl must be finite and positive.")
        # Split existing coupon/nearfield intervals without moving their end
        # points or changing the already frozen outer-vacuum/PML partitions.
        lower=int(np.searchsorted(y,-COUPON_HALF_Y,side='right')-1)
        upper=int(np.searchsorted(y,COUPON_HALF_Y,side='left'))
        refined=[]
        for left,right in zip(y[lower:upper],y[lower+1:upper+1]):
            count=max(1,int(np.ceil((right-left)/coupon_y_dl-1e-12)))
            # The fixed physical H contour crosses the centers at ±3.5 mm.
            # Odd subdivision preserves those centers exactly; an even count
            # would turn a valid half-grid contour into an off-grid contour.
            if any(abs(.5*(left+right)-point)<1e-14 for point in (-.0035,.0035)) and count%2==0:
                count+=1
            refined.extend(np.linspace(left,right,count+1)[:-1])
        y=np.r_[y[:lower],refined,y[upper:]]
    if boundary_cell_widths is not None:
        widths=np.asarray(boundary_cell_widths,dtype=np.float64)
        if widths.shape!=(3,2) or not np.isfinite(widths).all() or not (widths>0).all():
            raise ValueError("boundary_cell_widths must contain positive finite left/right widths for three axes.")
        axes=[]
        for nodes,(left,right),protected in zip((x,y,z),widths,(.032,.020,.017)):
            low,high=nodes[0],nodes[-1]
            if low+left>=-protected or high-right<=protected:
                raise ValueError("boundary_cell_widths must stay outside the fixed Huygens surface and physical anchors.")
            inside=nodes[(nodes>low+left+1e-14)&(nodes<high-right-1e-14)]
            minimum_gap=.5*min(air_dl,left,right)
            # Avoid an artificial sliver where a retained endpoint cell meets
            # a separately partitioned air interval. The physical surface and
            # conductor anchors lie farther inside the protected region.
            while inside.size and inside[0]-(low+left)<minimum_gap:inside=inside[1:]
            while inside.size and (high-right)-inside[-1]<minimum_gap:inside=inside[:-1]
            axes.append(np.r_[low,low+left,inside,high-right,high])
        x,y,z=axes
    structures=[mw.Structure(_box((0.,0.,-SUBSTRATE_HEIGHT/2),
        (2*COUPON_HALF_X,2*COUPON_HALF_Y,SUBSTRATE_HEIGHT)),
        mw.Material(eps_r=SUBSTRATE_EPS,sigma_e=dielectric_sigma),name="substrate")]
    for name,lo,hi in (("positive_arm",FEED_GAP/2,ARM_END),("negative_arm",-ARM_END,-FEED_GAP/2)):
        if metal=="pec":
            geometry=mw.PlanarPEC.rectangle(((lo,hi),(-STRIP_WIDTH/2,STRIP_WIDTH/2)),
                coordinate=0.,physical_thickness=35e-6)
            material=mw.Material.pec()
        else:
            geometry=_box(((lo+hi)/2,0.,0.),(hi-lo,STRIP_WIDTH,0.))
            material=mw.CopperFoil(thickness=35e-6,frequency_range=(.1e9,10e9))
        structures.append(mw.Structure(geometry,material,name=name))
    port=mw.TerminalPort("feed",mw.TerminalRef("positive_arm"),mw.TerminalRef("negative_arm"),
        mw.AxisPath("x"),.0005,current_surface=_box((.0005,0.,0.),(0.,.007,.005)),
        voltage_aperture=_box((0.,0.,0.),(FEED_GAP,STRIP_WIDTH,0.)))
    monitors=[]
    if radiation_monitor:
        monitors.append(mw.ClosedSurfaceMonitor.box("radiation",position=(0.,0.,0.),
            size=(.064,.040,.034),frequencies=tuple(frequencies)))
    if dielectric_sigma:
        monitors.append(mw.PowerLossMonitor("substrate_loss",position=(0.,0.,-SUBSTRATE_HEIGHT/2),
            size=(2*COUPON_HALF_X,2*COUPON_HALF_Y,SUBSTRATE_HEIGHT),
            frequencies=tuple(frequencies),channels=("conduction",)))
    return mw.Scene(domain=mw.Domain(bounds=bounds),grid=mw.GridSpec.custom(x,y,z),
        boundary=mw.BoundarySpec.pml(num_layers=pml_layers),structures=structures,ports=(port,),
        monitors=monitors,device=device,metadata={"physical_model":"fixed-geometry-printed-dipole-v1",
            "metal":metal,"physical_copper_thickness_m":35e-6,"dielectric_sigma_S_per_m":dielectric_sigma,
            "lateral_dl_m":lateral_dl,"edge_dl_m":edge_dl,"substrate_cells":int(substrate_cells),
            "air_dl_m":air_dl,"air_padding_m":air_padding,"pml_layers":pml_layers,
            "coupon_y_dl_m":coupon_y_dl,
            "boundary_cell_widths_m":None if boundary_cell_widths is None else widths.tolist(),
            "vacuum_reference_y_nodes_m":None if vacuum_reference_y_nodes is None else reference.tolist()})
