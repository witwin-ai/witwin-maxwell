"""Native component constitutive lowering and capability registry for charts."""
from __future__ import annotations

import math
import numpy as np
import torch

from ..coordinates import BendCoordinates


def copper_half_metric_average(chart, surface_u, width, side, component):
    """Exact mean constitutive metric over one split normal dual interval."""
    lower=surface_u-width if side<0 else surface_u
    h0=1-chart.curvature*lower
    shift=chart.curvature*width
    if component==0:
        return 1. if shift==0 else -math.log1p(-shift/h0)/shift
    return h0-shift/2


def validate_coordinate_scene(scene, *, method="fdtd"):
    chart = getattr(scene,"coordinates",None)
    if chart is None:
        return
    if not isinstance(chart,BendCoordinates):
        raise TypeError("Scene.coordinates must be BendCoordinates or None.")
    if method!="fdtd":
        raise NotImplementedError("BendCoordinates currently requires native FDTD.")
    chart.validate_bounds(scene.domain.bounds)
    if chart.curvature and scene.boundary.axis_kind("y") in ("periodic","bloch"):
        raise NotImplementedError("The varying normal metric cannot be identified across periodic u faces.")
    if scene.sources:
        raise NotImplementedError("Mapped field sources need physical Piola/Huygens lowering; use a registered RF feed.")
    if scene.thin_wires or scene.material_regions or scene.electrostatic_terminals or scene.charge_densities:
        raise NotImplementedError("Mapped wires, density regions and electrostatic sources are not registered.")
    if any(value is not None for value in scene.symmetry):
        raise NotImplementedError("Mapped symmetry planes require physical image validation.")
    from ..ports import LumpedPort,TerminalPort
    if any(not isinstance(port,(LumpedPort,TerminalPort)) for port in scene.ports):
        raise NotImplementedError("Mapped modal ports require a qualified covariant cross-section solve.")
    from ..monitors import PointMonitor,PowerLossMonitor,FinitePlaneMonitor,ClosedSurfaceMonitor
    if any(not isinstance(monitor,(PointMonitor,PowerLossMonitor,FinitePlaneMonitor,ClosedSurfaceMonitor)) for monitor in scene.monitors):
        raise NotImplementedError("Mapped observations require points, finite chart faces, closed chart surfaces or static conduction selections.")
    if any(isinstance(monitor,PowerLossMonitor) and monitor.channels!=("conduction",) for monitor in scene.monitors):
        raise NotImplementedError("Mapped loss currently registers automatic static electric conduction only.")
    for structure in scene.structures:
        if not getattr(structure,"enabled",True):
            continue
        material=structure.material
        if getattr(structure.geometry,"kind",None)=="surface_mesh":
            raise NotImplementedError("A mapped chart does not enable arbitrary SurfaceMesh split cells.")
        for feature in ("is_anisotropic","is_dispersive","is_nonlinear","is_gyromagnetic","is_medium2d",
                        "is_lossy_metal","is_surface_impedance"):
            if getattr(material,feature,False):
                raise NotImplementedError(f"Mapped physical material capability {feature} is not registered.")
        if getattr(material,"breakdown",None) is not None or getattr(material,"modulation",None) is not None:
            raise NotImplementedError("Mapped dynamic conductivity/modulation requires joint transformed state updates.")
        if getattr(material,"is_copper_foil",False):
            if abs(chart.curvature)*float(material.thickness)>2e-3:
                raise NotImplementedError("Mapped copper exceeds the independently bounded curvature*thickness range.")
            from .ports import _box_values
            position,size=_box_values(structure.geometry)
            if size[1]!=0 or size[0]<=0 or size[2]<=0:
                raise NotImplementedError("Mapped copper requires a true chart-normal u plane.")


def attach_coordinate_metric(scene, model):
    chart=getattr(scene,"coordinates",None)
    if chart is None:
        return model
    validate_coordinate_scene(scene)
    validate_coordinate_observers(scene)
    if "coordinate_frame" in model:
        raise RuntimeError("A coordinate metric must be compiled exactly once.")
    from .materials import _refresh_model_summary_aliases
    node_u=torch.as_tensor(scene.y_nodes64,device=scene.device,dtype=torch.float64)
    node_h=1-chart.curvature*node_u
    for key in ("eps_components","mu_components","sigma_e_components","sigma_m_components"):
        model[key]={axis:values*(1/node_h if axis=="x" else node_h).to(values.dtype)[None,:,None]
                    for axis,values in model[key].items()}
    edge=model.get("edge_components")
    if edge is None:
        raise NotImplementedError("Mapped materials require native electric/magnetic component sampling.")
    for key,magnetic in (("eps",False),("sigma_e",False),("mu",True),("sigma_m",True)):
        transformed={}
        for name,values in edge[key].items():
            own="xyz".index(name[1].lower())
            half=(own==1)!=magnetic
            u=torch.as_tensor(scene.y_half64 if half else scene.y_nodes64,device=values.device,dtype=torch.float64)
            h=1-chart.curvature*u
            factor=1/h if own==0 else h
            transformed[name]=values*factor.to(values.dtype)[None,:,None]
        edge[key]=transformed
    model["coordinate_frame"]=chart.snapshot()
    return _refresh_model_summary_aliases(model)


def _coordinate_foil_observation_regions(scene):
    """Actual staggered interpolation neighborhoods of active copper edges."""
    cached=getattr(scene,"_coordinate_foil_observation_regions",None)
    if cached is not None:
        return cached
    from .copper_foil import compile_copper_faces
    from ..fdtd.coords import component_coords
    regions=[]
    for face in compile_copper_faces(scene,device="cpu"):
        name="E"+"xyz"[face.component]
        axes=component_coords(scene,name)
        indices=face.indices.numpy()
        footprint=[]
        for axis in (0,2):
            values=np.asarray(axes[axis])
            lower=max(0,int(indices[:,axis].min())-1)
            upper=min(len(values)-1,int(indices[:,axis].max())+1)
            footprint.append((float(values[lower]),float(values[upper])))
        normal=face.normal_index
        nodes=np.asarray(scene.y_nodes64);half=np.asarray(scene.y_half64)
        # E_t uses the stored left trace: interpolation below the interface is
        # valid, while the right interval needs the independent right trace.
        # H_t interpolates across its jump between the adjacent half nodes.
        for component in (name,"H"+"zyx"[face.component],"Ey"):
            electric=component==name
            interval=(float(nodes[normal]),float(nodes[normal+1])) if electric else (float(half[normal-1]),float(half[normal]))
            regions.append((component,(footprint[0],interval,footprint[1]),electric))
    cached=tuple(regions)
    scene._coordinate_foil_observation_regions=cached
    return cached


def validate_coordinate_observation_bounds(scene, bounds=None, *, points=None,components=("Ex","Ey","Ez","Hx","Hy","Hz")):
    """Reject ordinary interpolation that needs unregistered one-sided traces.

    The normal limits use actual Yee coordinates, without a padding radius.
    Tangential support extends only to the immediate neighbor of each active
    copper-edge lattice. Magnetic normal continuity needs no jump correction;
    electric normal interpolation is conservatively excluded across the charge
    sheet. Bulk loss has separate two-trace accounting and is unaffected.
    """
    if getattr(scene,"coordinates",None) is None:
        return
    for component,exclusion,electric in _coordinate_foil_observation_regions(scene):
        if component not in components:
            continue
        if points is None:
            overlap=all(float(bounds[i][1])>exclusion[i][0] and float(bounds[i][0])<exclusion[i][1] for i in (0,2))
            overlap=overlap and float(bounds[1][0])<exclusion[1][1] and (float(bounds[1][1])>=exclusion[1][0] if electric else float(bounds[1][1])>exclusion[1][0])
        else:
            limits=points.new_tensor(exclusion)
            mask=((points>limits[:,0]) & (points<limits[:,1])).all(-1)
            if electric:
                mask=((points[:,(0,2)]>limits[(0,2),0]) & (points[:,(0,2)]<limits[(0,2),1])).all(-1) & (points[:,1]>=limits[1,0]) & (points[:,1]<limits[1,1])
            overlap=bool(mask.any())
        if overlap:
            raise NotImplementedError("Mapped ordinary field observations intersect an unresolved copper Yee interpolation stencil; one-sided electric and magnetic trace reconstruction is not registered.")


def validate_coordinate_observers(scene):
    from ..monitors import PointMonitor,FinitePlaneMonitor,ClosedSurfaceMonitor
    for monitor in scene.monitors:
        faces=monitor.faces if isinstance(monitor,ClosedSurfaceMonitor) else (monitor,)
        for face in faces:
            if isinstance(face,PointMonitor):
                bounds=tuple((value,value) for value in face.position)
            elif isinstance(face,FinitePlaneMonitor):
                bounds=tuple((face.position[i]-face.size[i]/2,face.position[i]+face.size[i]/2) for i in range(3))
            else:
                continue
            validate_coordinate_observation_bounds(scene,bounds,components=face.fields)
