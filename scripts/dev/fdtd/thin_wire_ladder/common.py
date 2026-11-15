"""Pure CPU run manifest, geometry, and diagnostic estimators."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import fnmatch
import math

import numpy as np

C0 = 299792458.0
F0 = 2.4e9
LAMBDA0 = C0/F0
DELTA0 = LAMBDA0/40
RADIUS = LAMBDA0/800
FREQUENCIES = np.linspace(1.6e9, 3.2e9, 33)
CHECK_FREQUENCIES = np.array([2.11e9, 2.40e9, 2.69e9])
LEVELS = (1., 1.5, 2., 3.)
PML = 12
GIB = 1024**3
MEMORY_BYTES_PER_CELL = 256
MEMORY_OVERHEAD = 256*1024**2


@dataclass(frozen=True)
class Run:
    antenna: str
    level: float
    gap_rule: str
    binding: str
    control: str = "primary"

    @property
    def name(self):
        return f"{self.antenna}_{self.gap_rule}_{self.binding}_r{self.level:g}_{self.control}"

    @property
    def delta(self):
        return DELTA0/self.level

    @property
    def radius(self):
        return .05*self.delta if self.control == "scaled-radius" else RADIUS

    @property
    def gap(self):
        return {"G1": self.delta, "G2": 2*self.delta,
                "G-phys": DELTA0, "G-phys075": .75*DELTA0}[self.gap_rule]

    @property
    def length(self):
        return {"D1": .47, "D2": .25, "L1": 1.}[self.antenna]*LAMBDA0

    @property
    def cells(self):
        extent = (0., 0., self.length) if self.antenna != "L1" else (self.length/math.pi,)*2+(0.,)
        # Symmetric bounds on exact integer cells keep center alignment fixed.
        return tuple(2*math.ceil((span/2 + .5*LAMBDA0)/self.delta - 1e-10)+2*PML for span in extent)

    @property
    def estimated_bytes(self):
        return math.prod(self.cells)*MEMORY_BYTES_PER_CELL+MEMORY_OVERHEAD

    def manifest(self):
        return dict(asdict(self), name=self.name, delta_m=self.delta, radius_m=self.radius,
                    gap_m=self.gap, gap_cells=self.gap/self.delta, length_m=self.length,
                    cells=self.cells, total_cells=math.prod(self.cells),
                    estimated_peak_bytes=self.estimated_bytes)


def runs(pattern="*"):
    """66 primary configurations plus six explicitly labeled D1 controls.

    G-phys r1 aliases G1 r1; alias metadata preserves the full 72-arm matrix.
    Controls use G1/nodes for scaled radius and nodes for the subcell gap.
    """
    result = [Run(a, r, g, b) for a in ("D1", "D2", "L1") for r in LEVELS
              for g in ("G1", "G2", "G-phys") for b in ("nodes", "gap")
              if not (g == "G-phys" and r == 1)]
    result += [Run("D1", r, "G1", "nodes", "scaled-radius") for r in (1., 2.)]
    result += [Run("D1", r, "G-phys075", "nodes", "subcell-gap") for r in LEVELS]
    patterns = pattern.split(",")
    return [r for r in result if any(fnmatch.fnmatchcase(r.name, p) for p in patterns)]


def wire_geometry(run):
    """Return source polylines and feed source-point references.

    Dipole tips stay fixed. Loop is in xy, radius L/(2pi); its straight feed
    chord has length exactly g. Circle vertices have spacing at most delta/2;
    their angles away from the feed are identical between binding variants.
    A node-bound loop is one open path whose two ends form the feed.
    """
    if run.antenna != "L1":
        count = math.ceil((run.length-run.gap)/(2*run.delta))
        negative = [(0., 0., float(z)) for z in np.linspace(-run.length/2, -run.gap/2, count+1)]
        positive = [(0., 0., float(z)) for z in np.linspace(run.gap/2, run.length/2, count+1)]
        if run.binding == "nodes":
            return {"negative": negative, "positive": positive}, ("negative", count), ("positive", 0)
        return {"wire": negative+positive}, ("wire", count), ("wire", count+1)
    radius = run.length/(2*math.pi)
    angle = math.asin(run.gap/(2*radius))
    count = math.ceil((2*math.pi-2*angle)*radius/(run.delta/2))
    # Negative feed terminal is below the x axis. Path runs from positive
    # terminal around the back of the circle to the negative terminal.
    theta = np.linspace(angle, 2*math.pi-angle, count+1)
    points = [(radius*math.cos(t), radius*math.sin(t), 0.) for t in theta]
    # Set symmetric feed coordinates explicitly to eliminate trig roundoff.
    xfeed = math.sqrt(radius**2-(run.gap/2)**2)
    points[0] = (xfeed, run.gap/2, 0.)
    points[-1] = (xfeed, -run.gap/2, 0.)
    if run.binding == "gap":
        points.append(points[0])
        return {"wire": points}, ("wire", count), ("wire", count+1)
    return {"wire": points}, ("wire", count), ("wire", 0)


def build_scene(run):
    import witwin.maxwell as mw
    geometry, negative, positive = wire_geometry(run)
    wires = tuple(mw.rf.ThinWire(name, points, run.radius, mw.rf.WireConductor.pec(), snap="continuous")
                  for name, points in geometry.items())
    refs = dict(negative=mw.rf.WireNodeRef(*negative), positive=mw.rf.WireNodeRef(*positive))
    binding = getattr(mw.rf.WirePortBinding, run.binding)(**refs)
    monitors = tuple(mw.rf.WireMonitor(f"wire_{name}", name, tuple(CHECK_FREQUENCIES),
                                  quantities=("current", "charge")) for name in geometry)
    # Domain is the physical air box: Scene preparation adds PML externally.
    bounds = tuple((-(n-2*PML)*run.delta/2, (n-2*PML)*run.delta/2) for n in run.cells)
    # Guard ceil(span/dl) against an extra cell from floating-point division.
    # Bounds fix the actual spacing at delta; nextafter only selects cell count.
    grid_step = math.nextafter(run.delta, math.inf)
    return mw.Scene(domain=mw.Domain(bounds=bounds), grid=mw.GridSpec.uniform(grid_step),
                    boundary=mw.BoundarySpec.pml(num_layers=PML), thin_wires=wires,
                    ports=(mw.LumpedPort("feed", wire_binding=binding, reference_impedance=50.),),
                    monitors=monitors)


def complex_pairs(values):
    values = np.asarray(values)
    return np.stack((values.real, values.imag), axis=-1).tolist()


def from_pairs(values):
    values = np.asarray(values)
    return values[..., 0]+1j*values[..., 1]


def resonance(frequencies, impedance, target=F0):
    """Linear X=0 interpolation, nearest target crossing; never extrapolate.

    No F4 estimator or p_fit.csv/gap_cells.csv is present in this checkout.
    This explicit estimator must be compared with that analysis before claiming
    estimator identity. Analysis tracks the previous level's crossing.
    """
    f, z = np.asarray(frequencies), np.asarray(impedance)
    candidates = []
    for j in range(len(f)-1):
        x, y = z[j].imag, z[j+1].imag
        if not np.isfinite([x, y]).all() or x*y > 0 or x == y:
            continue
        t = -x/(y-x)
        candidates.append((float(f[j]+t*(f[j+1]-f[j])), complex(z[j]+t*(z[j+1]-z[j]))))
    return min(candidates, key=lambda pair: abs(pair[0]-target)) if candidates else (None, None)
