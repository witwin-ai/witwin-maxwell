"""F-C class layout: an array of F-B interleaved transformers (thick surface_admittance metal) on one stack.

Every device is the F-B transformer (admittance_fixture_a1.fixture_fb geometry: primary and secondary
two-turn windings, under-pass at 302 um, guard ring 430/490 um) shifted by a pitch on a rows x cols grid,
with its own four lumped ports (d<k>p1..d<k>p4). At max/edge 40/10 um one device is 12640 unknowns, so a
4 x 4 array is the >= 2e5-unknown F-C target; per-body interior work stays that of one F-B body.

Usage:
    fixture_fc.py OUT.json --rows R --cols C [--maximum 40 --edge 10] [--pitch 600]
        [--backend hmatrix|matrix_free] [--frequencies 1e9] [--cap-gib G]
Writes OUT.json (sizes, stage times, torch and device memory peaks, solver statistics, per-device
L/R/M/k from the open-circuit device block of Z) and OUT.y.pt (Y per frequency).
"""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import witwin.maxwell as mw

UNIT = round(1e-6 * 2**60) / 2**60


def fixture_fc(*, rows, cols, maximum, edge, pitch=600., backend="hmatrix"):
    from tests.support.rfic_fixtures import FB_PRIMARY_XY as primary, FB_SECONDARY_XY as secondary
    stack = mw.mom.LayerStack((mw.mom.Layer("silicon", 300*UNIT, 11.9, 10.), mw.mom.Layer("oxide", 6*UNIT, 4.1)),
                              mw.mom.HalfSpace(11.9, 10.), mw.mom.HalfSpace())
    structures, ports = [], []
    material = mw.media.LossyMetalMedium(conductivity=5.8e7)

    for row in range(rows):
        for col in range(cols):
            ox, oy = (col - (cols - 1) / 2) * pitch, (row - (rows - 1) / 2) * pitch

            def strip(a, b, z, thickness):
                x0, x1 = sorted((a[0], b[0]))
                y0, y1 = sorted((a[1], b[1]))
                vertices = torch.tensor(((x0-10+ox, y0-10+oy), (x1+10+ox, y0-10+oy), (x1+10+ox, y1+10+oy),
                                         (x0-10+ox, y1+10+oy)), dtype=torch.float64)*UNIT
                structures.append(mw.Structure(mw.geometry.PlanarPEC(vertices, normal_axis="z", coordinate=z*UNIT,
                                                                       physical_thickness=thickness*UNIT), material))
            for path in (primary, secondary):
                strip((195, path[0][1]), path[0], 305, 2.)
                for a, b in zip(path[:-1], path[1:]):
                    strip(a, b, 305, 2.)
                x, y = path[-1]
                strip((x, y), (195, y), 302, 1.)
                strip((195, y), (195, y), 305, 2.)
                for vx in (x, 195):
                    structures.append(mw.Structure(mw.Box(
                        position=torch.tensor((vx+ox, y+oy, 303.5), dtype=torch.float64)*UNIT,
                        size=torch.tensor((4, 4, 3), dtype=torch.float64)*UNIT), material))
            outer = torch.tensor(((-245+ox, -245+oy), (245+ox, -245+oy), (245+ox, 245+oy), (-245+ox, 245+oy)),
                                 dtype=torch.float64)*UNIT
            inner = torch.tensor(((-215+ox, -215+oy), (215+ox, -215+oy), (215+ox, 215+oy), (-215+ox, 215+oy)),
                                 dtype=torch.float64)*UNIT
            structures.append(mw.Structure(mw.geometry.PlanarPEC(outer, holes=[inner], normal_axis="z",
                                                                   coordinate=305*UNIT, physical_thickness=2.*UNIT),
                                             material))
            device = row * cols + col
            for name, y in (("p1", 170), ("p2", 70), ("p3", 145), ("p4", 45)):
                ports.append(mw.LumpedPort(f"d{device}{name}", positive=((205+ox)*UNIT, (y+oy)*UNIT, 305*UNIT),
                                             negative=((215+ox)*UNIT, (y+oy)*UNIT, 305*UNIT),
                                             voltage_path=mw.AxisPath("x"), reference_impedance=50.))
    scene = mw.Scene(background=stack, structures=structures, ports=ports)
    config = mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(
        bridge_width_m=20*UNIT, max_edge_length_m=maximum*UNIT, edge_cell_m=edge*UNIT, growth_ratio=1.5,
        conductor_model="surface_admittance", operator_backend=backend))
    return scene, config


def device_metrics(result, devices):
    """Per device: D^T Z_dd D of its open-circuit block (D = ((1,0),(-1,0),(0,1),(0,-1))); L1, L2, M, k, R1, R2."""
    z = torch.linalg.inv(result.network.to_y())
    d = torch.tensor(((1., 0.), (-1., 0.), (0., 1.), (0., -1.)), dtype=z.dtype, device=z.device)
    omega = 2 * math.pi * result.network.frequencies.to(z.device)
    out = []
    for k in range(devices):
        zw = d.mT @ z[:, 4*k:4*k+4, 4*k:4*k+4] @ d
        lmat = -zw.imag / omega[:, None, None]
        l1, l2, m = lmat[:, 0, 0], lmat[:, 1, 1], lmat[:, 0, 1]
        out.append(dict(L1_H=l1.tolist(), L2_H=l2.tolist(), M_H=m.tolist(), k=(m / torch.sqrt(l1 * l2)).tolist(),
                        R1_ohm=zw.real[:, 0, 0].tolist(), R2_ohm=zw.real[:, 1, 1].tolist()))
    return out


class DevicePeak:
    """Device memory peak of this process: nvidia-smi by pid where visible, else total minus free."""

    def __init__(self):
        self.pid, self.peak, self.used_peak, self.stage, self.stages = str(os.getpid()), 0, 0, "start", {}
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            free, total = torch.cuda.mem_get_info()
            used = (total - free) // 2**20
            self.used_peak = max(self.used_peak, used)
            mine = 0
            try:
                text = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
                for line in text.splitlines():
                    q = [x.strip() for x in line.split(",")]
                    if q[0] == self.pid:
                        mine = int(q[1])
            except FileNotFoundError:
                pass
            self.peak = max(self.peak, mine)
            self.stages[self.stage] = max(self.stages.get(self.stage, 0), mine or used)
            time.sleep(0.5)


def main(args):
    if args.cap_gib:
        torch.cuda.set_per_process_memory_fraction(args.cap_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    peak = DevicePeak()
    print("witwin.maxwell", mw.__file__, flush=True)
    scene, config = fixture_fc(rows=args.rows, cols=args.cols, maximum=args.maximum, edge=args.edge,
                               pitch=args.pitch, backend=args.backend)
    record = dict(rows=args.rows, cols=args.cols, maximum_um=args.maximum, edge_um=args.edge, pitch_um=args.pitch,
                  backend=args.backend, frequencies=args.frequencies, device=torch.cuda.get_device_name(),
                  alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    started = time.perf_counter()
    peak.stage = "prepare"
    simulation = mw.Simulation.mom(scene, frequencies=tuple(args.frequencies), config=config).prepare()
    native = simulation.solver.prepared
    torch.cuda.synchronize()
    record.update(prepare_seconds=time.perf_counter() - started, unknowns=native.layout.basis.n_edges,
                  faces=len(native.layout.mesh.faces), ports=native.excitation.shape[1],
                  admittance_bodies=len(native.admittance.bodies),
                  body_faces=[len(b.faces) for b in native.admittance.bodies],
                  prepared_allocated=torch.cuda.memory_allocated(), prepared_peak=torch.cuda.max_memory_allocated())
    print("PREPARED", json.dumps(record), flush=True)
    args.output.write_text(json.dumps(record, indent=1, default=str))
    peak.stage = "run"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = simulation.run()
    torch.cuda.synchronize()
    metadata = result.network.metadata
    record.update(run_seconds=time.perf_counter() - started, torch_peak=torch.cuda.max_memory_allocated(),
                  device_peak_MiB=peak.peak, device_used_peak_MiB=peak.used_peak, stage_MiB=dict(peak.stages),
                  devices=device_metrics(result, args.rows * args.cols),
                  metadata={k: v for k, v in metadata.items() if isinstance(v, (int, float, str, tuple, list, dict))})
    y = result.network.to_y().cpu()
    record["reciprocity"] = float((y - y.mT).abs().max() / y.abs().max())
    torch.save(dict(y=y, frequencies=result.network.frequencies.cpu()), str(args.output) + ".y.pt")
    args.output.write_text(json.dumps(record, indent=1, default=str))
    print("DONE", json.dumps({k: v for k, v in record.items() if k not in ("metadata", "devices")}, default=str),
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--maximum", type=float, default=40.)
    p.add_argument("--edge", type=float, default=10.)
    p.add_argument("--pitch", type=float, default=600.)
    p.add_argument("--backend", choices=("hmatrix", "matrix_free"), default="hmatrix")
    p.add_argument("--frequencies", type=float, nargs="+", default=(1e9,))
    p.add_argument("--cap-gib", type=float, default=0.)
    main(p.parse_args())
