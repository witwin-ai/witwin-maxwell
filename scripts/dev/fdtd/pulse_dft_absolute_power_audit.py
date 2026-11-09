"""Capture copper pulse outputs or audit per-bin conversions of frozen payloads.

Run every command through the GPU lane wrapper. ``capture`` imports the selected
source tree and uses the existing copper bulk-loss scene. ``compare`` also reads
the payloads of the two existing port bit-identity scene scripts. No pins change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


AMPLITUDES = ("voltage", "current", "a", "b")
POWERS = ("available_power", "incident_power", "reflected_power", "accepted_power", "delivered_power")


def capture(tree, output):
    import witwin.maxwell as mw
    assert Path(mw.__file__).resolve().is_relative_to(tree.resolve()), mw.__file__
    # Tests supply the same scene to both package trees, without copying solvers.
    sys.path.insert(1, str(Path(__file__).resolve().parents[3]))
    from tests.materials.surface_impedance.test_copper_foil_power_loss import _simulation
    from witwin.maxwell.postprocess.fdtd_loss import conductor_region_power

    result = _simulation(graph=True, steps=512, pulse=True, normalize_source=False).run()
    assert result.stats()["normalize_source"] is False
    rows = {}

    def retain(name, value, exponent):
        rows[name] = dict(frequencies=list(result.frequencies), exponent=exponent,
                          value=torch.as_tensor(value).detach().cpu().clone())

    port = result.port("feed")
    for name in AMPLITUDES + POWERS:
        retain("port/" + name, getattr(port, name), 1 if name in AMPLITUDES else 2)
    record = result.conductor_losses["foil"]
    retain("copper/surface_power", record["surface_power"], 2)
    retain("copper/regional_power", conductor_region_power(record, ((-.02, .02),)*3), 2)
    for trace in record["bulk_electric_traces"]:
        retain("copper/trace_norm/" + trace["component"],
               torch.linalg.vector_norm(trace["electric_fields"].flatten(1), dim=1), 1)
    for name, value in result.power_loss("bulk").channel_power.items():
        retain("bulk/" + name, value, 2)
    torch.save(dict(module=mw.__file__, rows=rows), output)
    print(json.dumps(dict(module=mw.__file__, rows=len(rows))), flush=True)


def rows_from_payload(payload):
    if "rows" in payload:
        return payload["rows"]
    rows = {}
    for key, value in payload.items():
        name = key.rsplit("/", 1)[-1]
        if name not in AMPLITUDES + POWERS or not isinstance(value, torch.Tensor):
            continue
        if "/ports/" in key:
            frequency = payload[key.rsplit("/", 1)[0] + "/frequencies"]
        elif key.startswith(("direct/", "sweep/")):
            frequency = (.4e9, .6e9)
        else:
            continue
        if len(frequency) < 2:
            continue
        rows[key] = dict(frequencies=frequency, exponent=1 if name in AMPLITUDES else 2,
                         value=value.movedim(-1, 0))
    return rows


def compare(old_path, new_path, output, old_overlay=None):
    old_payload = torch.load(old_path, weights_only=False, map_location="cpu")
    new_payload = torch.load(new_path, weights_only=False, map_location="cpu")
    if old_overlay is not None:
        overlay = torch.load(old_overlay, weights_only=False, map_location="cpu")
        old_payload.update({key: value for key, value in overlay.items() if key != "__module"})
    old = rows_from_payload(old_payload)
    new = rows_from_payload(new_payload)
    assert old and old.keys() == new.keys(), (old.keys() - new.keys(), new.keys() - old.keys())
    report = []
    for name, before in old.items():
        frequency = torch.as_tensor(before["frequencies"], dtype=torch.float64)
        torch.testing.assert_close(frequency, torch.as_tensor(new[name]["frequencies"], dtype=torch.float64), rtol=0., atol=0.)
        center = float((frequency.min() + frequency.max()) / 2)
        factor = (frequency / center).pow(before["exponent"])
        a, b = before["value"], new[name]["value"]
        expected = a * factor.reshape((-1,) + (1,)*(a.ndim - 1))
        nonzero = expected != 0
        assert torch.equal(b[~nonzero], expected[~nonzero]), name
        error = torch.zeros_like(expected.real, dtype=torch.float64)
        error[nonzero] = ((b[nonzero] - expected[nonzero]) / expected[nonzero]).abs()
        per_bin = error.reshape(len(frequency), -1).amax(dim=1)
        maximum = float(per_bin.max())
        report.append(dict(name=name, frequencies_Hz=frequency.tolist(), center_Hz=center,
            exponent=before["exponent"], expected_new_over_old=factor.tolist(),
            max_relative_error_by_bin=per_bin.tolist(), max_relative_error=maximum,
            scalar_values=a.numel(), nonzero_values=int(nonzero.sum())))
    summary = dict(old=str(old_path), new=str(new_path),
        old_overlay=str(old_overlay) if old_overlay is not None else None,
        old_module=old_payload.get("module", old_payload.get("__module")),
        new_module=new_payload.get("module", new_payload.get("__module")), quantities=len(report),
        scalar_values=sum(row["scalar_values"] for row in report),
        max_relative_error=max(row["max_relative_error"] for row in report), rows=report)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}), flush=True)
    assert summary["max_relative_error"] <= 1e-6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("tree", type=Path)
    cap.add_argument("output", type=Path)
    comp = commands.add_parser("compare")
    comp.add_argument("old", type=Path)
    comp.add_argument("new", type=Path)
    comp.add_argument("output", type=Path)
    comp.add_argument("--old-overlay", type=Path, help="Copper payload used by the graph gate")
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.tree, args.output)
    else:
        compare(args.old, args.new, args.output, args.old_overlay)


if __name__ == "__main__":
    main()
