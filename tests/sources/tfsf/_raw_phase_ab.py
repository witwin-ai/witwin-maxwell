"""Fresh-process finite-record A/B spectra; CPU uses frozen test face kernels.

Run this same script with PYTHONPATH selecting each package tree, from outside
the repository. It records measurements without re-pinning any numerical gate.
"""
import argparse
from pathlib import Path
import sys

import torch
import witwin.maxwell as mw

# The selected Maxwell package is already imported; share its test oracles.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.fdtd.test_observer_origin_providers import execution
from witwin.maxwell.fdtd.runtime.spectral import source_normalization_spectrum


CASES = ("axis_pulse", "axis_cw", "special_pulse", "special_cw", "oblique_pulse",
         "slab_pulse", "slab_cw", "oblique_cw", "beam_cw", "astigmatic_cw", "beam_pulse")


def capture(args):
    torch.set_num_threads(1)
    cases = args.cases or ((*CASES, "grating_cw") if args.device == "cuda" else CASES)
    payload = {"module_file": mw.__file__, "steps": args.steps, "cases": {}}
    for case in cases:
        prepared, electric, magnetic = execution(case, device=args.device, steps=args.steps)
        if args.device == "cpu":
            from tests.fdtd.cuda import torch_reference
            torch_reference._require_cuda_tensors = lambda *values: None
            prepared.solver.fdtd_module = torch_reference.NativeFDTDModule()
        result = prepared.run()
        payload["cases"][case] = {
            "normalized": torch.stack([result.raw_monitor("point")["components"][field].cpu()
                                       for field in (electric, magnetic)]),
            "reference": source_normalization_spectrum(result.solver, (2e9,),
                steps_run=args.steps, window_type="none").cpu(),
        }
        print(case, payload["cases"][case]["normalized"], flush=True)
    torch.save(payload, args.output)


def compare(args):
    old = torch.load(args.baseline, weights_only=False)
    new = torch.load(args.output, weights_only=False)
    assert old["steps"] == new["steps"]
    assert old["cases"].keys() == new["cases"].keys()
    print("case,bitwise,max_relative_change,max_change_in_fp32_epsilon")
    for case in old["cases"]:
        a, b = (p["cases"][case]["normalized"] for p in (old, new))
        relative = ((a - b).abs() / a.abs()).max().item()
        print(f"{case},{torch.equal(a, b)},{relative:.12g},{relative / torch.finfo(torch.float32).eps:.12g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "compare"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--cases", nargs="+", choices=(*CASES, "grating_cw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    (capture if args.action == "capture" else compare)(args)
