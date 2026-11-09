"""Run the offline factorial cases against production binary pole compilation.

Use CUDA_VISIBLE_DEVICES=0 and the workstation's shared-GPU wrapper. The default
evidence directory supplies the original cases, Mie reference and calibrated
metric; arm 0 deliberately leaves the production material coefficients intact.
"""
import os
import argparse
import contextlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "coverage-dispersive-factorial")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--case", choices=("drude", "lorentz"), nargs="+", default=("drude", "lorentz"))
    parser.add_argument("--k", type=int, choices=(1, 2, 3), nargs="+", default=(1, 2, 3))
    args = parser.parse_args()
    tree = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(args.evidence.resolve()))
    sys.path.insert(0, str(tree))
    import factorial_run as runner
    import witwin.maxwell as mw

    @contextlib.contextmanager
    def cpu_material_compiler(_mw, enabled):
        # Retain the factorial's preparation-memory protocol, including the new
        # compiled per-edge weights. No material coefficients are replaced.
        from witwin.maxwell.scene import PreparedScene, prepare_scene
        original = PreparedScene.compile_materials
        def compile_on_cpu(scene, *positional, **keywords):
            model = original(prepare_scene(scene.clone(device="cpu")), *positional, **keywords)
            model["edge_components"] = runner.move(model["edge_components"], scene.device)
            for prefix in ("", "mu_"):
                for kind in ("debye", "drude", "lorentz"):
                    for entry in model[prefix+kind+"_poles"]:
                        entry["edge_weights"] = runner.move(entry["edge_weights"], scene.device)
            for key in ("pec_occupancy", "pec_volume_occupancy", "pec_sheet_edges", "pec_edge_fill"):
                if key in model:
                    model[key] = runner.move(model[key], scene.device)
            return model
        if enabled:
            PreparedScene.compile_materials = compile_on_cpu
        try:
            yield
        finally:
            PreparedScene.compile_materials = original
    runner.cpu_material_compiler = cpu_material_compiler
    expected = {"drude": (.4126, .3282, .1533), "lorentz": (.2047, .08023, .03944)}
    for k in args.k:
        for case in args.case:
            sys.argv = [str(args.evidence/"factorial_run.py"), str(tree), "0", case, str(k),
                        "--results", str(args.results.resolve()), "--force"]
            runner.main()
            record = json.loads((args.results/f"0_{case}_k{k}.json").read_text())
            assert record["status"] == "complete", record
            metric = record["metrics"]["material"]
            assert metric["metric_version"] == "vacuum-calibrated-v1"
            display = metric["field_l2"]["display"]
            print(f"{case} k={k}: display L2={display:.6g}; offline arm a={expected[case][k-1]:.6g}", flush=True)
            assert abs(display/expected[case][k-1]-1) < .15
            if case == "lorentz" and k == 3:
                # The measured arm-a scattering error is 3.269%; about 3%.
                assert max(abs(v) for v in metric["cross_relative_error"].values()) <= .04


if __name__ == "__main__":
    main()
