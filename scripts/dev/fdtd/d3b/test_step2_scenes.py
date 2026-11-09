"""Long-run copper gates; each variant has its own capped GPU-0 process."""
import sys
import json
import subprocess

import pytest
import torch

from scripts.dev.fdtd.d3b.step2_scenes import SCENES, compare


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("scenario", SCENES)
def test_e2_two_thousand_steps(scenario, tmp_path):
    for kernel in ("fused", "torch"):
        argv = ["step2_scenes", "snapshot", "--scenario", scenario,
                "--kernel", kernel, "--out", str(tmp_path / (kernel+".pt"))]
        code = ("import os,sys; "
                "import sysconfig; os.environ['CUDA_HOME']=os.path.join(sysconfig.get_paths()['purelib'],'nvidia','cu13'); "
                "os.environ['PATH']=os.environ['CUDA_HOME']+'/bin:'+os.environ['PATH']; "
                "os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; "
                "import runpy; sys.argv="+repr(argv)+"; "
                "runpy.run_module('scripts.dev.fdtd.d3b.step2_scenes',run_name='__main__')")
        subprocess.run(["env", "CUDA_VISIBLE_DEVICES=0",
                        sys.executable, "-c", code], check=True)
    from argparse import Namespace
    output = tmp_path / "metrics.json"
    compare(Namespace(fused=tmp_path/"fused.pt", reference=tmp_path/"torch.pt", out=output))
    metrics = json.loads(output.read_text())
    print(dict(E2=metrics), flush=True)
