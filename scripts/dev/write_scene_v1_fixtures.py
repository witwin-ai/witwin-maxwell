"""Write the witwin.scene/1 fixtures of tests/api/public/test_scene_migrate.py with the last /1 writer.

    python scripts/dev/write_scene_v1_fixtures.py <witwin-maxwell tree at 3c2631231>

The old tree's witwin.maxwell writes the documents; the builders come from this tree's
tests/api/public/scene_v1_fixtures.py. Output: tests/api/public/data/scene_v1/<name>.json, and
<name>.arrays.json + <name>.arrays.pt for documents with arrays-table references.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(OLD))
import torch  # noqa: E402
import witwin.maxwell as mw  # noqa: E402

assert Path(mw.__file__).resolve() == OLD / "witwin/maxwell/__init__.py", mw.__file__
spec = importlib.util.spec_from_file_location("scene_v1_fixtures", ROOT / "tests/api/public/scene_v1_fixtures.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
out = ROOT / "tests/api/public/data/scene_v1"
out.mkdir(parents=True, exist_ok=True)
for name, build in {**fixtures.EXACT, **fixtures.LOSSY}.items():
    scene = build()
    document = scene.to_dict()
    assert document["schema"] == "witwin.scene/1"
    (out / f"{name}.json").write_text(json.dumps(document, indent=1) + "\n")
    arrays = {}
    with_arrays = scene.to_dict(arrays=arrays)
    if arrays:
        (out / f"{name}.arrays.json").write_text(json.dumps(with_arrays, indent=1) + "\n")
        torch.save({key: value.detach().cpu() for key, value in arrays.items()}, out / f"{name}.arrays.pt")
    print(name, len(arrays))
