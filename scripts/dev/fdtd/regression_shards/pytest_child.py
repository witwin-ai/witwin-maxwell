"""Single-use pytest process. Completion is durable; abrupt exits stay abrupt."""
from pathlib import Path
import json
import os
import sys
import time
import traceback
from shard_common import runtime_identity, write_json

request = json.loads(Path(sys.argv[1]).read_text())
directory = Path(request["attempt_dir"])
os.chdir(request["source_root"])
sys.path.insert(0, request["source_root"])
sys.path.insert(0, request["control_root"])
os.environ["PCB_SHARD_REQUEST"] = str(Path(sys.argv[1]).resolve())
record = {"pid": os.getpid(), "ppid": os.getppid(), "sid": os.getsid(0),
          "started_unix_s": time.time(), "pytest_args": request["pytest_args"]}
try:
    import pytest
    record["pytest_exit_code"] = int(pytest.main(request["pytest_args"]))
except BaseException as error:
    record.update(pytest_exit_code=3, python_exception=repr(error), traceback=traceback.format_exc())
try:
    after = runtime_identity(request)
    write_json(directory / "identity-after.json", after)
    before_path = directory / "identity-before.json"
    before = json.loads(before_path.read_text()) if before_path.exists() else None
    record["identities_unchanged"] = bool(before and before["binding"] == after["binding"]
                                        and before["versions"] == after["versions"]
                                        and before.get("gsp_environment") == after.get("gsp_environment") == request["expected_gsp_environment"])
except BaseException as error:
    record.update(identities_unchanged=False, identity_exception=repr(error), identity_traceback=traceback.format_exc())
record["finished_unix_s"] = time.time()
write_json(directory / "process-result.json", record)
sys.exit(record["pytest_exit_code"] if record["identities_unchanged"] else 4)
