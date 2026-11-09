"""CPU-only CLI and synthetic verdict controls; never execute a timing arm."""
import pytest

from scripts.dev.fdtd import guard_cost_ab as cost
from witwin.maxwell.fdtd.runtime.passivity import SAMPLE_INTERVAL


def test_defaults():
    args = cost.parse_args([])
    assert args.steps == 2048 and args.steps >= 8*SAMPLE_INTERVAL
    assert args.device_index == 0
    assert args.phase_timeout == 600 and args.phase_timeout_prepare == 1200 and args.memory_gib == 12
    assert args.scene == list(cost.SCENES)
    assert cost.REPEATS == 5


def test_overrides(tmp_path):
    args = cost.parse_args(["--steps", "512", "--device-index", "1", "--phase-timeout", "90",
                           "--phase-timeout-prepare", "240", "--memory-gib", "6", "--scene", "copper_patch",
                           "--json", str(tmp_path/"a.json")])
    assert (args.steps, args.device_index, args.phase_timeout, args.memory_gib) == (512, 1, 90, 6)
    assert args.phase_timeout_prepare == 240
    assert args.scene == ["copper_patch"] and args.json == tmp_path/"a.json"


@pytest.mark.parametrize("argv", [["--steps", "255"], ["--phase-timeout", "0"], ["--phase-timeout-prepare", "0"],
                                 ["--memory-gib", "0"], ["--device-index", "-1"]])
def test_invalid_arguments(argv):
    with pytest.raises(SystemExit):
        cost.parse_args(argv)


@pytest.mark.parametrize("fraction,expected", [(0., "PASS"), (.02, "PASS"), (.020001, "CHECK"),
                                             (.05, "CHECK"), (.050001, "FAIL")])
def test_verdict(fraction, expected):
    assert cost.verdict(fraction) == expected
