from __future__ import annotations

import argparse
import subprocess
import tempfile
import zipfile
from pathlib import Path


EXPECTED_SASS = ("70", "75", "80", "86", "89", "90", "100", "101", "120")
EXPECTED_PTX_TARGET = "sm_120"


def _matches(path: Path, stems: tuple[str, ...]) -> bool:
    return path.suffix in {".pyd", ".so"} and any(path.name.startswith(stem) for stem in stems)


def _collect_binaries(inputs: list[Path], stems: tuple[str, ...], extract_root: Path) -> list[Path]:
    binaries: list[Path] = []
    for input_path in inputs:
        if input_path.suffix == ".whl":
            with zipfile.ZipFile(input_path) as wheel:
                for name in wheel.namelist():
                    member = Path(name)
                    if not _matches(member, stems):
                        continue
                    wheel.extract(name, extract_root)
                    binaries.append(extract_root / member)
            continue
        if input_path.is_dir():
            binaries.extend(path for path in input_path.rglob("*") if _matches(path, stems))
            continue
        if _matches(input_path, stems):
            binaries.append(input_path)
    return sorted(set(path.resolve() for path in binaries))


def _cuobjdump(flag: str, binary: Path) -> str:
    result = subprocess.run(
        ["cuobjdump", flag, str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return f"{result.stdout}\n{result.stderr}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CUDA SASS and PTX targets in release binaries.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--stem", action="append", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="witwin_cuda_arch_verify_") as temp_dir:
        binaries = _collect_binaries(args.inputs, tuple(args.stem), Path(temp_dir))
        if not binaries:
            raise SystemExit(f"No native binaries matching {args.stem!r} were found.")

        for binary in binaries:
            elf_listing = _cuobjdump("--list-elf", binary)
            missing_sass = [arch for arch in EXPECTED_SASS if f"sm_{arch}" not in elf_listing]
            if missing_sass:
                raise SystemExit(f"{binary} is missing SASS targets: {', '.join(missing_sass)}")

            ptx_dump = _cuobjdump("--dump-ptx", binary)
            if f".target {EXPECTED_PTX_TARGET}" not in ptx_dump:
                raise SystemExit(f"{binary} is missing compute 12.0 PTX.")

            print(f"Verified CUDA architectures in {binary}")


if __name__ == "__main__":
    main()
