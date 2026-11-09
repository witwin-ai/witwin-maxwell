import argparse
import sys

from benchmark.runner import main
from benchmark.validation_catalog import inventory_markdown


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mom":
        selector = argparse.ArgumentParser(description="Surface solver acceptance workflows")
        selector.add_argument("workflow", choices=("convergence", "pilot", "cross", "algebra", "fields", "modes", "auxiliary", "volume-reference", "block", "matrix-free", "archive"))
        selected = selector.parse_args(sys.argv[2:3])
        rest = sys.argv[3:]
        if selected.workflow == "convergence":
            from benchmark.mom_convergence import main as mom_main
        elif selected.workflow == "pilot":
            from benchmark.mom_pilot import main as mom_main
        elif selected.workflow == "cross":
            from benchmark.mom_fdtd import main as mom_main
        elif selected.workflow == "algebra":
            from benchmark.mom_algebra_acceptance import main as mom_main
        elif selected.workflow == "block":
            from benchmark.mom_block_acceptance import main as mom_main
        elif selected.workflow == "matrix-free":
            from benchmark.mom_matrix_free_acceptance import main as mom_main
        elif selected.workflow == "fields":
            from benchmark.mom_field_convergence import main as mom_main
        elif selected.workflow == "modes":
            from benchmark.mom_modal_convergence import main as mom_main
        elif selected.workflow == "auxiliary":
            from benchmark.mom_auxiliary_acceptance import main as mom_main
        elif selected.workflow == "volume-reference":
            from benchmark.mom_volume_reference import main as mom_main
        else:
            from benchmark.mom_archive import main as mom_main
        sys.exit(mom_main(rest))

    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        from benchmark.batch import main as batch_main

        batch_main(sys.argv[2:])
        sys.exit(0)

    # `python -m benchmark rf [scenes...]` dispatches to the RF wave-level
    # validation harness; everything else keeps the field-vs-reference runner.
    if len(sys.argv) > 1 and sys.argv[1] == "rf":
        from benchmark.rf_validation import main as rf_main

        rf_main(sys.argv[2:])
        sys.exit(0)

    # `python -m benchmark sar [scenes...]` dispatches to the SAR phantom exposure
    # validation harness (conservation-law / analytic gates, no external solver).
    if len(sys.argv) > 1 and sys.argv[1] == "sar":
        from benchmark.sar_validation import main as sar_main

        sar_main(sys.argv[2:])
        sys.exit(0)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inventory", action="store_true")
    known, _ = parser.parse_known_args()
    if known.inventory:
        print(inventory_markdown())
    else:
        main()
