import argparse
import sys

from benchmark.runner import main
from benchmark.validation_catalog import inventory_markdown


if __name__ == "__main__":
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
