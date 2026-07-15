import argparse

from benchmark.runner import main
from benchmark.validation_catalog import inventory_markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inventory", action="store_true")
    known, _ = parser.parse_known_args()
    if known.inventory:
        print(inventory_markdown())
    else:
        main()
