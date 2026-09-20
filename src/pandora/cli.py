"""Console entry points; stdout is reserved for protocol responses in control mode."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from .config import load_config
from .experiment import METHODS, run_experiment
from .integration import JsonlSession, load_controller
from .offline import train_from_directory
from .plotting import plot_run
from .traces import import_sinr_csv


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pandora", description="Pandora federated contract experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "ablate", "sweep"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/smoke.yaml")
        command.add_argument("--output", required=True)
        command.add_argument("--seeds", nargs="+", type=int)
        command.add_argument(
            "--methods", nargs="+", choices=METHODS, default=["independent", "adaptive", "pandora"]
        )
        if name == "sweep":
            command.add_argument("--values", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    plot = sub.add_parser("plot")
    plot.add_argument("directory")
    validate = sub.add_parser("validate")
    validate.add_argument("--config", default="configs/paper.yaml")
    trace = sub.add_parser("import-trace")
    trace.add_argument("csv")
    trace.add_argument("--output", required=True)
    for name in ("slots", "users", "cells"):
        trace.add_argument(f"--{name}", required=True, type=int)
    trace.add_argument("--bandwidth-mhz", type=float, default=20)
    control = sub.add_parser("control")
    control.add_argument("--checkpoint", required=True)
    control.add_argument("--calibration", required=True)
    control.add_argument("--slices", required=True, help="Comma-separated UE slice IDs (0,1,2)")
    control.add_argument("--audit", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument(
        "--data", required=True, help="Directory with site-N/train-*.npz and calibration-*.npz"
    )
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            print(json.dumps(load_config(args.config).to_dict(), indent=2))
        elif args.command == "plot":
            print(plot_run(args.directory))
        elif args.command == "import-trace":
            capacity = import_sinr_csv(
                args.csv, args.slots, args.users, args.cells, args.bandwidth_mhz
            )
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, capacity_mbps=capacity)
            print(destination)
        elif args.command == "train":
            print(train_from_directory(load_config(args.config), args.data, args.output, args.seed))
        elif args.command == "control":
            controller = load_controller(
                args.checkpoint, args.calibration, [int(x) for x in args.slices.split(",")]
            )
            session = JsonlSession(controller, args.audit)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("Expected a JSON object")
                    response = session.handle(request)
                except (ValueError, TypeError, KeyError) as error:
                    response = {"version": 1, "error": str(error)}
                print(json.dumps(response, allow_nan=False), flush=True)
        else:
            config = load_config(args.config)
            if args.seeds is not None:
                config.seeds = args.seeds
            methods = args.methods
            if args.command == "ablate":
                methods = [m for m in METHODS if m.startswith("pandora")]
            if args.command == "sweep":
                if len(set(args.values)) != len(args.values):
                    raise ValueError("Sweep values must be unique")
                for value in args.values:
                    current = copy.deepcopy(config)
                    current.simulation.aggressiveness = value
                    destination = Path(args.output) / f"gamma-{value:g}"
                    run_experiment(current, destination, methods)
                    plot_run(destination)
            else:
                run_experiment(config, args.output, methods)
                print(plot_run(args.output))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"pandora: {error}\n")


if __name__ == "__main__":
    main()
