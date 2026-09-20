"""Exercise the real JSONL subprocess protocol with the local sandbox acting as a RAN.

Usage: python examples/external_loop.py --run runs/smoke --output runs/external-audit.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pandora.config import load_config
from pandora.simulator import AnalyticalRAN
from pandora.xapps import ReferenceXApps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    config = load_config(run / "config.json")
    seed = config.seeds[0]
    env = AnalyticalRAN(config.simulation, seed, 0, config.simulation.train_episodes + 1)
    apps = ReferenceXApps(env.space, config.simulation)
    local = run / f"seed-{seed}" / "site-0"
    command = [
        sys.executable,
        "-m",
        "pandora",
        "control",
        "--checkpoint",
        str(local / "pandora.pt"),
        "--calibration",
        str(local / f"calibration-{config.simulation.train_episodes}.npz"),
        "--slices",
        ",".join(map(str, env.slices)),
        "--audit",
        args.output,
    ]
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8"
    ) as process:
        feedback = None
        for slot in range(config.simulation.slots + 1):
            message = {"version": 1, "slot": slot, "feedback": feedback}
            if slot == config.simulation.slots:
                message["type"] = "finish"
            else:
                telemetry = env.observe()
                message.update(
                    observation=telemetry.vector(config.simulation.buffer_mbit).tolist(),
                    proposal=apps.propose(telemetry).tolist(),
                )
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            if "error" in response:
                raise RuntimeError(response["error"])
            if response.get("finished"):
                break
            executed = np.array(response["action"])
            feedback = {
                "slot": slot,
                "executed": executed.tolist(),
                "kpis": env.step(executed).tolist(),
            }
        process.stdin.close()
        if process.wait(timeout=30):
            raise RuntimeError("Policy subprocess failed")
    print(f"Recorded {config.simulation.slots} acknowledged transitions in {args.output}")


if __name__ == "__main__":
    main()
